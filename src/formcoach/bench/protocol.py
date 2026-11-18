"""The Bench WebSocket protocol, version :data:`SCHEMA_VERSION` (Bench design §5.3).

One WebSocket (``/ws``) carries everything between the server and a page. This module builds
and parses the messages; it is pure (no Starlette, no OpenCV), so another front end can import
it, and ``tests/golden/bench_messages.json`` pins one example of every message.

**Server to page, text frames (JSON objects, each with a ``type``):**

- ``hello`` ``{v, state}``: the first message on every connection; ``state`` as below.
- ``state`` ``{state}``: the whole state again, after anything in it changed. ``state`` holds
  the keys of :data:`STATE_KEYS`: ``v``, ``exercise``, ``exercises``, ``tuned_exercises``,
  ``gate_source``, ``gate_sources``, ``manual_gate``, ``params`` (sliders of the current
  exercise, :class:`formcoach.app.tuning.Param` as dicts), ``values`` (overrides in force),
  ``rules`` (rules file name), ``totals``, ``link`` (``LinkManager.status()``), ``capture``
  (``{state, labels, summary, last}``), ``video`` (``{on, width}``), ``source`` (``"camera 0"``,
  ``"video x.mp4"``, ``"fake"`` or ``"ended"``), ``forms``, ``arms``.
- ``imu`` ``{t, ax, ay, az, gx, gy, gz, flags, seq}``: one batch of wearable samples as
  parallel lists; ``t`` is the laptop arrival time (s, session clock), accelerations m/s²,
  rates rad/s.
- ``event`` ``{kind, t, payload}``: a session event (``rep``, ``fault``, ``gate_open``,
  ``capture``, ``params_changed``, ...); ``t`` is seconds on the frame clock.
- ``link`` ``{status}``: watch link health (``LinkManager.status()``), about twice a second
  and whenever it changes.
- ``scan`` ``{kind, devices}``: scan results, ``devices`` a list of ``{kind, id, name, rssi,
  kit}``.
- ``log`` ``{level, text, t}``: a link log line (``info``/``warn``/``error``; ``t`` session
  clock, s).
- ``error`` ``{text, cmd}``: a command failed or was malformed (``cmd`` its name, or ``null``).

**Server to page, binary frames:** one video frame, :func:`encode_frame`: a 4-byte big-endian
header length, the UTF-8 JSON header (:func:`frame_header`, ``type`` ``"frame"``), then the
JPEG bytes, or nothing when ``has_image`` is false. Every frame carries its landmarks.

**Page to server:** text frames ``{"cmd": name, ...fields}``; :data:`COMMANDS` maps each name to
its required fields. Optional fields: ``save_rules`` takes ``overwrite`` (default false). Its
``path`` must be relative to the directory the bench was started in, stay inside it and end in
``.yaml``/``.yml``; an existing file is replaced only with ``overwrite: true``. Success is an
``event`` of kind ``rules_saved`` ``{path (relative, /-separated), overwritten}``.
Non-finite floats are sent as ``null`` (JSON has no NaN).
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from formcoach.app.events import Event
    from formcoach.app.link import DeviceInfo
    from formcoach.app.session import Snapshot
    from formcoach.io.source import SISample

SCHEMA_VERSION = 1
LM_DECIMALS = 4  # landmark x, y and visibility (normalised units)
IMU_DECIMALS = 4  # IMU time (s), acceleration (m/s²) and rate (rad/s)

STATE_KEYS = (
    "v",
    "exercise",
    "exercises",
    "tuned_exercises",
    "gate_source",
    "gate_sources",
    "manual_gate",
    "params",
    "values",
    "rules",
    "totals",
    "link",
    "capture",
    "video",
    "source",
    "forms",
    "arms",
)

# command -> required fields (optional ones are listed in the module doc)
COMMANDS: dict[str, tuple[str, ...]] = {
    "scan": ("kind",),
    "connect": ("kind", "id"),
    "disconnect": (),
    "device": ("action",),
    "label_kit": ("id", "kit"),
    "set_exercise": ("exercise",),
    "set_gate_source": ("source",),
    "set_manual_gate": ("open",),
    "set_param": ("key", "value"),
    "reset_params": (),
    "save_rules": ("path",),  # optional: overwrite
    "reset_counters": (),
    "set_video": ("on",),
    "capture_start": ("labels", "delay_s", "auto_stop"),
    "capture_stop": (),
    "capture_keep": ("true_reps",),
    "capture_discard": (),
    "cue_shown": ("arm", "rep_id"),
}

_HEADER_LEN = struct.Struct(">I")


class CommandError(ValueError):
    """A page sent something that is not a valid command; ``cmd`` is its name when known."""

    def __init__(self, text: str, cmd: str | None = None):
        super().__init__(text)
        self.cmd = cmd


def jsonable(obj):
    """``obj`` as plain JSON types: NumPy scalars and arrays to Python numbers and lists,
    non-finite floats to ``None``, paths and other objects to strings, dict keys to strings."""
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def dumps(msg: dict) -> str:
    """A message as compact JSON text (non-finite floats become ``null``)."""
    try:
        return json.dumps(msg, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(jsonable(msg), separators=(",", ":"), allow_nan=False)


def _check_state(state: dict) -> dict:
    missing = [k for k in STATE_KEYS if k not in state]
    if missing:
        raise ValueError(f"state is missing {', '.join(missing)}")
    return state


def hello(state: dict) -> dict:
    """The first message on a connection: ``{"type": "hello", "v": 1, "state": state}``.

    Raises:
        ValueError: ``state`` lacks a key of :data:`STATE_KEYS`.
    """
    return {"type": "hello", "v": SCHEMA_VERSION, "state": jsonable(_check_state(state))}


def state(state: dict) -> dict:
    """``{"type": "state", "state": state}``: the whole state, after it changed.

    Raises:
        ValueError: ``state`` lacks a key of :data:`STATE_KEYS`.
    """
    return {"type": "state", "state": jsonable(_check_state(state))}


def _landmarks(img) -> list | None:
    if img is None:
        return None
    a = np.asarray(img, dtype=np.float64)
    out = []
    for x, y, vis in zip(a[:, 0], a[:, 1], a[:, 3], strict=True):
        out.append([_round(x, LM_DECIMALS), _round(y, LM_DECIMALS), _round(vis, LM_DECIMALS)])
    return out


def _round(x, nd: int) -> float | None:
    f = float(x)
    return round(f, nd) if math.isfinite(f) else None


def frame_header(snap: Snapshot, *, has_image: bool, capture_state: str = "idle") -> dict:
    """The JSON header of one video frame from a :class:`~formcoach.app.session.Snapshot`.

    Fields: ``i`` (frame index), ``t`` (frame clock, s), ``t_mono`` (session clock, s),
    ``gate``, ``gate_source``, ``gates``, ``energy`` ((m/s²)² or ``null``), ``pose_ran``,
    ``lm`` (33 × ``[x, y, visibility]`` image landmarks, normalised, 4 decimals; ``null`` when
    pose did not run or found nobody), ``counters``, ``totals``, ``fps``, ``exercise``,
    ``link_state``, ``capture_state`` (the session's, passed in) and ``has_image``.
    """
    return {
        "type": "frame",
        "i": int(snap.index),
        "t": jsonable(snap.t),
        "t_mono": jsonable(snap.t_mono),
        "gate": bool(snap.gate),
        "gate_source": snap.gate_source,
        "gates": jsonable(snap.gates),
        "energy": jsonable(snap.energy),
        "pose_ran": bool(snap.pose_ran),
        "lm": _landmarks(snap.image_landmarks),
        "counters": jsonable(snap.counters),
        "totals": jsonable(snap.totals),
        "fps": jsonable(round(float(snap.fps), 2)),
        "exercise": snap.exercise,
        "link_state": snap.link_state,
        "capture_state": capture_state,
        "has_image": bool(has_image),
    }


def encode_frame(header: dict, jpeg: bytes | None) -> bytes:
    """One binary frame: the header's UTF-8 JSON length as 4 big-endian bytes, the JSON, then
    ``jpeg`` (nothing when ``None``)."""
    head = dumps(header).encode("utf-8")
    return _HEADER_LEN.pack(len(head)) + head + (jpeg or b"")


def decode_frame(data: bytes) -> tuple[dict, bytes | None]:
    """Split a frame made by :func:`encode_frame` into its header and JPEG bytes (``None`` when
    there are none).

    Raises:
        ValueError: ``data`` is too short for its header length, or the header is not JSON.
    """
    if len(data) < _HEADER_LEN.size:
        raise ValueError(f"frame of {len(data)} bytes has no header length")
    (n,) = _HEADER_LEN.unpack_from(data)
    end = _HEADER_LEN.size + n
    if len(data) < end:
        raise ValueError(f"frame header says {n} bytes but only {len(data) - 4} follow")
    try:
        header = json.loads(data[_HEADER_LEN.size : end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"frame header is not JSON: {exc}") from None
    rest = data[end:]
    return header, (bytes(rest) if rest else None)


def imu(samples: Sequence[SISample]) -> dict:
    """One batch of wearable samples as parallel lists: ``t`` (laptop arrival time, s, ``null``
    when a source did not stamp it), ``ax``/``ay``/``az`` (m/s²), ``gx``/``gy``/``gz`` (rad/s),
    ``flags`` and ``seq``."""

    def col(name: str) -> list:
        return [_round(getattr(s, name), IMU_DECIMALS) for s in samples]

    return {
        "type": "imu",
        "t": [None if s.t_arrival is None else _round(s.t_arrival, IMU_DECIMALS) for s in samples],
        **{name: col(name) for name in ("ax", "ay", "az", "gx", "gy", "gz")},
        "flags": [int(s.flags) for s in samples],
        "seq": [int(s.seq) for s in samples],
    }


def event(e: Event) -> dict:
    """A session event: ``{"type": "event", "kind", "t" (s, frame clock), "payload"}``."""
    return {"type": "event", "kind": e.kind, "t": jsonable(e.t), "payload": jsonable(e.payload)}


def link(status: dict) -> dict:
    """Watch link health: ``{"type": "link", "status": LinkManager.status()}``."""
    return {"type": "link", "status": jsonable(status)}


def scan(kind: str, devices: Sequence[DeviceInfo]) -> dict:
    """Scan results: ``{"type": "scan", "kind", "devices": [{kind, id, name, rssi, kit}]}``."""
    return {
        "type": "scan",
        "kind": kind,
        "devices": [
            {"kind": d.kind, "id": d.id, "name": d.name, "rssi": d.rssi, "kit": d.kit}
            for d in devices
        ],
    }


def log(level: str, text: str, t: float) -> dict:
    """A link log line: ``level`` ``info``/``warn``/``error``, ``t`` session clock (s)."""
    return {"type": "log", "level": level, "text": text, "t": jsonable(t)}


def error(text: str, cmd: str | None = None) -> dict:
    """A failed or malformed command: ``{"type": "error", "text", "cmd"}``."""
    return {"type": "error", "text": text, "cmd": cmd}


def parse_command(text: str) -> dict:
    """Parse one command from a page: a JSON object whose ``cmd`` is a key of :data:`COMMANDS`
    and which has every field that command requires. Returns the object.

    Raises:
        CommandError (a ``ValueError``): not JSON, not an object, an unknown ``cmd`` or a missing
            field; ``.cmd`` is the command's name when it was readable.
    """
    try:
        msg = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise CommandError(f"command is not valid JSON: {exc}") from None
    if not isinstance(msg, dict):
        raise CommandError("a command must be a JSON object with a 'cmd' field")
    cmd = msg.get("cmd")
    if not isinstance(cmd, str) or cmd not in COMMANDS:
        name = cmd if isinstance(cmd, str) else None
        raise CommandError(f"unknown command {cmd!r}; known: {', '.join(COMMANDS)}", name)
    missing = [f for f in COMMANDS[cmd] if f not in msg]
    if missing:
        raise CommandError(f"{cmd} needs {', '.join(missing)}", cmd)
    return msg
