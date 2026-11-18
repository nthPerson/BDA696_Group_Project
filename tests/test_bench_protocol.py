"""The Bench WebSocket protocol (Bench plan Task 8): message builders, the binary frame format
and command parsing. ``tests/golden/bench_messages.json`` pins one example of every server
message built from fixed inputs, plus one encoded frame as hex (``frame_hex``) that the pages'
JavaScript test decodes.

Regenerate the golden file after a deliberate protocol change (and bump
``protocol.SCHEMA_VERSION`` if pages written for the old one would break)::

    FORMCOACH_REGEN_GOLDEN=1 uv run pytest tests/test_bench_protocol.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from formcoach.app.events import Event
from formcoach.app.link import STATUS_KEYS, DeviceInfo
from formcoach.app.session import Snapshot
from formcoach.app.tuning import Param
from formcoach.bench import protocol
from formcoach.io.source import SISample

GOLDEN_PATH = Path(__file__).parent / "golden" / "bench_messages.json"
REGEN_ENV = "FORMCOACH_REGEN_GOLDEN"
FAKE_JPEG = b"\xff\xd8JPEGDATA\xff\xd9"  # fixed bytes, not a real encoder: the file stays stable
FRAME_KEYS = {
    "type", "i", "t", "t_mono", "gate", "gate_source", "gates", "energy", "pose_ran", "lm",
    "counters", "totals", "fps", "exercise", "link_state", "capture_state", "has_image",
}  # fmt: skip


def landmarks() -> np.ndarray:
    """Fixed ``(33, 4)`` image landmarks: x, y, z and visibility on simple ramps."""
    i = np.arange(33, dtype=np.float64)
    return np.stack([0.3 + i / 97.0, 0.2 + i / 61.0, np.zeros(33), 1.0 - i / 200.0], axis=1)


def snapshot(*, img: np.ndarray | None = None, energy: float | None = 0.8125) -> Snapshot:
    """A fixed curl snapshot (both arms counted, gate open from the watch)."""
    counter = {"angle": 95.5, "angle_raw": 96.25, "enter": 70.0, "exit": 140.0, "armed": True}
    return Snapshot(
        index=7,
        t=0.2333,
        t_mono=1000.25,
        frame=None,
        image_landmarks=img,
        world_landmarks=None,
        pose_ran=img is not None,
        gate=True,
        gate_source="device",
        gates={"device": True, "energy": False},
        energy=energy,
        angles={"elbow_l": 95.5},
        counters={"l": {**counter, "visibility": 0.98}, "r": {**counter, "visibility": 0.5}},
        totals={
            "l": {"reps": 2, "good": 1, "flagged": 1, "faults_by_code": {"CURL_TOO_FAST": 1}},
            "r": {"reps": 1, "good": 1, "flagged": 0, "faults_by_code": {}},
        },
        completed=[],
        fps=29.9712,
        exercise="curl",
        link_state="ok",
    )


def link_status() -> dict:
    status = dict.fromkeys(STATUS_KEYS)
    status.update(
        kind="ble", id="AA:BB:CC:DD:EE:FF", name="FormCoach-8428", kit="K2", state="ok",
        firmware="0.4.1", uptime_s=321, calibrated=True, rate_hz=50.0, samples=1600, dropped=2,
        bad_packets=0,
    )  # fmt: skip
    return status


def fixed_state() -> dict:
    param = Param("rule.curl.CURL_TOO_FAST.duration_s", "CURL_TOO_FAST: duration_s", "rules",
                  "s", 0.0, 2.5, 0.1, 1.0)  # fmt: skip
    return {
        "v": protocol.SCHEMA_VERSION,
        "exercise": "curl",
        "exercises": ["curl", "press", "raise", "squat", "idle"],
        "tuned_exercises": ["curl"],
        "gate_source": "device",
        "gate_sources": ["device", "energy", "always_on", "manual"],
        "manual_gate": False,
        "params": [asdict(param)],
        "values": {"rule.curl.CURL_TOO_FAST.duration_s": 0.8},
        "rules": "rules.mediapipe-webcam.yaml",
        "totals": snapshot().totals,
        "link": link_status(),
        "capture": {
            "state": "idle",
            "labels": {
                "subject": "S1",
                "exercise": "curl",
                "form": "clean",
                "arm": "both",
                "notes": "",
            },
            "summary": None,
            "last": {
                "dir": "data/team/S1/20261118-141500",
                "check": {"problems": []},
                "warnings": [],
            },
        },
        "video": {"on": True, "width": 640},
        "source": "fake",
        "forms": ["clean", "partial", "no_lockout", "bent_elbow", "shallow"],
        "arms": ["both", "l", "r"],
    }


def golden_messages() -> dict:
    """One message of every server type, built by :mod:`protocol` from fixed inputs, plus one
    encoded frame (hex) with :data:`FAKE_JPEG` as its image."""
    samples = [
        SISample(1.0, 0.1234567, -0.5, 9.81, 0.01, -0.02, 0.5, 3, 41, 1000.1),
        SISample(1.02, 0.2, -0.4, 9.7, 0.0, 0.0, 0.25, 1, 42, 1000.12),
    ]
    rep = Event(
        "rep", 3.5,
        {"arm": "l", "rep_id": 0, "t_start": 1.5, "duration_s": 2.0, "faults": [],
         "metrics": {"elbow_min": 52.25, "elbow_max": np.float64(151.5)}},
    )  # fmt: skip
    devices = [
        DeviceInfo("ble", "AA:BB:CC:DD:EE:FF", "FormCoach-8428", -48, "K2"),
        DeviceInfo("ble", "11:22:33:44:55:66", "FormCoach-8428", None, None),
    ]
    header = protocol.frame_header(snapshot(img=landmarks()), has_image=True, capture_state="idle")
    frame = protocol.encode_frame(header, FAKE_JPEG)
    return {
        "hello": protocol.hello(fixed_state()),
        "state": protocol.state(fixed_state()),
        "frame": protocol.frame_header(snapshot(img=landmarks()), has_image=False),
        "imu": protocol.imu(samples),
        "event": protocol.event(rep),
        "link": protocol.link(link_status()),
        "scan": protocol.scan("ble", devices),
        "log": protocol.log("warn", "link silent", 1002.5),
        "error": protocol.error("unknown parameter 'x'", "set_param"),
        "frame_hex": frame.hex(),
        "frame_jpeg_len": len(FAKE_JPEG),
    }


def test_frame_round_trips_header_and_jpeg():
    header = protocol.frame_header(
        snapshot(img=landmarks()), has_image=True, capture_state="review"
    )
    data = protocol.encode_frame(header, FAKE_JPEG)

    got, jpeg = protocol.decode_frame(data)
    assert jpeg == FAKE_JPEG
    assert got == json.loads(json.dumps(header))
    assert set(got) == FRAME_KEYS
    assert got["type"] == "frame" and got["has_image"] is True and got["capture_state"] == "review"
    assert got["i"] == 7 and got["t"] == 0.2333 and got["t_mono"] == 1000.25
    assert got["fps"] == 29.97 and got["gates"] == {"device": True, "energy": False}
    assert len(got["lm"]) == 33 and all(len(p) == 3 for p in got["lm"])
    assert got["lm"][5] == [round(0.3 + 5 / 97, 4), round(0.2 + 5 / 61, 4), 0.975]
    assert int.from_bytes(data[:4], "big") == len(data) - 4 - len(FAKE_JPEG)

    with pytest.raises(ValueError):
        protocol.decode_frame(data[:2])
    with pytest.raises(ValueError):
        protocol.decode_frame(data[:10])  # cut inside the header


def test_frame_without_image_round_trips():
    header = protocol.frame_header(snapshot(img=None, energy=float("nan")), has_image=False)
    data = protocol.encode_frame(header, None)

    got, jpeg = protocol.decode_frame(data)
    assert jpeg is None
    assert got["lm"] is None and got["pose_ran"] is False and got["has_image"] is False
    assert got["energy"] is None  # NaN is not JSON: sent as null
    assert got["capture_state"] == "idle"
    assert len(data) == 4 + int.from_bytes(data[:4], "big")


def test_golden_messages_are_stable():
    built = json.loads(json.dumps(golden_messages()))
    if os.environ.get(REGEN_ENV):
        GOLDEN_PATH.write_text(json.dumps(built, indent=1) + "\n", encoding="utf-8")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert built == golden
    types = {v["type"] for k, v in golden.items() if isinstance(v, dict) and "type" in v}
    assert types == {"hello", "state", "frame", "imu", "event", "link", "scan", "log", "error"}
    header, jpeg = protocol.decode_frame(bytes.fromhex(golden["frame_hex"]))
    assert jpeg == FAKE_JPEG and len(jpeg) == golden["frame_jpeg_len"]
    assert header["has_image"] is True and len(header["lm"]) == 33
    assert set(golden["hello"]["state"]) == set(protocol.STATE_KEYS)


def test_parse_command_rejects_bad_json_unknown_cmd_and_missing_fields():
    for text in ("{", "", "null", "[1, 2]", '"set_video"'):
        with pytest.raises(ValueError) as info:
            protocol.parse_command(text)
        assert info.value.cmd is None
    for text in ('{"cmd": "launch_rocket"}', '{"no_cmd": 1}', '{"cmd": 5}'):
        with pytest.raises(ValueError, match="unknown command"):
            protocol.parse_command(text)
    with pytest.raises(ValueError, match="set_param needs value") as info:
        protocol.parse_command('{"cmd": "set_param", "key": "seg.min_range_deg"}')
    assert info.value.cmd == "set_param"
    with pytest.raises(ValueError, match="labels, delay_s, auto_stop"):
        protocol.parse_command('{"cmd": "capture_start"}')

    # every command parses with its fields present (null is a value: connect to the first found)
    for name, fields in protocol.COMMANDS.items():
        cmd = {"cmd": name, **dict.fromkeys(fields)}
        assert protocol.parse_command(json.dumps(cmd)) == cmd
    with pytest.raises(ValueError):
        protocol.state({"v": 1})  # a state without its keys is refused, not sent


def test_protocol_module_imports_without_starlette_or_cv2():
    code = (
        "import sys\n"
        "for name in ('starlette', 'uvicorn', 'websockets', 'cv2'):\n"
        "    sys.modules[name] = None  # any import of them fails\n"
        "from formcoach.bench import protocol\n"
        'msg = protocol.parse_command(\'{"cmd": "reset_counters"}\')\n'
        "assert msg == {'cmd': 'reset_counters'}\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
