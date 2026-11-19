"""``Controller``: page commands in, session output out (Bench design §5.3).

The controller subscribes to one :class:`~formcoach.app.session.LiveSession` and turns its
snapshots, IMU batches and events into protocol messages on a :class:`~.hub.Hub`, and turns the
pages' commands into session calls. It also joins the watch link (:class:`LinkManager`) to the
session: after a connect the session gets the new board, before a disconnect it lets go of it.

Threads:

- **Session thread.** The snapshot, IMU and event listeners run there, and every command that
  touches the session is queued with :meth:`LiveSession.call` and runs there between frames.
  Video frames are JPEG-encoded there (at most ``video_max_fps`` per second on the session
  clock, and only while a page is connected), and link status goes out about twice a second
  from there.
- **Link worker.** ``scan``, ``connect``, ``disconnect``, ``device`` and ``label_kit`` can block
  for seconds on the radio or the disk, so they run in order on one worker thread.
- **Caller (the server's loop).** :meth:`Controller.handle` only checks the command and queues
  the work; it returns at once and never raises: every failure goes out as an ``error``
  message naming the command, followed (for session commands) by a fresh ``state``.

When the frame source ends (:meth:`Controller.source_ended`) the session thread stops stepping;
session commands then run directly on the caller's thread, one at a time, so a take in review
can still be kept, and a timer thread takes over the link status (about twice a second) until
:meth:`Controller.close`.

A watch change (``connect``, ``disconnect``) is refused while a take records (the session's
rule): the link worker asks the session first and leaves the link alone, sending an ``error``.
"""

from __future__ import annotations

import copy
import itertools
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath, PureWindowsPath

from formcoach.app.capture import ARMS, FORMS, TakeLabels
from formcoach.app.events import Event
from formcoach.app.link import DeviceInfo, LinkManager
from formcoach.app.session import EXERCISES, GATE_SOURCES, LiveSession, Snapshot
from formcoach.bench import protocol
from formcoach.bench.hub import Hub
from formcoach.io.protocol import Command

TUNED_EXERCISES = ("curl",)  # exercises whose webcam rule thresholds have been tuned (ADR-0033)
LINK_PERIOD_S = 0.5  # link status goes out this often (session clock, s)
BOARD_WAIT_S = 2.0  # how long a link job waits for the session to let go of the board (s)
DEVICE_ACTIONS: dict[str, Command | None] = {
    "led_good": Command.LED_GOOD_REP,
    "led_fault": Command.LED_FAULT,
    "calibrate": Command.CALIBRATE,
    "identify": None,  # LinkManager.identify
}
LINK_COMMANDS = frozenset({"scan", "connect", "disconnect", "device", "label_kit"})
# session events after which the state is rebuilt and published
STATE_EVENTS = frozenset(
    {"exercise", "gate_source", "params_changed", "counters_reset", "capture", "link"}
)


def _describe(exc: BaseException) -> str:
    """A one-line message for a failed command: the text alone for the errors session methods
    raise on bad input or the wrong state, else the exception type and text."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    if isinstance(exc, ValueError | RuntimeError | OSError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _as_bool(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false, got {value!r}")
    return value


class Controller:
    """Joins a :class:`LiveSession`, a :class:`LinkManager` and a :class:`Hub` (see the module
    doc). Subscribes to the session and takes over ``links.on_log`` on construction.

    ``video_width`` is the width (px) frames are resized to before JPEG encoding at
    ``jpeg_quality`` (0-100), at most ``video_max_fps`` images per second; ``team_root`` is
    where kept takes are written; ``clock`` is the session's clock (s), read when a
    ``cue_shown`` arrives and to stamp log lines. Construct it before the session thread starts.
    """

    def __init__(
        self,
        session: LiveSession,
        links: LinkManager,
        hub: Hub,
        *,
        video_width: int,
        jpeg_quality: int,
        video_max_fps: float,
        team_root: Path,
        rules_dir_base: Path,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.session = session
        self.links = links
        self.hub = hub
        self.video_width = int(video_width)
        self.jpeg_quality = int(jpeg_quality)
        self.video_max_fps = float(video_max_fps)
        self.team_root = Path(team_root)
        self.rules_dir_base = Path(rules_dir_base)
        self.clock = clock

        self._lock = threading.Lock()  # guards the cached state and the fields state() reads
        self._source = session.capture_context.get("frame_source") or "unknown"
        self._context_link: tuple | None = None  # the link (kind, id) _sync_context last saw
        self._totals: dict = session.pipeline.arm_totals() if session.pipeline else {}
        self._manual_gate = False
        self._video_on = True
        self._capture: dict = {"state": "idle", "labels": None, "summary": None, "last": None}
        self._scans: dict[str, list[DeviceInfo]] = {}
        self._cv2 = None  # the cv2 module once imported; False when OpenCV is not installed
        self._next_image = float("-inf")  # session clock (s) from which the next image may go
        self._last_link: float | None = None
        self._in_command = False  # a session command is running (session thread)

        self._call_lock = threading.Lock()  # guards _ended and _pending
        self._direct = threading.RLock()  # serialises session work after the source ended
        self._ended = False
        self._pending: dict[int, Callable[[LiveSession], None]] = {}
        self._ids = itertools.count()
        # link jobs run in order on one daemon thread (a radio call stuck at exit must not
        # hold the process); None stops it
        self._jobs: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._closed = False
        self._stop_timer = threading.Event()  # stops the link status timer (after source end)
        self._worker = threading.Thread(target=self._work, name="bench-link", daemon=True)
        self._worker.start()

        self._cached = self._build_state()
        links.on_log = self._log_line
        self._unsubscribe = session.subscribe(
            on_snapshot=self._on_snapshot, on_imu=self._on_imu, on_event=self._on_event
        )

    # ---- state ---------------------------------------------------------------------------------

    def state(self) -> dict:
        """Everything a new page needs (the keys of :data:`protocol.STATE_KEYS`): the state as
        of the last change, with the latest totals, the link status read now and the source.
        Any thread."""
        link = self.links.status()
        with self._lock:
            out = dict(self._cached)
            out["totals"] = self._totals
            out["source"] = self._source
        out["link"] = link
        return out

    def _params(self, exercise: str) -> list[dict]:
        overrides = self.session.overrides
        if overrides is None:
            return []
        try:
            params = overrides.params(exercise)
        except KeyError:  # idle, or an exercise the rules file lacks: gate and counter only
            known = overrides.base.exercises
            params = [p for p in overrides.params(known[0]) if p.group != "rules"] if known else []
        return [asdict(p) for p in params]

    def _build_state(self) -> dict:
        """The state from the session (session thread, or when no session thread runs)."""
        s = self.session
        overrides = s.overrides
        capture = copy.deepcopy(self._capture)
        capture["state"] = s.capture_state
        return {
            "v": protocol.SCHEMA_VERSION,
            "exercise": s.exercise,
            "exercises": list(EXERCISES),
            "tuned_exercises": list(TUNED_EXERCISES),
            "gate_source": s.gate_source,
            "gate_sources": list(GATE_SOURCES),
            "manual_gate": self._manual_gate,
            "params": self._params(s.exercise),
            "values": overrides.values if overrides is not None else {},
            "rules": s.rules_name,
            "totals": s.pipeline.arm_totals() if s.pipeline is not None else {},
            "link": None,  # read in state()
            "capture": capture,
            "video": {"on": self._video_on, "width": self.video_width},
            "source": None,  # set in state()
            "forms": list(FORMS),
            "arms": list(ARMS),
        }

    def _refresh(self) -> None:
        """Rebuild the cached state and publish it (session thread, or after the source
        ended)."""
        built = self._build_state()
        with self._lock:
            self._cached = built
            self._totals = built["totals"]
        self.hub.publish(protocol.state(self.state()))

    # ---- session output ------------------------------------------------------------------------

    def _on_snapshot(self, snap: Snapshot) -> None:
        with self._lock:
            self._totals = snap.totals
        if self._last_link is None or snap.t_mono - self._last_link >= LINK_PERIOD_S:
            self._last_link = snap.t_mono
            self._sync_context(self._publish_link())
        if self.hub.client_count() == 0:
            return  # nobody is watching: no encoding
        jpeg = self._encode(snap)
        header = protocol.frame_header(
            snap, has_image=jpeg is not None, capture_state=self.session.capture_state
        )
        self.hub.publish_frame(protocol.encode_frame(header, jpeg))

    def _encode(self, snap: Snapshot) -> bytes | None:
        """The frame as JPEG bytes, resized to ``video_width``; ``None`` while video is off,
        between images, without a frame or without OpenCV.

        Images are due every ``1 / video_max_fps`` seconds of session clock; a frame up to a
        quarter period early still counts (camera timing jitters), so a source at exactly
        ``video_max_fps`` sends every frame and a faster one is thinned evenly."""
        frame = snap.frame
        period = 1.0 / self.video_max_fps
        if not self._video_on or frame is None or snap.t_mono < self._next_image - period / 4:
            return None
        cv2 = self._opencv()
        if cv2 is None:
            return None
        h, w = frame.shape[:2]
        if w != self.video_width and w > 0:
            size = (self.video_width, max(1, round(h * self.video_width / w)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None
        due = self._next_image
        # on schedule: the next image is one period after this one was due; after a gap (or
        # the first image) one period from now
        self._next_image = due + period if snap.t_mono - due < period else snap.t_mono + period
        return buf.tobytes()

    def _opencv(self):
        if self._cv2 is None:
            try:
                import cv2
            except ImportError:
                cv2 = False
            self._cv2 = cv2
        return self._cv2 or None

    def _on_imu(self, samples: list) -> None:
        self.hub.publish(protocol.imu(samples))

    def _on_event(self, e: Event) -> None:
        self.hub.publish(protocol.event(e))
        if e.kind == "capture":
            self._note_capture(e.payload)
        if e.kind in STATE_EVENTS and not self._in_command:
            self._refresh()

    def _note_capture(self, payload: dict) -> None:
        c = self._capture
        if "labels" in payload:
            c["labels"] = payload["labels"]
        if "summary" in payload:
            c["summary"] = payload["summary"]
        if payload.get("state") == "idle":
            c["summary"] = None
            if "dir" in payload:
                c["last"] = {k: payload.get(k) for k in ("dir", "check", "warnings")}

    def _log_line(self, level: str, text: str) -> None:
        self.hub.publish(protocol.log(level, text, self.clock()))

    def _publish_link(self) -> dict:
        status = self.links.status()
        self.hub.publish(protocol.link(status))
        return status

    def _sync_context(self, status: dict) -> None:
        """Record the link in ``session.capture_context`` (session thread): ``source`` (the
        link's kind, ``"none"`` without one), ``device`` (its identifier, or its name when it
        was connected as "first found"), ``firmware`` and ``kit``. While the link stays the same
        (same kind and identifier) a value it does not know yet, such as the firmware before the
        watch's status packet, never erases one it knew; a new link or no link replaces them."""
        ctx = self.session.capture_context
        if status["state"] == "none":
            ctx.update(source="none", device=None, firmware=None, kit=None)
            self._context_link = None
            return
        link = (status["kind"], status["id"])
        known = {"device": status["id"] or status["name"], "firmware": status["firmware"]}
        if link == self._context_link:
            known = {k: v for k, v in known.items() if v is not None}
        self._context_link = link
        ctx.update(source=status["kind"], kit=status["kit"], **known)

    # ---- running work on the session ------------------------------------------------------------

    def _call(self, fn: Callable[[LiveSession], None]) -> None:
        """Run ``fn(session)`` on the session thread, or here and now (one at a time) once the
        source has ended."""
        with self._call_lock:
            if not self._ended:
                key = next(self._ids)
                self._pending[key] = fn
                self.session.call(lambda _s, key=key: self._run_pending(key))
                return
        with self._direct:
            fn(self.session)

    def _run_pending(self, key: int) -> None:
        with self._call_lock:
            fn = self._pending.pop(key, None)
        if fn is not None:
            fn(self.session)

    def _call_and_wait(self, fn: Callable[[LiveSession], None], timeout: float) -> None:
        done = threading.Event()

        def run(session: LiveSession) -> None:
            try:
                fn(session)
            finally:
                done.set()

        self._call(run)
        done.wait(timeout)

    def _session_command(self, name: str, fn: Callable[[LiveSession], None]) -> None:
        """Queue ``fn`` as command ``name``: a failure becomes an ``error`` message; a fresh
        ``state`` is published afterwards either way."""

        def run(session: LiveSession) -> None:
            self._in_command = True
            try:
                fn(session)
            except Exception as exc:  # a failing command must never raise in the session
                self.hub.publish(protocol.error(_describe(exc), name))
            finally:
                self._in_command = False
                self._refresh()

        self._call(run)

    def source_ended(self, error: str | None = None) -> None:
        """The frame source ran out (or failed with ``error``): call on the session thread
        after its loop returned. Commands queued but not run are run now; from here on session
        commands run directly. ``state.source`` becomes ``"ended"``. Link status keeps going
        out every :data:`LINK_PERIOD_S` seconds (wall clock) from a timer thread until
        :meth:`close`, so the watch's health stays live without frames."""
        with self._call_lock:
            self._ended = True
            leftovers = list(self._pending.values())
            self._pending.clear()
        with self._direct:
            for fn in leftovers:
                fn(self.session)
            with self._lock:
                self._source = "ended"
            if error is not None:
                self.hub.publish(protocol.error(f"the frame source stopped: {error}"))
            self._refresh()
        if not self._closed:
            threading.Thread(target=self._link_timer, name="bench-link-status", daemon=True).start()

    def _link_timer(self) -> None:
        """Publish the link status about twice a second until :meth:`close` (after the source
        ended, when no snapshot does it)."""
        while not self._stop_timer.wait(LINK_PERIOD_S):
            with self._direct:
                self._sync_context(self._publish_link())

    def close(self) -> None:
        """Stop listening to the session, stop the link status timer and stop the link worker
        after its current job; link commands are refused from here on."""
        self._unsubscribe()
        self._closed = True
        self._stop_timer.set()
        self._jobs.put(None)

    def _work(self) -> None:
        while (job := self._jobs.get()) is not None:
            if not self._closed:
                job()

    # ---- commands -------------------------------------------------------------------------------

    def handle(self, cmd: dict) -> None:
        """Act on one parsed command (:func:`protocol.parse_command`). Returns at once: session
        work is queued for the session thread, link work for the link worker. Never raises;
        errors go out as ``error`` messages."""
        name = cmd.get("cmd") if isinstance(cmd, dict) else None
        try:
            if name not in protocol.COMMANDS:
                raise protocol.CommandError(f"unknown command {name!r}")
            missing = [f for f in protocol.COMMANDS[name] if f not in cmd]
            if missing:
                raise protocol.CommandError(f"{name} needs {', '.join(missing)}", name)
            if name in LINK_COMMANDS:
                self._link_command(name, cmd)
            else:
                self._command(name, cmd)
        except Exception as exc:
            self.hub.publish(protocol.error(_describe(exc), name))

    def _command(self, name: str, cmd: dict) -> None:
        if name == "cue_shown":
            t = self.clock()
            arm, rep_id = cmd["arm"], int(cmd["rep_id"])
            self._call(lambda s: self._quietly(name, lambda: s.note_cue_shown(arm, rep_id, t)))
            return
        self._session_command(name, self._session_action(name, cmd))

    def _quietly(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            self.hub.publish(protocol.error(_describe(exc), name))

    def _session_action(self, name: str, cmd: dict) -> Callable[[LiveSession], None]:
        """The session-thread function of a session command (checked fields; errors raised
        inside it are reported by :meth:`_session_command`)."""
        if name == "set_exercise":
            exercise = cmd["exercise"]
            return lambda s: s.set_exercise(exercise)
        if name == "set_gate_source":
            source = cmd["source"]
            return lambda s: s.set_gate_source(source)
        if name == "set_manual_gate":
            open_ = _as_bool(cmd["open"], "open")

            def manual(s: LiveSession) -> None:
                s.set_manual_gate(open_)
                self._manual_gate = open_

            return manual
        if name == "set_param":
            key, value = cmd["key"], cmd["value"]
            return lambda s: s.set_param(key, value)
        if name == "reset_params":
            return lambda s: s.reset_params()
        if name == "reset_counters":
            return lambda s: s.reset_counters()
        if name == "save_rules":
            path, overwrite = cmd["path"], _as_bool(cmd.get("overwrite", False), "overwrite")
            return lambda s: self._save_rules(s, path, overwrite)
        if name == "set_video":
            on = _as_bool(cmd["on"], "on")

            def video(_s: LiveSession) -> None:
                self._video_on = on
                self._next_image = float("-inf")

            return video
        if name == "capture_start":
            labels, auto_stop = cmd["labels"], _as_bool(cmd["auto_stop"], "auto_stop")
            if not isinstance(labels, dict):
                raise ValueError("labels must be an object {subject, exercise, form, arm, notes}")
            delay_s = float(cmd["delay_s"])

            def start(s: LiveSession) -> None:
                take = TakeLabels(**labels)
                s.start_capture(take, delay_s=delay_s, auto_stop=auto_stop)
                self._capture["labels"] = asdict(take)

            return start
        if name == "capture_stop":
            return lambda s: s.stop_capture()
        if name == "capture_keep":
            true_reps = cmd["true_reps"]
            if true_reps is not None and (
                isinstance(true_reps, bool) or not isinstance(true_reps, int) or true_reps < 0
            ):
                raise ValueError(
                    f"true_reps must be a whole number >= 0 or null, got {true_reps!r}"
                )

            def keep(s: LiveSession) -> None:
                self._capture["last"] = s.keep_capture(true_reps=true_reps, root=self.team_root)

            return keep
        if name == "capture_discard":
            return lambda s: s.discard_capture()
        raise protocol.CommandError(f"unknown command {name!r}")  # pragma: no cover

    def _rules_target(self, path: str, overwrite: bool) -> tuple[Path, str]:
        """Where ``save_rules`` may write ``path``: the resolved file inside
        :attr:`rules_dir_base` and its path relative to it (``/``-separated).

        Raises:
            ValueError: ``path`` is not text, is absolute or has a drive, leaves the base
                directory (``..``, or a symlink pointing out), does not end in ``.yaml`` or
                ``.yml``, or names an existing file and ``overwrite`` is false.
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("save_rules needs a file name such as tuned.yaml")
        if PureWindowsPath(path).anchor or PurePosixPath(path).anchor:
            raise ValueError(
                f"{path!r} is not allowed: give a path relative to the start directory"
            )
        if Path(path).suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"{path!r} is not allowed: a rules file ends in .yaml or .yml")
        base = self.rules_dir_base.resolve()
        target = (base / path).resolve()
        if base not in target.parents:
            raise ValueError(f"{path!r} is not allowed: it leaves the start directory {base}")
        if target.exists() and not overwrite:
            raise ValueError(f"{path!r} already exists; send overwrite: true to replace it")
        return target, target.relative_to(base).as_posix()

    def _save_rules(self, s: LiveSession, path: str, overwrite: bool) -> None:
        if s.overrides is None:
            raise RuntimeError("this session has no rules, so there is nothing to save")
        target, rel = self._rules_target(path, overwrite)
        existed = target.exists()
        s.overrides.save(target)  # still refuses the packaged rules directory
        s.emit("rules_saved", path=rel, overwritten=existed)

    # ---- the watch link (link worker) -----------------------------------------------------------

    def _link_command(self, name: str, cmd: dict) -> None:
        if name == "scan":
            kind = cmd["kind"]
            job = lambda: self._scan(kind)  # noqa: E731
        elif name == "connect":
            kind, ident = cmd["kind"], cmd["id"]
            if kind not in self.links.factories:  # refused before the session lets go of a board
                raise ValueError(
                    f"unknown link kind {kind!r}; choose {', '.join(self.links.factories)}"
                )
            job = lambda: self._connect(kind, ident)  # noqa: E731
        elif name == "disconnect":
            job = self._disconnect
        elif name == "device":
            action = cmd["action"]
            if action not in DEVICE_ACTIONS:
                raise ValueError(
                    f"unknown device action {action!r}; choose {', '.join(DEVICE_ACTIONS)}"
                )
            job = lambda: self._device(action)  # noqa: E731
        else:  # label_kit
            ident, kit = cmd["id"], cmd["kit"]
            if not isinstance(ident, str) or not (kit is None or isinstance(kit, str)):
                raise ValueError("label_kit needs id as text and kit as text or null")
            job = lambda: self._label_kit(ident, kit or None)  # noqa: E731

        def run() -> None:
            try:
                job()
            except Exception as exc:
                self.hub.publish(protocol.error(_describe(exc), name))

        if self._closed:
            raise RuntimeError("the bench is closing")
        self._jobs.put(run)

    def _scan(self, kind: str) -> None:
        devices = self.links.scan(kind)
        with self._lock:
            self._scans[kind] = devices
        self.hub.publish(protocol.scan(kind, devices))

    def _release_board(self) -> None:
        """Let the session drop the board before the link closes or replaces it.

        Raises:
            RuntimeError: the session refuses a watch change now (a take is recording); the
                link is left as it is.
        """
        refused: list[RuntimeError] = []

        def release(s: LiveSession) -> None:
            try:
                s.check_board_change()
                if s.board is not None:
                    s.set_board(None)
            except RuntimeError as exc:
                refused.append(exc)

        self._call_and_wait(release, BOARD_WAIT_S)
        if refused:
            raise refused[0]

    def _connect(self, kind: str, ident: str | None) -> None:
        self._release_board()
        try:
            board = self.links.connect(kind, ident)
        except Exception:
            self._link_changed("connect")
            raise

        def attach(s: LiveSession) -> None:
            s.set_board(board)
            self._sync_context(self._publish_link())

        self._session_command("connect", attach)

    def _disconnect(self) -> None:
        self._release_board()
        self.links.disconnect()
        self._link_changed("disconnect")

    def _link_changed(self, name: str) -> None:
        """Publish the link status now, then record it on the session and publish the state."""
        status = self._publish_link()
        self._session_command(name, lambda _s: self._sync_context(status))

    def _device(self, action: str) -> None:
        command = DEVICE_ACTIONS[action]
        if command is None:
            self.links.identify()
        else:
            self.links.send(command)

    def _label_kit(self, ident: str, kit: str | None) -> None:
        self.links.set_kit_label(ident, kit)
        with self._lock:
            self._scans = {
                kind: [replace(d, kit=kit) if d.id == ident else d for d in devices]
                for kind, devices in self._scans.items()
            }
            scans = dict(self._scans)
        for kind, devices in scans.items():
            self.hub.publish(protocol.scan(kind, devices))
        self._link_changed("label_kit")
