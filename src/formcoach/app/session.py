"""``LiveSession``: the live loop as an object, with no UI in it (Bench design §5.1).

A session owns one :class:`formcoach.app.pipeline.Pipeline` (``per_arm=True``), one
:class:`formcoach.app.events.EventLog`, an optional wearable (a
:class:`formcoach.app.board.BoardLink` or anything shaped like it) and a detector. Each call to
:meth:`LiveSession.step` takes one camera frame, runs the gate, pose, rep counting and rules on
it, sends the wearable its LED command and returns a :class:`Snapshot` of everything a front end
draws. Front ends subscribe to snapshots, IMU batches and events; ``run_live`` (the OpenCV
window and the headless demo) is one such client, the web Bench is another.

Gate sources (``gate_source``): ``device`` follows the wearable's gate flag
(``board.gate_open()``); ``energy`` follows a laptop-side :class:`formcoach.app.gate.EnergyGate`
fed from the wearable's IMU samples, and only while the board is alive; ``always_on`` is always
open; ``manual`` is opened and closed by hand (closed until set). ``device`` and ``energy`` are
closed without a board. The energy gate is computed whenever a board is connected, whatever the
source, so its state can be shown next to the device's.

Threads: the loop (``step``/``run``) and every listener callback run on the session's thread.
:meth:`LiveSession.call`, :meth:`LiveSession.stop` and :meth:`LiveSession.subscribe` may be
called from any thread. A listener that raises never stops the loop: its first error is
published as an ``error`` event and later ones are ignored.

Timing: ``t`` is seconds on the frame source's clock; ``t_mono``, the FPS readout and the LED
fault-flash guard use the injectable ``clock`` (wall seconds).
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from formcoach.app.events import Event, EventLog
from formcoach.app.gate import EnergyGate
from formcoach.app.pipeline import Pipeline, PipelineResult, RepMetrics, _arm_min_visibility
from formcoach.io import protocol
from formcoach.io.replay import PoseFrame
from formcoach.pose.skeletons import MEDIAPIPE33

GATE_SOURCES = ("device", "energy", "always_on", "manual")
MAX_BUFFER_FRAMES = 3600  # per-arm angle buffer cap (about two minutes at 30 fps)
REP_PAD_S = 1.0  # metrics window reaches this far back to the rest position (see Pipeline)
FPS_WINDOW = 30  # frames averaged for the FPS readout
FAULT_FLASH_S = 0.8  # firmware fault flash (ui.cpp): a newer LED command cancels it, so a good-rep
# command is held back for this long after a fault one (wall seconds on the injectable clock)

DetectFn = Callable[[np.ndarray, int], "tuple[np.ndarray | None, np.ndarray | None]"]


def pose_frame_from_detection(
    index: int, t: float, img: np.ndarray | None, world: np.ndarray | None
) -> PoseFrame:
    """Wrap one detector result as a :class:`PoseFrame` (MediaPipe-33 skeleton).

    ``img`` is ``(33, 4)`` normalised image x, y, z and visibility; ``world`` is ``(33, 3)``
    metres. Either being ``None`` (no person found) gives an invalid frame: ``valid=False``,
    ``world`` all NaN, ``visibility`` zeros, ``image`` ``None``. ``t`` is seconds.
    """
    if img is None or world is None:
        return PoseFrame(
            index, float(t), np.full((33, 3), np.nan), np.zeros(33), False, MEDIAPIPE33, None
        )
    img = np.asarray(img)
    return PoseFrame(
        index,
        float(t),
        np.asarray(world, dtype=np.float64),
        img[:, 3].astype(np.float64),
        True,
        MEDIAPIPE33,
        img,
    )


@dataclass(slots=True)
class Snapshot:
    """Everything a front end needs to draw one frame.

    ``index`` is the source's frame index; ``t`` seconds on the frame source's clock; ``t_mono``
    the session ``clock()`` (s) when :meth:`LiveSession.step` received the frame. ``frame`` is
    the raw BGR image, unmirrored. ``image_landmarks`` ``(33, 4)`` (normalised x, y, z,
    visibility) and ``world_landmarks`` ``(33, 3)`` (metres) are ``None`` when pose did not run
    or found nobody. ``gate`` is the gate the pipeline used for this frame and ``gate_source``
    where it came from (one of :data:`GATE_SOURCES`); ``gates`` holds both wearable lanes,
    ``{"device": bool, "energy": bool}``, each ``None`` without a board. ``energy`` is the
    latest ``var(|a|)`` window value of the energy gate in (m/s²)², ``None`` before the first
    window or without a board. ``angles`` holds every tracked joint angle of this frame in
    degrees (``{}`` when pose did not run or found nobody). ``counters`` is keyed ``"l"``/``"r"``
    (arm exercises) or ``"all"`` (squat): ``{"angle": deg | None, "enter": deg, "exit": deg,
    "armed": bool, "visibility": 0-1 | None}``, where ``angle`` is the counter's primary angle
    on its latest processed frame and ``visibility`` the arm's lowest landmark visibility
    (``None`` for ``"all"`` and when there are no landmarks). ``totals`` is
    :meth:`Pipeline.arm_totals`; ``completed`` the reps finished on this frame; ``fps`` frames/s
    over the last :data:`FPS_WINDOW` frames; ``link_state`` ``"none"`` (no board),
    ``"waiting"``, ``"ok"`` or ``"silent"``.
    """

    index: int
    t: float
    t_mono: float
    frame: np.ndarray | None
    image_landmarks: np.ndarray | None
    world_landmarks: np.ndarray | None
    pose_ran: bool
    gate: bool
    gate_source: str
    gates: dict[str, bool | None]
    energy: float | None
    angles: dict[str, float]
    counters: dict[str, dict]
    totals: dict[str, dict]
    completed: list[RepMetrics]
    fps: float
    exercise: str
    link_state: str


class LiveSession:
    """The live loop: frames in, snapshots, IMU batches and events out (see the module doc)."""

    def __init__(
        self,
        *,
        exercise: str,
        detect: DetectFn,
        rules=None,
        board=None,
        clock: Callable[[], float] = time.monotonic,
        rep_config: dict | None = None,
        gate_source: str | None = None,
        report_dropped: bool = True,
        on_event: Callable[[Event], None] | None = None,
    ):
        """``exercise``: ``curl``/``press``/``raise``/``squat``. ``detect(frame_bgr, t_ms)``
        returns ``(image_landmarks (33, 4), world_landmarks (33, 3))`` or ``(None, None)``;
        ``t_ms`` is strictly increasing. ``rules``: a rules engine (or ``None``). ``board``: a
        :class:`formcoach.app.board.BoardLink` or anything with ``poll``/``gate_open``/``state``/
        ``send``/``name`` (``alive`` is used when present). ``clock``: wall seconds.
        ``rep_config`` overrides the rep segmenter's settings. ``gate_source``: one of
        :data:`GATE_SOURCES`, default ``"device"`` with a board, else ``"always_on"``.
        ``report_dropped``: whether reps the pipeline throws away are published as events.
        ``on_event`` is subscribed as an event listener.

        Raises:
            ValueError: ``gate_source`` is not one of :data:`GATE_SOURCES`.
        """
        if gate_source is None:
            gate_source = "device" if board is not None else "always_on"
        if gate_source not in GATE_SOURCES:
            raise ValueError(f"unknown gate source {gate_source!r}; choose from {GATE_SOURCES}")
        self.exercise = exercise
        self.detect = detect
        self.board = board
        self.clock = clock
        self.gate_source = gate_source
        self.report_dropped = report_dropped
        self.log = EventLog()

        self._lock = threading.Lock()
        self._listeners: dict[str, list[Callable]] = {"snapshot": [], "imu": [], "event": []}
        self._failed: list[Callable] = []  # listeners whose first error was already reported
        self._calls: queue.SimpleQueue[Callable[[LiveSession], None]] = queue.SimpleQueue()
        self._stopped = threading.Event()
        self._manual_gate = False  # the "manual" source's state: closed until set
        self._energy = EnergyGate()  # shadow gate, fed from the board's samples
        self._last_ms = -1
        self._stamps: deque[float] = deque(maxlen=FPS_WINDOW)
        self._fault_led_at: float | None = None  # when LED_FAULT was last sent (clock seconds)
        self._t_first: float | None = None
        self._t_last: float | None = None
        self._n_frames = 0

        if on_event is not None:
            self.subscribe(on_event=on_event)
        self.pipeline = Pipeline(
            exercise=exercise,
            rules=rules,
            log=self.log,
            on_event=self._publish_event,
            rep_config=rep_config,
            per_arm=True,
            max_buffer_frames=MAX_BUFFER_FRAMES,
            pad_rep_start_s=REP_PAD_S,
            gate_name=gate_source,
        )

    # ---- listeners and commands --------------------------------------------------------------

    def subscribe(
        self,
        *,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        on_imu: Callable[[list], None] | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> Callable[[], None]:
        """Add listeners: ``on_snapshot(Snapshot)`` once per frame, ``on_imu(list[SISample])``
        with each non-empty batch of wearable samples (SI units), ``on_event(Event)`` for every
        pipeline and session event. They run on the session thread. Returns a function that
        removes them again. Thread-safe."""
        added = [
            (kind, fn)
            for kind, fn in (("snapshot", on_snapshot), ("imu", on_imu), ("event", on_event))
            if fn is not None
        ]
        with self._lock:
            for kind, fn in added:
                self._listeners[kind].append(fn)

        def unsubscribe() -> None:
            with self._lock:
                for kind, fn in added:
                    with contextlib.suppress(ValueError):
                        self._listeners[kind].remove(fn)

        return unsubscribe

    def call(self, fn: Callable[[LiveSession], None]) -> None:
        """Queue ``fn(session)`` to run on the session thread at the start of the next
        :meth:`step`. Thread-safe. If ``fn`` raises, an ``error`` event is published."""
        self._calls.put(fn)

    def stop(self) -> None:
        """Make :meth:`run` return after the frame being processed. Thread-safe."""
        self._stopped.set()

    def emit(self, kind: str, t: float | None = None, **payload) -> Event:
        """Log a session-level event and send it to the event listeners. ``t`` (s, frame
        clock) defaults to the latest frame's time, or 0.0 before the first frame. Call it on
        the session thread (from inside :meth:`call` when on another thread)."""
        if t is None:
            t = self._t_last if self._t_last is not None else 0.0
        e = self.log.event(kind, t, **payload)
        self._dispatch("event", e)
        return e

    def _publish_event(self, e: Event) -> None:
        self._dispatch("event", e)

    def _dispatch(self, kind: str, arg) -> None:
        with self._lock:
            listeners = tuple(self._listeners[kind])
        for fn in listeners:
            try:
                fn(arg)
            except Exception as exc:  # a broken listener must never stop the loop
                if any(f is fn for f in self._failed):  # by identity: fn may be unhashable
                    continue
                self._failed.append(fn)
                self.emit("error", where=f"on_{kind}", error=f"{type(exc).__name__}: {exc}")

    def _run_calls(self) -> None:
        while True:
            try:
                fn = self._calls.get_nowait()
            except queue.Empty:
                return
            try:
                fn(self)
            except Exception as exc:
                self.emit("error", where="call", error=f"{type(exc).__name__}: {exc}")

    # ---- the loop ----------------------------------------------------------------------------

    def _board_alive(self) -> bool:
        alive = getattr(self.board, "alive", None)
        return bool(alive()) if alive is not None else self.board.state() == "ok"

    def _read_gates(self) -> dict[str, bool | None]:
        if self.board is None:
            return {"device": None, "energy": None}
        return {
            "device": bool(self.board.gate_open()),
            "energy": bool(self._energy.state and self._board_alive()),
        }

    def _choose_gate(self, gates: dict[str, bool | None]) -> bool:
        if self.gate_source == "always_on":
            return True
        if self.gate_source == "manual":
            return self._manual_gate
        return bool(gates[self.gate_source])  # device/energy; None (no board) is closed

    def _send_led(self, completed: list[RepMetrics]) -> None:
        """ONE command per frame, so arms finishing together cannot overwrite each other's LED
        flash: any fault wins. A good-rep command is skipped while a fault flash is still
        showing (an arm finishing a frame or two later would cancel it). Send errors are
        ignored."""
        if self.board is None:
            return
        now = self.clock()
        if any(rep.faults for rep in completed):
            cmd = protocol.Command.LED_FAULT
            self._fault_led_at = now
        elif self._fault_led_at is None or now - self._fault_led_at >= FAULT_FLASH_S:
            cmd = protocol.Command.LED_GOOD_REP
        else:
            return
        with contextlib.suppress(Exception):
            self.board.send(cmd)

    def _counters(self, pf: PoseFrame | None) -> dict[str, dict]:
        latest = self.pipeline.latest_angles()
        bands = self.pipeline.thresholds()
        out: dict[str, dict] = {}
        for track in self.pipeline.tracks:
            key = track.arm if track.arm is not None else "all"
            angle = latest[key]
            enter, exit_ = bands[key]
            visibility = None
            if track.arm is not None and pf is not None and pf.valid:
                visibility = _arm_min_visibility(pf, track.side)
            out[key] = {
                "angle": None if np.isnan(angle) else float(angle),
                "enter": float(enter),
                "exit": float(exit_),
                "armed": bool(track.seg.armed),
                "visibility": visibility,
            }
        return out

    def step(self, index: int, t: float, frame: np.ndarray | None) -> Snapshot:
        """Process one frame: run queued calls, poll the board (samples go to ``on_imu``
        listeners and the energy gate), choose the gate, run pose and the pipeline when it is
        open, send the LED command for reps finished here, then publish and return the
        :class:`Snapshot`. ``index`` is the source's frame index, ``t`` seconds on its clock,
        ``frame`` BGR. Exceptions from ``detect`` propagate; the frame is still counted."""
        t_mono = self.clock()
        self._stamps.append(t_mono)
        if self._t_first is None:
            self._t_first = t
        self._t_last = t
        self._n_frames += 1

        self._run_calls()

        if self.board is not None:
            samples = self.board.poll()
            if samples:
                for s in samples:
                    self._energy.update(s)
                self._dispatch("imu", samples)
        gates = self._read_gates()
        gate = self._choose_gate(gates)
        self.pipeline.set_gate(gate, t)

        img = world = None
        pf = None
        if gate:
            t_ms = max(int(t * 1000), self._last_ms + 1)
            self._last_ms = t_ms
            img, world = self.detect(frame, t_ms)
            pf = pose_frame_from_detection(index, t, img, world)
            completed = self.pipeline.on_frame(pf)
        else:
            self.pipeline.skip_frame(t)
            completed = []
        if completed:
            self._send_led(completed)

        found = pf is not None and pf.valid
        fps = 0.0
        if len(self._stamps) > 1 and self._stamps[-1] > self._stamps[0]:
            fps = (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])
        frame_angles = self.pipeline.last_frame_angles if found else None
        snap = Snapshot(
            index=index,
            t=t,
            t_mono=t_mono,
            frame=frame,
            image_landmarks=np.asarray(img) if found else None,
            world_landmarks=np.asarray(world) if found else None,
            pose_ran=gate,
            gate=gate,
            gate_source=self.gate_source,
            gates=gates,
            energy=self._energy.last_value if self.board is not None else None,
            angles=dict(frame_angles) if frame_angles is not None else {},
            counters=self._counters(pf),
            totals=self.pipeline.arm_totals(),
            completed=completed,
            fps=fps,
            exercise=self.exercise,
            link_state=self.board.state() if self.board is not None else "none",
        )
        self._dispatch("snapshot", snap)
        return snap

    def run(self, frames: Iterable[tuple[int, float, np.ndarray]]) -> None:
        """:meth:`step` through ``frames`` (``(index, t_seconds, frame_bgr)``) until they run
        out or :meth:`stop` is called. Exceptions from ``frames`` or ``detect`` propagate."""
        if self._stopped.is_set():
            return
        for index, t, frame in frames:
            self.step(index, t, frame)
            if self._stopped.is_set():
                break

    def finish(self) -> PipelineResult:
        """Close the run: :meth:`Pipeline.finish` (no IMU cross-check) with the summary keys
        ``live`` (True), ``board`` (its name or ``None``), ``fps_mean`` (frames/s over the frame
        clock) and ``duration_s`` (first to last frame, s) added in that order."""
        if self._t_first is not None and self._t_last is not None:
            duration = self._t_last - self._t_first
        else:
            duration = 0.0
        result = self.pipeline.finish(compute_imu=False, end_t=self._t_last)
        n = self._n_frames
        result.summary["live"] = True
        result.summary["board"] = self.board.name if self.board is not None else None
        result.summary["fps_mean"] = round((n - 1) / duration, 1) if duration > 0 else 0.0
        result.summary["duration_s"] = round(duration, 2)
        return result
