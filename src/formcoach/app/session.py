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

Controls (session thread only; other threads wrap them in :meth:`LiveSession.call`):
:meth:`~LiveSession.set_gate_source`, :meth:`~LiveSession.set_manual_gate`,
:meth:`~LiveSession.set_energy_gate`, :meth:`~LiveSession.set_exercise`,
:meth:`~LiveSession.reset_counters`, :meth:`~LiveSession.set_board`,
:meth:`~LiveSession.set_param` and :meth:`~LiveSession.reset_params` (tuning: an overrides set
over the rules, :mod:`formcoach.app.tuning`; a rule change re-judges the reps already counted
from their stored metrics, without re-running pose). Changing the exercise or
resetting the counters replaces the pipeline with a fresh one on the same event log, rules engine
and rep settings, so the log stays one continuous record and an open gate stays open (no second
``gate_open``). Exercise ``idle`` has no pipeline: no counting and no rules, but the gate, pose,
the frame log and gate events go on (the recording protocol's false-wake measurement).

Threads: the loop (``step``/``run``) and every listener callback run on the session's thread.
:meth:`LiveSession.call`, :meth:`LiveSession.stop` and :meth:`LiveSession.subscribe` may be
called from any thread. A listener that raises never stops the loop: its first error is
published as an ``error`` event and later ones are ignored.

Timing: ``t`` is seconds on the frame source's clock; ``t_mono``, the FPS readout and the LED
fault-flash guard use the injectable ``clock`` (wall seconds). ``clock`` is the one laptop clock:
the wearable's ``SISample.t_arrival`` stamps must come from it too.

Capture (Bench design §7; :mod:`formcoach.app.capture`): :meth:`~LiveSession.start_capture`,
:meth:`~LiveSession.stop_capture`, :meth:`~LiveSession.keep_capture`,
:meth:`~LiveSession.discard_capture` and :meth:`~LiveSession.note_cue_shown` record one set as a
:class:`~formcoach.app.capture.Take`. ``capture_state`` goes ``idle`` → (``countdown`` →)
``recording`` → ``review`` → ``idle``; each change emits a ``capture`` event ``{state, ...}``.
While recording, pose runs on **every** frame (counting still follows the gate) and every
snapshot, IMU batch and event goes into the take; nothing is written before Keep.
"""

from __future__ import annotations

import contextlib
import math
import queue
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from formcoach.app.capture import Take, TakeLabels
from formcoach.app.events import Event, EventLog
from formcoach.app.gate import EnergyGate
from formcoach.app.pipeline import (
    Pipeline,
    PipelineResult,
    RepMetrics,
    UpEstimator,
    arm_min_visibility,
    frame_gate_summary,
    frame_joint_angles,
)
from formcoach.app.tuning import Overrides, rejudge
from formcoach.io import protocol, session_check
from formcoach.io.recorder import TEAM_ROOT
from formcoach.io.replay import PoseFrame
from formcoach.pose.skeletons import MEDIAPIPE33
from formcoach.rules.engine import RuleEngine

GATE_SOURCES = ("device", "energy", "always_on", "manual")
CAPTURE_STATES = ("idle", "countdown", "recording", "review")
AUTO_STOP_S = 5.0  # with auto-stop on, a take ends this long (clock seconds) after the gate closes
EXERCISES = ("curl", "press", "raise", "squat", "idle")  # idle: no counting, no rules
MAX_BUFFER_FRAMES = 3600  # per-arm angle buffer cap (about two minutes at 30 fps)
REP_PAD_S = 1.0  # metrics window reaches this far back to the rest position (see Pipeline)
FPS_WINDOW = 30  # frames averaged for the FPS readout
FAULT_FLASH_S = 0.8  # firmware fault flash (ui.cpp): a newer LED command cancels it, so a good-rep
# command is held back for this long after a fault one (wall seconds on the injectable clock)

MIN_RANGE_KEY = "seg.min_range_deg"
# tuning keys that set the energy gate, and the set_energy_gate argument each one sets
ENERGY_GATE_KEYS = {
    "gate.energy_threshold": "threshold",
    "gate.on_windows": "on_windows",
    "gate.off_windows": "off_windows",
}

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
    or found nobody. ``pose_ran``: whether pose ran on this frame: while the gate is open, and
    on every frame during a capture's recording. ``gate`` is the gate used for this frame (the
    pipeline processed the frame exactly when it is open) and ``gate_source``
    where it came from (one of :data:`GATE_SOURCES`); ``gates`` holds both wearable lanes,
    ``{"device": bool, "energy": bool}``, each ``None`` without a board. ``energy`` is the
    latest ``var(|a|)`` window value of the energy gate in (m/s²)², ``None`` before the first
    window or without a board. ``angles`` holds every tracked joint angle of this frame in
    degrees (``{}`` when pose did not run or found nobody).
    ``counters`` is keyed ``"l"``/``"r"`` (arm exercises) or ``"all"`` (squat):
    ``{"angle": deg | None, "angle_raw": deg | None, "enter": deg, "exit": deg, "armed": bool,
    "visibility": 0-1 | None}``. ``angle`` is the value the rep segmenter compared with
    ``enter``/``exit`` on the counter's latest processed frame (the 5-frame median of the
    primary angle), so a trace of it crosses the band exactly where reps are counted;
    ``angle_raw`` is the unsmoothed primary angle on that frame; either is ``None`` when that
    frame had no usable angle or nothing was processed since the gate last closed.
    ``visibility`` is the arm's lowest landmark visibility (``None`` for ``"all"`` and when
    there are no landmarks). ``totals`` is :meth:`Pipeline.arm_totals`. ``counters`` and
    ``totals`` are ``{}`` for ``idle``. ``completed`` the reps finished on this frame; ``fps``
    frames/s over the last :data:`FPS_WINDOW` frames; ``link_state`` ``"none"`` (no board),
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
        rules_name: str | None = None,
        now: Callable[[], datetime] = datetime.now,
    ):
        """``exercise``: one of :data:`EXERCISES`. ``detect(frame_bgr, t_ms)``
        returns ``(image_landmarks (33, 4), world_landmarks (33, 3))`` or ``(None, None)``;
        ``t_ms`` is strictly increasing. ``rules``: a rules engine (or ``None``). ``board``: a
        :class:`formcoach.app.board.BoardLink` or anything with ``poll``/``gate_open``/``state``/
        ``send``/``name`` (``alive`` is used when present). ``clock``: wall seconds.
        ``rep_config`` overrides the rep segmenter's settings. ``gate_source``: one of
        :data:`GATE_SOURCES`, default ``"device"`` with a board, else ``"always_on"``.
        ``report_dropped``: whether reps the pipeline throws away are published as events.
        ``on_event`` is subscribed as an event listener. ``rules_name``: the rules file's name,
        for :attr:`overrides` (``"rules"`` when not given) and ``meta.json``. ``now``: the local
        wall clock, which names a capture's folder.

        :attr:`capture_context` is what front ends know about the setup and a kept take records
        in ``meta.json``: ``model``, ``camera``, ``firmware``, ``device``, ``kit``, ``source``
        (empty to start; ``device`` defaults to the board's name).

        :attr:`overrides` is the session's tuning (:class:`formcoach.app.tuning.Overrides` over
        ``rules``), ``None`` without rules or when ``rules`` is not a
        :class:`~formcoach.rules.engine.RuleEngine` (a stand-in with only ``evaluate`` judges
        reps but cannot be tuned).

        Raises:
            ValueError: ``exercise`` is not one of :data:`EXERCISES` or ``gate_source`` not one
                of :data:`GATE_SOURCES`.
        """
        if gate_source is None:
            gate_source = "device" if board is not None else "always_on"
        _check_gate_source(gate_source)
        _check_exercise(exercise)
        self.exercise = exercise
        self.detect = detect
        self.rules = rules
        self.rules_name = rules_name
        self.overrides: Overrides | None = (
            Overrides(rules, rules_name or "rules") if isinstance(rules, RuleEngine) else None
        )
        self.board = board
        self.clock = clock
        self.rep_config = rep_config
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
        # the gate as last applied, kept here as well as in the pipeline so it survives a
        # pipeline swap (and so idle, which has no pipeline, can log gate changes)
        self._gate = False
        self._gate_since: float | None = None  # start of the open gate segment (s)
        self._segments: list[tuple[float, float]] = []  # closed gate segments (s)
        # the session's own vertical estimate, for the angles of frames the pipeline did not
        # process (idle has no pipeline; a capture runs pose while the gate is closed)
        self._own_up = UpEstimator()

        self.now = now
        self.capture_state = "idle"
        self.capture_context: dict = {}
        self._take: Take | None = None
        self._pending: TakeLabels | None = None  # the labels of a take in its countdown
        self._countdown_until: float | None = None  # clock seconds
        self._auto_stop = False
        self._gate_seen_open = False  # in this take (auto-stop waits for the set to start)
        self._closed_since: float | None = None  # clock seconds the gate has been closed since
        self._mono_offset = 0.0  # clock minus frame time of the latest frame (s)
        self._rep_mono: dict[tuple[str | None, int], float] = {}  # (arm, rep_id) -> frame clock

        if on_event is not None:
            self.subscribe(on_event=on_event)
        self.pipeline: Pipeline | None = self._new_pipeline(exercise)

    def _new_pipeline(self, exercise: str) -> Pipeline | None:
        """A fresh per-arm pipeline for ``exercise`` (``None`` for ``idle``) on the session's
        log, rules (with the tuning overrides) and rep settings, continuing the session's gate
        state and segments."""
        self._rep_mono.clear()  # rep ids start again at 0
        if exercise == "idle":
            return None
        pipe = Pipeline(
            exercise=exercise,
            rules=self.overrides.engine() if self.overrides is not None else self.rules,
            log=self.log,
            on_event=self._publish_event,
            rep_config=self.rep_config,
            per_arm=True,
            max_buffer_frames=MAX_BUFFER_FRAMES,
            pad_rep_start_s=REP_PAD_S,
            gate_name=self.gate_source,
            report_dropped=self.report_dropped,
        )
        pipe.resume_gate(self._gate, self._gate_since, self._segments)
        if self.overrides is not None and MIN_RANGE_KEY in self.overrides.values:
            pipe.set_min_range(self.overrides.values[MIN_RANGE_KEY])
        return pipe

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
        self._publish_event(e)
        return e

    def _publish_event(self, e: Event) -> None:
        """Send a logged event to the listeners, and into the take while recording (at its time
        on the session clock, through the latest frame's clock-minus-frame-time offset)."""
        if self.capture_state == "recording":
            self._take.add_event(e, e.t + self._mono_offset)
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

    # ---- controls (session thread; other threads use call) -----------------------------------

    def set_gate_source(self, source: str) -> None:
        """Choose where the gate comes from (one of :data:`GATE_SOURCES`) from the next frame
        on; emits ``gate_source`` ``{source}``.

        Raises:
            ValueError: ``source`` is not one of :data:`GATE_SOURCES`.
        """
        _check_gate_source(source)
        self.gate_source = source
        if self.pipeline is not None:
            self.pipeline.gate_name = source
        self.emit("gate_source", source=source)

    def set_manual_gate(self, open_: bool) -> None:
        """Open or close the ``manual`` source's gate (it starts closed). Used only while
        ``gate_source`` is ``manual``; the state is kept when switching away and back."""
        self._manual_gate = bool(open_)

    def set_energy_gate(
        self,
        *,
        threshold: float | None = None,
        on_windows: int | None = None,
        off_windows: int | None = None,
    ) -> None:
        """Tune the laptop-side energy gate: ``threshold`` on ``var(|a|)`` in (m/s²)²,
        ``on_windows``/``off_windows`` consecutive active/idle windows (0.5 s apart) to open/
        close it. ``None`` leaves a setting as it is. Takes effect from the next window.

        Raises:
            ValueError: a negative ``threshold`` or a window count below 1 (nothing changes).
        """
        if threshold is not None and not threshold >= 0:
            raise ValueError(f"energy threshold must be >= 0 (m/s²)², got {threshold!r}")
        for name, n in (("on_windows", on_windows), ("off_windows", off_windows)):
            if n is not None and n < 1:
                raise ValueError(f"{name} must be >= 1, got {n!r}")
        if threshold is not None:
            self._energy.threshold = float(threshold)
        if on_windows is not None:
            self._energy.hyst.on_windows = int(on_windows)
        if off_windows is not None:
            self._energy.hyst.off_windows = int(off_windows)

    def set_exercise(self, exercise: str) -> None:
        """Switch to ``exercise`` (one of :data:`EXERCISES`): a fresh pipeline (``None`` for
        ``idle``) on the same event log, so the counters start again at zero; emits
        ``exercise`` ``{exercise}``.

        Raises:
            ValueError: ``exercise`` is not one of :data:`EXERCISES`.
            RuntimeError: a capture is under way (``capture_state`` is not ``idle``).
        """
        self._refuse_during_capture("change the exercise")
        _check_exercise(exercise)
        self.exercise = exercise
        self.pipeline = self._new_pipeline(exercise)
        self.emit("exercise", exercise=exercise)

    def reset_counters(self) -> None:
        """Start the counters again at zero (a fresh pipeline, same exercise and event log);
        emits ``counters_reset``.

        Raises:
            RuntimeError: a capture is under way (``capture_state`` is not ``idle``).
        """
        self._refuse_during_capture("reset the counters")
        self.pipeline = self._new_pipeline(self.exercise)
        self.emit("counters_reset")

    def _refuse_during_capture(self, what: str) -> None:
        if self.capture_state != "idle":
            raise RuntimeError(f"cannot {what} while a capture is in {self.capture_state!r}")

    def set_board(self, board) -> None:
        """Use ``board`` (BoardLink-shaped, see :meth:`__init__`) from the next frame on, or no
        board with ``None``; the gate source is kept, so ``device`` and ``energy`` stay closed
        until a board is set. The energy gate starts over (same settings). The session does not
        close the old board. Emits ``link`` ``{state, name}``: the new board's ``state()`` and
        ``name``, or ``"none"`` and ``None``."""
        old = self._energy
        self._energy = EnergyGate(
            old.threshold,
            window=old.window,
            stride=old.stride,
            on_windows=old.hyst.on_windows,
            off_windows=old.hyst.off_windows,
        )
        self.board = board
        self._fault_led_at = None
        if board is None:
            self.emit("link", state="none", name=None)
        else:
            self.emit("link", state=board.state(), name=board.name)

    def set_param(self, key: str, value: float) -> None:
        """Set one tuning parameter (keys and ranges: :mod:`formcoach.app.tuning`); ``value``
        is clamped to its range. A ``rule.*`` key swaps the pipeline's rules engine and
        re-judges every rep the pipeline holds (both arms) from its stored metrics; a
        ``gate.*`` key retunes the energy gate (:meth:`set_energy_gate`);
        ``seg.min_range_deg`` sets the rep counter's minimum range. Emits ``params_changed``
        ``{key, value, values}`` (the value applied and every override now in force), then,
        for a rule key, ``rejudged`` ``{totals, reps: [{arm, rep_id, faults}]}``.

        Re-judging rewrites ``faults`` on the pipeline's :class:`RepMetrics`, so
        :meth:`Pipeline.arm_totals` and later snapshots follow; ``rep`` and ``fault`` events
        already logged are left as they were (``rejudged`` is the record). The overrides also
        apply to the pipelines that :meth:`set_exercise` and :meth:`reset_counters` build.

        Raises:
            RuntimeError: the session has no rules (``overrides`` is ``None``).
            KeyError: ``key`` is not a parameter.
            ValueError: ``value`` is NaN or not a number. Nothing changes on any error.
        """
        overrides = self._require_overrides()
        value = overrides.set(key, value)
        if key.startswith("rule."):
            self._swap_engine()
        elif key in ENERGY_GATE_KEYS:
            arg = ENERGY_GATE_KEYS[key]
            self.set_energy_gate(**{arg: value if arg == "threshold" else int(value)})
        elif key == MIN_RANGE_KEY and self.pipeline is not None:
            self.pipeline.set_min_range(value)
        self.emit("params_changed", key=key, value=value, values=overrides.values)
        if key.startswith("rule."):
            self._emit_rejudged()

    def reset_params(self) -> None:
        """Drop every tuning override: the base rules engine judges again (every rep is
        re-judged), the energy gate and the rep counter's minimum range go back to the rules
        file's values. Emits ``params_changed`` ``{key: None, value: None, values: {}}`` and
        ``rejudged`` (see :meth:`set_param`).

        Raises:
            RuntimeError: the session has no rules (``overrides`` is ``None``).
        """
        overrides = self._require_overrides()
        overrides.reset()
        self._swap_engine()
        self.set_energy_gate(
            threshold=overrides.default("gate.energy_threshold"),
            on_windows=int(overrides.default("gate.on_windows")),
            off_windows=int(overrides.default("gate.off_windows")),
        )
        if self.pipeline is not None:
            self.pipeline.set_min_range(overrides.default(MIN_RANGE_KEY))
        self.emit("params_changed", key=None, value=None, values=overrides.values)
        self._emit_rejudged()

    def _require_overrides(self) -> Overrides:
        if self.overrides is None:
            raise RuntimeError("this session has no rules, so there is nothing to tune")
        return self.overrides

    def _session_reps(self) -> list[RepMetrics]:
        """Every rep the current pipeline holds, all tracks, in order of completion."""
        if self.pipeline is None:
            return []
        return sorted((r for t in self.pipeline.tracks for r in t.reps), key=lambda r: r.t_end)

    def _swap_engine(self) -> None:
        """Give the pipeline the overrides' engine and re-judge its reps in place."""
        if self.pipeline is None:
            return
        engine = self.overrides.engine()
        self.pipeline.rules = engine
        reps = self._session_reps()
        for rep, faults in zip(reps, rejudge(reps, engine), strict=True):
            rep.faults = faults

    def _emit_rejudged(self) -> None:
        reps = [
            {"arm": r.arm, "rep_id": r.rep_id, "faults": r.faults} for r in self._session_reps()
        ]
        totals = self.pipeline.arm_totals() if self.pipeline is not None else {}
        self.emit("rejudged", totals=totals, reps=reps)

    # ---- capture (session thread; other threads use call) ------------------------------------

    def start_capture(
        self, labels: TakeLabels, *, delay_s: float = 0.0, auto_stop: bool = False
    ) -> None:
        """Start recording one take labelled ``labels``: after ``delay_s`` seconds on the
        session clock (``countdown``, emits ``capture`` ``{state: "countdown", delay_s}``), or
        at once. When recording starts the counters are reset (rep ids start at 0) and
        ``capture`` ``{state: "recording", t_mono, labels}`` is emitted. With ``auto_stop`` the
        take moves to ``review`` by itself once the gate, having been open, has stayed closed
        for :data:`AUTO_STOP_S` seconds; ignored while ``gate_source`` is ``always_on``.

        Raises:
            RuntimeError: ``capture_state`` is not ``idle``.
            ValueError: ``labels.exercise`` is not the session's exercise, or ``delay_s`` is
                negative.
        """
        if self.capture_state != "idle":
            raise RuntimeError(f"a capture is already in {self.capture_state!r}")
        if labels.exercise != self.exercise:
            raise ValueError(
                f"take labelled {labels.exercise!r} but the session runs {self.exercise!r}"
            )
        if not delay_s >= 0:
            raise ValueError(f"delay_s must be >= 0 s, got {delay_s!r}")
        self._pending = labels
        self._auto_stop = bool(auto_stop)
        if delay_s > 0:
            self._countdown_until = self.clock() + float(delay_s)
            self.capture_state = "countdown"
            self.emit("capture", state="countdown", delay_s=float(delay_s))
        else:
            self._begin_recording(self.clock())

    def _begin_recording(self, t0_mono: float) -> None:
        labels = self._pending
        self._pending = self._countdown_until = None
        self.pipeline = self._new_pipeline(self.exercise)  # counters start at zero
        self._gate_seen_open = False
        self._closed_since = None
        self._take = Take(labels, t0_mono=t0_mono, context=self._capture_ctx(), started=self.now())
        self.capture_state = "recording"
        self.emit(
            "capture", t0_mono - self._mono_offset, state="recording", t_mono=t0_mono,
            labels=asdict(labels),
        )  # fmt: skip

    def _capture_ctx(self) -> dict:
        ctx = {
            "rules": self.rules_name,
            "overrides": dict(self.overrides.values) if self.overrides is not None else {},
            "gate_source": self.gate_source,
            "device": self.board.name if self.board is not None else None,
        }
        ctx.update(self.capture_context)
        return ctx

    def stop_capture(self) -> dict:
        """Stop the take. From ``recording``: to ``review``, emits ``capture`` ``{state:
        "review", summary}`` and returns :meth:`Take.summary`. From ``countdown``: straight back
        to ``idle`` with nothing recorded (emits ``capture`` ``{state: "idle"}``), returns
        ``{}``.

        Raises:
            RuntimeError: ``capture_state`` is neither ``countdown`` nor ``recording``.
        """
        if self.capture_state == "countdown":
            self._pending = self._countdown_until = None
            self.capture_state = "idle"
            self.emit("capture", state="idle")
            return {}
        if self.capture_state != "recording":
            raise RuntimeError(f"no take is recording (capture is {self.capture_state!r})")
        self.capture_state = "review"
        summary = self._take.summary()
        self.emit("capture", state="review", summary=summary)
        return summary

    def keep_capture(self, *, true_reps: int | None, root: Path = TEAM_ROOT) -> dict:
        """Write the take in review to ``<root>/<subject>/<YYYYMMDD-HHMMSS>/``
        (:meth:`Take.write`; ``true_reps`` is the operator's count per arm, or ``None``), run
        ``session check`` on it and go back to ``idle``. Returns and emits (``capture``
        ``{state: "idle", dir, check}``) ``{"dir": str, "check": dict}``.

        Raises:
            RuntimeError: ``capture_state`` is not ``review``.
            OSError, ValueError: the take could not be written (see :meth:`Take.write`); the
                take stays in ``review`` and nothing was written.
        """
        if self.capture_state != "review":
            raise RuntimeError(f"no take to keep (capture is {self.capture_state!r})")
        folder = self._take.write(Path(root), true_reps=true_reps)
        with warnings.catch_warnings():
            # a take without watch samples: today's check takes the mean of nothing (NaN)
            warnings.simplefilter("ignore", RuntimeWarning)
            report = session_check.check(folder)
        report = {  # NaN is not JSON a browser can read
            k: None if isinstance(v, float) and not math.isfinite(v) else v
            for k, v in report.items()
        }
        self._take = None
        self.capture_state = "idle"
        self.emit("capture", state="idle", dir=str(folder), check=report)
        return {"dir": str(folder), "check": report}

    def discard_capture(self) -> None:
        """Drop the take in review (nothing was written) and go back to ``idle``; emits
        ``capture`` ``{state: "idle", discarded: True}``.

        Raises:
            RuntimeError: ``capture_state`` is not ``review``.
        """
        if self.capture_state != "review":
            raise RuntimeError(f"no take to discard (capture is {self.capture_state!r})")
        self._take = None
        self.capture_state = "idle"
        self.emit("capture", state="idle", discarded=True)

    def note_cue_shown(self, arm: str | None, rep_id: int, t_mono: float) -> None:
        """A front end drew the cue of rep ``rep_id`` of ``arm`` (``"l"``/``"r"``, ``None`` for
        squat) at ``t_mono`` (session clock, s). Emits ``cue_shown`` ``{arm, rep_id,
        latency_ms}``, the latency from the frame that completed the rep, and stores it in the
        take (recording or review). A rep the current counters do not know is ignored."""
        t_frame = self._rep_mono.get((arm, int(rep_id)))
        if t_frame is None:
            return
        latency_ms = (float(t_mono) - t_frame) * 1000.0
        self.emit("cue_shown", arm=arm, rep_id=int(rep_id), latency_ms=latency_ms)
        if self._take is not None:
            self._take.note_latency(arm, int(rep_id), latency_ms)

    def _check_auto_stop(self, gate: bool, t_mono: float) -> None:
        if not self._auto_stop or self.gate_source == "always_on":
            return
        if gate:
            self._gate_seen_open = True
            self._closed_since = None
        elif self._gate_seen_open:
            if self._closed_since is None:
                self._closed_since = t_mono
            elif t_mono - self._closed_since >= AUTO_STOP_S:
                self.stop_capture()

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

    def _apply_gate(self, gate: bool, t: float) -> None:
        """Give the pipeline this frame's gate (it emits ``gate_open``/``gate_close``); without
        one (``idle``) the session emits them. Keeps the session's own copy of the gate and its
        segments in step, for the next pipeline and for :meth:`finish`."""
        if self.pipeline is not None:
            self.pipeline.set_gate(gate, t)
        elif gate != self._gate:
            self.emit("gate_open" if gate else "gate_close", t)
        if gate == self._gate:
            return
        if gate:
            self._gate_since = t
        else:
            start = self._gate_since if self._gate_since is not None else t
            self._segments.append((start, t))
            self._gate_since = None
        self._gate = gate

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
        if self.pipeline is None:
            return {}
        raw = self.pipeline.latest_angles()
        bands = self.pipeline.thresholds()
        out: dict[str, dict] = {}
        for track in self.pipeline.tracks:
            key = track.arm if track.arm is not None else "all"
            enter, exit_ = bands[key]
            visibility = None
            if track.arm is not None and pf is not None and pf.valid:
                visibility = arm_min_visibility(pf, track.side)
            out[key] = {
                "angle": _finite_or_none(track.last_p),
                "angle_raw": _finite_or_none(raw[key]),
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
        ``frame`` BGR. Exceptions from ``detect`` propagate; the frame is still counted.

        During a capture: a countdown that has run out starts recording on this frame; while
        recording, pose runs whatever the gate, and the samples, the snapshot and the events
        go into the take; auto-stop is checked after the snapshot is taken."""
        t_mono = self.clock()
        self._stamps.append(t_mono)
        if self._t_first is None:
            self._t_first = t
        self._t_last = t
        self._mono_offset = t_mono - t
        self._n_frames += 1

        self._run_calls()
        if self.capture_state == "countdown" and t_mono >= self._countdown_until:
            self._begin_recording(t_mono)
        recording = self.capture_state == "recording"

        if self.board is not None:
            samples = self.board.poll()
            if samples:
                for s in samples:
                    self._energy.update(s)
                if recording:
                    self._take.add_imu(samples)
                self._dispatch("imu", samples)
        gates = self._read_gates()
        gate = self._choose_gate(gates)
        self._apply_gate(gate, t)

        pipe = self.pipeline
        img = world = None
        pf = None
        completed: list[RepMetrics] = []
        if gate or recording:
            t_ms = max(int(t * 1000), self._last_ms + 1)
            self._last_ms = t_ms
            img, world = self.detect(frame, t_ms)
            pf = pose_frame_from_detection(index, t, img, world)
        if gate:
            if pipe is not None:
                completed = pipe.on_frame(pf)
            else:
                self.log.frame(t, True, True)
        elif pipe is not None:
            pipe.skip_frame(t)
        else:
            self.log.frame(t, False, False)
        for rep in completed:
            self._rep_mono[(rep.arm, rep.rep_id)] = t_mono
        if completed:
            self._send_led(completed)

        found = pf is not None and pf.valid
        fps = 0.0
        if len(self._stamps) > 1 and self._stamps[-1] > self._stamps[0]:
            fps = (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])
        frame_angles = None
        if found:  # a frame the pipeline did not process: on the session's own vertical estimate
            if pipe is not None and gate:
                frame_angles = pipe.last_frame_angles
            else:
                frame_angles = frame_joint_angles(pf, self._own_up)
        snap = Snapshot(
            index=index,
            t=t,
            t_mono=t_mono,
            frame=frame,
            image_landmarks=np.asarray(img) if found else None,
            world_landmarks=np.asarray(world) if found else None,
            pose_ran=gate or recording,
            gate=gate,
            gate_source=self.gate_source,
            gates=gates,
            energy=self._energy.last_value if self.board is not None else None,
            angles=dict(frame_angles) if frame_angles is not None else {},
            counters=self._counters(pf),
            totals=pipe.arm_totals() if pipe is not None else {},
            completed=completed,
            fps=fps,
            exercise=self.exercise,
            link_state=self.board.state() if self.board is not None else "none",
        )
        if recording:
            self._take.add_frame(snap)
            self._check_auto_stop(gate, t_mono)
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
        clock) and ``duration_s`` (first to last frame, s) added in that order. The reps and
        totals are the current pipeline's (since the last exercise change or counter reset); the
        events and frame counts cover the whole session. For ``idle`` the same summary keys are
        built from the session's log, with no reps and ``arms`` ``{}``."""
        if self._t_first is not None and self._t_last is not None:
            duration = self._t_last - self._t_first
        else:
            duration = 0.0
        if self.pipeline is not None:
            result = self.pipeline.finish(compute_imu=False, end_t=self._t_last)
        else:
            result = self._idle_result()
        n = self._n_frames
        result.summary["live"] = True
        result.summary["board"] = self.board.name if self.board is not None else None
        result.summary["fps_mean"] = round((n - 1) / duration, 1) if duration > 0 else 0.0
        result.summary["duration_s"] = round(duration, 2)
        return result

    def _idle_result(self) -> PipelineResult:
        """The :meth:`Pipeline.finish` summary for a session without a pipeline (``idle``):
        the same keys, from the session's log and gate segments, with no reps."""
        segments = list(self._segments)
        if self._gate and self._gate_since is not None:
            end = self._t_last if self._t_last is not None else self._gate_since
            segments.append((self._gate_since, end))
        log = self.log
        summary: dict = {
            "exercise": self.exercise,
            "gate": self.gate_source,
            "reps": 0,
            "reps_pose": 0,
            "faults": 0,
            **frame_gate_summary(log, segments),
            "arms": {},
        }
        return PipelineResult(log.events, log.frames_total, log.frames_processed, [], summary)


def _check_gate_source(source: str) -> None:
    if source not in GATE_SOURCES:
        raise ValueError(f"unknown gate source {source!r}; choose from {GATE_SOURCES}")


def _check_exercise(exercise: str) -> None:
    if exercise not in EXERCISES:
        raise ValueError(f"unknown exercise {exercise!r}; choose from {EXERCISES}")


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None
