"""Replay a Bench take "as recorded" (Bench design §5.1, §7).

A take folder (:mod:`formcoach.app.capture`) holds what the live session saw: every camera
frame's landmarks (``pose.parquet``), the gate it used per frame and the frame source's own time
(``frames.parquet``: ``t``, ``gate_state``, ``t_source``), the wearable's samples
(``imu.parquet``), the session's events (``events.parquet``) and ``meta.json`` with a ``"bench"``
marker, the rules file's name and the tuning overrides. :func:`replay_as_recorded` rebuilds the
pipeline the way :class:`formcoach.app.session.LiveSession` built it (one track per arm, the same
metrics window and buffer cap, the take's rules and overrides, no ``rep_dropped`` events) and
feeds it those frames, so a take is judged by the logic that produced it.

**Clocks.** The pipeline is fed each frame's ``t_source``, the time the live pipeline was given,
so rep timing and durations are the live ones. Everything the result reports (event times, the
``t_start``/``t_end`` in event payloads and on the reps, the frame log) is on the take clock, the
``t`` of ``pose.parquet`` and ``imu.parquet``: each frame's ``t - t_source`` is added, as
:meth:`formcoach.app.capture.Take.add_event` does for the live events.

**Rules.** The packaged rules file named in ``meta["rules"]`` (the packaged webcam rules, with a
note, when the take names any other file), then ``meta["overrides"]``, then every
``params_changed`` event of the take in order. The live session re-judges every rep whenever a
rule threshold moves, so every rep is judged with the final thresholds; a change of
``seg.min_range_deg`` takes effect from its event's time, as it did live. A ``rules`` engine given
by the caller is used as it is and the take's overrides are ignored.

**Gates.** ``recorded``: each frame gets the gate stored for it. ``always_on``: open on every
frame. ``energy``, ``laptop``, ``device``: recomputed from ``imu.parquet`` with
:func:`formcoach.app.gate.make_gate`, the samples merged with the frames by take-clock time the
way :func:`formcoach.app.pipeline.run_pipeline` merges them (frames after the last sample are
skipped). A take without IMU samples supports only ``recorded`` and ``always_on``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.app import gate as gate_mod
from formcoach.app.events import Event, EventLog
from formcoach.app.pipeline import Pipeline, PipelineResult, frame_gate_summary
from formcoach.app.session import (
    MAX_BUFFER_FRAMES,
    MIN_RANGE_KEY,
    REP_PAD_S,
    pose_frame_from_detection,
)
from formcoach.app.tuning import PACKAGE_RULES_DIR, Overrides
from formcoach.io.source import SISample
from formcoach.pose.store import PoseSequence, read_pose
from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

RECORDED = "recorded"
REPLAY_GATES = (RECORDED, *gate_mod.GATE_NAMES)
NO_IMU_GATES = (RECORDED, "always_on")  # the gates a take without IMU samples can replay with
_SHIFTED_PAYLOAD_KEYS = ("t_start", "t_end")  # frame-clock times inside event payloads (s)
_IMU_COLS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]


def is_bench_session(session: Path) -> bool:
    """Whether ``session`` is a folder written by Bench capture: its ``meta.json`` has a
    ``"bench"`` key. ``False`` for anything else, including a missing or unreadable file."""
    return "bench" in _read_meta(Path(session))


def _read_meta(session: Path) -> dict:
    try:
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


class _TakeClockLog(EventLog):
    """An :class:`EventLog` that moves the times it is given (frame-source clock) onto the take
    clock by adding ``shift`` seconds, to events, their payload times and frames alike."""

    shift = 0.0

    def event(self, kind: str, t: float, **payload) -> Event:
        for key in _SHIFTED_PAYLOAD_KEYS:
            value = payload.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                payload[key] = float(value) + self.shift
        return super().event(kind, t + self.shift, **payload)

    def frame(self, t: float, processed: bool, gate_state: bool, **extra: float) -> None:
        super().frame(t + self.shift, processed, gate_state, **extra)


class _GateOnly:
    """The pipeline's stand-in for an ``idle`` take, which the live session runs without one:
    follows the gate (``gate_open``/``gate_close`` events), logs every frame, counts nothing."""

    def __init__(self, *, gate: gate_mod.GateBase | None, gate_name: str, log: EventLog, on_event):
        self.gate, self.gate_name, self.log, self.on_event = gate, gate_name, log, on_event
        self.gate_state = False
        self._since: float | None = None
        self._segments: list[tuple[float, float]] = []
        self._last_imu_t: float | None = None

    def on_imu(self, s: SISample, t: float) -> None:
        self._last_imu_t = s.t
        self.set_gate(self.gate.update(s), t)

    def set_gate(self, open_: bool, t: float) -> None:
        if bool(open_) == self.gate_state:
            return
        self.gate_state = bool(open_)
        if self.gate_state:
            self._since = t
        else:
            self._segments.append((self._since if self._since is not None else t, t))
        e = self.log.event("gate_open" if self.gate_state else "gate_close", t)
        if self.on_event is not None:
            self.on_event(e)

    def on_frame(self, fr) -> list:
        self.log.frame(fr.t, self.gate_state, self.gate_state)
        return []

    def skip_frame(self, t: float) -> None:
        self.log.frame(t, False, False)

    def finish(self, *, end_t: float | None = None, compute_imu: bool = False) -> PipelineResult:
        del compute_imu
        segments = list(self._segments)
        if self.gate_state and self._since is not None:
            end = self._last_imu_t if self._last_imu_t is not None else end_t
            segments.append((self._since, end if end is not None else self._since))
        summary = {
            "exercise": "idle",
            "gate": self.gate.name if self.gate is not None else self.gate_name,
            "reps": 0,
            "reps_pose": 0,
            "faults": 0,
            **frame_gate_summary(self.log, segments),
            "arms": {},
        }
        log = self.log
        return PipelineResult(log.events, log.frames_total, log.frames_processed, [], summary)


def _base_rules(name, notes: list[str]) -> tuple[RuleEngine, str]:
    """The packaged rules file ``name`` (a file name in ``src/formcoach/rules/``), or the
    packaged webcam rules with a note when ``name`` is anything else."""
    if isinstance(name, str) and name and Path(name).name == name:
        path = PACKAGE_RULES_DIR / name
        if path.suffix == ".yaml" and path.is_file():
            return RuleEngine.from_yaml(path), name
    notes.append(
        f"the take names rules file {name!r}, which is not a packaged rules file; "
        f"judged with {WEBCAM_RULES.name}"
    )
    return RuleEngine.from_yaml(WEBCAM_RULES), WEBCAM_RULES.name


def _take_rules(
    meta: dict, events: pd.DataFrame, notes: list[str]
) -> tuple[RuleEngine, float | None, list[tuple[float, float]]]:
    """The engine the take's reps are judged with (base rules, ``meta["overrides"]``, then every
    ``params_changed`` event), the minimum range (deg) in force at the start (``None``: the
    rules file's) and the later minimum-range changes as ``(take-clock s, deg)``."""
    base, name = _base_rules(meta.get("rules"), notes)
    overrides = Overrides(base, name)

    def apply(key: str, value) -> float | None:
        try:
            return overrides.set(key, value)
        except (KeyError, ValueError) as exc:
            notes.append(f"override {key}={value!r} not applied: {exc}")
            return None

    for key, value in (meta.get("overrides") or {}).items():
        apply(key, value)
    start_range = overrides.values.get(MIN_RANGE_KEY)
    changes: list[tuple[float, float]] = []
    changed = events[events["kind"] == "params_changed"].sort_values("t", kind="stable")
    for t, raw in zip(changed["t"], changed["payload"], strict=True):
        payload = json.loads(raw)
        key = payload.get("key")
        if key is None:  # reset_params: back to the rules file's values
            overrides.reset()
            changes.append((float(t), overrides.default(MIN_RANGE_KEY)))
            continue
        value = apply(key, payload.get("value"))
        if key == MIN_RANGE_KEY and value is not None:
            changes.append((float(t), value))
    return overrides.engine(), start_range, changes


def _pose_frame(seq: PoseSequence, i: int, t: float):
    """Frame ``i`` of a take's ``pose.parquet`` as the live session built it from the detector
    output (:func:`formcoach.app.session.pose_frame_from_detection`), at time ``t`` (s)."""
    if not seq.valid[i]:
        return pose_frame_from_detection(int(seq.frames[i]), t, None, None)
    return pose_frame_from_detection(int(seq.frames[i]), t, seq.image[i], seq.world[i])


def replay_as_recorded(
    session: Path,
    *,
    gate: str = RECORDED,
    rules: RuleEngine | None = None,
    gate_kwargs: dict | None = None,
    out_dir: Path | None = None,
    on_event: Callable[[Event], None] | None = None,
) -> PipelineResult:
    """Replay the take in ``session`` through a pipeline built as the live session built it
    (see the module doc). ``gate`` is one of :data:`REPLAY_GATES`; ``gate_kwargs`` go to
    :func:`formcoach.app.gate.make_gate` for a recomputed gate. ``rules`` replaces the take's
    rules and overrides. ``on_event`` sees every event as it is logged (take clock). With
    ``out_dir``, ``events.parquet``, ``frames.parquet`` and ``session.json`` are written there.

    The summary is :meth:`Pipeline.finish`'s without the IMU cross-check (``arms`` holds the
    per-arm totals; ``{}`` for ``idle``, which counts nothing) plus ``as_recorded`` (True),
    ``imu_seconds`` (span of the take's IMU samples, s), ``notes`` (what was substituted or not
    applied), ``session`` and ``meta``.

    Raises:
        ValueError: ``gate`` is unknown; or it must be recomputed and the take has no IMU
            samples; or ``pose.parquet`` and ``frames.parquet`` disagree on the frame count.
        FileNotFoundError: a take file is missing (or the ``laptop`` gate's model).
    """
    session = Path(session)
    if gate not in REPLAY_GATES:
        raise ValueError(f"unknown gate {gate!r}; choose from {REPLAY_GATES}")
    meta = _read_meta(session)
    frames = pd.read_parquet(session / "frames.parquet")
    seq = read_pose(session / "pose.parquet")
    imu = pd.read_parquet(session / "imu.parquet")
    events = pd.read_parquet(session / "events.parquet")
    if gate not in NO_IMU_GATES and len(imu) == 0:
        raise ValueError(
            f"{session} has no IMU samples, so the {gate!r} gate cannot be recomputed; "
            f"replay it with one of {NO_IMU_GATES}"
        )
    if len(seq.t) != len(frames):
        raise ValueError(
            f"{session}: pose.parquet has {len(seq.t)} frames but frames.parquet {len(frames)}"
        )

    notes: list[str] = []
    exercise = meta.get("exercise", "curl")
    t_take = frames["t"].to_numpy(dtype=np.float64)
    # a take with no frames has no diagnostic columns at all, t_source among them
    t_src = frames.get("t_source", frames["t"]).to_numpy(dtype=np.float64)
    shift = t_take - t_src  # take clock minus frame clock, per frame (s)
    g = None if gate in NO_IMU_GATES else gate_mod.make_gate(gate, **(gate_kwargs or {}))
    log = _TakeClockLog()

    range_changes: list[tuple[float, float]] = []
    if exercise == "idle":
        sink = _GateOnly(gate=g, gate_name=gate, log=log, on_event=on_event)
    else:
        start_range = None
        if rules is None:
            rules, start_range, range_changes = _take_rules(meta, events, notes)
        sink = Pipeline(
            exercise=exercise, gate=g, gate_name=gate, rules=rules, log=log, on_event=on_event,
            per_arm=True, max_buffer_frames=MAX_BUFFER_FRAMES, pad_rep_start_s=REP_PAD_S,
            report_dropped=False,
        )  # fmt: skip
        if start_range is not None:
            sink.set_min_range(start_range)

    rep_shift: dict[int, float] = {}  # id(rep) -> its frame's shift, to report it on the take clock

    def feed(i: int, gate_open: bool | None = None) -> None:
        """Frame ``i``, after the minimum-range changes due by its time (live: commands run
        before the frame) and, when given, its gate."""
        while range_changes and range_changes[0][0] <= t_take[i] + 1e-6:
            sink.set_min_range(range_changes.pop(0)[1])
        log.shift = float(shift[i])
        if gate_open is not None:
            sink.set_gate(gate_open, t_src[i])
        for rep in sink.on_frame(_pose_frame(seq, i, t_src[i])):
            rep_shift[id(rep)] = float(shift[i])

    n = len(frames)
    if g is None:
        recorded = frames["gate_state"].to_numpy(dtype=bool)
        for i in range(n):
            feed(i, bool(recorded[i]) if gate == RECORDED else True)
    else:
        t_imu = imu["t"].to_numpy(dtype=np.float64)
        imu_shift = np.interp(t_imu, t_take, shift) if n else np.zeros_like(t_imu)
        cols = imu[_IMU_COLS].to_numpy(dtype=np.float64)
        flags = imu["flags"].to_numpy(dtype=np.int64)
        seqs = imu["seq"].to_numpy(dtype=np.int64)
        i = 0
        for k in range(len(imu)):
            t_k = float(t_imu[k] - imu_shift[k])  # the sample's time on the frame clock
            s = SISample(t_k, *cols[k, 1:].tolist(), int(flags[k]), int(seqs[k]))
            log.shift = float(imu_shift[k])
            sink.on_imu(s, t_k)
            while i < n and t_take[i] <= t_imu[k]:
                feed(i)
                i += 1
        for j in range(i, n):  # after the last sample: no gate information
            log.shift = float(shift[j])
            sink.skip_frame(t_src[j])

    result = sink.finish(end_t=float(t_src[-1]) if n else None, compute_imu=False)
    result.reps = [
        replace(r, t_start=r.t_start + rep_shift[id(r)], t_end=r.t_end + rep_shift[id(r)])
        for r in result.reps
    ]
    t_all = imu["t"].to_numpy(dtype=np.float64)
    summary = result.summary
    summary["as_recorded"] = True
    summary["imu_seconds"] = round(float(t_all[-1] - t_all[0]) if len(t_all) > 1 else 0.0, 2)
    summary["notes"] = notes
    summary["session"] = str(session)
    summary["meta"] = meta
    if out_dir is not None:
        log.write(Path(out_dir), summary)
        result.out_dir = Path(out_dir)
    return result
