"""The FormCoach pipeline, source-agnostic (docs/02 §1, §4).

``run_pipeline`` merges an IMU sample stream and a pose-frame stream by time. Every IMU sample
updates the gate; every pose frame is *processed* (angles → rep segmentation → rules) only while
the gate is open, otherwise logged as *skipped*. Reps are detected from the exercise's primary
joint angle (:mod:`formcoach.pose.reps`); when a rep completes, :class:`RepMetrics` (angles
resampled to 30 steps + scalar metrics) is emitted as a ``rep`` event and handed to the rules
engine if one is attached (Checkpoint 6). At the end, IMU peak-detection reps are computed on
the gated segments as the cross-check (``reps_imu``, ``reps_fused`` in the summary).

``run_replay`` wires a session directory to this with :class:`ReplaySource` / :class:`PoseReplay`
— that is what ``formcoach demo --source replay --headless`` runs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from formcoach.app import gate as gate_mod
from formcoach.app.events import Event, EventLog
from formcoach.io.replay import FIXTURE_SESSION, PoseFrame, PoseReplay, ReplaySource
from formcoach.io.source import IMUSource, SISample
from formcoach.pose import angles as angles_mod
from formcoach.pose.reps import DEFAULT_REP_CONFIG, PoseRep, RepSegmenter
from formcoach.pose.skeletons import Skeleton, joint

REP_STEPS = 30
TRACKED = angles_mod.ANGLE_NAMES
MAX_GAP_FRAMES = 3  # docs/02 §4.3: bridge gaps <= 3 frames, longer gaps are invalid
MIN_ARM_VISIBILITY = 0.5  # docs/02 §4.3: an arm is tracked only if shoulder/elbow/wrist >= this
TWO_ARM_OVERLAP = 0.5  # fraction of the shorter rep's duration two arms must overlap to pair


@dataclass
class RepMetrics:
    exercise: str
    rep_id: int
    t_start: float
    t_end: float
    duration_s: float
    source: str  # pose | imu | fused
    angles: dict[str, np.ndarray]  # each (REP_STEPS,)
    metrics: dict[str, float]
    faults: list[dict] = field(default_factory=list)
    arm: str | None = None  # "l" | "r" | None (squat, or when per-arm tracking is off)

    def as_payload(self) -> dict:
        return {
            "exercise": self.exercise,
            "rep_id": self.rep_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration_s": self.duration_s,
            "source": self.source,
            "metrics": self.metrics,
            "faults": self.faults,
        }


@dataclass
class PipelineResult:
    events: list[Event]
    frames_total: int
    frames_processed: int
    reps: list[RepMetrics]
    summary: dict
    out_dir: Path | None = None


class _UpEstimator:
    """Running estimate of the vertical from hip→shoulder directions (EMA)."""

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self.up: np.ndarray | None = None

    def update(self, world: np.ndarray, sk: Skeleton) -> np.ndarray:
        v = joint(world[None], sk, "shoulder_c")[0] - joint(world[None], sk, "hip_c")[0]
        n = np.linalg.norm(v)
        if n > 1e-9 and np.isfinite(n):
            v = v / n
            self.up = v if self.up is None else (1 - self.alpha) * self.up + self.alpha * v
            self.up = self.up / np.linalg.norm(self.up)
        return self.up if self.up is not None else np.array([0.0, 1.0, 0.0])


def primary_angle(exercise: str, a: dict[str, float], side: str = "both") -> float:
    key = DEFAULT_REP_CONFIG[exercise]["angle"]
    if side == "both":
        return float(np.nanmean([a[f"{key}_l"], a[f"{key}_r"]]))
    return float(a[f"{key}_{side}"])


def resample_rep_angles(
    rep: PoseRep, ts: np.ndarray, series: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Every tracked angle series (degrees, or torso units for the offset metrics) over
    ``[i_start, i_end]`` resampled to ``REP_STEPS`` points: ``{name: (REP_STEPS,) float}``."""
    i0, i1 = rep.i_start, rep.i_end
    t_seg = ts[i0 : i1 + 1]
    grid = np.linspace(t_seg[0], t_seg[-1], REP_STEPS)
    res: dict[str, np.ndarray] = {}
    for k, v in series.items():
        seg = v[i0 : i1 + 1]
        ok = np.isfinite(seg)
        res[k] = (
            np.interp(grid, t_seg[ok], seg[ok]) if ok.sum() >= 2 else np.full(REP_STEPS, np.nan)
        )
    return res


def metrics_from_angles(
    res: dict[str, np.ndarray], duration_s: float, side: str = "both"
) -> dict[str, float]:
    """Scalar rep metrics (degrees, seconds, torso units) from the 30-step angle series; the
    keys are the metric names rules.yaml refers to.

    ``side`` = ``"both"`` (default, bit-identical to the original behaviour) averages every
    paired series 0.5*(left + right); ``"l"``/``"r"`` uses that side's series alone, so a
    one-arm rep is not diluted by an idle arm. ``elbow_min_l/_r``, ``elbow_asymmetry`` and
    ``elbow_top_asymmetry`` always compare both raw sides regardless of ``side``.
    """

    def mn(k):
        return float(np.nanmin(res[k])) if np.isfinite(res[k]).any() else float("nan")

    def mx(k):
        return float(np.nanmax(res[k])) if np.isfinite(res[k]).any() else float("nan")

    pairs = (
        "elbow",
        "shoulder_abd",
        "knee",
        "upper_arm_trunk",
        "elbow_drift",
        "knee_track",
        "hip_knee_height",
    )
    if side == "both":
        both = {k: 0.5 * (res[f"{k}_l"] + res[f"{k}_r"]) for k in pairs}
    else:
        both = {k: res[f"{k}_{side}"] for k in pairs}
    bottom = int(np.nanargmin(both["knee"])) if np.isfinite(both["knee"]).any() else 0
    peak_abd = (
        int(np.nanargmax(both["shoulder_abd"])) if np.isfinite(both["shoulder_abd"]).any() else 0
    )
    return {
        "duration_s": float(duration_s),
        "elbow_min": float(np.nanmin(both["elbow"])),
        "elbow_max": float(np.nanmax(both["elbow"])),
        "elbow_min_l": mn("elbow_l"),
        "elbow_min_r": mn("elbow_r"),
        "elbow_max_l": mx("elbow_l"),
        "elbow_max_r": mx("elbow_r"),
        "elbow_asymmetry": abs(mn("elbow_l") - mn("elbow_r")),
        "elbow_top_asymmetry": abs(mx("elbow_l") - mx("elbow_r")),
        "upper_arm_trunk_range": float(
            np.nanmax(both["upper_arm_trunk"]) - np.nanmin(both["upper_arm_trunk"])
        ),
        "upper_arm_trunk_drift": float(
            np.nanmax(np.abs(both["upper_arm_trunk"] - both["upper_arm_trunk"][0]))
        ),
        "elbow_drift_max": float(np.nanmax(both["elbow_drift"] - both["elbow_drift"][0])),
        "trunk_incl_max": mx("trunk_incl"),
        "abd_peak": float(np.nanmax(both["shoulder_abd"])),
        "elbow_at_abd_peak": float(both["elbow"][peak_abd]),
        "knee_min": float(np.nanmin(both["knee"])),
        "trunk_incl_at_bottom": float(res["trunk_incl"][bottom]),
        "knee_track_at_bottom": float(
            np.nanmin([res["knee_track_l"][bottom], res["knee_track_r"][bottom]])
        ),
        "hip_knee_height_at_bottom": float(both["hip_knee_height"][bottom]),
    }


def compute_rep_metrics(
    exercise: str,
    rep: PoseRep,
    ts: np.ndarray,
    series: dict[str, np.ndarray],
    source: str = "pose",
    side: str = "both",
) -> RepMetrics:
    """Resample every tracked angle over the rep to ``REP_STEPS`` and derive scalar metrics
    (documented keys consumed by rules.yaml; see docs/02 §7). ``side`` is forwarded to
    :func:`metrics_from_angles` (see there)."""
    res = resample_rep_angles(rep, ts, series)
    return RepMetrics(exercise, rep.rep_id, rep.t_start, rep.t_end, rep.duration_s, source, res,
                      metrics_from_angles(res, rep.duration_s, side=side))  # fmt: skip


def _arm_joint_names(side: str) -> tuple[str, str, str]:
    return (f"shoulder_{side}", f"elbow_{side}", f"wrist_{side}")


def _arm_visible(fr: PoseFrame, side: str) -> bool:
    """Whether ``side``'s shoulder, elbow and wrist all meet ``MIN_ARM_VISIBILITY`` in ``fr``."""
    idx = [fr.skeleton.index(name) for name in _arm_joint_names(side)]
    return bool(np.min(fr.visibility[idx]) >= MIN_ARM_VISIBILITY)


class _Track:
    """One rep-counting lane: its own adaptive segmenter, angle buffer and bridge/NaN gap state.

    ``side`` selects which arm's (or both arms') series feeds the primary angle and the rep
    metrics (:func:`primary_angle`, :func:`compute_rep_metrics`); ``arm`` is the label
    (``"l"``/``"r"``/``None``) attached to this track's reps and events.
    """

    def __init__(self, *, side: str, arm: str | None, cfg: dict):
        self.side = side
        self.arm = arm
        self.cfg = cfg
        self.buf_t: list[float] = []
        self.buf_a: dict[str, list[float]] = {k: [] for k in TRACKED}
        self.smooth: deque[float] = deque(maxlen=5)
        self.last_angles: dict[str, float] | None = None
        self.gap = 0
        self.offset = 0  # frames trimmed from the front of buf_t/buf_a so far
        self.bridged = 0
        self.invalid = 0
        self.reps: list[RepMetrics] = []
        self.seg = self._new_segmenter()

    def _new_segmenter(self) -> RepSegmenter:
        cfg = self.cfg
        return RepSegmenter(
            enter=cfg["enter"], exit=cfg["exit"], mode=cfg["mode"], adaptive=cfg["adaptive"],
            window_s=cfg.get("window_s", 6.0), min_range=cfg.get("min_range", 25.0),
            frac=cfg.get("frac", 0.3),
        )  # fmt: skip

    def reset(self) -> None:
        """Drop all buffered frames and start a fresh segmenter (on gate close)."""
        self.seg = self._new_segmenter()
        self.last_angles, self.gap = None, 0
        self.buf_t.clear()
        for v in self.buf_a.values():
            v.clear()
        self.smooth.clear()
        self.offset = 0


class Pipeline:
    """Stateful gate → angles → reps → rules pipeline (one or two :class:`_Track`\\ s).

    ``on_imu``/``set_gate`` drive the gate; ``on_frame``/``skip_frame`` feed pose frames;
    ``finish`` closes the run and returns a :class:`PipelineResult`. ``run_pipeline`` is a thin
    wrapper that merges an :class:`IMUSource` with a pose-frame iterator the way the replay path
    always has; a live loop (no IMU) drives the same object directly.
    """

    def __init__(
        self,
        *,
        exercise: str,
        gate: gate_mod.GateBase | None = None,
        gate_name: str = "always_on",
        rules=None,
        log: EventLog | None = None,
        on_event: Callable[[Event], None] | None = None,
        side: str = "both",
        rep_config: dict | None = None,
        view: str = "frontal",
        per_arm: bool = False,
        max_buffer_frames: int | None = None,
    ):
        self.exercise = exercise
        self.gate = gate
        self.gate_name = gate_name
        self.rules = rules
        self.log = log or EventLog()
        self.on_event = on_event
        self.side = side
        self.view = view
        self.per_arm = per_arm
        self.max_buffer_frames = max_buffer_frames

        cfg = {**DEFAULT_REP_CONFIG[exercise], "adaptive": True}
        if rules is not None and hasattr(rules, "rep_config"):
            cfg.update(rules.rep_config(exercise))
        if rep_config:
            cfg.update(rep_config)
        self.cfg = cfg

        self.up = _UpEstimator()
        self.gate_state = False
        self.gated_seg_start: float | None = None
        self.gated_segments: list[tuple[float, float]] = []
        self.imu_t: list[float] = []
        self.imu_acc: list[tuple[float, float, float]] = []

        if per_arm and exercise in ("curl", "press", "raise"):
            self.tracks = [_Track(side="l", arm="l", cfg=cfg), _Track(side="r", arm="r", cfg=cfg)]
        else:
            self.tracks = [_Track(side=side, arm=None, cfg=cfg)]

    @property
    def gate_open(self) -> bool:
        """Current gate state."""
        return self.gate_state

    @property
    def buffered_frames(self) -> int:
        """Longest per-track angle buffer currently held (frames)."""
        return max((len(t.buf_t) for t in self.tracks), default=0)

    def latest_angles(self) -> dict[str, float]:
        """Each track's primary angle (degrees) on its most recently processed frame, keyed like
        :meth:`arm_totals` (``"l"``/``"r"``/``"all"``). ``NaN`` when that frame had no usable
        angle, or when nothing has been processed since the last gate close. For the live HUD."""
        out: dict[str, float] = {}
        for track in self.tracks:
            key = track.arm if track.arm is not None else "all"
            if not track.buf_t:
                out[key] = float("nan")
                continue
            last = {k: v[-1] for k, v in track.buf_a.items()}
            out[key] = primary_angle(self.exercise, last, track.side)
        return out

    def thresholds(self) -> dict[str, tuple[float, float]]:
        """Each track's current ``(enter, exit)`` rep thresholds in degrees (the adaptive values
        once the segmenter has seen enough range, else the exercise defaults), keyed like
        :meth:`arm_totals`. For the live HUD's trace band."""
        return {
            (t.arm if t.arm is not None else "all"): (
                t.seg.sign * t.seg.enter,
                t.seg.sign * t.seg.exit,
            )
            for t in self.tracks
        }

    def on_imu(self, s: SISample, t: float | None = None) -> bool:
        """Update the gate from one IMU sample; returns the new gate state. Also stores
        ``s.t``/acceleration for the end-of-run IMU cross-check (``finish(compute_imu=True)``)."""
        self.imu_t.append(s.t)
        self.imu_acc.append((s.ax, s.ay, s.az))
        state = self.gate.update(s)
        self.set_gate(state, t if t is not None else s.t)
        return state

    def set_gate(self, open_: bool, t: float) -> None:
        """Force the gate open/closed at time ``t`` (s); emits ``gate_open``/``gate_close`` and
        resets every track's buffers on close. A no-op if the state does not change."""
        open_ = bool(open_)
        if open_ == self.gate_state:
            return
        self.gate_state = open_
        if open_:
            self.gated_seg_start = t
            self._emit("gate_open", t)
        else:
            start = self.gated_seg_start if self.gated_seg_start is not None else t
            self.gated_segments.append((start, t))
            self._emit("gate_close", t)
            for track in self.tracks:
                track.reset()

    def _angles(self, fr: PoseFrame) -> dict[str, float] | None:
        """Joint angles (degrees) for one frame, or ``None`` when invalid/non-finite."""
        if not fr.valid or not np.isfinite(fr.world).all():
            return None
        u = self.up.update(fr.world, fr.skeleton)
        return {
            k: float(v[0])
            for k, v in angles_mod.joint_angles(fr.world[None], fr.skeleton, u).items()
        }

    def on_frame(self, fr: PoseFrame) -> list[RepMetrics]:
        """Log the frame; process it (angles → reps → rules, per track) only while the gate is
        open. Returns the reps completed by this frame, across all tracks."""
        self.log.frame(fr.t, self.gate_state, self.gate_state)
        completed: list[RepMetrics] = []
        if not self.gate_state:
            return completed
        a_global = self._angles(fr)
        for track in self.tracks:
            rm = self._push_track(track, fr, a_global)
            if rm is not None:
                completed.append(rm)
        return completed

    def skip_frame(self, t: float) -> None:
        """Log a frame as skipped (no gate information), e.g. trailing frames after the last
        IMU sample."""
        self.log.frame(t, False, False)

    def _trim_track(self, track: _Track) -> None:
        if self.max_buffer_frames is None:
            return
        n = len(track.buf_t)
        if n > 2 * self.max_buffer_frames:
            drop = n - self.max_buffer_frames
            track.buf_t = track.buf_t[drop:]
            for k in track.buf_a:
                track.buf_a[k] = track.buf_a[k][drop:]
            track.offset += drop

    def _push_track(
        self, track: _Track, fr: PoseFrame, a_global: dict[str, float] | None
    ) -> RepMetrics | None:
        """Angles for one frame on one track; invalid/low-visibility frames reuse the track's
        last good angles for up to MAX_GAP_FRAMES consecutive frames (docs/02 §4.3), longer gaps
        become NaN. Returns the rep this frame completed on this track, if any."""
        needs_visibility = self.per_arm and track.arm is not None
        visible = a_global is not None and (not needs_visibility or _arm_visible(fr, track.side))
        if visible:
            track.gap = 0
            a = dict(a_global)
            track.last_angles = a
        else:
            track.gap += 1
            if track.last_angles is not None and track.gap <= MAX_GAP_FRAMES:
                a = dict(track.last_angles)
                track.bridged += 1
            else:
                a = dict.fromkeys(TRACKED, float("nan"))
                track.invalid += 1
        track.buf_t.append(fr.t)
        for k in TRACKED:
            track.buf_a[k].append(a[k])
        self._trim_track(track)

        p = primary_angle(self.exercise, a, track.side)
        if np.isfinite(p):
            track.smooth.append(p)
            p = float(np.median(track.smooth))
        rep = track.seg.push(fr.t, p)
        if rep is None:
            return None

        i_start = max(0, rep.i_start - track.offset)
        i_end = max(0, rep.i_end - track.offset)
        if i_start != rep.i_start or i_end != rep.i_end:
            rep = replace(rep, i_start=i_start, i_end=i_end)
        ts = np.asarray(track.buf_t)
        series = {k: np.asarray(v) for k, v in track.buf_a.items()}
        rm = compute_rep_metrics(self.exercise, rep, ts, series, side=track.side)
        rm.arm = track.arm
        if self.rules is not None:
            context = self._context(track, rm)
            rm.faults = [f.as_dict() for f in self.rules.evaluate(rm, context=context)]
        track.reps.append(rm)
        self._emit_rep(track, rm)
        return rm

    def _other_arm_track(self, track: _Track) -> _Track | None:
        for t in self.tracks:
            if t is not track and t.arm is not None:
                return t
        return None

    def _paired(self, other: _Track, rep: RepMetrics) -> bool:
        """Whether ``other``'s last completed rep, or its rep in progress, overlaps ``rep`` by
        at least ``TWO_ARM_OVERLAP`` of the shorter one's duration (so two-arm rules only fire
        on reps the two arms actually did together)."""
        if other.reps:
            lr = other.reps[-1]
            shorter = min(rep.duration_s, lr.duration_s)
            overlap = min(rep.t_end, lr.t_end) - max(rep.t_start, lr.t_start)
            if shorter > 0 and overlap >= TWO_ARM_OVERLAP * shorter:
                return True
        start = other.seg.rep_start_t
        if start is not None:
            denom = min(rep.duration_s, rep.t_end - start)
            overlap = rep.t_end - max(rep.t_start, start)
            if denom > 0 and overlap >= 0.5 * denom:
                return True
        return False

    def _context(self, track: _Track, rm: RepMetrics) -> set[str]:
        context: set[str] = set()
        if self.view == "frontal":
            context.add("frontal_view")
        if track.arm is None:
            if track.side == "both":
                context.add("two_arm")
        else:
            other = self._other_arm_track(track)
            if other is not None and self._paired(other, rm):
                context.add("two_arm")
        return context

    def _emit(self, kind: str, t: float, **payload) -> Event:
        e = self.log.event(kind, t, **payload)
        if self.on_event:
            self.on_event(e)
        return e

    def _emit_rep(self, track: _Track, rm: RepMetrics) -> None:
        payload = rm.as_payload()
        if self.per_arm:
            payload["arm"] = track.arm
        self._emit("rep", rm.t_end, **payload)
        for f in rm.faults:
            fault_payload = dict(f)
            if self.per_arm:
                fault_payload["arm"] = track.arm
            self._emit("fault", rm.t_end, rep_id=rm.rep_id, **fault_payload)

    def arm_totals(self) -> dict[str, dict]:
        """Per-track totals keyed ``"l"``/``"r"`` (or ``"all"`` for a single whole-body track):
        ``{"reps": int, "good": int, "flagged": int, "faults_by_code": {code: n}}``. Good = no
        fault of any severity; flagged = at least one."""
        out: dict[str, dict] = {}
        for track in self.tracks:
            key = track.arm if track.arm is not None else "all"
            good = sum(1 for r in track.reps if not r.faults)
            by_code: dict[str, int] = {}
            for r in track.reps:
                for f in r.faults:
                    by_code[f["code"]] = by_code.get(f["code"], 0) + 1
            out[key] = {
                "reps": len(track.reps),
                "good": good,
                "flagged": len(track.reps) - good,
                "faults_by_code": by_code,
            }
        return out

    def finish(self, *, end_t: float | None = None, compute_imu: bool = True) -> PipelineResult:
        """Close the run and return the :class:`PipelineResult`. An open gate segment is closed
        at the last IMU sample's time if there is one, else ``end_t``, else the segment start.
        ``compute_imu=False`` (live mode: one board on one wrist cannot validate both arms)
        omits ``reps_imu``/``reps_fused``/``imu_seconds`` from the summary."""
        if self.gate_state and self.gated_seg_start is not None:
            if self.imu_t:
                seg_end = self.imu_t[-1]
            elif end_t is not None:
                seg_end = end_t
            else:
                seg_end = self.gated_seg_start
            self.gated_segments.append((self.gated_seg_start, seg_end))

        all_reps = sorted((r for t in self.tracks for r in t.reps), key=lambda r: r.t_end)
        bridged = sum(t.bridged for t in self.tracks)
        invalid = sum(t.invalid for t in self.tracks)
        summary = {
            "exercise": self.exercise,
            "gate": self.gate.name if self.gate is not None else self.gate_name,
            "reps": len(all_reps),
            "reps_pose": len(all_reps),
            "faults": sum(len(r.faults) for r in all_reps),
            "frames_total": self.log.frames_total,
            "frames_processed": self.log.frames_processed,
            "frames_processed_pct": round(
                100.0 * self.log.frames_processed / max(self.log.frames_total, 1), 1
            ),
            "frames_bridged": bridged,
            "frames_invalid": invalid,
            "gated_seconds": round(sum(b - a for a, b in self.gated_segments), 2),
        }
        if compute_imu:
            imu_t = np.asarray(self.imu_t)
            imu_acc = np.asarray(self.imu_acc)
            imu_reps = _imu_reps(imu_t, imu_acc, self.gated_segments)
            fused = _fuse(all_reps, imu_reps)
            summary["reps_imu"] = len(imu_reps)
            summary["reps_fused"] = fused
            summary["imu_seconds"] = round((imu_t[-1] - imu_t[0]) if len(imu_t) > 1 else 0.0, 2)
        if self.per_arm:
            summary["arms"] = self.arm_totals()
        return PipelineResult(
            self.log.events, self.log.frames_total, self.log.frames_processed, all_reps, summary
        )


def run_pipeline(
    imu: IMUSource,
    frames: Iterable[PoseFrame],
    *,
    exercise: str,
    gate: gate_mod.GateBase,
    rules=None,
    log: EventLog | None = None,
    on_event: Callable[[Event], None] | None = None,
    side: str = "both",
    rep_config: dict | None = None,
    view: str = "frontal",
) -> PipelineResult:
    """Run IMU + pose through gate → angles → reps → rules; returns events and counts.

    ``side`` = ``both`` (two-arm exercise, enables ``requires: two_arm`` rules) or ``l``/``r``;
    ``view`` = ``frontal`` (enables ``requires: frontal_view`` rules) or ``side``. Thin wrapper
    around :class:`Pipeline` that merges ``imu`` and ``frames`` by time, the way replay always
    has: every IMU sample updates the gate, every pending pose frame up to that time is fed in.
    """
    pipe = Pipeline(
        exercise=exercise, gate=gate, rules=rules, log=log, on_event=on_event,
        side=side, rep_config=rep_config, view=view,
    )  # fmt: skip
    frame_iter = iter(frames)
    pending = next(frame_iter, None)
    for s in imu.iter_samples():
        pipe.on_imu(s)
        while pending is not None and pending.t <= s.t:
            pipe.on_frame(pending)
            pending = next(frame_iter, None)
    imu.close()
    # trailing frames after the last IMU sample are skipped (no gate information)
    while pending is not None:
        pipe.skip_frame(pending.t)
        pending = next(frame_iter, None)
    return pipe.finish()


def _imu_reps(t: np.ndarray, acc: np.ndarray, segments: list[tuple[float, float]]) -> list:
    from formcoach.signal.reps import count_reps
    from formcoach.signal.resample import resample_uniform

    out = []
    for a, b in segments:
        m = (t >= a) & (t <= b)
        if m.sum() < 100:
            continue
        tu, xu = resample_uniform(t[m], acc[m], 50.0)
        out.extend(count_reps(xu, tu, 50.0))
    return out


def _fuse(pose_reps: list[RepMetrics], imu_reps: list) -> int:
    """Count pose reps that overlap an IMU rep by ≥ 50 % of the pose rep's duration."""
    n = 0
    for r in pose_reps:
        for ir in imu_reps:
            ov = min(r.t_end, ir.t_end) - max(r.t_start, ir.t_start)
            if ov >= 0.5 * max(r.duration_s, 1e-6):
                n += 1
                break
    return n


def run_replay(
    session: Path | str = FIXTURE_SESSION,
    *,
    gate: str = "always_on",
    exercise: str | None = None,
    headless: bool = True,
    out_dir: Path | None = None,
    rules=None,
    speed: float = 0.0,
    on_event: Callable[[Event], None] | None = None,
    gate_kwargs: dict | None = None,
    side: str = "both",
    view: str = "frontal",
) -> PipelineResult:
    """Replay a session directory through the pipeline; writes events/frames/session.json to
    ``out_dir`` when given. ``side``/``view`` default to the session's ``meta.json`` values."""
    session = Path(session)
    src = ReplaySource(session, speed=speed)
    exercise = exercise or src.meta.get("exercise", "curl")
    pose_path = session / "pose.parquet"
    frames: Iterable[PoseFrame] = PoseReplay(pose_path) if pose_path.exists() else []
    g = gate_mod.make_gate(gate, **(gate_kwargs or {}))
    log = EventLog()
    if not headless:
        from formcoach.app.overlay import Overlay  # lazy: needs opencv

        overlay = Overlay(exercise)
        frames = overlay.wrap(frames)
        prev = on_event
        on_event = lambda e: (overlay.on_event(e), prev(e) if prev else None)  # noqa: E731
    result = run_pipeline(
        src, frames, exercise=exercise, gate=g, rules=rules, log=log, on_event=on_event
    )
    result.summary["session"] = str(session)
    result.summary["meta"] = src.meta
    if out_dir is not None:
        log.write(Path(out_dir), result.summary)
        result.out_dir = Path(out_dir)
    return result
