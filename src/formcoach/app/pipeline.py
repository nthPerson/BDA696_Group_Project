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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from formcoach.app import gate as gate_mod
from formcoach.app.events import Event, EventLog
from formcoach.io.replay import FIXTURE_SESSION, PoseFrame, PoseReplay, ReplaySource
from formcoach.io.source import IMUSource
from formcoach.pose import angles as angles_mod
from formcoach.pose.reps import DEFAULT_REP_CONFIG, PoseRep, RepSegmenter
from formcoach.pose.skeletons import Skeleton, joint

REP_STEPS = 30
TRACKED = angles_mod.ANGLE_NAMES
MAX_GAP_FRAMES = 3  # docs/02 §4.3: bridge gaps <= 3 frames, longer gaps are invalid


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


def compute_rep_metrics(
    exercise: str, rep: PoseRep, ts: np.ndarray, series: dict[str, np.ndarray], source: str = "pose"
) -> RepMetrics:
    """Resample every tracked angle over the rep to ``REP_STEPS`` and derive scalar metrics
    (documented keys consumed by rules.yaml; see docs/02 §7)."""
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

    def mn(k):
        return float(np.nanmin(res[k])) if np.isfinite(res[k]).any() else float("nan")

    def mx(k):
        return float(np.nanmax(res[k])) if np.isfinite(res[k]).any() else float("nan")

    both = {
        k: 0.5 * (res[f"{k}_l"] + res[f"{k}_r"])
        for k in (
            "elbow",
            "shoulder_abd",
            "knee",
            "upper_arm_trunk",
            "elbow_drift",
            "knee_track",
            "hip_knee_height",
        )
    }
    bottom = int(np.nanargmin(both["knee"])) if np.isfinite(both["knee"]).any() else 0
    peak_abd = (
        int(np.nanargmax(both["shoulder_abd"])) if np.isfinite(both["shoulder_abd"]).any() else 0
    )
    m = {
        "duration_s": float(rep.duration_s),
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
    return RepMetrics(exercise, rep.rep_id, rep.t_start, rep.t_end, rep.duration_s, source, res, m)


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
) -> PipelineResult:
    """Run IMU + pose through gate → angles → reps → rules; returns events and counts."""
    cfg = {**DEFAULT_REP_CONFIG[exercise], "adaptive": True}
    if rep_config:
        cfg.update(rep_config)
    log = log or EventLog()
    frame_iter = iter(frames)
    pending = next(frame_iter, None)
    up = _UpEstimator()

    def new_segmenter() -> RepSegmenter:
        return RepSegmenter(
            enter=cfg["enter"], exit=cfg["exit"], mode=cfg["mode"], adaptive=cfg["adaptive"]
        )

    seg = new_segmenter()
    buf_t: list[float] = []
    buf_a: dict[str, list[float]] = {k: [] for k in TRACKED}
    smooth: deque[float] = deque(maxlen=5)
    reps: list[RepMetrics] = []
    gate_state = False
    gated_segments: list[tuple[float, float]] = []
    seg_start: float | None = None
    imu_t: list[float] = []
    imu_acc: list[tuple[float, float, float]] = []

    def emit(kind: str, t: float, **payload) -> None:
        e = log.event(kind, t, **payload)
        if on_event:
            on_event(e)

    last_angles: dict[str, float] | None = None
    gap = 0
    counts = {"invalid": 0, "bridged": 0}

    def process_frame(fr: PoseFrame) -> None:
        """Angles for one frame; invalid frames reuse the last good angles for up to
        MAX_GAP_FRAMES consecutive frames (docs/02 §4.3), longer gaps become NaN."""
        nonlocal reps, last_angles, gap
        if not fr.valid or not np.isfinite(fr.world).all():
            gap += 1
            if last_angles is not None and gap <= MAX_GAP_FRAMES:
                a = dict(last_angles)
                counts["bridged"] += 1
            else:
                a = dict.fromkeys(TRACKED, float("nan"))
                counts["invalid"] += 1
        else:
            gap = 0
            u = up.update(fr.world, fr.skeleton)
            a = {
                k: float(v[0])
                for k, v in angles_mod.joint_angles(fr.world[None], fr.skeleton, u).items()
            }
            last_angles = a
        buf_t.append(fr.t)
        for k in TRACKED:
            buf_a[k].append(a[k])
        p = primary_angle(exercise, a, side)
        if np.isfinite(p):
            smooth.append(p)
            p = float(np.median(smooth))
        rep = seg.push(fr.t, p)
        if rep is not None:
            ts = np.asarray(buf_t)
            series = {k: np.asarray(v) for k, v in buf_a.items()}
            rm = compute_rep_metrics(exercise, rep, ts, series)
            if rules is not None:
                rm.faults = [f.as_dict() for f in rules.evaluate(rm)]
            reps.append(rm)
            emit("rep", rep.t_end, **rm.as_payload())
            for f in rm.faults:
                emit("fault", rep.t_end, rep_id=rm.rep_id, **f)

    def reset_pose_state() -> None:
        nonlocal seg, last_angles, gap
        seg = new_segmenter()
        last_angles, gap = None, 0
        buf_t.clear()
        for v in buf_a.values():
            v.clear()
        smooth.clear()

    for s in imu.iter_samples():
        imu_t.append(s.t)
        imu_acc.append((s.ax, s.ay, s.az))
        new_state = gate.update(s)
        if new_state != gate_state:
            gate_state = new_state
            if gate_state:
                seg_start = s.t
                emit("gate_open", s.t)
            else:
                gated_segments.append((seg_start if seg_start is not None else s.t, s.t))
                emit("gate_close", s.t)
                reset_pose_state()
        while pending is not None and pending.t <= s.t:
            log.frame(pending.t, gate_state, gate_state)
            if gate_state:
                process_frame(pending)
            pending = next(frame_iter, None)
    imu.close()
    if gate_state and seg_start is not None:
        gated_segments.append((seg_start, imu_t[-1] if imu_t else seg_start))
    # trailing frames after the last IMU sample are skipped (no gate information)
    while pending is not None:
        log.frame(pending.t, False, False)
        pending = next(frame_iter, None)

    imu_reps = _imu_reps(np.asarray(imu_t), np.asarray(imu_acc), gated_segments)
    fused = _fuse(reps, imu_reps)
    summary = {
        "exercise": exercise,
        "gate": gate.name,
        "reps": len(reps),
        "reps_pose": len(reps),
        "reps_imu": len(imu_reps),
        "reps_fused": fused,
        "faults": sum(len(r.faults) for r in reps),
        "frames_total": log.frames_total,
        "frames_processed": log.frames_processed,
        "frames_processed_pct": round(100.0 * log.frames_processed / max(log.frames_total, 1), 1),
        "frames_bridged": counts["bridged"],
        "frames_invalid": counts["invalid"],
        "gated_seconds": round(sum(b - a for a, b in gated_segments), 2),
        "imu_seconds": round((imu_t[-1] - imu_t[0]) if len(imu_t) > 1 else 0.0, 2),
    }
    return PipelineResult(log.events, log.frames_total, log.frames_processed, reps, summary)


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
) -> PipelineResult:
    """Replay a session directory through the pipeline; writes events/frames/session.json to
    ``out_dir`` when given."""
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
