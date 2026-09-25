"""``formcoach eval rules`` (docs/02 §8): validate rules.yaml on MM-Fit 3-D pose.

For every MM-Fit set of the four v1 exercises the pose_3d segment is run through the same
angle → adaptive rep segmentation → metrics path as the live pipeline. MM-Fit reps are taken
as correct form, so the **pass rate** of each rule (fraction of reps it does *not* flag) is
its false-alarm complement. The **perturbation table** applies each synthetic fault to every
rep's 30-step angle series (ROM × 0.7, +25° trunk lean, speed × 2, elbow shifted 0.2 torso,
bent elbow at the peak, knee valgus) and reports the detection rate of the intended rule.
``--calibrate-out`` writes a rules.yaml copy whose thresholds sit at the 5th/95th percentile
of the correct-form distribution for this pose source (docs/02 §7).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from formcoach.app.pipeline import REP_STEPS, RepMetrics, metrics_from_angles, resample_rep_angles
from formcoach.data import mmfit
from formcoach.eval import report
from formcoach.pose import angles as angles_mod
from formcoach.pose import normalize
from formcoach.pose.reps import RepSegmenter
from formcoach.rules.engine import RuleEngine

DEFAULT_OUT = report.REPORTS_DIR / "rules_validation.md"
PRIMARY = {"curl": "elbow", "press": "elbow", "raise": "shoulder_abd", "squat": "knee"}
# perturbation name -> (exercise, expected code, function(angles, duration) -> (angles, duration))
PERTURBATIONS: dict[str, tuple[str, str]] = {
    "rom_x0.7": ("curl", "CURL_PARTIAL_ROM"),
    "elbow_shift_0.2": ("curl", "CURL_SWING"),
    "speed_x2": ("curl", "CURL_TOO_FAST"),
    "press_no_lockout": ("press", "PRESS_NO_LOCKOUT"),
    "press_trunk_lean_25": ("press", "PRESS_TRUNK_LEAN"),
    "raise_bent_elbow": ("raise", "RAISE_BENT_ELBOW"),
    "raise_partial": ("raise", "RAISE_PARTIAL"),
    "squat_shallow": ("squat", "SQUAT_SHALLOW"),
    "trunk_lean_25": ("squat", "SQUAT_FORWARD_LEAN"),
    "knee_valgus": ("squat", "SQUAT_KNEE_VALGUS"),
    "squat_speed_x2": ("squat", "SQUAT_TOO_FAST"),
}


def _perturb(
    name: str, a: dict[str, np.ndarray], duration: float
) -> tuple[dict[str, np.ndarray], float]:
    a = {k: v.copy() for k, v in a.items()}
    if name in ("rom_x0.7", "press_no_lockout", "squat_shallow"):
        key = {"rom_x0.7": "elbow", "press_no_lockout": "elbow", "squat_shallow": "knee"}[name]
        for s in ("l", "r"):
            v = a[f"{key}_{s}"]
            rest = np.nanmax(v) if name != "press_no_lockout" else np.nanmin(v)
            a[f"{key}_{s}"] = rest + (v - rest) * 0.7
        if name == "squat_shallow":
            for s in ("l", "r"):
                a[f"hip_knee_height_{s}"] = a[f"hip_knee_height_{s}"] + 0.3
    elif name == "elbow_shift_0.2":
        for s in ("l", "r"):
            a[f"elbow_drift_{s}"] = a[f"elbow_drift_{s}"] + np.linspace(0, 0.25, REP_STEPS)
            a[f"upper_arm_trunk_{s}"] = a[f"upper_arm_trunk_{s}"] + np.linspace(0, 25, REP_STEPS)
    elif name in ("speed_x2", "squat_speed_x2"):
        duration = duration / 2.0
    elif name in ("press_trunk_lean_25", "trunk_lean_25"):
        a["trunk_incl"] = a["trunk_incl"] + 25.0
    elif name == "raise_bent_elbow":
        for s in ("l", "r"):
            a[f"elbow_{s}"] = np.minimum(a[f"elbow_{s}"], 120.0)
    elif name == "raise_partial":
        for s in ("l", "r"):
            v = a[f"shoulder_abd_{s}"]
            a[f"shoulder_abd_{s}"] = np.nanmin(v) + (v - np.nanmin(v)) * 0.5
    elif name == "knee_valgus":
        for s in ("l", "r"):
            a[f"knee_track_{s}"] = a[f"knee_track_{s}"] - 0.15
    else:
        raise KeyError(name)
    return a, duration


def _reps_for_set(xyz, t, sk, exercise: str, engine: RuleEngine) -> list[RepMetrics]:
    up = normalize.estimate_up_vector(xyz, sk)
    a = angles_mod.joint_angles(xyz.astype(np.float64), sk, up)
    key = PRIMARY[exercise]
    primary = normalize.smooth_angles(0.5 * (a[f"{key}_l"] + a[f"{key}_r"]))
    cfg = engine.rep_config(exercise)
    seg = RepSegmenter(
        enter=cfg["enter"],
        exit=cfg["exit"],
        mode=cfg["mode"],
        adaptive=cfg["adaptive"],
        window_s=cfg["window_s"],
        min_range=cfg["min_range"],
        frac=cfg["frac"],
    )
    series = {k: np.asarray(v) for k, v in a.items()}
    out = []
    for i in range(len(t)):
        rep = seg.push(float(t[i]), float(primary[i]))
        if rep is not None:
            res = resample_rep_angles(rep, t, series)
            out.append(
                RepMetrics(
                    exercise,
                    rep.rep_id,
                    rep.t_start,
                    rep.t_end,
                    rep.duration_s,
                    "pose",
                    res,
                    metrics_from_angles(res, rep.duration_s),
                )
            )
    return out


def collect_reps(
    raw_root: Path, engine: RuleEngine, workouts: list[str] | None = None
) -> tuple[list[RepMetrics], pd.DataFrame]:
    """All correct-form reps from MM-Fit sets of the four exercises + per-set segmentation table."""
    from formcoach.pose.skeletons import H36M17

    reps: list[RepMetrics] = []
    rows = []
    for w in workouts or mmfit.list_workouts(raw_root):
        try:
            _frames, xyz, t = mmfit.load_pose3d(raw_root, w)
        except FileNotFoundError:
            continue
        sets = mmfit.load_sets(raw_root, w)
        for s in sets.itertuples(index=False):
            if s.exercise not in PRIMARY:
                continue
            m = (t >= s.t_start) & (t <= s.t_end)
            if m.sum() < 30:
                continue
            found = _reps_for_set(xyz[m], t[m], H36M17, s.exercise, engine)
            rows.append(
                {
                    "workout": w,
                    "subject": s.subject,
                    "set_id": s.set_id,
                    "exercise": s.exercise,
                    "reps_true": int(s.reps),
                    "reps_pose": len(found),
                }
            )
            reps.extend(found)
    return reps, pd.DataFrame(rows)


def evaluate(
    raw_root: Path | None = None,
    out: Path = DEFAULT_OUT,
    *,
    rules_path: Path | None = None,
    calibrate_out: Path | None = None,
    workouts: list[str] | None = None,
) -> dict:
    raw_root = raw_root or mmfit.DEFAULT_ROOT
    engine = RuleEngine.from_yaml(rules_path)
    reps, seg = collect_reps(raw_root, engine, workouts)
    result: dict = {"reps": {}, "pass_rate": {}, "perturbation": {}}
    by_ex: dict[str, list[RepMetrics]] = {}
    for r in reps:
        by_ex.setdefault(r.exercise, []).append(r)
    # rep segmentation agreement
    seg_rows = []
    for ex, g in seg.groupby("exercise"):
        err = g["reps_pose"] - g["reps_true"]
        seg_rows.append(
            {
                "exercise": ex,
                "sets": len(g),
                "reps_true": int(g["reps_true"].sum()),
                "reps_pose": int(g["reps_pose"].sum()),
                "mae": float(err.abs().mean()),
                "exact_pct": float((err == 0).mean() * 100),
            }
        )
        result["reps"][ex] = int(g["reps_pose"].sum())
    seg_table = pd.DataFrame(seg_rows)
    # pass rates
    pass_rows = []
    for ex in PRIMARY:
        g = by_ex.get(ex, [])
        for rule in engine.rules_for(ex):
            flagged = sum(any(f.code == rule.code for f in engine.evaluate(r)) for r in g)
            rate = 1 - flagged / len(g) if g else float("nan")
            result["pass_rate"][(ex, rule.code)] = rate
            thr = " or ".join(f"{c.metric} {c.op} {c.threshold:g}" for c in rule.when)
            pass_rows.append(
                {
                    "exercise": ex,
                    "rule": rule.code,
                    "reps": len(g),
                    "flagged": flagged,
                    "pass_rate": rate,
                    "condition": thr,
                }
            )
    pass_table = pd.DataFrame(pass_rows)
    # perturbations
    pert_rows = []
    for name, (ex, code) in PERTURBATIONS.items():
        g = by_ex.get(ex, [])
        hits = 0
        for r in g:
            a2, d2 = _perturb(name, r.angles, r.duration_s)
            m2 = metrics_from_angles(a2, d2)
            hits += any(
                f.code == code
                for f in engine.evaluate(RepMetrics(ex, 0, 0, d2, d2, "pose", a2, m2))
            )
        rate = hits / len(g) if g else float("nan")
        result["perturbation"][(ex, name, code)] = rate
        pert_rows.append(
            {
                "exercise": ex,
                "perturbation": name,
                "expected": code,
                "reps": len(g),
                "detected": hits,
                "detection_rate": rate,
            }
        )
    pert_table = pd.DataFrame(pert_rows)
    # metric distributions (for calibration)
    dist_rows = []
    keys = (
        "duration_s",
        "elbow_min",
        "elbow_max",
        "upper_arm_trunk_drift",
        "elbow_drift_max",
        "trunk_incl_max",
        "abd_peak",
        "elbow_at_abd_peak",
        "knee_min",
        "trunk_incl_at_bottom",
        "knee_track_at_bottom",
        "hip_knee_height_at_bottom",
        "elbow_asymmetry",
    )
    pct: dict[tuple[str, str], tuple[float, float, float]] = {}
    for ex, g in by_ex.items():
        for k in keys:
            v = np.array([r.metrics[k] for r in g], dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            pct[(ex, k)] = (float(p5), float(p50), float(p95))
            dist_rows.append(
                {"exercise": ex, "metric": k, "n": len(v), "p5": p5, "median": p50, "p95": p95}
            )
    dist_table = pd.DataFrame(dist_rows)
    fig = out.parent / "figures" / "rules_validation_metrics.png"
    _fig_metrics(by_ex, fig)
    parts = [
        report.header(
            "Rules validation (rules.yaml on MM-Fit 3-D pose)",
            "formcoach eval rules",
            extra={
                "rules": str(rules_path or "src/formcoach/rules/rules.yaml"),
                "pose source": "mmfit-pose3d (Human3.6M 17 joints, lifted from video)",
                "reps": sum(len(g) for g in by_ex.values()),
                "sets": len(seg),
            },
        ),
        "## Rep segmentation (adaptive pose thresholds vs MM-Fit set counts)",
        "",
        report.md_table(seg_table, floatfmt=".2f"),
        "",
        "## Pass rate per rule (MM-Fit reps taken as correct form; low = fires on normal reps)",
        "",
        report.md_table(pass_table, floatfmt=".3f"),
        "",
        "## Perturbation detection (synthetic faults applied to the same reps)",
        "",
        report.md_table(pert_table, floatfmt=".3f"),
        "",
        "## Metric distributions on correct-form reps (5th / 50th / 95th percentile)",
        "",
        report.md_table(dist_table, floatfmt=".2f"),
        "",
        report.relative_figure(fig, out),
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    if calibrate_out is not None:
        write_calibrated(engine, pct, calibrate_out)
    return result


def write_calibrated(engine: RuleEngine, pct: dict, path: Path) -> Path:
    """rules.yaml copy with each threshold at the 5th/95th percentile of correct-form reps:
    ``gt`` rules take the 95th percentile, ``lt`` rules the 5th (docs/02 §7 procedure)."""
    doc = copy.deepcopy(engine.to_doc())
    doc["pose_source"] = "mmfit-pose3d"
    doc["calibrated_from"] = {
        "pose_source": "mmfit-pose3d",
        "method": "5th/95th percentile of MM-Fit reps",
        "command": "formcoach eval rules --calibrate-out",
    }
    for ex, spec in doc["exercises"].items():
        for rule in spec["rules"]:
            for cond in rule["when"]:
                key = (ex, cond["metric"])
                if key not in pct:
                    continue
                p5, _, p95 = pct[key]
                if cond["op"] in ("gt", "ge"):
                    cond["threshold"] = round(p95, 3)
                elif cond["op"] in ("lt", "le"):
                    cond["threshold"] = round(p5, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _fig_metrics(by_ex: dict[str, list[RepMetrics]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("curl", "elbow_min"),
        ("press", "elbow_max"),
        ("raise", "abd_peak"),
        ("squat", "knee_min"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, (ex, k) in zip(axes, panels, strict=True):
        v = np.array([r.metrics[k] for r in by_ex.get(ex, [])], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) > 1 and np.ptp(v) > 1e-6:
            ax.hist(v, bins=min(20, len(v)), color="tab:blue")
        ax.set_title(f"{ex}: {k} (n={len(v)})", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
