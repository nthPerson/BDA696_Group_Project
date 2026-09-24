"""``formcoach eval repcount``: peak-detection rep counts vs MM-Fit set labels (docs/02 §8).

For every labelled set in every MM-Fit workout, the watch stream inside the set's time span is
counted with :func:`formcoach.signal.reps.count_reps`; the report gives MAE, mean signed error,
% of sets exactly right and % within ±1 per exercise (canonical and raw) and per device.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.data import mmfit
from formcoach.eval import report
from formcoach.signal import reps
from formcoach.signal.resample import resample_stream

DEFAULT_OUT = report.REPORTS_DIR / "baseline_repcount.md"
FS = 50.0


def count_sets(
    processed_root: Path,
    raw_root: Path,
    devices: tuple[str, ...] = ("sw_l", "sw_r"),
    mode: str = "axis",
    prominence_g: float = reps.DEFAULT_PROMINENCE_G,
    min_distance_s: float = reps.DEFAULT_MIN_DISTANCE_S,
) -> pd.DataFrame:
    """One row per (workout, device, set): ``reps_true, reps_pred, error``."""
    rows = []
    for w in mmfit.list_workouts(raw_root):
        sets = mmfit.load_sets(raw_root, w)
        for dev in devices:
            path = processed_root / "mmfit" / "streams" / f"{mmfit.subject_id(w)}-{w}-{dev}.parquet"
            if not path.exists():
                continue
            df = resample_stream(pd.read_parquet(path), FS)
            t = df["t"].to_numpy()
            acc = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
            for s in sets.itertuples(index=False):
                m = (t >= s.t_start) & (t <= s.t_end)
                found = reps.count_reps(acc[m], t[m], FS, prominence_g=prominence_g,
                                        min_distance_s=min_distance_s, mode=mode)  # fmt: skip
                rows.append(
                    {
                        "workout": w,
                        "subject": s.subject,
                        "device": dev,
                        "set_id": s.set_id,
                        "activity": s.activity,
                        "exercise": s.exercise,
                        "duration_s": float(s.t_end - s.t_start),
                        "reps_true": int(s.reps),
                        "reps_pred": len(found),
                    }
                )
    out = pd.DataFrame(rows)
    out["error"] = out["reps_pred"] - out["reps_true"]
    return out


def summarize(counts: pd.DataFrame, by: str) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        e = g["error"]
        return pd.Series(
            {
                "n_sets": len(g),
                "mae": float(e.abs().mean()),
                "bias": float(e.mean()),
                "exact_pct": float((e == 0).mean() * 100),
                "within1_pct": float((e.abs() <= 1).mean() * 100),
            }
        )

    table = counts.groupby(by, sort=True).apply(agg, include_groups=False).reset_index()
    total = agg(counts)
    total[by] = "all"
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)[
        [by, "n_sets", "mae", "bias", "exact_pct", "within1_pct"]
    ]


def evaluate(
    processed_root: Path,
    raw_root: Path,
    out: Path = DEFAULT_OUT,
    devices: tuple[str, ...] = ("sw_l", "sw_r"),
    mode: str = "axis",
    prominence_g: float = reps.DEFAULT_PROMINENCE_G,
    min_distance_s: float = reps.DEFAULT_MIN_DISTANCE_S,
) -> pd.DataFrame:
    """Run the count on every set, write the report, return the per-exercise summary table."""
    counts = count_sets(processed_root, raw_root, devices, mode, prominence_g, min_distance_s)
    by_ex = summarize(counts, "exercise")
    by_raw = summarize(counts, "activity")
    by_dev = summarize(counts, "device")
    fig = out.parent / "figures" / "baseline_repcount_errors.png"
    _fig_errors(counts, fig)
    parts = [
        report.header(
            "Rep-count baseline (peak detection on MM-Fit smartwatch streams)",
            "formcoach eval repcount --source peaks",
            extra={
                "method": f"band-pass {reps.BAND[0]}-{reps.BAND[1]} Hz, mode={mode}, prominence "
                f"{prominence_g} g, min distance {min_distance_s} s (docs/02 §4.4 defaults)",
                "devices": ", ".join(devices),
                "sets": len(counts),
            },
        ),
        "## Per canonical exercise",
        "",
        report.md_table(by_ex, floatfmt=".2f"),
        "",
        "## Per MM-Fit activity",
        "",
        report.md_table(by_raw, floatfmt=".2f"),
        "",
        "## Per device",
        "",
        report.md_table(by_dev, floatfmt=".2f"),
        "",
        report.relative_figure(fig, out),
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    counts.to_csv(out.with_suffix(".csv"), index=False)
    return by_ex


def _fig_errors(counts: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    order = sorted(counts["activity"].unique())
    data = [counts.loc[counts["activity"] == a, "error"].to_numpy() for a in order]
    ax.boxplot(data, tick_labels=order, showfliers=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("predicted - true reps")
    ax.set_title("Rep-count error per set")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
