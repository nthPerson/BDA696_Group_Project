"""``formcoach data profile`` → ``reports/data_profile.md`` (+ ``reports/figures/data_profile_*.png``).

Per dataset: subjects, sessions, minutes, sampling rate, duplicate/missing timestamps, class
balance in canonical labels; one figure of minutes per canonical label per dataset and one
30-second MM-Fit example with the labelled sets shaded. Numbers come from the loaders'
``describe()`` so the report and the code never disagree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from formcoach.data import loader, schema
from formcoach.eval import report

DEFAULT_OUT = report.REPORTS_DIR / "data_profile.md"


def _minutes_table(roots: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, root in roots.items():
        mod = loader(name)
        if not mod.available(root):
            continue
        d = mod.describe(root)
        if name == "mmfit":
            sets = d["sets_per_exercise"]
            # minutes per label from the streams themselves
            mins = {}
            for w in mod.list_workouts(root):
                s = mod.load_sets(root, w)
                for r in s.itertuples(index=False):
                    mins[r.exercise] = mins.get(r.exercise, 0.0) + (r.t_end - r.t_start) / 60.0
            total_min = sum(v["minutes"] for k, v in d["devices"].items() if k == "sw_l")
            mins["idle"] = max(total_min - sum(mins.values()), 0.0)
            for k in schema.CANONICAL_EXERCISES:
                rows.append((name, k, round(mins.get(k, 0.0), 1), sets.get(k, 0)))
        elif name == "recofit":
            for k in schema.CANONICAL_EXERCISES:
                rows.append((name, k, d["multi"]["minutes_per_exercise"].get(k, 0.0), ""))
        elif name == "recgym":
            from formcoach.data import labels

            mins = {}
            for raw, m in d["minutes_per_workout"].items():
                c = labels.canonical("recgym", raw)
                mins[c] = mins.get(c, 0.0) + m
            for k in schema.CANONICAL_EXERCISES:
                rows.append((name, k, round(mins.get(k, 0.0), 1), ""))
    return pd.DataFrame(rows, columns=["dataset", "label", "minutes", "sets"])


def _fig_minutes(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = table.pivot(index="label", columns="dataset", values="minutes").reindex(
        list(schema.CANONICAL_EXERCISES)
    )
    ax = piv.plot.bar(figsize=(7, 3.5), logy=True)
    ax.set_ylabel("minutes (log)")
    ax.set_title("Minutes per canonical label")
    ax.figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(path, dpi=120)
    plt.close(ax.figure)


def _fig_mmfit_example(root: Path, path: Path) -> bool:
    mod = loader("mmfit")
    if not mod.available(root):
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = mod.list_workouts(root)[0]
    df = mod.load_stream(root, w, "sw_l")
    sets = mod.load_sets(root, w)
    first = sets.iloc[0]
    t0, t1 = first.t_start - 5, first.t_start + 25
    seg = df[(df["t"] >= t0) & (df["t"] <= t1)]
    fig, ax = plt.subplots(figsize=(8, 3))
    for c in ("ax", "ay", "az"):
        ax.plot(seg["t"], seg[c], lw=0.7, label=c)
    for s in sets.itertuples(index=False):
        if s.t_end >= t0 and s.t_start <= t1:
            ax.axvspan(s.t_start, s.t_end, color="orange", alpha=0.2)
            ax.text(s.t_start, ax.get_ylim()[1] * 0.9, f"{s.exercise} x{s.reps}", fontsize=8)
    ax.set_xlabel("t (s, session clock)")
    ax.set_ylabel("m/s²")
    ax.set_title(f"MM-Fit {w} sw_l accelerometer, 30 s around the first set")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _mmfit_section(d: dict) -> str:
    dev = pd.DataFrame(
        [
            {
                "device": k,
                "placement": v["placement"],
                "workouts": v["workouts"],
                "minutes": v["minutes"],
                "rate_hz": v["rate_hz"],
                "duplicate_ts": v["duplicate_timestamps"],
            }
            for k, v in d["devices"].items()
        ]
    )
    sets = pd.DataFrame(
        [
            {"label": k, "sets": d["sets_per_exercise"].get(k, 0), "reps": d["reps_per_exercise"].get(k, 0)}
            for k in ("curl", "press", "raise", "squat", "other")
        ]
    )
    missing = ", ".join(f"{w}: {'/'.join(m)}" for w, m in d["missing_modalities"].items()) or "none"
    return "\n".join(
        [
            "## MM-Fit",
            "",
            f"- workouts: {d['workouts']} · subjects: {d['subjects']} · labelled minutes: "
            f"{d['minutes_labeled']}",
            f"- missing modalities: {missing}",
            "",
            report.md_table(dev, floatfmt=".1f"),
            "",
            report.md_table(sets),
            "",
        ]
    )


def _recofit_section(d: dict) -> str:
    m = d["multi"]
    mins = pd.DataFrame(
        [{"label": k, "minutes": v} for k, v in m["minutes_per_exercise"].items()]
    )
    single = d.get("single", {})
    return "\n".join(
        [
            "## RecoFit",
            "",
            f"- subjects: {m['subjects']} · visits: {m['visits']} · hours: {m['hours']} · "
            f"rate: {m['rate_hz']} Hz · activity labels: {d['activities']} · "
            f"junk-labelled minutes (dropped): {m['junk_minutes']}",
            f"- single-activity file: {single.get('recordings', 'n/a')} recordings, "
            f"cell matrix {single.get('shape', 'n/a')}",
            "",
            report.md_table(mins, floatfmt=".1f"),
            "",
        ]
    )


def _recgym_section(d: dict) -> str:
    mins = pd.DataFrame(
        [{"workout": k, "minutes": v} for k, v in d["minutes_per_workout"].items()]
    )
    return "\n".join(
        [
            "## RecGym",
            "",
            f"- rows: {d['rows']:,} · subjects: {d['subjects']} · sessions: {d['sessions']} · "
            f"positions: {', '.join(d['positions'])} · rate: {d['rate_hz']} Hz · "
            f"hours: {d['hours']}",
            f"- units: {d['units']} (value range {d['value_range']}) — see ADR-0014",
            "",
            report.md_table(mins, floatfmt=".1f"),
            "",
        ]
    )


def write_profile(roots: dict[str, Path], out_md: Path = DEFAULT_OUT) -> Path:
    """Write the profile report for every dataset whose raw files exist; return its path."""
    fig_dir = out_md.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    parts = [report.header("Data profile", "formcoach data profile")]
    present = {n: r for n, r in roots.items() if loader(n).available(r)}
    parts.append(
        "Datasets present: " + (", ".join(present) or "none") + ". Raw roots are gitignored; "
        "run `make data DATASET=<name>` to fetch."
    )
    parts.append("")
    for name, root in present.items():
        d = loader(name).describe(root)
        parts.append({"mmfit": _mmfit_section, "recofit": _recofit_section, "recgym": _recgym_section}[name](d))
    table = _minutes_table(present)
    parts.append("## Minutes per canonical label")
    parts.append("")
    parts.append(report.md_table(table, floatfmt=".1f"))
    parts.append("")
    if not table.empty:
        fig = fig_dir / "data_profile_minutes.png"
        _fig_minutes(table, fig)
        parts.append(report.relative_figure(fig, out_md))
        parts.append("")
    if "mmfit" in present:
        fig = fig_dir / "data_profile_mmfit_example.png"
        if _fig_mmfit_example(present["mmfit"], fig):
            parts.append("## MM-Fit example")
            parts.append("")
            parts.append(report.relative_figure(fig, out_md))
            parts.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return out_md


def default_roots() -> dict[str, Path]:
    return {n: loader(n).DEFAULT_ROOT for n in ("mmfit", "recofit", "recgym")}


