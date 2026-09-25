"""``formcoach eval gating`` (docs/02 §8): gated vs always-on on identical recorded sessions.

Every session directory is replayed once per gate. The report lists frames processed %, wall
time, CPU % (psutil when installed, else process time / wall time), rep and fault counts, and
whether the rep/fault events match the always-on run of the same session (the correctness
side of the efficiency claim)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from formcoach.app import pipeline
from formcoach.eval import report
from formcoach.rules.engine import RuleEngine

DEFAULT_OUT = report.REPORTS_DIR / "gating.md"
GATES = ("always_on", "energy", "laptop", "device")


def _cpu_percent_start():
    try:
        import psutil

        p = psutil.Process(os.getpid())
        p.cpu_percent(None)
        return p
    except ImportError:
        return None


def run_session(session: Path, gate: str, rules: RuleEngine | None) -> dict:
    proc = _cpu_percent_start()
    t_cpu0 = time.process_time()
    t0 = time.perf_counter()
    res = pipeline.run_replay(session, gate=gate, headless=True, rules=rules)
    wall = time.perf_counter() - t0
    cpu_pct = (
        proc.cpu_percent(None) if proc else 100.0 * (time.process_time() - t_cpu0) / max(wall, 1e-9)
    )
    reps = [e for e in res.events if e.kind == "rep"]
    faults = sorted(e.payload.get("code", "") for e in res.events if e.kind == "fault")
    return {
        "session": session.name,
        "gate": gate,
        "frames_total": res.frames_total,
        "frames_processed": res.frames_processed,
        "frames_processed_pct": round(100.0 * res.frames_processed / max(res.frames_total, 1), 1),
        "gated_s": res.summary["gated_seconds"],
        "imu_s": res.summary["imu_seconds"],
        "wall_s": round(wall, 3),
        "cpu_pct": round(float(cpu_pct), 1),
        "reps": len(reps),
        "faults": len(faults),
        "_faults": faults,
    }


def evaluate(
    sessions: list[Path],
    gates: tuple[str, ...] = ("always_on", "energy"),
    out: Path = DEFAULT_OUT,
    rules_path: Path | None = None,
) -> pd.DataFrame:
    rules = RuleEngine.from_yaml(rules_path)
    rows = []
    for s in sessions:
        base = None
        for g in gates:
            try:
                r = run_session(Path(s), g, rules)
            except FileNotFoundError as exc:  # e.g. laptop gate without a trained model
                rows.append({"session": Path(s).name, "gate": g, "error": str(exc)})
                continue
            if g == "always_on":
                base = r
            ref = base
            r["reps_match"] = bool(ref is not None and r["reps"] == ref["reps"])
            r["faults_match"] = bool(ref is not None and r["_faults"] == ref["_faults"])
            rows.append(r)
    table = pd.DataFrame(rows).drop(columns=["_faults"], errors="ignore")
    if "reps_match" in table.columns:
        for c in ("reps_match", "faults_match"):
            table[c] = table[c].astype("boolean")
    parts = [
        report.header(
            "Gating benchmark (replayed sessions)",
            "formcoach eval gating",
            extra={
                "sessions": len(sessions),
                "gates": ", ".join(gates),
                "note": "identical events vs always_on are the correctness check; CPU % is "
                "process CPU during the replay (psutil when installed)",
            },
        ),
        report.md_table(table, floatfmt=".1f"),
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return table
