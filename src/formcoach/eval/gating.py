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

from formcoach.app import pipeline, replay_recorded
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


def _without_imu(session: Path) -> bool:
    """Whether ``session``'s ``imu.parquet`` has no rows (a take recorded without a watch)."""
    try:
        return len(pd.read_parquet(session / "imu.parquet", columns=["t"])) == 0
    except (OSError, ValueError):
        return False  # the replay reports the file itself


def evaluate(
    sessions: list[Path],
    gates: tuple[str, ...] = ("always_on", "energy"),
    out: Path = DEFAULT_OUT,
    rules_path: Path | None = None,
) -> pd.DataFrame:
    """Replay every session once per gate and write the Markdown table to ``out``; returns it.

    Rules: ``rules_path`` (``--rules``) judges every session. Without it a Bench take is judged
    with its own rules and overrides (as recorded) and any other session with the default
    ``rules.yaml``. A gate that cannot run on a session is a row with an ``error`` instead of
    numbers: a Bench take without IMU samples skips every gate that needs them, and a replay
    that fails on a missing file (e.g. the laptop gate's model) or bad input is reported.
    """
    rules = RuleEngine.from_yaml(rules_path)
    rows = []
    for s in sessions:
        s = Path(s)
        bench = replay_recorded.is_bench_session(s)
        session_rules = None if bench and rules_path is None else rules
        no_imu = bench and _without_imu(s)
        base = None
        for g in gates:
            if no_imu and g not in replay_recorded.NO_IMU_GATES:
                rows.append(
                    {"session": s.name, "gate": g, "error": "skipped: the take has no IMU samples"}
                )
                continue
            try:
                r = run_session(s, g, session_rules)
            except (FileNotFoundError, ValueError) as exc:  # e.g. laptop gate without a model
                rows.append({"session": s.name, "gate": g, "error": str(exc)})
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
