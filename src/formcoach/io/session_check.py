"""``formcoach session check <dir>``: quality report for a recorded or replay session —
sample count, rate, drops (from ``seq``), gate timeline, pose frames, reps vs expected.

A session whose ``meta.json`` labels the working ``arm`` (a Bench take: ``both``, ``l`` or ``r``)
is checked per arm (Bench design §7): ``expected_reps`` is the count for each working arm, so a
clean two-arm set of 10 is 10 left and 10 right, not 20 rep events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARM_NAMES = {"l": "left", "r": "right"}


def check(session_dir: Path) -> dict:
    """The quality report of one session folder (``imu.parquet`` required; ``meta.json``,
    ``pose.parquet``, ``frames.parquet``, ``events.parquet`` used when present): ``n_samples``,
    ``duration_s`` (s), ``rate_hz``, ``dropped`` (from ``seq``), ``gate_open_pct`` (% of samples
    with the gate flag; ``None`` without samples), ``pose_frames``, ``frames_processed_pct``,
    ``reps`` and ``faults`` (event counts), ``expected_reps`` and ``problems``.

    With an ``arm`` in ``meta.json`` the report adds ``reps_by_arm`` (``{"l": n, "r": n}``, or
    ``{"all": n}`` for squat, ``{}`` for idle; ``None`` without events) and ``notes``. Each
    working arm's count (both arms for ``both``) is compared with ``expected_reps``, and a
    session without IMU samples gets the note ``"no IMU samples"`` rather than a problem."""
    session_dir = Path(session_dir)
    imu = pd.read_parquet(session_dir / "imu.parquet")
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    t = imu["t"].to_numpy()
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    rate = float(len(t) / duration) if duration > 0 else 0.0
    dropped = None
    if "seq" in imu.columns and len(imu) > 1:
        gaps = (np.diff(imu["seq"].to_numpy().astype(np.int64)) - 1) % 65536
        dropped = int(gaps[gaps < 32768].sum())
    gate_pct = None
    if "flags" in imu.columns and len(imu):
        gate_pct = float((imu["flags"].to_numpy().astype(int) & 1).mean() * 100)
    report = {
        "session": str(session_dir),
        "subject": meta.get("subject", imu["subject"].iloc[0] if len(imu) else None),
        "exercise": meta.get("exercise", imu["exercise"].iloc[0] if len(imu) else None),
        "n_samples": len(imu),
        "duration_s": duration,
        "rate_hz": rate,
        "dropped": dropped,
        "gate_open_pct": gate_pct,
        "pose_frames": None,
        "frames_processed_pct": None,
        "reps": None,
        "expected_reps": meta.get("expected_reps"),
        "faults": None,
        "problems": [],
    }
    if (session_dir / "pose.parquet").exists():
        pose = pd.read_parquet(session_dir / "pose.parquet", columns=["frame"])
        report["pose_frames"] = len(pose)
    if (session_dir / "frames.parquet").exists():
        fr = pd.read_parquet(session_dir / "frames.parquet")
        report["frames_processed_pct"] = float(fr["processed"].mean() * 100) if len(fr) else 0.0
    ev = None
    if (session_dir / "events.parquet").exists():
        ev = pd.read_parquet(session_dir / "events.parquet")
        report["reps"] = int((ev["kind"] == "rep").sum())
        report["faults"] = int((ev["kind"] == "fault").sum())
    per_arm = "arm" in meta
    if per_arm:
        report["reps_by_arm"] = None if ev is None else _reps_by_arm(report["exercise"], ev)
        report["notes"] = [] if len(imu) else ["no IMU samples"]
    if rate and abs(rate - 50) > 5 and meta.get("source") in ("serial", "ble", "fake"):
        report["problems"].append(f"sample rate {rate:.1f} Hz is not 50 Hz")
    if dropped:
        report["problems"].append(f"{dropped} dropped samples")
    if per_arm:
        report["problems"].extend(_arm_problems(report, meta.get("arm")))
    elif (
        report["reps"] is not None
        and report["expected_reps"] is not None
        and report["reps"] != report["expected_reps"]
    ):
        report["problems"].append(f"reps {report['reps']} != expected {report['expected_reps']}")
    return report


def _reps_by_arm(exercise: str | None, ev: pd.DataFrame) -> dict[str, int]:
    """Rep events per counter: ``{"all": n}`` for squat, ``{}`` for idle, else ``{"l", "r"}``
    from each event's ``arm``."""
    if exercise == "idle":
        return {}
    reps = ev[ev["kind"] == "rep"]
    if exercise == "squat":
        return {"all": len(reps)}
    arms = [json.loads(p).get("arm") for p in reps["payload"]]
    return {k: arms.count(k) for k in ARM_NAMES}


def _arm_problems(report: dict, arm: str | None) -> list[str]:
    """Each working arm's count against ``expected_reps``: for ``arm`` ``l`` or ``r`` that arm,
    for ``both`` (or anything else) each arm; for squat its one counter, ``all``."""
    by_arm, expected = report["reps_by_arm"], report["expected_reps"]
    if not by_arm or expected is None:
        return []
    keys = ["all"] if "all" in by_arm else [arm] if arm in ARM_NAMES else list(ARM_NAMES)
    out = []
    for k in keys:
        n = by_arm.get(k, 0)
        if n != expected:
            who = "reps" if k == "all" else f"{ARM_NAMES[k]} arm reps"
            out.append(f"{who} {n} != expected {expected}")
    return out


def format_report(r: dict) -> str:
    def fmt(v, unit=""):
        return "n/a" if v is None else (f"{v:.1f}{unit}" if isinstance(v, float) else f"{v}{unit}")

    lines = [
        f"session  {r['session']}",
        f"subject  {r['subject']}   exercise {r['exercise']}",
        f"samples  {r['n_samples']}   duration {fmt(r['duration_s'], ' s')}   "
        f"rate {fmt(r['rate_hz'], ' Hz')}",
        f"dropped  {fmt(r['dropped'])}   gate open {fmt(r['gate_open_pct'], '%')}",
        f"pose     frames {fmt(r['pose_frames'])}   "
        f"processed {fmt(r['frames_processed_pct'], '%')}",
        f"reps     {fmt(r['reps'])} (expected {fmt(r['expected_reps'])})   "
        f"faults {fmt(r['faults'])}",
    ]
    if r.get("reps_by_arm"):
        lines[-1] += "   by arm " + ", ".join(f"{k} {n}" for k, n in r["reps_by_arm"].items())
    if r.get("notes"):
        lines.append("notes    " + "; ".join(r["notes"]))
    lines.append("problems " + ("; ".join(r["problems"]) if r["problems"] else "none"))
    return "\n".join(lines)
