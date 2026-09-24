"""``formcoach session check <dir>``: quality report for a recorded or replay session —
sample count, rate, drops (from ``seq``), gate timeline, pose frames, reps vs expected."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def check(session_dir: Path) -> dict:
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
    if "flags" in imu.columns:
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
    if (session_dir / "events.parquet").exists():
        ev = pd.read_parquet(session_dir / "events.parquet")
        report["reps"] = int((ev["kind"] == "rep").sum())
        report["faults"] = int((ev["kind"] == "fault").sum())
    if rate and abs(rate - 50) > 5 and meta.get("source") in ("serial", "ble", "fake"):
        report["problems"].append(f"sample rate {rate:.1f} Hz is not 50 Hz")
    if dropped:
        report["problems"].append(f"{dropped} dropped samples")
    if (
        report["reps"] is not None
        and report["expected_reps"] is not None
        and report["reps"] != report["expected_reps"]
    ):
        report["problems"].append(f"reps {report['reps']} != expected {report['expected_reps']}")
    return report


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
    lines.append("problems " + ("; ".join(r["problems"]) if r["problems"] else "none"))
    return "\n".join(lines)
