"""``formcoach eval device --log <file>``: on-device metrics from a captured serial log.

Firmware v1.1 prints ``# infer us=<n> arena=<bytes> class=<k> p_active=<p>`` after every gate
inference and ``# flash app=<bytes>`` at boot. This parses those comment lines into
``reports/device.md`` (inference-time distribution, arena bytes, flash) so the on-device
numbers in the report come from a log file checked into ``reports/logs/``."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.eval import report

DEFAULT_OUT = report.REPORTS_DIR / "device.md"
INFER = re.compile(
    r"#\s*infer\s+us=(\d+)\s+arena=(\d+)(?:\s+class=(\d+))?(?:\s+p_active=([0-9.]+))?"
)
FLASH = re.compile(r"#\s*flash\s+app=(\d+)")
FW = re.compile(r"#\s*FormCoach fw\s+(\S+)")


def parse(text: str) -> dict:
    """Parse a serial capture: ``infer_us`` (per-inference microseconds, array), ``arena``
    (max tensor-arena bytes), ``flash`` (app bytes), ``firmware`` version, ``classes``."""
    us, arena, classes = [], [], []
    flash = None
    fw = None
    for line in text.splitlines():
        m = INFER.search(line)
        if m:
            us.append(int(m.group(1)))
            arena.append(int(m.group(2)))
            if m.group(3) is not None:
                classes.append(int(m.group(3)))
            continue
        m = FLASH.search(line)
        if m:
            flash = int(m.group(1))
        m = FW.search(line)
        if m:
            fw = m.group(1)
    return {"n": len(us), "infer_us": np.array(us), "arena": max(arena) if arena else None,
            "flash": flash, "firmware": fw, "classes": np.array(classes)}  # fmt: skip


def evaluate(log: Path, out: Path = DEFAULT_OUT) -> Path:
    """Write ``reports/device.md`` (inference ms median/p95/max, arena and flash bytes) from
    ``log`` and return its path."""
    d = parse(Path(log).read_text(encoding="utf-8", errors="replace"))
    rows = []
    if d["n"]:
        us = d["infer_us"]
        med, p95, mx = np.median(us) / 1000, np.percentile(us, 95) / 1000, us.max() / 1000
        rows.append(
            {
                "metric": "inference time (ms) median / p95 / max",
                "value": f"{med:.2f} / {p95:.2f} / {mx:.2f}",
            }
        )
    rows.append({"metric": "inferences parsed", "value": d["n"]})
    rows.append(
        {
            "metric": "tensor arena used (bytes)",
            "value": d["arena"] if d["arena"] is not None else "n/a",
        }
    )
    rows.append(
        {"metric": "app flash (bytes)", "value": d["flash"] if d["flash"] is not None else "n/a"}
    )
    rows.append({"metric": "firmware", "value": d["firmware"] or "n/a"})
    if len(d["classes"]):
        counts = np.bincount(d["classes"])
        rows.append(
            {
                "metric": "class histogram",
                "value": ", ".join(f"{i}: {c}" for i, c in enumerate(counts)),
            }
        )
    parts = [
        report.header("On-device gate metrics", f"formcoach eval device --log {log}", seed=None),
        report.md_table(pd.DataFrame(rows)),
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
