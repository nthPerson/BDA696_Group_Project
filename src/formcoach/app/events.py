"""Session event log: rep / fault / gate events, per-frame processed-or-skipped log, and the
``events.parquet`` + ``frames.parquet`` + ``session.json`` files (docs/02 §4.6)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Event:
    kind: str  # gate_open | gate_close | rep | fault | info
    t: float
    payload: dict = field(default_factory=dict)


class EventLog:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.frames: list[tuple[float, bool, bool]] = []  # t, processed, gate_state

    def event(self, kind: str, t: float, **payload) -> Event:
        e = Event(kind, float(t), payload)
        self.events.append(e)
        return e

    def frame(self, t: float, processed: bool, gate_state: bool) -> None:
        self.frames.append((float(t), bool(processed), bool(gate_state)))

    @property
    def frames_total(self) -> int:
        return len(self.frames)

    @property
    def frames_processed(self) -> int:
        return sum(p for _, p, _ in self.frames)

    def write(self, out_dir: Path, summary: dict) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ev = pd.DataFrame(
            {
                "kind": pd.Series([e.kind for e in self.events], dtype="str"),
                "t": pd.Series([e.t for e in self.events], dtype="float64"),
                "payload": pd.Series(
                    [json.dumps(e.payload, default=_json_default) for e in self.events], dtype="str"
                ),
            }
        )
        ev.to_parquet(out_dir / "events.parquet", index=False)
        fr = pd.DataFrame(self.frames, columns=["t", "processed", "gate_state"])
        fr.to_parquet(out_dir / "frames.parquet", index=False)
        (out_dir / "session.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
