"""Session event log: rep / fault / gate events, per-frame processed-or-skipped log, and the
``events.parquet`` + ``frames.parquet`` + ``session.json`` files (docs/02 §4.6)."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Event:
    # pipeline: gate_open | gate_close | rep | fault | rep_dropped (live only, opt-in);
    # live session: error | gate_source | exercise | counters_reset | link; also info
    kind: str
    t: float
    payload: dict = field(default_factory=dict)


class EventLog:
    """Events and per-frame rows of one run.

    By default everything is kept. ``max_events`` / ``max_frames`` bound what is held (the
    newest are kept; ``max_frames=0`` keeps no frame rows), for a session that runs for hours
    and is never written (the Bench). :attr:`frames_total` and :attr:`frames_processed` always
    count every frame logged, held or not.
    """

    def __init__(self, *, max_events: int | None = None, max_frames: int | None = None) -> None:
        self.events: list[Event] | deque[Event] = (
            [] if max_events is None else deque(maxlen=max_events)
        )
        # t, processed, gate_state
        self.frames: list[tuple[float, bool, bool]] | deque[tuple[float, bool, bool]] = (
            [] if max_frames is None else deque(maxlen=max_frames)
        )
        # per-frame diagnostics, parallel to frames
        self._extras: list[dict[str, float]] | deque[dict[str, float]] = (
            [] if max_frames is None else deque(maxlen=max_frames)
        )
        self._n_frames = 0
        self._n_processed = 0

    def event(self, kind: str, t: float, **payload) -> Event:
        e = Event(kind, float(t), payload)
        self.events.append(e)
        return e

    def frame(self, t: float, processed: bool, gate_state: bool, **extra: float) -> None:
        """Log one frame. ``extra`` (live mode: segmenter input, thresholds, visibility) becomes
        extra ``frames.parquet`` columns, NaN on frames that gave none; with no extras at all
        (replay) the file keeps exactly ``t, processed, gate_state``."""
        self.frames.append((float(t), bool(processed), bool(gate_state)))
        self._extras.append(extra)
        self._n_frames += 1
        self._n_processed += bool(processed)

    @property
    def frames_total(self) -> int:
        """Every frame logged (also those a bounded log no longer holds)."""
        return self._n_frames

    @property
    def frames_processed(self) -> int:
        """Every processed frame logged (also those a bounded log no longer holds)."""
        return self._n_processed

    def write(self, out_dir: Path, summary: dict) -> None:
        """Write ``events.parquet``, ``frames.parquet`` (:meth:`write_tables`) and
        ``session.json`` (``summary``) into ``out_dir``, creating it."""
        self.write_tables(out_dir)
        (out_dir / "session.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )

    def write_tables(self, out_dir: Path) -> None:
        """Write ``events.parquet`` (``kind, t, payload`` as JSON) and ``frames.parquet``
        (``t, processed, gate_state`` plus one column per frame diagnostic) into ``out_dir``,
        creating it. ``t`` is seconds, as logged."""
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
        fr = pd.DataFrame(list(self.frames), columns=["t", "processed", "gate_state"])
        for key in sorted({k for e in self._extras for k in e}):
            fr[key] = pd.Series([e.get(key, float("nan")) for e in self._extras], dtype="float64")
        fr.to_parquet(out_dir / "frames.parquet", index=False)


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
