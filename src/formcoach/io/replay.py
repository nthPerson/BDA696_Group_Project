"""Replay recorded sessions with no hardware (docs/02 §4.1 ``ReplaySource``).

A *session directory* holds ``imu.parquet`` (IMUStream schema), optionally ``pose.parquet``
(Pose schema) and ``meta.json``. The committed fixture ``data/fixtures/replay/mmfit_w00_curls``
is 30 s of MM-Fit; ``SessionRecorder`` (Checkpoint 4) writes the same layout for team sessions.
``speed=0`` replays as fast as possible; ``speed=1`` paces to real time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.io.source import SISample
from formcoach.pose.skeletons import Skeleton
from formcoach.pose.store import read_pose

FIXTURE_SESSION = (
    Path(__file__).resolve().parents[3] / "data" / "fixtures" / "replay" / "mmfit_w00_curls"
)


class ReplaySource:
    """Yield :class:`SISample` from ``imu.parquet`` (or any IMUStream DataFrame)."""

    def __init__(self, session: Path | str = FIXTURE_SESSION, speed: float = 0.0):
        session = Path(session)
        self.session_dir = session if session.is_dir() else session.parent
        imu_path = session / "imu.parquet" if session.is_dir() else session
        self.df = pd.read_parquet(imu_path)
        self.speed = speed
        meta_path = self.session_dir / "meta.json"
        self.meta: dict = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )

    @classmethod
    def from_stream(cls, df: pd.DataFrame, speed: float = 0.0) -> ReplaySource:
        obj = cls.__new__(cls)
        obj.session_dir = None
        obj.df = df.reset_index(drop=True)
        obj.speed = speed
        obj.meta = {}
        return obj

    def iter_samples(self) -> Iterator[SISample]:
        cols = self.df[["t", "ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float64)
        flags = (
            self.df["flags"].to_numpy() if "flags" in self.df.columns else np.zeros(len(cols), int)
        )
        wall0 = time.monotonic()
        t0 = cols[0, 0] if len(cols) else 0.0
        for i, row in enumerate(cols):
            if self.speed > 0:
                due = wall0 + (row[0] - t0) / self.speed
                lag = due - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
            yield SISample(*row.tolist(), int(flags[i]), i)

    def close(self) -> None:
        pass


@dataclass(slots=True)
class PoseFrame:
    frame: int
    t: float
    world: np.ndarray  # (J, 3)
    visibility: np.ndarray  # (J,)
    valid: bool
    skeleton: Skeleton


class PoseReplay:
    """Iterate :class:`PoseFrame` from a ``pose.parquet`` in time order."""

    def __init__(self, path: Path):
        self.seq = read_pose(path)

    @property
    def skeleton(self) -> Skeleton:
        return self.seq.skeleton

    def __len__(self) -> int:
        return len(self.seq.t)

    def __iter__(self) -> Iterator[PoseFrame]:
        s = self.seq
        for i in range(len(s.t)):
            yield PoseFrame(
                int(s.frames[i]),
                float(s.t[i]),
                s.world[i],
                s.visibility[i],
                bool(s.valid[i]),
                s.skeleton,
            )
