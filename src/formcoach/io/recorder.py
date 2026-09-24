"""``SessionRecorder`` (docs/02 §4.1): writes ``data/team/<S#>/<session_id>/imu.parquet`` in the
IMUStream schema (+ ``flags``, ``seq`` columns) and ``meta.json``. Video and pose files are
added by the app when a camera is used; ``video.mp4`` never leaves the recording laptop."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

from formcoach import __version__
from formcoach.data import schema
from formcoach.io.source import SISample

TEAM_ROOT = Path(__file__).resolve().parents[3] / "data" / "team"


class SessionRecorder:
    def __init__(
        self,
        subject: str,
        exercise: str,
        root: Path = TEAM_ROOT,
        *,
        source: str = "serial",
        firmware: str | None = None,
        device_name: str | None = None,
        camera: int | None = None,
        notes: str = "",
        placement: str = "wrist_l",
    ):
        if exercise not in schema.CANONICAL_EXERCISES:
            raise ValueError(f"exercise must be one of {schema.CANONICAL_EXERCISES}")
        self.subject, self.exercise, self.root = subject, exercise, Path(root)
        self.meta = {
            "subject": subject,
            "exercise": exercise,
            "placement": placement,
            "source": source,
            "firmware": firmware,
            "device_name": device_name,
            "camera": camera,
            "notes": notes,
            "laptop": platform.node(),
            "os": platform.platform(),
            "formcoach": __version__,
        }
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.root / subject / self.session_id
        self._rows: list[tuple] = []
        self._started: str | None = None

    def __enter__(self) -> SessionRecorder:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.finish()

    def start(self) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._started = datetime.now().isoformat(timespec="seconds")
        return self.session_dir

    def add(self, s: SISample) -> None:
        self._rows.append((s.t, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq, s.t_arrival))

    def finish(self, extra_meta: dict | None = None) -> Path:
        arr = np.array(self._rows, dtype=np.float64) if self._rows else np.zeros((0, 10))
        t = arr[:, 0]
        keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.zeros(0, bool)
        arr = arr[keep]
        df = schema.make_imu_stream(
            dataset="team",
            subject=self.subject,
            session=self.session_id,
            device="formcoach",
            placement=self.meta["placement"],
            t=arr[:, 0],
            acc=arr[:, 1:4],
            gyr=arr[:, 4:7],
            exercise=np.full(len(arr), self.exercise, dtype=object),
        )
        df["flags"] = arr[:, 7].astype(np.int16)
        df["seq"] = arr[:, 8].astype(np.int32)
        df["t_arrival"] = arr[:, 9]
        schema.validate_imu_stream(df)
        df.to_parquet(self.session_dir / "imu.parquet", index=False)
        seq = df["seq"].to_numpy()
        gaps = int(((np.diff(seq) - 1) % 65536).clip(0, 32767).sum()) if len(seq) > 1 else 0
        meta = {
            **self.meta,
            "session_id": self.session_id,
            "started": self._started,
            "ended": datetime.now().isoformat(timespec="seconds"),
            "n_samples": len(df),
            "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) > 1 else 0.0,
            "dropped": gaps,
            **(extra_meta or {}),
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return self.session_dir
