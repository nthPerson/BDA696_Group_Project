"""RecGym loader (Bian & Lukowicz, UCI 1128 / Kaggle mirror) → IMUStream.

Verified against ``RecGym.csv`` on 2026-09-24 (docs/04-datasets.md §3):

* Columns are ``Subject, Position, Session, A_x, A_y, A_z, G_x, G_y, G_z, C_1, Workout``
  (4,703,320 rows; the docs' ``Object`` column and 4,432,070 count refer to an older release).
* 10 subjects × 5 sessions × 3 positions (``wrist``, ``pocket``, ``leg``); 12 workouts; 20 Hz.
* **All signal columns are min-max normalised to [0, 1]**: there are no physical units and no
  timestamps. The loader centres them (``x - 0.5``), sets ``units == "normalized"`` and builds a
  synthetic 20 Hz clock. RecGym is therefore only a within-dataset robustness check for the
  resampler and classifier (ADR-0014); it is never mixed into SI training data.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.data import labels, schema

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "external" / "recgym"
CSV_NAME = "RecGym.csv"
FS_HZ = 20.0
PLACEMENT = {"wrist": "wrist", "pocket": "pocket", "leg": "calf"}
SIGNALS = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]
DTYPES = {"Subject": "int16", "Session": "int8", "Position": "str", "Workout": "str"}
DTYPES.update({c: "float32" for c in SIGNALS + ["C_1"]})


def available(root: Path = DEFAULT_ROOT) -> bool:
    return (root / CSV_NAME).exists()


@functools.lru_cache(maxsize=1)
def load_csv(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """The whole CSV (~4.7 M rows, ~250 MB in memory), cached per process."""
    return pd.read_csv(root / CSV_NAME, dtype=DTYPES)


def list_sessions(root: Path = DEFAULT_ROOT) -> list[tuple[int, str, int]]:
    """``(subject, position, session)`` triples in file order."""
    df = load_csv(root)
    keys = df[["Subject", "Position", "Session"]].drop_duplicates()
    return [(int(s), str(p), int(v)) for s, p, v in keys.itertuples(index=False)]


def load_stream(root: Path, subject: int, position: str, session: int) -> pd.DataFrame:
    """IMUStream for one (subject, position, session) block.

    ``subject`` → ``G<n>``, ``session`` → ``s<n>-<position>`` (one stream per position),
    ``t`` = row index / 20 Hz (the file has no timestamps), signals centred on 0.5 in
    normalised units (``units == "normalized"``). ``set_id`` numbers contiguous non-idle runs.
    Extra column ``label_raw`` keeps the RecGym workout name.
    """
    df = load_csv(root)
    m = (df["Subject"] == subject) & (df["Position"] == position) & (df["Session"] == session)
    block = df.loc[m]
    if block.empty:
        raise KeyError(f"no rows for subject={subject} position={position} session={session}")
    n = len(block)
    raw = block["Workout"].to_numpy(dtype=object)
    exercise = np.array([labels.canonical("recgym", r) for r in raw], dtype=object)
    runs = np.concatenate([[True], raw[1:] != raw[:-1]]).cumsum() - 1
    set_id = np.where(exercise == "idle", -1, runs).astype(np.int32)
    sig = block[SIGNALS].to_numpy(dtype=np.float32) - 0.5
    out = schema.make_imu_stream(
        dataset="recgym",
        subject=f"G{subject}",
        session=f"s{session}-{position}",
        device="recgym_unit",
        placement=PLACEMENT[position],
        t=np.arange(n, dtype=np.float64) / FS_HZ,
        acc=sig[:, :3],
        gyr=sig[:, 3:],
        exercise=exercise,
        set_id=set_id,
        units="normalized",
    )
    out["label_raw"] = pd.Series(raw, dtype="str")
    return out


def iter_streams(root: Path = DEFAULT_ROOT, positions: tuple[str, ...] = ("wrist",)):
    """Yield ``((subject, position, session), IMUStream)`` for the requested positions."""
    for key in list_sessions(root):
        if key[1] in positions:
            yield key, load_stream(root, *key)


def describe(root: Path = DEFAULT_ROOT) -> dict:
    df = load_csv(root)
    minutes = (df.groupby("Workout").size() / FS_HZ / 60.0).round(1)
    return {
        "dataset": "recgym",
        "root": str(root),
        "rows": int(len(df)),
        "subjects": int(df["Subject"].nunique()),
        "sessions": int(df[["Subject", "Session"]].drop_duplicates().shape[0]),
        "positions": sorted(df["Position"].unique().tolist()),
        "rate_hz": FS_HZ,
        "hours": round(len(df) / FS_HZ / 3600.0, 1),
        "minutes_per_workout": {str(k): float(v) for k, v in minutes.items()},
        "units": "normalized [0,1]; no timestamps",
        "value_range": [float(df[SIGNALS].min().min()), float(df[SIGNALS].max().max())],
    }
