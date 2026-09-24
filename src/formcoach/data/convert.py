"""Convert raw downloads into IMUStream Parquet files (``formcoach data convert``).

Output layout: ``data/processed/<dataset>/streams/<subject>-<session>[-<device>].parquet``,
one file per stream, schema per docs/02-system-design.md §5 with the extra columns each loader
adds. Existing files are skipped unless ``force=True`` so re-running is cheap. Every later
stage (``features build``, ``eval …``, replay) reads these files instead of the raw formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from formcoach.data import loader, schema

PROCESSED_ROOT = Path(__file__).resolve().parents[3] / "data" / "processed"


@dataclass(frozen=True)
class StreamFile:
    dataset: str
    subject: str
    session: str
    device: str
    path: Path


def _stem(subject: str, session: str, device: str | None) -> str:
    return f"{subject}-{session}" + (f"-{device}" if device else "")


def _write(df: pd.DataFrame, path: Path) -> None:
    schema.validate_imu_stream(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def convert_dataset(
    name: str, raw_root: Path | None, out_root: Path = PROCESSED_ROOT, *, force: bool = False
) -> list[Path]:
    """Write every stream of ``name`` under ``out_root/<name>/streams/``; return files written."""
    mod = loader(name)
    root = raw_root or mod.DEFAULT_ROOT
    if not mod.available(root):
        raise FileNotFoundError(f"{name}: raw data not found under {root}; run `make data`")
    out_dir = out_root / name / "streams"
    written: list[Path] = []
    if name == "mmfit":
        for w in mod.list_workouts(root):
            for dev in ("sw_l", "sw_r"):
                path = out_dir / f"{_stem(mod.subject_id(w), w, dev)}.parquet"
                if path.exists() and not force:
                    continue
                try:
                    df = mod.load_stream(root, w, dev)
                except FileNotFoundError:
                    continue
                _write(df, path)
                written.append(path)
    elif name == "recofit":
        for si, v in mod.list_visits(root):
            df = None
            # subject id is only known after loading; derive the stem from the loader
            df = mod.load_stream(root, si, v)
            path = out_dir / f"{_stem(df['subject'].iloc[0], df['session'].iloc[0], None)}.parquet"
            if path.exists() and not force:
                continue
            _write(df, path)
            written.append(path)
    elif name == "recgym":
        for key in mod.list_sessions(root):
            subject, position, session = key
            path = out_dir / f"{_stem(f'G{subject}', f's{session}-{position}', None)}.parquet"
            if path.exists() and not force:
                continue
            _write(mod.load_stream(root, *key), path)
            written.append(path)
    return written


def list_streams(out_root: Path = PROCESSED_ROOT, dataset: str | None = None) -> list[StreamFile]:
    """Every converted stream file, parsed from its name (``subject-session[-device]``)."""
    out: list[StreamFile] = []
    for ds_dir in sorted(out_root.glob("*")):
        if not ds_dir.is_dir() or (dataset and ds_dir.name != dataset):
            continue
        for p in sorted((ds_dir / "streams").glob("*.parquet")):
            parts = p.stem.split("-")
            subject, session = parts[0], parts[1]
            device = parts[2] if len(parts) > 2 and ds_dir.name == "mmfit" else ""
            if ds_dir.name == "recgym":  # session is "s1-wrist"
                session = "-".join(parts[1:3])
            out.append(StreamFile(ds_dir.name, subject, session, device, p))
    return out


def read_stream(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    schema.validate_imu_stream(df)
    return df
