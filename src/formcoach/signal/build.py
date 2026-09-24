"""``formcoach features build``: converted streams → windows + features Parquet.

Reads ``data/processed/<dataset>/streams/*.parquet`` (from ``data convert``), resamples to
``fs``, windows (2 s / 1 s stride by default), featurises, and writes
``data/processed/<dataset>/windows/<same stem>.parquet``. :func:`load_windows` concatenates a
dataset's window files for training and evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from formcoach.data import convert
from formcoach.signal import features, windows

WINDOW_META = [
    "dataset",
    "subject",
    "session",
    "device",
    "units",
    "label_exercise",
    "label_active",
    "label_purity",
]


def _build_one(args: tuple[Path, Path, float, float, float]) -> Path:
    src, out, fs, win_s, stride_s = args
    df = pd.read_parquet(src)
    w = windows.make_windows(df, fs=fs, win_s=win_s, stride_s=stride_s)
    w = features.featurize_windows(w, fs)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(out, index=False)
    return out


def build_features(
    processed_root: Path = convert.PROCESSED_ROOT,
    dataset: str | None = None,
    *,
    fs: float = windows.DEFAULT_FS,
    win_s: float = windows.DEFAULT_WIN_S,
    stride_s: float = windows.DEFAULT_STRIDE_S,
    force: bool = False,
    jobs: int = 1,
) -> list[Path]:
    """Window + featurise every stream (optionally one dataset); returns files written.

    ``jobs > 1`` processes stream files in a process pool (the per-window feature loop is
    pure Python); results are identical to ``jobs=1``.
    """
    todo = []
    for s in convert.list_streams(processed_root, dataset):
        out = processed_root / s.dataset / "windows" / s.path.name
        if out.exists() and not force:
            continue
        todo.append((s.path, out, fs, win_s, stride_s))
    if jobs > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_build_one, todo))
    return [_build_one(a) for a in todo]


def load_windows(
    processed_root: Path,
    dataset: str,
    *,
    min_purity: float = 0.8,
    devices: tuple[str, ...] | None = None,
    with_x: bool = True,
) -> pd.DataFrame:
    """Concatenate ``windows/*.parquet`` of ``dataset``; keep windows with purity ≥ threshold."""
    files = sorted((processed_root / dataset / "windows").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no window files under {processed_root / dataset / 'windows'}; run `make features`"
        )
    parts = []
    for f in files:
        w = pd.read_parquet(f) if with_x else pd.read_parquet(f).drop(columns=["x"])
        if devices and w["device"].iloc[0] not in devices:
            continue
        parts.append(w[w["label_purity"] >= min_purity])
    out = pd.concat(parts, ignore_index=True)
    return out
