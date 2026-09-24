"""Fixed-length windows over IMU streams (docs/02-system-design.md §5 ``Window`` schema).

``make_windows`` resamples to ``fs`` (default 50 Hz) if needed and slides a ``win_s`` window
with ``stride_s`` stride. Each row carries the majority canonical label, its purity (fraction
of samples carrying that label), ``label_active`` (1 unless the majority label is ``idle``) and
``x``: the raw ``(win, 6)`` float32 samples flattened row-major into a list column so Parquet
can store it. Windows overlapping RecoFit junk labels are dropped by default (ADR-0015).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from formcoach.data import labels, schema
from formcoach.signal.resample import SIGNALS, resample_stream

DEFAULT_FS = 50
DEFAULT_WIN_S = 2.0
DEFAULT_STRIDE_S = 1.0
JUNK = labels.RECOFIT_JUNK


def _needs_resample(t: np.ndarray, fs: float) -> bool:
    if len(t) < 3:
        return True
    dt = np.diff(t)
    return not np.allclose(dt, 1.0 / fs, rtol=1e-3, atol=1e-6)


def make_windows(
    df: pd.DataFrame,
    fs: float = DEFAULT_FS,
    win_s: float = DEFAULT_WIN_S,
    stride_s: float = DEFAULT_STRIDE_S,
    drop_junk: bool = True,
) -> pd.DataFrame:
    """Window one IMUStream. Returns a ``Window`` DataFrame (may be empty for short streams)."""
    if _needs_resample(df["t"].to_numpy(), fs):
        df = resample_stream(df, fs)
    n = len(df)
    win = round(win_s * fs)
    stride = max(round(stride_s * fs), 1)
    t = df["t"].to_numpy()
    sig = df[SIGNALS].to_numpy(dtype=np.float32)
    ex = df["exercise"].to_numpy(dtype=object)
    raw = df["label_raw"].to_numpy(dtype=object) if "label_raw" in df.columns else None
    const = (
        {c: df[c].iloc[0] for c in ("dataset", "subject", "session", "device", "units")}
        if n
        else {}
    )
    rows = []
    wid = 0
    for start in range(0, n - win + 1, stride):
        stop = start + win
        if drop_junk and raw is not None and any(r in JUNK for r in raw[start:stop]):
            continue
        vals, counts = np.unique(ex[start:stop], return_counts=True)
        k = int(np.argmax(counts))
        label = str(vals[k])
        rows.append(
            {
                **const,
                "window_id": wid,
                "t_start": float(t[start]),
                "t_end": float(t[stop - 1]),
                "label_exercise": label,
                "label_active": int(schema.is_active(label)),
                "label_purity": float(counts[k] / win),
                "x": sig[start:stop].reshape(-1),
            }
        )
        wid += 1
    cols = [*schema.WINDOW_COLUMNS, "label_purity", "units", "x"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    out = pd.DataFrame(rows)
    return out[cols]


def windows_to_array(w: pd.DataFrame, channels: int = 6) -> np.ndarray:
    """Stack the ``x`` column into an ``(n, win, channels)`` float32 array."""
    if len(w) == 0:
        return np.zeros((0, 0, channels), dtype=np.float32)
    x = np.stack([np.asarray(v, dtype=np.float32) for v in w["x"]])
    return x.reshape(len(w), -1, channels)
