"""Uniform resampling of IMU streams (``t`` seconds → fixed ``fs`` Hz).

Datasets arrive at 100 Hz (MM-Fit watches), 50 Hz (RecoFit) and 20 Hz (RecGym); the gate and
the firmware run at 50 Hz (``protocol.SAMPLE_RATE_HZ``). Linear interpolation on a uniform
grid; duplicate timestamps are averaged and unsorted input is sorted first. Label columns are
carried by nearest original sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from formcoach.data import schema

SIGNALS = list(schema.SIGNAL_COLUMNS)


def _dedupe(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    uniq, idx, counts = np.unique(t, return_index=True, return_counts=True)
    if len(uniq) == len(t):
        return t, x
    # average the samples sharing a timestamp
    sums = np.add.reduceat(x, idx, axis=0)
    return uniq, sums / counts[:, None]


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``x`` (n, c) sampled at times ``t`` (s) onto a uniform ``fs`` grid.

    Returns ``(t_u, x_u)`` with ``t_u`` from ``ceil(t[0]·fs)/fs`` to ``floor(t[-1]·fs)/fs``
    inclusive; never extrapolates and never produces NaN. Raises ``ValueError`` for fewer than
    two distinct timestamps.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if len(t) != len(x):
        raise ValueError("t and x must have the same length")
    t, x = _dedupe(t, x)
    if len(t) < 2:
        raise ValueError("need at least two distinct timestamps to resample")
    k0 = int(np.ceil(t[0] * fs - 1e-9))
    k1 = int(np.floor(t[-1] * fs + 1e-9))
    if k1 < k0:
        raise ValueError("input shorter than one output sample")
    t_u = np.arange(k0, k1 + 1, dtype=np.float64) / fs
    x_u = np.column_stack([np.interp(t_u, t, x[:, c]) for c in range(x.shape[1])])
    return t_u, x_u


def resample_stream(df: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Return an IMUStream resampled to ``fs`` Hz (signals interpolated, labels by nearest).

    Constant columns (dataset, subject, …, units) are kept; extra columns such as ``label_raw``,
    ``set_id``, ``frame`` are filled from the nearest original sample. ``df.attrs["fs"]`` is
    set on the result.
    """
    t = df["t"].to_numpy(dtype=np.float64)
    sig = df[SIGNALS].to_numpy(dtype=np.float64)
    t_u, x_u = resample_uniform(t, sig, fs)
    nearest = np.clip(np.searchsorted(t, t_u), 0, len(t) - 1)
    left = np.clip(nearest - 1, 0, len(t) - 1)
    use_left = np.abs(t[left] - t_u) < np.abs(t[nearest] - t_u)
    nearest = np.where(use_left, left, nearest)
    out = df.iloc[nearest].reset_index(drop=True).copy()
    out["t"] = t_u
    for i, c in enumerate(SIGNALS):
        out[c] = x_u[:, i].astype(np.float32)
    out.attrs["fs"] = fs
    return out
