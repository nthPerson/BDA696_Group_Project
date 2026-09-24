"""Hand-crafted per-window features for the RF / energy baselines (docs/02 §6.2).

For each of the six channels and the acceleration magnitude ``amag``: mean, std, min, max,
energy (mean square), dominant frequency (Hz, 0.3-8 Hz band), spectral entropy (bits,
normalised power spectrum) and the autocorrelation peak lag (s) and height in the 0.3-3 s
range. 7 channels x 9 statistics = 63 features named ``feat_<channel>_<stat>``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "amag")
STATS = ("mean", "std", "min", "max", "energy", "domfreq", "spec_entropy", "ac_lag", "ac_peak")
FEATURE_NAMES: tuple[str, ...] = tuple(f"feat_{c}_{s}" for c in CHANNELS for s in STATS)
F_LO, F_HI = 0.3, 8.0
LAG_LO_S, LAG_HI_S = 0.3, 3.0


def _channel_stats(v: np.ndarray, fs: float) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    mean = float(v.mean())
    std = float(v.std())
    energy = float(np.mean(v * v))
    d = v - mean
    spec = np.abs(np.fft.rfft(d)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= F_LO) & (freqs <= F_HI)
    if band.any() and spec[band].max() > 0:
        domfreq = float(freqs[band][np.argmax(spec[band])])
    else:
        domfreq = 0.0
    p = spec[1:]
    if p.sum() > 0:
        p = p / p.sum()
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
    else:
        entropy = 0.0
    lag_lo, lag_hi = int(LAG_LO_S * fs), min(int(LAG_HI_S * fs), n - 2)
    if std > 0 and lag_hi > lag_lo:
        ac = np.correlate(d, d, mode="full")[n - 1 :]
        ac = ac / ac[0]
        seg = ac[lag_lo : lag_hi + 1]
        k = int(np.argmax(seg))
        ac_lag, ac_peak = float((lag_lo + k) / fs), float(seg[k])
    else:
        ac_lag, ac_peak = 0.0, 0.0
    return [mean, std, float(v.min()), float(v.max()), energy, domfreq, entropy, ac_lag, ac_peak]


def window_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Features for one window ``x`` of shape ``(win, 6)`` (ax..gz in SI) → ``{name: value}``."""
    x = np.asarray(x, dtype=np.float64)
    amag = np.linalg.norm(x[:, :3], axis=1)
    values: list[float] = []
    for c in range(6):
        values.extend(_channel_stats(x[:, c], fs))
    values.extend(_channel_stats(amag, fs))
    return dict(zip(FEATURE_NAMES, values, strict=True))


def featurize_windows(w: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Append ``feat_*`` columns to a Window DataFrame (``x`` stays)."""
    if len(w) == 0:
        for name in FEATURE_NAMES:
            w[name] = pd.Series(dtype=np.float32)
        return w
    from formcoach.signal.windows import windows_to_array

    xs = windows_to_array(w)
    feats = np.array([list(window_features(xi, fs).values()) for xi in xs], dtype=np.float32)
    out = w.copy()
    for j, name in enumerate(FEATURE_NAMES):
        out[name] = feats[:, j]
    return out
