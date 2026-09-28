"""IMU-primary rep segmentation by peak detection (docs/02 §4.4).

Band-pass 0.3-3 Hz the linear acceleration, take the dominant axis (max variance) or |a|,
find peaks with prominence ≥ 0.15 g and minimum spacing 0.8 s **(default, calibrate)**; a rep
spans the troughs on either side of a peak. Pure NumPy/SciPy; no pose imports (rule 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from formcoach.signal.filters import bandpass

G = 9.80665
DEFAULT_PROMINENCE_G = 0.15
DEFAULT_MIN_DISTANCE_S = 0.8
BAND = (0.3, 3.0)


@dataclass(frozen=True)
class Rep:
    rep_id: int
    t_start: float
    t_peak: float
    t_end: float
    duration_s: float
    prominence: float  # m/s² on the rep signal


def rep_signal(acc: np.ndarray, fs: float, mode: str = "axis", band=BAND) -> np.ndarray:
    """1-D oscillation signal (m/s²) from ``acc`` (n, 3): band-passed dominant axis or |a|."""
    acc = np.asarray(acc, dtype=np.float64)
    if mode == "axis":
        bp = bandpass(acc, fs, *band)
        return bp[:, int(np.argmax(bp.var(axis=0)))]
    if mode == "magnitude":
        mag = np.linalg.norm(acc, axis=1)
        return bandpass(mag, fs, *band)
    raise ValueError("mode must be 'axis' or 'magnitude'")


def count_reps(
    acc: np.ndarray,
    t: np.ndarray,
    fs: float,
    *,
    prominence_g: float = DEFAULT_PROMINENCE_G,
    min_distance_s: float = DEFAULT_MIN_DISTANCE_S,
    mode: str = "axis",
) -> list[Rep]:
    """Reps in an accelerometer segment (``acc`` m/s², ``t`` s, uniform ``fs``)."""
    acc = np.asarray(acc, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if len(t) < int(2 * min_distance_s * fs) + 2:
        return []
    s = rep_signal(acc, fs, mode)
    peaks, props = find_peaks(
        s, prominence=prominence_g * G, distance=max(int(min_distance_s * fs), 1)
    )
    if len(peaks) == 0:
        return []
    troughs = []
    for i, p in enumerate(peaks):
        lo = peaks[i - 1] if i > 0 else max(p - int(min_distance_s * fs), 0)
        troughs.append(lo + int(np.argmin(s[lo : p + 1])))
    last = peaks[-1]
    hi = min(last + int(min_distance_s * fs), len(s) - 1)
    troughs.append(last + int(np.argmin(s[last : hi + 1])))
    reps = []
    for i, p in enumerate(peaks):
        a, b = troughs[i], troughs[i + 1]
        if b <= a:
            b = min(a + 1, len(s) - 1)
        reps.append(
            Rep(
                i,
                float(t[a]),
                float(t[p]),
                float(t[b]),
                float(t[b] - t[a]),
                float(props["prominences"][i]),
            )
        )
    return reps
