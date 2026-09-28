"""Zero-phase Butterworth filters for offline processing (``scipy.signal.sosfiltfilt``).

The live pipeline and the firmware use causal filters; these are for datasets, reports and
rep segmentation on recorded sessions. Inputs are ``(n,)`` or ``(n, c)`` arrays; ``fs`` Hz.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    padlen = min(3 * (2 * sos.shape[0] + 1), max(n - 1, 0))
    if n < 4:
        return x.copy()
    return sps.sosfiltfilt(sos, x, axis=0, padlen=padlen)


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Low-pass ``x`` at ``fc`` Hz (zero phase)."""
    sos = sps.butter(order, fc, btype="low", fs=fs, output="sos")
    return _apply(sos, x)


def highpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    sos = sps.butter(order, fc, btype="high", fs=fs, output="sos")
    return _apply(sos, x)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Band-pass ``x`` between ``lo`` and ``hi`` Hz (zero phase). ``hi`` is clipped below
    Nyquist so 20 Hz input with a 3 Hz upper edge still works."""
    hi = min(hi, 0.45 * fs)
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return _apply(sos, x)
