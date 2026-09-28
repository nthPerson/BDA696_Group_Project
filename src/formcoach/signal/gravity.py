"""Gravity / linear-acceleration separation and acceleration magnitude (m/s²)."""

from __future__ import annotations

import numpy as np

from formcoach.signal.filters import lowpass

G = 9.80665


def split_gravity(acc: np.ndarray, fs: float, fc: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Split ``acc`` (n, 3) m/s² into ``(gravity, linear)`` with a ``fc`` Hz low-pass estimate
    of gravity (offline, zero phase). Short inputs (< 4 samples) return the mean as gravity."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.shape[0] < 4:
        g = np.repeat(acc.mean(axis=0, keepdims=True), acc.shape[0], axis=0)
        return g, acc - g
    g = lowpass(acc, fs, fc, order=2)
    return g, acc - g


def magnitude(acc: np.ndarray) -> np.ndarray:
    """``|a|`` per sample for an (n, 3) array."""
    return np.linalg.norm(np.asarray(acc, dtype=np.float64), axis=1)
