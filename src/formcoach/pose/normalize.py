"""Camera-independent normalisation of pose sequences (docs/02 §4.3).

* :func:`estimate_up_vector`: median direction hip-centre → shoulder-centre over the sequence
  (people stand roughly upright on average), so no assumption about the source's axes.
* :func:`normalize_sequence`: hip-centred, scaled to torso units, low-visibility joints
  interpolated over gaps ≤ ``max_gap`` frames, longer gaps flagged ``valid == False``.
* :func:`smooth_angles`: 5-frame median then Savitzky–Golay (window 7, order 2).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from formcoach.pose.angles import torso_length
from formcoach.pose.skeletons import Skeleton, joint

DEFAULT_REQUIRED = ("shoulder_l", "shoulder_r", "hip_l", "hip_r")


def estimate_up_vector(xyz: np.ndarray, sk: Skeleton) -> np.ndarray:
    """Unit vector of the median hip→shoulder direction (the sequence's vertical)."""
    v = joint(xyz, sk, "shoulder_c") - joint(xyz, sk, "hip_c")
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) == 0:
        return np.array([0.0, 1.0, 0.0])
    med = np.median(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9), axis=0)
    return med / max(np.linalg.norm(med), 1e-9)


def _fill_gaps(x: np.ndarray, bad: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate ``x`` (n, …) over runs of ``bad`` frames of length ≤ max_gap.
    Returns (filled, still_bad)."""
    x = x.copy()
    n = len(x)
    still = bad.copy()
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        gap = j - i
        if gap <= max_gap and i > 0 and j < n:
            w = np.linspace(0, 1, gap + 2)[1:-1]
            x[i:j] = x[i - 1] * (1 - w).reshape(-1, *([1] * (x.ndim - 1))) + x[j] * w.reshape(
                -1, *([1] * (x.ndim - 1))
            )
            still[i:j] = False
        i = j
    return x, still


def normalize_sequence(
    xyz: np.ndarray,
    sk: Skeleton,
    visibility: np.ndarray | None = None,
    *,
    required: tuple[str, ...] = DEFAULT_REQUIRED,
    vis_thr: float = 0.5,
    max_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz_norm, valid)``: hip-centred coordinates in torso units and a per-frame
    validity flag. A frame is bad when any ``required`` joint has ``visibility < vis_thr`` or a
    NaN coordinate; bad runs ≤ ``max_gap`` are interpolated, longer runs stay invalid."""
    xyz = np.asarray(xyz, dtype=np.float64)
    bad = ~np.isfinite(xyz).all(axis=(1, 2))
    if visibility is not None:
        vis = np.asarray(visibility, dtype=np.float64)
        idx = [sk.index(r) for r in required]
        bad |= (vis[:, idx] < vis_thr).any(axis=1)
    filled, still_bad = _fill_gaps(xyz, bad, max_gap)
    valid = ~still_bad
    centre = joint(filled, sk, "hip_c")[:, None, :]
    out = filled - centre
    torso = torso_length(filled, sk)
    scale = np.nanmedian(torso[valid]) if valid.any() else np.nanmedian(torso)
    out = out / max(float(scale), 1e-9)
    return out, valid


def smooth_angles(
    a: np.ndarray, median_k: int = 5, sg_window: int = 7, sg_order: int = 2
) -> np.ndarray:
    """Median + Savitzky–Golay smoothing of a 1-D angle series; NaNs are interpolated first."""
    a = np.asarray(a, dtype=np.float64).copy()
    n = len(a)
    if n == 0:
        return a
    nan = ~np.isfinite(a)
    if nan.all():
        return a
    if nan.any():
        idx = np.arange(n)
        a[nan] = np.interp(idx[nan], idx[~nan], a[~nan])
    if n >= median_k:
        a = median_filter(a, size=median_k, mode="nearest")
    if n >= sg_window:
        a = savgol_filter(a, sg_window, sg_order, mode="interp")
    return a
