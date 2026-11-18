"""Put the wearable's sample times on the laptop's clock (Bench design §7, "One clock").

Every wearable sample carries ``t`` (seconds on the device's own clock) and ``t_arrival`` (the
laptop's monotonic clock when the sample was received). The difference
``delta = t_arrival - t_device`` is the clock offset plus a latency that is never negative: the
firmware sends samples in batches of five, so the first sample of a batch waits four sample
periods, and the radio adds its own delay on top. The true mapping is therefore the *lower
envelope* of ``delta``. :func:`fit_device_clock` takes the smallest ``delta`` in each 1 s bin of
device time and fits a straight line through those minima (an offset and a drift);
:func:`to_laptop` applies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_S = 1.0  # width of a device-time bin (s); one lower-envelope point per bin
MIN_BINS = 3  # fewer bins than this: offset only, no drift


@dataclass(frozen=True)
class ClockFit:
    """``laptop = t_device * (1 + drift) + offset_s`` (seconds).

    ``offset_s`` seconds; ``drift`` dimensionless (s per s; 20e-6 is 20 ppm); ``residual_ms`` the
    median absolute deviation (ms) of the per-bin minima from the fitted line; ``n`` the number
    of samples the fit used; ``method`` ``"envelope"`` (offset and drift through the per-bin
    minima), ``"offset"`` (fewer than :data:`MIN_BINS` bins: drift 0, offset the smallest
    ``delta``) or ``"none"`` (no samples: all zeros).
    """

    offset_s: float
    drift: float
    residual_ms: float
    n: int
    method: str


def fit_device_clock(t_device: np.ndarray, t_arrival: np.ndarray) -> ClockFit:
    """Fit the device clock (``t_device``, s) to the laptop clock (``t_arrival``, s) from the
    lower envelope of ``t_arrival - t_device`` (see the module doc). Pairs where either time is
    not finite (a sample without an arrival stamp) are left out.

    Raises:
        ValueError: the two arrays differ in length.
    """
    td = np.asarray(t_device, dtype=np.float64).ravel()
    ta = np.asarray(t_arrival, dtype=np.float64).ravel()
    if td.shape != ta.shape:
        raise ValueError(f"t_device and t_arrival differ in length: {td.shape} vs {ta.shape}")
    ok = np.isfinite(td) & np.isfinite(ta)
    td, ta = td[ok], ta[ok]
    n = len(td)
    if n == 0:
        return ClockFit(0.0, 0.0, 0.0, 0, "none")
    delta = ta - td
    bins = np.floor((td - td.min()) / BIN_S).astype(np.int64)
    order = np.lexsort((delta, bins))  # by bin, smallest delta first within each bin
    first = np.concatenate([[True], np.diff(bins[order]) != 0])
    pick = order[first]  # the sample with the smallest delta in each bin
    x, y = td[pick], delta[pick]
    if len(pick) < MIN_BINS:
        offset = float(delta.min())
        resid = y - offset
        return ClockFit(offset, 0.0, float(np.median(np.abs(resid)) * 1000.0), n, "offset")
    x0 = float(x.mean())  # centred for a well-conditioned fit on long device uptimes
    drift, at_x0 = np.polyfit(x - x0, y, 1)
    offset = float(at_x0 - drift * x0)
    resid = y - (drift * x + offset)
    return ClockFit(offset, float(drift), float(np.median(np.abs(resid)) * 1000.0), n, "envelope")


def to_laptop(fit: ClockFit, t_device: np.ndarray) -> np.ndarray:
    """Device times (s) on the laptop clock (s): ``t_device * (1 + drift) + offset_s``."""
    return np.asarray(t_device, dtype=np.float64) * (1.0 + fit.drift) + fit.offset_s
