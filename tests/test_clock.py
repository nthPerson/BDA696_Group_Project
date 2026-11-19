"""``formcoach.io.clock``: the wearable's clock mapped onto the laptop's (Bench plan Task 6)."""

from __future__ import annotations

import numpy as np

from formcoach.io.clock import ClockFit, fit_device_clock, to_laptop


def _batched(seconds: float, *, offset: float, drift: float, seed: int = 0):
    """Device times at 50 Hz and their arrival times: offset, drift, five-sample batching (the
    first sample of a batch waits 80 ms, the last 0) and 0-5 ms of radio jitter."""
    rng = np.random.default_rng(seed)
    td = np.arange(round(seconds * 50)) / 50.0
    i = np.arange(len(td))
    ta = td * (1 + drift) + offset + (4 - i % 5) * 0.02 + rng.uniform(0, 0.005, len(td))
    return td, ta


def test_envelope_fit_recovers_offset_and_drift_despite_batching():
    td, ta = _batched(60.0, offset=1234.5, drift=20e-6)
    fit = fit_device_clock(td, ta)
    assert fit.method == "envelope"
    assert abs(fit.offset_s - 1234.5) < 0.006
    assert abs(fit.drift - 20e-6) < 2e-4
    assert fit.residual_ms < 5
    assert fit.n == len(td)
    # mapped device times sit on the lower envelope: never later than the arrival
    assert np.all(to_laptop(fit, td) <= ta + 0.006)


def test_short_recording_uses_minimum_offset():
    td, ta = _batched(1.5, offset=50.0, drift=0.0)  # two 1 s bins
    fit = fit_device_clock(td, ta)
    delta = ta - td
    assert fit.method == "offset"
    assert fit.drift == 0.0
    assert fit.offset_s == float(delta.min())
    assert fit.n == len(td)
    np.testing.assert_allclose(to_laptop(fit, td), td + delta.min())


def test_no_samples_gives_method_none():
    fit = fit_device_clock(np.zeros(0), np.zeros(0))
    assert fit == ClockFit(0.0, 0.0, 0.0, 0, "none")
    # samples without an arrival stamp cannot be placed either
    fit = fit_device_clock(np.array([0.0, 0.02]), np.array([np.nan, np.nan]))
    assert fit.method == "none" and fit.n == 0
    assert to_laptop(fit, np.zeros(0)).shape == (0,)
