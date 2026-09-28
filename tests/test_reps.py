"""IMU rep counter (signal/reps.py) and the MM-Fit rep-count evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.data import convert
from formcoach.data.fixtures import FIXTURE_ROOT
from formcoach.eval import repcount
from formcoach.signal import reps

FS = 50.0


def _sine_acc(f_hz: float, n_s: float, amp: float = 3.0, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_s, 1 / FS)
    acc = np.column_stack(
        [amp * np.sin(2 * np.pi * f_hz * t), np.zeros_like(t), np.full_like(t, 9.81)]
    )
    return t, acc + rng.normal(0, noise, acc.shape)


def test_count_reps_on_clean_sine():
    t, acc = _sine_acc(1.0, 10.0)
    found = reps.count_reps(acc, t, FS)
    assert 9 <= len(found) <= 10
    r = found[3]
    assert r.t_start < r.t_peak < r.t_end
    assert 0.7 < r.duration_s < 1.3
    assert found[0].rep_id == 0 and found[-1].rep_id == len(found) - 1


def test_min_distance_suppresses_double_peaks_and_noise_gives_none():
    t, acc = _sine_acc(1.0, 10.0)
    acc[:, 0] += 0.5 * np.sin(2 * np.pi * 4.0 * t)  # fast wobble
    assert 9 <= len(reps.count_reps(acc, t, FS)) <= 11
    rng = np.random.default_rng(1)
    still = np.column_stack([np.zeros(500), np.zeros(500), np.full(500, 9.81)]) + rng.normal(
        0, 0.05, (500, 3)
    )
    assert len(reps.count_reps(still, np.arange(500) / FS, FS)) == 0


def test_rep_signal_modes():
    t, acc = _sine_acc(1.0, 6.0)
    s_axis = reps.rep_signal(acc, FS, mode="axis")
    s_mag = reps.rep_signal(acc, FS, mode="magnitude")
    assert s_axis.shape == s_mag.shape == (len(t),)
    with pytest.raises(ValueError):
        reps.rep_signal(acc, FS, mode="nope")


def test_repcount_eval_on_fixture_writes_report(tmp_path):
    processed = tmp_path / "processed"
    convert.convert_dataset("mmfit", FIXTURE_ROOT / "mmfit" / "mm-fit", processed)
    out = tmp_path / "reports" / "baseline_repcount.md"
    table = repcount.evaluate(processed, FIXTURE_ROOT / "mmfit" / "mm-fit", out, devices=("sw_l",))
    assert {"exercise", "n_sets", "mae", "exact_pct"} <= set(table.columns)
    assert table.loc[table["exercise"] == "all", "mae"].iloc[0] < 2.0  # synthetic 1 Hz reps
    text = out.read_text(encoding="utf-8")
    assert "# Rep-count baseline" in text and "| curl |" in text
