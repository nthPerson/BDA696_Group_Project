"""signal/: resample, filters, gravity, windows, features. Units: t s, acc m/s², gyro rad/s."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formcoach.data import schema
from formcoach.signal import features, filters, gravity, resample, windows

FS = 50


def _stream(n_s: float = 10.0, fs: float = 100.0, label_switch_s: float = 5.0) -> pd.DataFrame:
    n = int(n_s * fs)
    t = np.arange(n) / fs
    acc = np.column_stack([3 * np.sin(2 * np.pi * 1.0 * t), np.zeros(n), np.full(n, 9.81)])
    gyr = np.column_stack([np.zeros(n), 2 * np.cos(2 * np.pi * 1.0 * t), np.zeros(n)])
    ex = np.where(t < label_switch_s, "idle", "curl")
    df = schema.make_imu_stream(
        dataset="fixture", subject="S1", session="0001", device="fake", placement="wrist_l",
        t=t, acc=acc, gyr=gyr, exercise=ex, set_id=np.where(t < label_switch_s, -1, 0),
    )  # fmt: skip
    df["label_raw"] = pd.Series(
        np.where(t < label_switch_s, "Non-Exercise", "Bicep Curl"), dtype="str"
    )
    return df


# ---- resample ---------------------------------------------------------------------------------
def test_resample_uniform_recovers_sine_from_irregular_samples_with_duplicates():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 900))
    t[100:105] = t[100]  # duplicate timestamps
    x = np.column_stack([np.sin(2 * np.pi * 1.0 * t), np.cos(2 * np.pi * 0.5 * t)])
    t_u, x_u = resample.resample_uniform(t, x, FS)
    assert np.allclose(np.diff(t_u), 1 / FS)
    assert t_u[0] >= t[0] and t_u[-1] <= t[-1]
    assert not np.isnan(x_u).any()
    assert np.abs(x_u[:, 0] - np.sin(2 * np.pi * 1.0 * t_u)).max() < 0.05


def test_resample_uniform_rejects_short_or_unsorted_input():
    with pytest.raises(ValueError):
        resample.resample_uniform(np.array([0.0]), np.zeros((1, 3)), FS)
    t = np.array([0.0, 0.5, 0.2, 1.0])
    t_u, _ = resample.resample_uniform(t, np.zeros((4, 1)), FS)  # unsorted is sorted first
    assert t_u[0] == 0.0


def test_resample_stream_keeps_schema_and_labels_by_nearest_sample():
    df = _stream(fs=100.0)
    out = resample.resample_stream(df, FS)
    schema.validate_imu_stream(out)
    assert np.allclose(np.diff(out["t"]), 1 / FS)
    assert set(out["exercise"]) == {"idle", "curl"}
    assert (out.loc[out["t"] < 4.9, "exercise"] == "idle").all()
    assert (out.loc[out["t"] > 5.1, "exercise"] == "curl").all()
    assert out["label_raw"].iloc[-1] == "Bicep Curl" and out["set_id"].iloc[-1] == 0
    assert out.attrs.get("fs") == FS


def test_resample_stream_upsamples_20hz_recgym_like_input():
    df = _stream(fs=20.0)
    out = resample.resample_stream(df, FS)
    assert len(out) == pytest.approx(len(df) * 2.5, rel=0.02)


# ---- filters / gravity ------------------------------------------------------------------------
def test_lowpass_removes_high_frequency_and_keeps_low():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    x = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 20.0 * t)
    y = filters.lowpass(x, fs, fc=5.0)
    core = slice(100, -100)
    assert np.abs(y[core] - np.sin(2 * np.pi * 1.0 * t[core])).max() < 0.05


def test_bandpass_2d_input_filters_each_column():
    fs = 50.0
    t = np.arange(0, 10, 1 / fs)
    x = np.column_stack([np.sin(2 * np.pi * 1.0 * t) + 5.0, np.sin(2 * np.pi * 0.05 * t)])
    y = filters.bandpass(x, fs, 0.3, 3.0)
    assert y.shape == x.shape
    assert abs(y[100:-100, 0].mean()) < 0.05  # DC removed
    assert np.abs(y[100:-100, 1]).max() < 0.1  # 0.05 Hz removed


def test_split_gravity_separates_constant_from_motion():
    fs = 50.0
    t = np.arange(0, 20, 1 / fs)
    acc = np.column_stack(
        [2 * np.sin(2 * np.pi * 1.0 * t), np.zeros_like(t), np.full_like(t, 9.81)]
    )
    g, lin = gravity.split_gravity(acc, fs)
    core = slice(200, -200)
    assert np.allclose(g[core, 2], 9.81, atol=0.05)
    assert np.abs(g[core, 0]).max() < 0.1
    assert np.abs(lin[core, 0] - 2 * np.sin(2 * np.pi * 1.0 * t[core])).max() < 0.1
    assert gravity.magnitude(acc).shape == (len(t),)


def test_split_gravity_short_input_does_not_crash():
    g, lin = gravity.split_gravity(np.zeros((20, 3)), 50.0)
    assert g.shape == lin.shape == (20, 3)


# ---- windows ----------------------------------------------------------------------------------
def test_make_windows_shapes_labels_and_purity():
    df = resample.resample_stream(_stream(fs=100.0), FS)  # 10 s at 50 Hz = 500 samples
    w = windows.make_windows(df, fs=FS, win_s=2.0, stride_s=1.0)
    assert len(w) == 9  # (500 - 100) / 50 + 1
    for c in schema.WINDOW_COLUMNS:
        assert c in w.columns, c
    assert w["window_id"].tolist() == list(range(9))
    assert w["t_start"].iloc[0] == pytest.approx(0.0) and w["t_end"].iloc[0] == pytest.approx(
        2.0 - 1 / FS
    )
    x = windows.windows_to_array(w)
    assert x.shape == (9, 100, 6) and x.dtype == np.float32
    assert w["label_exercise"].tolist()[:3] == ["idle", "idle", "idle"]
    assert w["label_exercise"].tolist()[-3:] == ["curl", "curl", "curl"]
    assert w["label_active"].tolist()[-1] == 1 and w["label_active"].tolist()[0] == 0
    assert (w["label_purity"] <= 1.0).all() and w["label_purity"].iloc[0] == 1.0
    boundary = w.iloc[4]  # 4..6 s straddles the 5 s switch
    assert boundary["label_purity"] == pytest.approx(0.5, abs=0.02)
    assert w["units"].iloc[0] == "si" and w["dataset"].iloc[0] == "fixture"


def test_make_windows_drops_junk_overlap_and_resamples_if_needed():
    df = _stream(fs=100.0)  # not 50 Hz: make_windows must resample
    df.loc[df["t"].between(2.0, 2.5), "label_raw"] = "Tap Left Device"
    w = windows.make_windows(df, fs=FS, win_s=2.0, stride_s=1.0, drop_junk=True)
    starts = w["t_start"].round(2).tolist()
    assert 1.0 not in starts and 2.0 not in starts  # windows overlapping 2.0-2.5 dropped
    assert 3.0 in starts
    w2 = windows.make_windows(df, fs=FS, win_s=2.0, stride_s=1.0, drop_junk=False)
    assert len(w2) == 9


def test_make_windows_normalized_units_carry_over():
    df = _stream(fs=50.0)
    df["units"] = "normalized"
    w = windows.make_windows(df, fs=FS)
    assert (w["units"] == "normalized").all()


# ---- features ---------------------------------------------------------------------------------
def test_window_features_names_and_dominant_frequency():
    t = np.arange(100) / FS
    x = np.zeros((100, 6), dtype=np.float32)
    x[:, 0] = 3 * np.sin(2 * np.pi * 1.5 * t)
    x[:, 2] = 9.81
    f = features.window_features(x, FS)
    assert len(f) == len(features.FEATURE_NAMES)
    assert all(k.startswith("feat_") for k in f)
    assert f["feat_ax_domfreq"] == pytest.approx(1.5, abs=0.3)
    assert f["feat_az_mean"] == pytest.approx(9.81, abs=0.01)
    assert f["feat_gy_energy"] == 0.0
    assert f["feat_amag_std"] > 0
    assert np.isfinite(list(f.values())).all()


def test_featurize_windows_adds_columns():
    df = resample.resample_stream(_stream(fs=100.0), FS)
    w = features.featurize_windows(windows.make_windows(df, fs=FS), FS)
    assert {"feat_ax_mean", "feat_amag_domfreq"} <= set(w.columns)
    assert len(w) == 9
