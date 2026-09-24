"""`formcoach features build`: converted streams -> windows + features Parquet."""

from __future__ import annotations

import pandas as pd

from formcoach.data import convert
from formcoach.data.fixtures import FIXTURE_ROOT
from formcoach.signal import build, features


def test_build_windows_for_every_stream_and_skip_existing(tmp_path):
    processed = tmp_path / "processed"
    convert.convert_dataset("mmfit", FIXTURE_ROOT / "mmfit" / "mm-fit", processed)
    convert.convert_dataset("recgym", FIXTURE_ROOT / "recgym", processed)
    written = build.build_features(processed, fs=50, win_s=2.0, stride_s=1.0)
    names = sorted(p.name for p in written)
    assert "P2-w00-sw_l.parquet" in names and "G1-s1-wrist.parquet" in names
    assert all("windows" in p.parts for p in written)
    w = pd.read_parquet(written[0])
    assert {"window_id", "label_exercise", "label_active", "label_purity", "x"} <= set(w.columns)
    assert set(features.FEATURE_NAMES) <= set(w.columns)
    assert len(w) > 5
    assert build.build_features(processed) == []  # nothing rewritten
    assert len(build.build_features(processed, dataset="recgym", force=True)) == 4


def test_load_windows_concatenates_and_filters_purity(tmp_path):
    processed = tmp_path / "processed"
    convert.convert_dataset("mmfit", FIXTURE_ROOT / "mmfit" / "mm-fit", processed)
    build.build_features(processed)
    w = build.load_windows(processed, "mmfit", min_purity=0.8)
    assert (w["label_purity"] >= 0.8).all()
    assert set(w["device"]) == {"sw_l", "sw_r"}
    assert set(w["subject"]) == {"P2"}
    w_all = build.load_windows(processed, "mmfit", min_purity=0.0)
    assert len(w_all) >= len(w)
