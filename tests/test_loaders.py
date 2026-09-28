"""Loader contracts on the committed fixtures (data/fixtures/<dataset>/), built by
`formcoach data make-fixtures`. Every loader returns the IMUStream schema (docs/02 §5)."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.data import labels, mmfit, recgym, recofit, schema
from formcoach.data.fixtures import FIXTURE_ROOT

MMFIT_ROOT = FIXTURE_ROOT / "mmfit" / "mm-fit"
RECOFIT_ROOT = FIXTURE_ROOT / "recofit"
RECGYM_ROOT = FIXTURE_ROOT / "recgym"


# ---- label maps --------------------------------------------------------------------------------
def test_canonical_label_maps():
    assert labels.canonical("mmfit", "bicep_curls") == "curl"
    assert labels.canonical("mmfit", "dumbbell_shoulder_press") == "press"
    assert labels.canonical("mmfit", "lateral_shoulder_raises") == "raise"
    assert labels.canonical("mmfit", "squats") == "squat"
    assert labels.canonical("mmfit", "jumping_jacks") == "other"
    assert (
        labels.canonical("recofit", "Two-arm Dumbbell Curl (both arms, not alternating)") == "curl"
    )
    assert labels.canonical("recofit", "Squat Rack Shoulder Press") == "press"
    assert labels.canonical("recofit", "Lateral Raise") == "raise"
    assert labels.canonical("recofit", "Squat (kettlebell / goblet)") == "squat"
    assert labels.canonical("recofit", "Non-Exercise") == "idle"
    assert labels.canonical("recofit", "Walk") == "idle"
    assert labels.canonical("recofit", "Plank") == "other"
    assert labels.canonical("recofit", "Tap Left Device") is None  # junk: dropped downstream
    assert labels.canonical("recgym", "ArmCurl") == "curl"
    assert labels.canonical("recgym", "Null") == "idle"
    assert labels.canonical("recgym", "Riding") == "other"
    with pytest.raises(KeyError):
        labels.canonical("recofit", "Not A Real Label")


def test_every_recofit_label_is_mapped():
    assert len(labels.RECOFIT_ACTIVITIES) == 75
    for name in labels.RECOFIT_ACTIVITIES:
        labels.canonical("recofit", name)  # no KeyError


# ---- MM-Fit ------------------------------------------------------------------------------------
def test_mmfit_fixture_exists_and_lists_workouts():
    assert mmfit.available(MMFIT_ROOT)
    assert mmfit.list_workouts(MMFIT_ROOT) == ["w00"]
    assert mmfit.WORKOUT_SUBJECT["w00"] == 2 and mmfit.WORKOUT_SUBJECT["w20"] == 9
    assert len(set(mmfit.WORKOUT_SUBJECT.values())) == 10
    assert mmfit.SPLITS["unseen_test"] == ["w00", "w05", "w12", "w13", "w20"]


def test_mmfit_stream_schema_units_and_labels():
    df = mmfit.load_stream(MMFIT_ROOT, "w00", device="sw_l")
    schema.validate_imu_stream(df)
    assert df["dataset"].iloc[0] == "mmfit" and df["subject"].iloc[0] == "P2"
    assert df["placement"].iloc[0] == "wrist_l" and df["device"].iloc[0] == "sw_l"
    assert "frame" in df.columns and "label_raw" in df.columns
    assert df["t"].iloc[0] > -0.1  # t = 0 is video frame 0 from the fitted session clock
    a = np.linalg.norm(df[["ax", "ay", "az"]].to_numpy(), axis=1)
    assert 8 < np.median(a) < 12  # already m/s² in the raw files
    sets = mmfit.load_sets(MMFIT_ROOT, "w00")
    assert list(sets["exercise"]) == ["curl", "squat"]
    assert sets["reps"].tolist() == [10, 8]
    curl = sets.iloc[0]
    inside = df[(df["frame"] >= curl.start_frame) & (df["frame"] <= curl.end_frame)]
    assert (inside["exercise"] == "curl").all() and (inside["set_id"] == 0).all()
    before = df[df["frame"] < curl.start_frame]
    assert (before["exercise"] == "idle").all() and (before["set_id"] == -1).all()
    assert (df["rep_id"] == -1).all()  # MM-Fit has set-level rep counts only


def test_mmfit_missing_modality_returns_none_and_right_wrist_loads():
    assert mmfit.load_modality(MMFIT_ROOT, "w00", "sp_l_acc") is None
    df = mmfit.load_stream(MMFIT_ROOT, "w00", device="sw_r")
    assert df["placement"].iloc[0] == "wrist_r"
    with pytest.raises(FileNotFoundError):
        mmfit.load_stream(MMFIT_ROOT, "w00", device="eb_l")  # not in the fixture


def test_mmfit_duplicate_timestamps_are_dropped_and_t_is_monotonic():
    raw = mmfit.load_modality(MMFIT_ROOT, "w00", "sw_l_acc")
    assert (np.diff(raw[:, 1]) == 0).sum() > 0  # fixture contains duplicates on purpose
    df = mmfit.load_stream(MMFIT_ROOT, "w00", device="sw_l")
    assert (np.diff(df["t"].to_numpy()) > 0).all()


def test_mmfit_pose3d_and_session_clock():
    frames, xyz, t = mmfit.load_pose3d(MMFIT_ROOT, "w00")
    assert xyz.shape[1:] == (17, 3) and len(frames) == len(t) == xyz.shape[0]
    assert mmfit.H36M17_JOINTS[13] == "wrist_r" and mmfit.H36M17_JOINTS[0] == "hip_c"
    clock = mmfit.session_clock(MMFIT_ROOT, "w00")
    assert 30 < clock.ms_per_frame < 36  # ~30 fps
    df = mmfit.load_stream(MMFIT_ROOT, "w00", device="sw_l")
    # a sensor row's frame and the pose frame at the same t agree within one frame
    i = len(df) // 2
    assert abs(clock.frame_to_t(df["frame"].iloc[i]) - df["t"].iloc[i]) < 0.05


def test_mmfit_describe_on_fixture():
    d = mmfit.describe(MMFIT_ROOT)
    assert d["workouts"] == 1 and d["subjects"] == 1
    assert d["sets_per_exercise"]["curl"] == 1
    assert d["devices"]["sw_l"]["rate_hz"] == pytest.approx(100, abs=2)


# ---- RecoFit -----------------------------------------------------------------------------------
def test_recofit_fixture_loads_with_scipy_and_lists_visits():
    assert recofit.available(RECOFIT_ROOT)
    visits = recofit.list_visits(RECOFIT_ROOT)
    assert visits == [(1, 1), (2, 1), (2, 2)]  # 1-based subject index / visit as in MATLAB


def test_recofit_stream_units_and_segments():
    df = recofit.load_stream(RECOFIT_ROOT, subject_index=1, visit=1)
    schema.validate_imu_stream(df)
    assert df["placement"].iloc[0] == "forearm_r" and df["subject"].iloc[0] == "R526"
    assert df["session"].iloc[0] == "v1"
    a = np.linalg.norm(df[["ax", "ay", "az"]].to_numpy(), axis=1)
    assert 8 < np.median(a) < 12  # g -> m/s²
    assert np.abs(df[["gx", "gy", "gz"]].to_numpy()).max() < 40  # dps -> rad/s
    assert df["t"].iloc[0] == pytest.approx(0.0)
    segs = recofit.load_segments(RECOFIT_ROOT, subject_index=1, visit=1)
    assert list(segs["label_raw"]) == ["Non-Exercise", "Tap Left Device", "Bicep Curl", "Squat"]
    assert list(segs["exercise"]) == ["idle", None, "curl", "squat"]
    assert segs["reps"].tolist() == [-1, -1, 20, 15]
    curl = segs.iloc[2]
    inside = df[(df["t"] >= curl.t_start) & (df["t"] < curl.t_end)]
    assert (inside["exercise"] == "curl").all() and (inside["set_id"] == 2).all()
    tap = df[df["label_raw"] == "Tap Left Device"]
    assert (
        len(tap) > 0 and (tap["exercise"] == "idle").all()
    )  # junk keeps raw label, canonical idle


def test_recofit_struct_array_visits_are_handled():
    df = recofit.load_stream(RECOFIT_ROOT, subject_index=2, visit=2)
    schema.validate_imu_stream(df)
    assert df["session"].iloc[0] == "v2"


def test_recofit_describe_on_fixture():
    d = recofit.describe(RECOFIT_ROOT)
    assert d["multi"]["subjects"] == 2 and d["multi"]["visits"] == 3
    assert d["single"]["recordings"] == 2
    assert "curl" in d["multi"]["minutes_per_exercise"]


# ---- RecGym ------------------------------------------------------------------------------------
def test_recgym_sessions_and_normalized_units():
    assert recgym.available(RECGYM_ROOT)
    sessions = recgym.list_sessions(RECGYM_ROOT)
    assert sessions[0] == (1, "wrist", 1)
    df = recgym.load_stream(RECGYM_ROOT, 1, "wrist", 1)
    schema.validate_imu_stream(df)
    assert (df["units"] == "normalized").all()
    assert df["placement"].iloc[0] == "wrist" and df["subject"].iloc[0] == "G1"
    assert df["t"].iloc[1] - df["t"].iloc[0] == pytest.approx(0.05)  # 20 Hz, synthetic clock
    assert abs(df["ax"].mean()) < 0.1  # centred around 0.5 -> 0
    assert set(df["exercise"]) == {"idle", "curl"}
    leg = recgym.load_stream(RECGYM_ROOT, 1, "leg", 1)
    assert leg["placement"].iloc[0] == "calf"


def test_recgym_describe_on_fixture():
    d = recgym.describe(RECGYM_ROOT)
    assert d["subjects"] == 2 and d["rows"] > 0 and d["positions"] == ["leg", "wrist"]
