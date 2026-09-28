"""pose/: skeleton maps, joint angles (hand-computed), normalisation, rep segmentation.
Pure NumPy — no MediaPipe needed."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.pose import angles, normalize, reps, skeletons


def _standing(n: int = 1) -> np.ndarray:
    """H3.6M-17 standing skeleton (mm), y up, x to the subject's left, z toward camera."""
    base = {
        "hip_c": (0, 0, 0), "hip_l": (100, 0, 0), "knee_l": (100, -450, 0),
        "foot_l": (100, -900, 0), "hip_r": (-100, 0, 0), "knee_r": (-100, -450, 0),
        "foot_r": (-100, -900, 0), "spine": (0, 250, 0), "thorax": (0, 500, 0),
        "neck": (0, 580, 0), "head": (0, 680, 0),
        "shoulder_r": (-200, 480, 0), "elbow_r": (-200, 180, 0), "wrist_r": (-200, -100, 0),
        "shoulder_l": (200, 480, 0), "elbow_l": (200, 180, 0), "wrist_l": (200, -100, 0),
    }  # fmt: skip
    sk = skeletons.H36M17
    xyz = np.zeros((n, len(sk.joints), 3))
    for name, p in base.items():
        xyz[:, sk.index(name), :] = p
    return xyz


def test_angle_deg_hand_computed_cases():
    a, b, c = np.array([1.0, 0, 0]), np.array([0.0, 0, 0]), np.array([0.0, 1, 0])
    assert angles.angle_deg(a, b, c) == pytest.approx(90.0)
    assert angles.angle_deg(np.array([1.0, 0, 0]), b, np.array([-1.0, 0, 0])) == pytest.approx(
        180.0
    )
    assert angles.angle_deg(np.array([1.0, 0, 0]), b, np.array([2.0, 0, 0])) == pytest.approx(0.0)
    assert angles.angle_deg(np.array([1.0, 1, 0]), b, np.array([1.0, 0, 0])) == pytest.approx(45.0)
    assert np.isnan(angles.angle_deg(b, b, c))  # zero-length segment -> nan, not crash
    batch = angles.angle_deg(np.tile(a, (4, 1)), np.tile(b, (4, 1)), np.tile(c, (4, 1)))
    assert batch.shape == (4,) and np.allclose(batch, 90.0)


def test_skeleton_lookup_and_aliases():
    sk = skeletons.H36M17
    assert sk.index("wrist_r") == 13 and sk.index("ankle_l") == sk.index("foot_l")
    mp = skeletons.MEDIAPIPE33
    assert mp.index("shoulder_l") == 11 and mp.index("ankle_r") == 28 and len(mp.joints) == 33
    with pytest.raises(KeyError):
        sk.index("tail")
    xyz = _standing(3)
    hip_c = skeletons.joint(xyz, sk, "hip_c")
    assert hip_c.shape == (3, 3)
    # derived centres exist for MediaPipe (no hip_c joint)
    xyz_mp = np.zeros((2, 33, 3))
    xyz_mp[:, 23] = (1, 0, 0)
    xyz_mp[:, 24] = (-1, 0, 0)
    assert np.allclose(skeletons.joint(xyz_mp, mp, "hip_c"), 0)


def test_joint_angles_on_standing_pose_with_bent_elbow():
    xyz = _standing(2)
    sk = skeletons.H36M17
    # bend the left elbow to 90°: wrist forward (toward camera) at elbow height
    xyz[1, sk.index("wrist_l")] = (200, 180, 280)
    up = np.array([0.0, 1.0, 0.0])
    a = angles.joint_angles(xyz, sk, up)
    assert a["elbow_l"][0] == pytest.approx(180.0, abs=1e-6)
    assert a["elbow_l"][1] == pytest.approx(90.0, abs=1e-6)
    assert a["elbow_r"][1] == pytest.approx(180.0, abs=1e-6)
    assert a["knee_l"][0] == pytest.approx(180.0)
    assert a["trunk_incl"][0] == pytest.approx(0.0, abs=1e-6)
    assert 10 < a["shoulder_abd_l"][0] < 25  # arm hangs slightly out from the hip
    assert a["hip_l"][0] == pytest.approx(180.0 - a["shoulder_abd_l"][0] * 0 - 0, abs=15)
    assert a["knee_track_l"][0] == pytest.approx(0.0, abs=1e-6)
    assert a["hip_knee_height_l"][0] > 0  # standing: hip above knee (torso units)
    assert set(angles.ANGLE_NAMES) <= set(a)


def test_trunk_lean_and_knee_valgus_are_detected():
    xyz = _standing(1)
    sk = skeletons.H36M17
    up = np.array([0.0, 1.0, 0.0])
    # lean the whole upper body 30° forward (rotate shoulders/spine/neck/head about x axis)
    ang = np.deg2rad(30)
    for j in (
        "spine",
        "thorax",
        "neck",
        "head",
        "shoulder_l",
        "shoulder_r",
        "elbow_l",
        "elbow_r",
        "wrist_l",
        "wrist_r",
    ):
        p = xyz[0, sk.index(j)]
        y, z = p[1], p[2]
        xyz[0, sk.index(j), 1] = y * np.cos(ang) - z * np.sin(ang)
        xyz[0, sk.index(j), 2] = y * np.sin(ang) + z * np.cos(ang)
    a = angles.joint_angles(xyz, sk, up)
    assert a["trunk_incl"][0] == pytest.approx(30.0, abs=1e-6)
    # left knee moves 100 mm toward the midline (valgus): knee_track_l negative in torso units
    xyz2 = _standing(1)
    xyz2[0, sk.index("knee_l"), 0] -= 100
    a2 = angles.joint_angles(xyz2, sk, up)
    torso = angles.torso_length(xyz2, sk)[0]
    assert a2["knee_track_l"][0] == pytest.approx(-100 / torso, abs=1e-6)


def test_estimate_up_vector_and_normalize_sequence():
    xyz = _standing(50) + np.random.default_rng(0).normal(0, 2, (50, 17, 3))
    sk = skeletons.H36M17
    up = normalize.estimate_up_vector(xyz, sk)
    assert np.allclose(up, [0, 1, 0], atol=0.05)
    xyz_n, valid = normalize.normalize_sequence(xyz, sk)
    assert valid.all() and xyz_n.shape == xyz.shape
    assert np.allclose(skeletons.joint(xyz_n, sk, "hip_c"), 0, atol=1e-6)  # hip-centred
    assert np.allclose(angles.torso_length(xyz_n, sk).mean(), 1.0, atol=0.05)  # torso units


def test_normalize_sequence_gap_interpolation_and_invalid_segments():
    xyz = _standing(40).astype(float)
    sk = skeletons.H36M17
    vis = np.ones((40, 17))
    vis[10:12, sk.index("wrist_l")] = 0.1  # 2-frame gap: interpolated
    vis[20:26, sk.index("wrist_l")] = 0.1  # 6-frame gap: invalid
    xyz[10:12, sk.index("wrist_l")] = np.nan
    xyz_n, valid = normalize.normalize_sequence(
        xyz, sk, visibility=vis, required=("wrist_l",), vis_thr=0.5, max_gap=3
    )
    assert valid[10:12].all() and not np.isnan(xyz_n[10:12]).any()
    assert not valid[20:26].any() and valid[26:].all()


def test_smooth_angles_keeps_length_and_removes_spikes():
    t = np.arange(120) / 30.0
    a = 120 + 40 * np.sin(2 * np.pi * 0.5 * t)
    noisy = a.copy()
    noisy[50] += 60  # one-frame spike
    s = normalize.smooth_angles(noisy)
    assert s.shape == a.shape
    assert abs(s[50] - a[50]) < 8
    assert np.abs(s[10:-10] - a[10:-10]).max() < 5


def test_segment_reps_curl_like_angle_series():
    t = np.arange(0, 12, 1 / 30)
    # elbow angle: 170 at rest, five excursions to 40°
    elbow = 170 - 130 * np.clip(np.sin(2 * np.pi * 0.5 * t), 0, None) ** 2
    found = reps.segment_reps(elbow, t, enter=60.0, exit=150.0, mode="below")
    assert len(found) == 6 or len(found) == 5
    r = found[1]
    assert r.t_start < r.t_extreme < r.t_end and r.extreme_value == pytest.approx(40, abs=2)
    assert 0.5 < r.duration_s < 2.5  # time below 150° and back, not the 2 s cycle
    # incremental segmenter gives the same reps
    seg = reps.RepSegmenter(enter=60.0, exit=150.0, mode="below")
    online = [
        rr for i in range(len(t)) if (rr := seg.push(float(t[i]), float(elbow[i]))) is not None
    ]
    assert [x.rep_id for x in online] == [x.rep_id for x in found]
    assert [x.t_end for x in online] == pytest.approx([x.t_end for x in found])


def test_segment_reps_above_mode_and_min_duration():
    t = np.arange(0, 10, 1 / 30)
    abd = 20 + 70 * np.clip(np.sin(2 * np.pi * 0.5 * t), 0, None)  # peaks at 90°
    found = reps.segment_reps(abd, t, enter=75.0, exit=40.0, mode="above")
    assert len(found) == 5
    fast = reps.segment_reps(abd, t, enter=75.0, exit=40.0, mode="above", min_duration_s=5.0)
    assert fast == []
    with pytest.raises(ValueError):
        reps.segment_reps(abd, t, enter=75.0, exit=40.0, mode="sideways")


def test_adaptive_segmenter_finds_reps_at_a_different_angle_scale():
    t = np.arange(0, 14, 1 / 30)
    # MM-Fit-like curl: elbow swings 140 -> 95 -> 140 (never below 60 or above 150)
    elbow = 140 - 45 * np.clip(np.sin(2 * np.pi * 0.5 * t), 0, None) ** 2
    fixed = reps.segment_reps(elbow, t, enter=60.0, exit=150.0, mode="below")
    assert fixed == []
    adaptive = reps.segment_reps(elbow, t, enter=60.0, exit=150.0, mode="below", adaptive=True)
    assert 5 <= len(adaptive) <= 7
    assert adaptive[2].extreme_value == pytest.approx(95, abs=3)
    # a series with too little range yields nothing
    flat = 140 - 5 * np.sin(2 * np.pi * 0.5 * t)
    assert reps.segment_reps(flat, t, enter=60.0, exit=150.0, mode="below", adaptive=True) == []
