"""Synthetic MediaPipe-33 pose frames for the live-demo pipeline tests (Tasks 3-4).

``pose_frame`` builds a finite, self-consistent standing figure: shoulders and hips fixed,
upper arms straight down from the shoulder, forearms rotated so the elbow angle is exactly the
requested degrees, legs built the same way for the knee angle. It never imports cv2/mediapipe.
"""

from __future__ import annotations

import math

import numpy as np

from formcoach.io.replay import PoseFrame
from formcoach.pose.skeletons import MEDIAPIPE33

UPPER_ARM = 0.3  # m, shoulder -> elbow
FOREARM = 0.3  # m, elbow -> wrist
UPPER_LEG = 0.45  # m, hip -> knee
LOWER_LEG = 0.45  # m, knee -> ankle


def _limb_end(root: np.ndarray, length: float, deg: float) -> np.ndarray:
    """A point ``length`` (m) from ``root``, at ``deg`` degrees from the straight-down
    continuation of the proximal segment (0 = folded back onto it, 180 = fully extended)."""
    rad = math.radians(deg)
    return root + length * np.array([math.sin(rad), math.cos(rad), 0.0])


def pose_frame(
    i: int,
    t: float,
    left_deg: float,
    right_deg: float,
    *,
    vis_l: float = 1.0,
    vis_r: float = 1.0,
    knee_deg: float = 175.0,
) -> PoseFrame:
    """A finite synthetic :class:`PoseFrame` (mediapipe33): shoulders at ``(+/-0.2, 0.5, 0)``,
    hips at ``(+/-0.1, 0, 0)``, elbow angle exactly ``left_deg``/``right_deg`` degrees, knee
    angle exactly ``knee_deg``. ``visibility`` is 1.0 everywhere except the shoulder/elbow/wrist
    of each side, set to ``vis_l``/``vis_r``. Always ``valid=True``.
    """
    sk = MEDIAPIPE33
    world = np.zeros((33, 3), dtype=np.float64)
    vis = np.ones(33, dtype=np.float64)

    shoulder_l = np.array([-0.2, 0.5, 0.0])
    shoulder_r = np.array([0.2, 0.5, 0.0])
    hip_l = np.array([-0.1, 0.0, 0.0])
    hip_r = np.array([0.1, 0.0, 0.0])
    elbow_l = shoulder_l + np.array([0.0, -UPPER_ARM, 0.0])
    elbow_r = shoulder_r + np.array([0.0, -UPPER_ARM, 0.0])
    wrist_l = _limb_end(elbow_l, FOREARM, left_deg)
    wrist_r = _limb_end(elbow_r, FOREARM, right_deg)
    knee_l = hip_l + np.array([0.0, -UPPER_LEG, 0.0])
    knee_r = hip_r + np.array([0.0, -UPPER_LEG, 0.0])
    ankle_l = _limb_end(knee_l, LOWER_LEG, knee_deg)
    ankle_r = _limb_end(knee_r, LOWER_LEG, knee_deg)

    joints = {
        "shoulder_l": shoulder_l, "shoulder_r": shoulder_r,
        "hip_l": hip_l, "hip_r": hip_r,
        "elbow_l": elbow_l, "elbow_r": elbow_r,
        "wrist_l": wrist_l, "wrist_r": wrist_r,
        "knee_l": knee_l, "knee_r": knee_r,
        "ankle_l": ankle_l, "ankle_r": ankle_r,
    }  # fmt: skip
    for name, xyz in joints.items():
        world[sk.index(name)] = xyz

    for side, v in (("l", vis_l), ("r", vis_r)):
        for part in ("shoulder", "elbow", "wrist"):
            vis[sk.index(f"{part}_{side}")] = v

    return PoseFrame(i, float(t), world, vis, True, sk)


def curl_deg(
    t: float,
    *,
    period_s: float = 2.5,
    top: float = 50.0,
    bottom: float = 150.0,
    phase_s: float = 0.0,
) -> float:
    """Elbow angle (degrees) of a sinusoidal curl at time ``t`` (s): starts at ``bottom``
    (arm extended) and swings to ``top`` (arm curled) and back every ``period_s`` seconds."""
    phase = 2 * math.pi * (t - phase_s) / period_s
    return (top + bottom) / 2 + (bottom - top) / 2 * math.cos(phase)
