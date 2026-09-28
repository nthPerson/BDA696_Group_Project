"""Joint-name → index maps so the angle code works on any pose source.

``H36M17`` is the Human3.6M 17-joint order used by MM-Fit ``pose_3d`` (docs/04 §1);
``MEDIAPIPE33`` is the MediaPipe PoseLandmarker order (docs/02 §4.3). Aliases let the angle
code ask for ``ankle_l`` on a skeleton that only has ``foot_l``; derived centres (``hip_c``,
``shoulder_c``) are computed when the skeleton has no such joint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MP_NAMES = (
    "nose", "eye_inner_l", "eye_l", "eye_outer_l", "eye_inner_r", "eye_r", "eye_outer_r",
    "ear_l", "ear_r", "mouth_l", "mouth_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "pinky_l", "pinky_r", "index_l", "index_r", "thumb_l", "thumb_r",
    "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r",
    "foot_index_l", "foot_index_r",
)  # fmt: skip
_H36M_NAMES = (
    "hip_c", "hip_l", "knee_l", "foot_l", "hip_r", "knee_r", "foot_r", "spine", "thorax",
    "neck", "head", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l",
)  # fmt: skip

_DERIVED = {"hip_c": ("hip_l", "hip_r"), "shoulder_c": ("shoulder_l", "shoulder_r")}


@dataclass(frozen=True)
class Skeleton:
    name: str
    joints: tuple[str, ...]
    aliases: dict[str, str] = field(default_factory=dict)

    def index(self, name: str) -> int:
        name = self.aliases.get(name, name)
        try:
            return self.joints.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.name} has no joint {name!r}") from exc

    def has(self, name: str) -> bool:
        return self.aliases.get(name, name) in self.joints


H36M17 = Skeleton("h36m17", _H36M_NAMES, {"ankle_l": "foot_l", "ankle_r": "foot_r"})
MEDIAPIPE33 = Skeleton("mediapipe33", _MP_NAMES)
SKELETONS = {s.name: s for s in (H36M17, MEDIAPIPE33)}


def joint(xyz: np.ndarray, skeleton: Skeleton, name: str) -> np.ndarray:
    """``(n, 3)`` coordinates of ``name`` from ``xyz`` ``(n, J, 3)``; derived centres computed."""
    if skeleton.has(name):
        return xyz[:, skeleton.index(name), :]
    if name in _DERIVED:
        a, b = _DERIVED[name]
        return 0.5 * (joint(xyz, skeleton, a) + joint(xyz, skeleton, b))
    raise KeyError(f"{skeleton.name} has no joint {name!r}")
