"""Joint angles from 3-D joints (docs/02-system-design.md §4.3). Pure NumPy.

All angles in degrees; distances in *torso units* (torso = mid-shoulder to mid-hip). ``up``
is the unit vertical (see :func:`formcoach.pose.normalize.estimate_up_vector`).

Names produced by :func:`joint_angles` (``ANGLE_NAMES``):

* ``elbow_l/r`` shoulder–elbow–wrist
* ``shoulder_abd_l/r`` hip–shoulder–wrist (abduction proxy)
* ``upper_arm_trunk_l/r`` angle between upper arm (shoulder→elbow) and trunk (shoulder_c→hip_c)
* ``hip_l/r`` shoulder–hip–knee · ``knee_l/r`` hip–knee–ankle
* ``trunk_incl`` angle between mid-hip→mid-shoulder and ``up``
* ``knee_track_l/r`` lateral offset of knee from ankle, torso units, negative = inward (valgus)
* ``elbow_drift_l/r`` horizontal offset of elbow from shoulder, torso units
* ``hip_knee_height_l/r`` (hip − knee)·up in torso units, ≤ 0 when the hip is below the knee
"""

from __future__ import annotations

import numpy as np

from formcoach.pose.skeletons import Skeleton, joint

ANGLE_NAMES: tuple[str, ...] = (
    "elbow_l", "elbow_r", "shoulder_abd_l", "shoulder_abd_r", "upper_arm_trunk_l",
    "upper_arm_trunk_r", "hip_l", "hip_r", "knee_l", "knee_r", "trunk_incl", "knee_track_l",
    "knee_track_r", "elbow_drift_l", "elbow_drift_r", "hip_knee_height_l", "hip_knee_height_r",
)  # fmt: skip
EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > EPS, v / np.maximum(n, EPS), np.nan)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray | float:
    """Angle at ``b`` between ``b→a`` and ``b→c`` in degrees; NaN for a zero-length segment.
    Broadcasts over leading dimensions (``(n, 3)`` inputs give ``(n,)``)."""
    a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
    u, v = _unit(a - b), _unit(c - b)
    cos = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    return float(out) if out.ndim == 0 else out


def vector_angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between vectors ``u`` and ``v`` (broadcast, degrees)."""
    cos = np.clip(np.sum(_unit(u) * _unit(v), axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def torso_length(xyz: np.ndarray, sk: Skeleton) -> np.ndarray:
    """``|shoulder_c − hip_c|`` per frame (same unit as ``xyz``)."""
    return np.linalg.norm(joint(xyz, sk, "shoulder_c") - joint(xyz, sk, "hip_c"), axis=-1)


def joint_angles(xyz: np.ndarray, sk: Skeleton, up: np.ndarray) -> dict[str, np.ndarray]:
    """All :data:`ANGLE_NAMES` for ``xyz`` ``(n, J, 3)``; each value is ``(n,)``."""
    xyz = np.asarray(xyz, dtype=np.float64)
    up = _unit(np.asarray(up, dtype=np.float64))
    j = {name: joint(xyz, sk, name) for name in (
        "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r", "hip_l", "hip_r",
        "knee_l", "knee_r", "ankle_l", "ankle_r", "hip_c", "shoulder_c",
    )}  # fmt: skip
    torso = np.maximum(torso_length(xyz, sk), EPS)
    trunk = j["shoulder_c"] - j["hip_c"]
    lateral = j["shoulder_l"] - j["shoulder_r"]
    lateral = lateral - np.sum(lateral * up, axis=-1, keepdims=True) * up  # horizontal
    lateral = _unit(lateral)

    def horiz(v: np.ndarray) -> np.ndarray:
        return v - np.sum(v * up, axis=-1, keepdims=True) * up

    out: dict[str, np.ndarray] = {}
    for s in ("l", "r"):
        out[f"elbow_{s}"] = angle_deg(j[f"shoulder_{s}"], j[f"elbow_{s}"], j[f"wrist_{s}"])
        out[f"shoulder_abd_{s}"] = angle_deg(j[f"hip_{s}"], j[f"shoulder_{s}"], j[f"wrist_{s}"])
        out[f"upper_arm_trunk_{s}"] = vector_angle_deg(j[f"elbow_{s}"] - j[f"shoulder_{s}"], -trunk)
        out[f"hip_{s}"] = angle_deg(j[f"shoulder_{s}"], j[f"hip_{s}"], j[f"knee_{s}"])
        out[f"knee_{s}"] = angle_deg(j[f"hip_{s}"], j[f"knee_{s}"], j[f"ankle_{s}"])
        sign = 1.0 if s == "l" else -1.0  # outward = +lateral for the left knee
        d = horiz(j[f"knee_{s}"] - j[f"ankle_{s}"])
        out[f"knee_track_{s}"] = sign * np.sum(d * lateral, axis=-1) / torso
        out[f"elbow_drift_{s}"] = (
            np.linalg.norm(horiz(j[f"elbow_{s}"] - j[f"shoulder_{s}"]), axis=-1) / torso
        )
        out[f"hip_knee_height_{s}"] = np.sum((j[f"hip_{s}"] - j[f"knee_{s}"]) * up, axis=-1) / torso
    out["trunk_incl"] = vector_angle_deg(trunk, np.broadcast_to(up, trunk.shape))
    return out
