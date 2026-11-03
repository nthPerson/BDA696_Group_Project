"""Per-side rep metrics and the segmenter's in-progress start time (live-demo plan Task 2).

``side="l"``/``"r"`` must use that arm's own series instead of the 0.5*(left+right) average,
so a one-arm or alternating rep is not diluted by an idle arm; ``side="both"`` stays
bit-identical to the pre-Task-2 behaviour.
"""

from __future__ import annotations

import numpy as np

from formcoach.app.pipeline import metrics_from_angles
from formcoach.pose import angles
from formcoach.pose.reps import RepSegmenter

N = 30  # REP_STEPS


def _res(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    """One ``N``-point series per name in ``angles.ANGLE_NAMES``: ``elbow_l``/``elbow_r`` from
    the arguments, every other angle held constant at 90 degrees."""
    res: dict[str, np.ndarray] = {}
    for name in angles.ANGLE_NAMES:
        if name == "elbow_l":
            res[name] = np.asarray(left, dtype=float)
        elif name == "elbow_r":
            res[name] = np.asarray(right, dtype=float)
        else:
            res[name] = np.full(N, 90.0)
    return res


def test_side_l_metrics_ignore_the_right_arm():
    left = np.linspace(50.0, 150.0, N)
    right = np.full(N, 150.0)
    res = _res(left, right)

    both = metrics_from_angles(res, 1.5, side="both")
    assert both["elbow_min"] == 100.0

    l_only = metrics_from_angles(res, 1.5, side="l")
    assert l_only["elbow_min"] == 50.0
    assert l_only["elbow_max"] == 150.0

    r_only = metrics_from_angles(res, 1.5, side="r")
    assert r_only["elbow_min"] == 150.0


def test_asymmetry_still_compares_both_sides():
    left = np.linspace(50.0, 150.0, N)
    right = np.full(N, 150.0)
    res = _res(left, right)
    m = metrics_from_angles(res, 1.5, side="l")
    assert m["elbow_asymmetry"] == 100.0


def test_rep_start_t_is_none_when_idle_and_set_during_a_rep():
    seg = RepSegmenter(enter=60.0, exit=150.0, mode="below")
    dt = 1.0 / 30.0  # 30 fps
    t = 0.0

    seg.push(t, 150.0)
    assert seg.rep_start_t is None

    t += dt
    seg.push(t, 50.0)
    assert isinstance(seg.rep_start_t, float)

    t += dt
    seg.push(t, 150.0)
    assert seg.rep_start_t is None
