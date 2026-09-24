"""Pose-based rep segmentation: one excursion of the primary joint angle beyond ``enter`` and
back past ``exit`` (docs/02 §4.4). ``mode="below"`` for angles that *decrease* during the rep
(elbow in a curl, knee in a squat); ``mode="above"`` for angles that increase (shoulder
abduction in a lateral raise). :class:`RepSegmenter` is the online form used by the live
pipeline; :func:`segment_reps` runs it over a whole series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseRep:
    rep_id: int
    t_start: float
    t_end: float
    t_extreme: float
    i_start: int
    i_end: int
    extreme_value: float
    duration_s: float


# per-exercise defaults (docs/02 §7); the rules engine may override from rules.yaml
DEFAULT_REP_CONFIG: dict[str, dict] = {
    "curl": {"angle": "elbow", "mode": "below", "enter": 60.0, "exit": 150.0},
    "press": {"angle": "elbow", "mode": "below", "enter": 90.0, "exit": 160.0},
    "raise": {"angle": "shoulder_abd", "mode": "above", "enter": 75.0, "exit": 40.0},
    "squat": {"angle": "knee", "mode": "below", "enter": 100.0, "exit": 160.0},
}

ADAPTIVE_DEFAULTS = {"window_s": 6.0, "min_range": 25.0, "frac": 0.3}


class RepSegmenter:
    """Hysteresis state machine over ``(t, angle)`` samples; ``push`` returns a rep when one
    completes. ``min_duration_s`` drops glitches.

    With ``adaptive=True`` the ``enter``/``exit`` thresholds are recomputed from the 10th/90th
    percentiles of the last ``window_s`` seconds of angle: ``enter = lo + frac*range``,
    ``exit = hi - frac*range`` (mirrored for ``mode="above"``), and nothing is a rep until the
    observed range exceeds ``min_range`` degrees. This makes segmentation independent of the
    pose source's absolute angle scale (MM-Fit lifted 3-D pose reads a curl as 95-140 deg, not
    60-150); the fixed values remain the initial thresholds (ADR-0018).
    """

    def __init__(
        self,
        *,
        enter: float,
        exit: float,
        mode: str = "below",
        min_duration_s: float = 0.4,
        adaptive: bool = False,
        window_s: float = ADAPTIVE_DEFAULTS["window_s"],
        min_range: float = ADAPTIVE_DEFAULTS["min_range"],
        frac: float = ADAPTIVE_DEFAULTS["frac"],
    ):
        if mode not in ("below", "above"):
            raise ValueError("mode must be 'below' or 'above'")
        self.sign = 1.0 if mode == "below" else -1.0
        self.enter = self.sign * enter
        self.exit = self.sign * exit
        self.min_duration_s = min_duration_s
        self.adaptive = adaptive
        self.window_s, self.min_range, self.frac = window_s, min_range, frac
        self._hist_t: list[float] = []
        self._hist_v: list[float] = []
        self.active = False
        self.i = -1
        self.n_reps = 0
        self._last_rest_i = 0
        self._last_rest_t = 0.0
        self._ext_v = np.inf
        self._ext_t = 0.0

    def _adapt(self, t: float, v: float) -> bool:
        """Update thresholds from the recent window; False while the range is too small."""
        self._hist_t.append(t)
        self._hist_v.append(v)
        while self._hist_t and t - self._hist_t[0] > self.window_s:
            self._hist_t.pop(0)
            self._hist_v.pop(0)
        if len(self._hist_v) < 5:
            return False
        lo, hi = np.percentile(self._hist_v, [10, 90])
        rng = hi - lo
        if rng < self.min_range:
            return False
        self.enter = lo + self.frac * rng
        self.exit = hi - self.frac * rng
        return True

    def push(self, t: float, angle: float) -> PoseRep | None:
        self.i += 1
        if not np.isfinite(angle):
            return None
        v = self.sign * angle  # rep = v drops below enter, ends when v rises above exit
        if self.adaptive and not self._adapt(t, v):
            self._last_rest_i, self._last_rest_t = self.i, t
            self.active = False
            return None
        if not self.active:
            if v >= self.exit:
                self._last_rest_i, self._last_rest_t = self.i, t
            if v < self.enter:
                self.active = True
                self._ext_v, self._ext_t = v, t
            return None
        if v < self._ext_v:
            self._ext_v, self._ext_t = v, t
        if v >= self.exit:
            self.active = False
            duration = t - self._last_rest_t
            start_i, start_t = self._last_rest_i, self._last_rest_t
            self._last_rest_i, self._last_rest_t = self.i, t
            if duration < self.min_duration_s:
                return None
            rep = PoseRep(
                self.n_reps, start_t, t, self._ext_t, start_i, self.i,
                float(self.sign * self._ext_v), float(duration),
            )  # fmt: skip
            self.n_reps += 1
            return rep
        return None


def segment_reps(
    angle: np.ndarray,
    t: np.ndarray,
    *,
    enter: float,
    exit: float,
    mode: str = "below",
    min_duration_s: float = 0.4,
    adaptive: bool = False,
    **adaptive_kw,
) -> list[PoseRep]:
    """Run :class:`RepSegmenter` over a whole angle series (degrees) with times ``t`` (s)."""
    seg = RepSegmenter(
        enter=enter, exit=exit, mode=mode, min_duration_s=min_duration_s, adaptive=adaptive,
        **adaptive_kw,
    )  # fmt: skip
    out = []
    for ti, ai in zip(np.asarray(t, dtype=float), np.asarray(angle, dtype=float), strict=True):
        rep = seg.push(float(ti), float(ai))
        if rep is not None:
            out.append(rep)
    return out
