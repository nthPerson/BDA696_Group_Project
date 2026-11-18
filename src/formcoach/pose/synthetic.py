"""A synthetic person and a synthetic camera, for tests, dry runs and the Bench without hardware.

``pose_frame`` builds a finite, self-consistent standing figure: shoulders and hips fixed,
upper arms straight down from the shoulder, forearms rotated so the elbow angle is exactly the
requested degrees, legs built the same way for the knee angle. :class:`FakeScene` plays that
figure doing an exercise on a schedule as a paced stream of dark frames plus a ``detect``
function shaped like the MediaPipe wrapper's. It never imports cv2 or mediapipe.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator

import numpy as np

from formcoach.io.fake import default_schedule
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


# (joint the exercise counts on, angle at rest, angle at the far end of the movement), degrees.
# The rest angle is where the rep counter's default thresholds (pose/reps.py) see "not started".
_MOTION = {
    "curl": ("elbow", 150.0, 50.0),
    "press": ("elbow", 170.0, 60.0),
    "raise": ("shoulder_abd", 20.0, 95.0),
    "squat": ("knee", 175.0, 80.0),
}
_STAND_ELBOW = 170.0
_STAND_KNEE = 175.0
_HIP_TILT_DEG = math.degrees(math.atan2(0.1, 0.5))  # hip-shoulder line vs straight down
_Y_CENTRE = -0.2  # m, middle of the figure (shoulders 0.5, ankles about -0.9)
_BODY_HEIGHT_FRAC = 0.7  # of the frame height, for a figure about 1.4 m tall


def _abduct(pf: PoseFrame, abd_l: float, abd_r: float) -> None:
    """Move both arms (straight, elbow 180 deg) so ``hip-shoulder-wrist`` reads ``abd_l``/``abd_r``
    degrees; edits ``pf.world`` in place."""
    sk = pf.skeleton
    for side, deg, sign in (("l", abd_l, -1.0), ("r", abd_r, 1.0)):
        phi = math.radians(deg - _HIP_TILT_DEG)  # from straight down, outward
        shoulder = pf.world[sk.index(f"shoulder_{side}")]
        step = np.array([sign * math.sin(phi), -math.cos(phi), 0.0]) * UPPER_ARM
        pf.world[sk.index(f"elbow_{side}")] = shoulder + step
        pf.world[sk.index(f"wrist_{side}")] = shoulder + step * (UPPER_ARM + FOREARM) / UPPER_ARM


def _with_extremities(pf: PoseFrame) -> None:
    """Place the face, hand and foot landmarks that ``pose_frame`` leaves at the origin, so the
    figure draws as a person (angles use none of them); edits ``pf.world`` in place."""
    sk = pf.skeleton
    w = pf.world
    for name in ("nose", "eye_inner_l", "eye_l", "eye_outer_l", "eye_inner_r", "eye_r",
                 "eye_outer_r", "ear_l", "ear_r", "mouth_l", "mouth_r"):  # fmt: skip
        w[sk.index(name)] = (0.0, 0.68, 0.0)
    for side in ("l", "r"):
        for hand in ("pinky", "index", "thumb"):
            w[sk.index(f"{hand}_{side}")] = w[sk.index(f"wrist_{side}")]
        w[sk.index(f"heel_{side}")] = w[sk.index(f"ankle_{side}")]
        w[sk.index(f"foot_index_{side}")] = w[sk.index(f"ankle_{side}")] + (0.0, 0.0, 0.1)


class FakeScene:
    """A synthetic webcam and pose detector: a standing figure that does ``exercise`` while
    ``schedule(t)`` is true and stands still otherwise (``t`` = seconds on ``clock`` since
    :meth:`frames` started).

    ``frames()`` yields ``(index, t_seconds, frame_bgr)`` paced to real time (``sleep`` is
    injectable for tests); ``detect(frame, t_ms)`` returns MediaPipe-shaped
    ``(image (33, 4), world (33, 3))`` landmarks: image x, y normalised to [0, 1] with
    visibility in column 3, world in metres.
    """

    def __init__(
        self,
        *,
        exercise: str = "curl",
        fps: float = 30.0,
        period_s: float = 2.5,
        schedule: Callable[[float], bool] = default_schedule,
        size: tuple[int, int] = (360, 640),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        duration_s: float | None = None,
    ):
        """``size`` is ``(height, width)`` in pixels, ``period_s`` one repetition in seconds,
        ``duration_s`` the length of the run in seconds (``None``: endless).

        Raises:
            ValueError: ``exercise`` is not one of curl, press, raise, squat.
        """
        if exercise not in _MOTION:
            raise ValueError(f"exercise must be one of {sorted(_MOTION)}, got {exercise!r}")
        self.exercise = exercise
        self.fps = fps
        self.period_s = period_s
        self.schedule = schedule
        self.size = size
        self.clock = clock
        self.sleep = sleep
        self.duration_s = duration_s
        self._t0: float | None = None
        self._active_since: float | None = None  # scene time the exercise last started (s)

    def _now(self) -> float:
        return 0.0 if self._t0 is None else self.clock() - self._t0

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Dark BGR frames at ``fps``, paced to real time; ``t`` is seconds since the first
        frame. Endless unless ``duration_s`` was given."""
        self._t0 = self.clock()
        height, width = self.size
        blank = np.zeros((height, width, 3), np.uint8)
        i = 0
        while self.duration_s is None or i / self.fps < self.duration_s:
            wait = i / self.fps - self._now()
            if wait > 0:
                self.sleep(wait)
            yield i, i / self.fps, blank
            i += 1

    def detect(self, frame, t_ms: int) -> tuple[np.ndarray, np.ndarray]:
        """Landmarks of the figure at frame time ``t_ms`` (ms): image ``(33, 4)`` (x, y in
        [0, 1], z, visibility) and world ``(33, 3)`` (m). Whether it exercises is decided by the
        schedule at the scene clock's time; the movement itself follows ``t_ms``."""
        t = t_ms / 1000.0
        if self.schedule(self._now()):
            if self._active_since is None:
                self._active_since = t
        else:
            self._active_since = None
        joint, rest, far = _MOTION[self.exercise]
        deg = rest
        if self._active_since is not None:
            deg = curl_deg(
                t, period_s=self.period_s, top=far, bottom=rest, phase_s=self._active_since
            )
        left_deg = right_deg = _STAND_ELBOW
        knee_deg = _STAND_KNEE
        if joint == "elbow":
            left_deg = right_deg = deg
        elif joint == "knee":
            knee_deg = deg
        pf = pose_frame(0, t, left_deg, right_deg, knee_deg=knee_deg)
        if joint == "shoulder_abd":
            _abduct(pf, deg, deg)
        _with_extremities(pf)
        return self._image_landmarks(pf), pf.world.astype(np.float32)

    def _image_landmarks(self, pf: PoseFrame) -> np.ndarray:
        """``(33, 4)`` image landmarks: the figure centred, about 70 % of the frame height."""
        height, width = self.size
        metres_to_unit_y = _BODY_HEIGHT_FRAC / 1.4
        img = np.zeros((33, 4), np.float32)
        img[:, 0] = 0.5 + pf.world[:, 0] * metres_to_unit_y * height / width
        img[:, 1] = 0.5 - (pf.world[:, 1] - _Y_CENTRE) * metres_to_unit_y
        img[:, 2] = pf.world[:, 2]
        img[:, 3] = pf.visibility
        return img
