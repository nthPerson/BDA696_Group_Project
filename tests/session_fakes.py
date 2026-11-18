"""Fakes for the ``LiveSession`` tests: blank frames, synthetic detectors and a fake board; no
camera, no model, no cv2, no Bluetooth."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from formcoach.app.session import LiveSession
from formcoach.io import protocol
from formcoach.io.source import SISample

from .synthetic_pose import curl_deg, pose_frame

BLANK = np.zeros((48, 64, 3), np.uint8)


def detection(t: float, left_deg: float, right_deg: float, *, knee_deg: float = 175.0):
    """MediaPipe-shaped ``(image (33, 4), world (33, 3))`` of the synthetic figure at ``t`` (s)
    with the given elbow angles and knee angle (degrees)."""
    pf = pose_frame(0, t, left_deg, right_deg, knee_deg=knee_deg)
    img = np.zeros((33, 4), np.float32)
    img[:, 0] = 0.5 + 0.5 * pf.world[:, 0]
    img[:, 1] = 0.5 - 0.5 * pf.world[:, 1]
    img[:, 3] = pf.visibility
    return img, pf.world.astype(np.float32)


def synthetic_detect(_frame, t_ms):
    """Both arms curl (2.5 s period) at frame time ``t_ms``; MediaPipe-shaped output."""
    t = t_ms / 1000.0
    return detection(t, curl_deg(t), curl_deg(t))


def scripted_detect(
    elbow_deg: Callable[[float], float], hidden: Callable[[float], bool] = lambda t: False
):
    """A detector whose two elbows follow ``elbow_deg(t)`` (degrees, ``t`` seconds); nobody is
    found on frames where ``hidden(t)`` is true."""

    def detect(_frame, t_ms):
        t = t_ms / 1000.0
        if hidden(t):
            return None, None
        deg = elbow_deg(t)
        return detection(t, deg, deg)

    return detect


def must_not_be_called(_frame, _t_ms):
    raise AssertionError("pose must not run while the gate is closed")


def feed(session: LiveSession, *, seconds: float, fps: float = 30.0, start: int = 0) -> list:
    """Step ``session`` through ``seconds`` of blank frames at ``fps``, frame indices from
    ``start`` (``t = index / fps``); returns the snapshots."""
    n = round(seconds * fps)
    return [session.step(i, i / fps, BLANK) for i in range(start, start + n)]


class FakeBoard:
    """BoardLink-shaped: always alive, a fixed gate flag, ``samples_per_poll`` samples per poll."""

    name = "FormCoach-TEST"

    def __init__(self, *, gate: bool = True, samples_per_poll: int = 0):
        self.gate = gate
        self.samples_per_poll = samples_per_poll
        self.seq = 0
        self.sent: list[protocol.Command] = []

    def accel(self, seq: int) -> tuple[float, float, float]:
        """Acceleration (m/s²) of sample ``seq``: at rest, gravity on z."""
        return 0.0, 0.0, 9.81

    def poll(self):
        out = []
        flags = protocol.Flag.GATE_STATE if self.gate else 0
        for _ in range(self.samples_per_poll):
            ax, ay, az = self.accel(self.seq)
            out.append(SISample(self.seq / 50.0, ax, ay, az, 0.0, 0.0, 0.0, flags, self.seq))
            self.seq += 1
        return out

    def alive(self):
        return True

    def gate_open(self):
        return self.gate

    def state(self):
        return "ok"

    def send(self, cmd):
        self.sent.append(cmd)
