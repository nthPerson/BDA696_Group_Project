"""``LiveSession`` (Bench plan Task 1): fake frames, a synthetic detector and a fake board; no
camera, no model, no cv2, no Bluetooth."""

from __future__ import annotations

import numpy as np

from formcoach.app.session import LiveSession
from formcoach.io import protocol
from formcoach.io.source import SISample
from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

from .synthetic_pose import curl_deg, pose_frame

BLANK = np.zeros((48, 64, 3), np.uint8)


def synthetic_detect(_frame, t_ms):
    """Both arms curl (2.5 s period) at frame time ``t_ms``; MediaPipe-shaped output."""
    t = t_ms / 1000.0
    pf = pose_frame(0, t, curl_deg(t), curl_deg(t))
    img = np.zeros((33, 4), np.float32)
    img[:, 0] = 0.5 + 0.5 * pf.world[:, 0]
    img[:, 1] = 0.5 - 0.5 * pf.world[:, 1]
    img[:, 3] = pf.visibility
    return img, pf.world.astype(np.float32)


def must_not_be_called(_frame, _t_ms):
    raise AssertionError("pose must not run while the gate is closed")


def feed(session: LiveSession, *, seconds: float, fps: float = 30.0) -> list:
    """Step ``session`` through ``seconds`` of blank frames at ``fps``; returns the snapshots."""
    return [session.step(i, i / fps, BLANK) for i in range(int(seconds * fps))]


class FakeBoard:
    """BoardLink-shaped: always alive, a fixed gate flag, ``samples_per_poll`` samples per poll."""

    name = "FormCoach-TEST"

    def __init__(self, *, gate: bool = True, samples_per_poll: int = 0):
        self.gate = gate
        self.samples_per_poll = samples_per_poll
        self.seq = 0
        self.sent: list[protocol.Command] = []

    def poll(self):
        out = []
        flags = protocol.Flag.GATE_STATE if self.gate else 0
        for _ in range(self.samples_per_poll):
            out.append(SISample(self.seq / 50.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, flags, self.seq))
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


def test_step_returns_landmarks_angles_and_counters_when_gate_open():
    s = LiveSession(exercise="curl", detect=synthetic_detect)  # no board -> always_on
    snap = feed(s, seconds=1.0)[-1]
    assert snap.pose_ran
    assert snap.image_landmarks.shape == (33, 4)
    assert snap.world_landmarks.shape == (33, 3)
    assert set(snap.counters) == {"l", "r"}
    assert {"angle", "enter", "exit", "armed", "visibility"} <= set(snap.counters["l"])
    assert snap.gate is True
    assert snap.gate_source == "always_on"
    assert snap.link_state == "none"


def test_ten_curls_are_counted_per_arm_and_published_as_events():
    events = []
    s = LiveSession(
        exercise="curl", detect=synthetic_detect, rules=RuleEngine.from_yaml(WEBCAM_RULES)
    )
    s.subscribe(on_event=events.append)
    feed(s, seconds=27.0)  # 2.5 s period
    totals = s.pipeline.arm_totals()
    assert totals["l"]["reps"] == 10 and totals["r"]["reps"] == 10
    assert sum(e.kind == "rep" for e in events) == 20


def test_imu_samples_reach_listeners_and_are_not_dropped():
    got = []
    s = LiveSession(exercise="curl", detect=synthetic_detect, board=FakeBoard(samples_per_poll=2))
    s.subscribe(on_imu=got.extend)
    feed(s, seconds=1.0, fps=30)
    assert len(got) == 60


def test_device_gate_closed_means_no_pose_and_no_landmarks():
    s = LiveSession(exercise="curl", detect=must_not_be_called, board=FakeBoard(gate=False))
    snap = feed(s, seconds=0.2)[-1]
    assert snap.gate is False
    assert snap.pose_ran is False
    assert snap.image_landmarks is None


def test_no_person_gives_a_snapshot_without_landmarks():
    s = LiveSession(exercise="curl", detect=lambda f, t: (None, None))
    snap = feed(s, seconds=0.2)[-1]
    assert snap.pose_ran is True
    assert snap.image_landmarks is None
    assert snap.counters["l"]["angle"] is None


def test_a_raising_listener_does_not_stop_the_loop():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    errors = []
    s.subscribe(on_snapshot=lambda snap: 1 / 0)
    s.subscribe(on_event=lambda e: errors.append(e) if e.kind == "error" else None)
    feed(s, seconds=0.5)
    assert len(errors) == 1


def test_call_runs_on_the_next_step_and_stop_ends_run():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    seen = []
    s.call(lambda sess: seen.append(sess.exercise))
    s.step(0, 0.0, None)
    assert seen == ["curl"]

    s.subscribe(on_snapshot=lambda snap: s.stop() if snap.index == 4 else None)
    s.run((i, i / 30.0, BLANK) for i in range(1, 30))
    assert s.log.frames_total == 5  # frame 0 above, then frames 1-4 until stop()
