"""``LiveSession`` (Bench plan Task 1): fake frames, a synthetic detector and a fake board; no
camera, no model, no cv2, no Bluetooth."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.app.session import LiveSession
from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

from .session_fakes import BLANK, FakeBoard, feed, must_not_be_called, synthetic_detect


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


def test_counter_angle_is_what_the_segmenter_saw_and_angle_raw_is_unsmoothed():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    snaps = feed(s, seconds=2.0)
    raw = [snap.counters["l"]["angle_raw"] for snap in snaps]
    for i in range(4, len(snaps)):
        # the segmenter compares the 5-frame median of the raw primary angle with enter/exit
        assert snaps[i].counters["l"]["angle"] == pytest.approx(
            float(np.median(raw[i - 4 : i + 1]))
        )
    last = snaps[-1].counters["l"]
    assert last["angle"] == pytest.approx(s.pipeline.tracks[0].last_p)
    assert last["angle_raw"] == pytest.approx(s.pipeline.latest_angles()["l"])
    assert any(abs(sn.counters["l"]["angle"] - sn.counters["l"]["angle_raw"]) > 1.0 for sn in snaps)


def test_unsubscribe_removes_every_listener_it_added():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    snaps, events = [], []
    unsubscribe = s.subscribe(on_snapshot=snaps.append, on_event=events.append)
    s.step(0, 0.0, BLANK)
    s.emit("note")
    assert len(snaps) == 1 and [e.kind for e in events] == ["gate_open", "note"]
    unsubscribe()
    s.step(1, 1 / 30, BLANK)
    s.emit("note")
    assert len(snaps) == 1 and len(events) == 2
    unsubscribe()  # a second call is harmless


def test_emit_logs_the_event_and_sends_it_to_listeners():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    got = []
    s.subscribe(on_event=got.append)
    before = s.emit("note", text="hello")  # before any frame: t = 0.0
    assert (before.kind, before.t, before.payload) == ("note", 0.0, {"text": "hello"})
    feed(s, seconds=0.5)
    latest = s.emit("note")  # defaults to the latest frame's time
    assert latest.t == pytest.approx(14 / 30)
    explicit = s.emit("note", t=3.0)
    assert explicit.t == 3.0
    notes = [e for e in s.log.events if e.kind == "note"]
    assert notes == [before, latest, explicit]
    assert [e for e in got if e.kind == "note"] == notes


def test_run_live_prints_a_broken_on_event_listener_through_say():
    from formcoach.app.live import run_live

    said = []

    def broken(e):
        raise RuntimeError("printer broke")

    frames = ((i, i / 30.0, BLANK) for i in range(10))
    run_live(frames, synthetic_detect, exercise="curl", on_event=broken, say=said.append)
    errors = [line for line in said if line.startswith("error in")]
    assert errors == ["error in on_event: RuntimeError: printer broke"]
