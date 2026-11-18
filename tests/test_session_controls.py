"""``LiveSession`` controls (Bench plan Task 2): gate sources, exercise switching, counter
reset, board changes and dropped-rep events. Fake frames, synthetic detectors, fake boards."""

from __future__ import annotations

import math

import pytest

from formcoach.app.session import LiveSession
from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

from .session_fakes import (
    BLANK,
    FakeBoard,
    feed,
    must_not_be_called,
    scripted_detect,
    synthetic_detect,
)
from .synthetic_pose import curl_deg

FPS = 30.0
SHAKE_MS2 = 3.0  # amplitude of the shaking board: var(|a|) = 9 (m/s²)², above the 3.0 default


class ShakingBoard(FakeBoard):
    """Gate flag always False; still for the first ``still`` samples, then |a| alternates
    9.81 +/- SHAKE_MS2. ``silent_after`` frames (polls) later it sends nothing and is not alive."""

    def __init__(self, *, still: int, silent_after: int | None = None):
        super().__init__(gate=False, samples_per_poll=10)
        self.still = still
        self.silent_after = silent_after
        self.polls = 0

    def accel(self, seq):
        if seq < self.still:
            return 0.0, 0.0, 9.81
        return 0.0, 0.0, 9.81 + (SHAKE_MS2 if seq % 2 else -SHAKE_MS2)

    def _silent(self):
        return self.silent_after is not None and self.polls > self.silent_after

    def poll(self):
        self.polls += 1
        return [] if self._silent() else super().poll()

    def alive(self):
        return not self._silent()

    def state(self):
        return "silent" if self._silent() else "ok"


def kinds(session: LiveSession) -> list[str]:
    return [e.kind for e in session.log.events]


def test_manual_gate_opens_and_closes_pose():
    s = LiveSession(exercise="curl", detect=synthetic_detect, gate_source="manual")
    assert feed(s, seconds=0.2)[-1].pose_ran is False  # manual starts closed
    s.set_manual_gate(True)
    snap = feed(s, seconds=0.2, start=6)[-1]
    assert snap.pose_ran is True and snap.gate is True and snap.image_landmarks is not None
    s.set_manual_gate(False)
    snap = feed(s, seconds=0.2, start=12)[-1]
    assert snap.pose_ran is False and snap.image_landmarks is None
    assert kinds(s) == ["gate_open", "gate_close"]


def test_energy_gate_follows_shaking_not_the_device_flag():
    still = 200  # samples (20 frames at 10 samples per frame)
    s = LiveSession(
        exercise="curl",
        detect=synthetic_detect,
        board=ShakingBoard(still=still),
        gate_source="energy",
    )
    snaps = feed(s, seconds=3.0)
    still_snaps = snaps[: still // 10]
    assert all(sn.gate is False and sn.pose_ran is False for sn in still_snaps)
    assert all(sn.gates == {"device": False, "energy": False} for sn in still_snaps)
    opened = next(i for i, sn in enumerate(snaps) if sn.gate)
    # two active windows (decided every 25 samples, 2.5 frames) after the shaking started
    assert opened * 10 >= still + 2 * 25 - 10
    last = snaps[-1]
    assert last.gates == {"device": False, "energy": True}
    assert last.gate is True and last.pose_ran is True and last.gate_source == "energy"
    assert last.energy > 3.0


def test_energy_gate_closes_when_the_board_goes_silent():
    board = ShakingBoard(still=0, silent_after=45)
    s = LiveSession(exercise="curl", detect=synthetic_detect, board=board, gate_source="energy")
    snaps = feed(s, seconds=2.0)
    assert snaps[44].gates["energy"] is True and snaps[44].gate is True
    # no new samples: the energy gate's own state is stale, so the silent board closes it
    assert snaps[-1].gates == {"device": False, "energy": False}
    assert snaps[-1].gate is False and snaps[-1].link_state == "silent"
    assert kinds(s)[-1] == "gate_close"


def test_set_energy_gate_threshold_changes_the_decision():
    s = LiveSession(
        exercise="curl", detect=synthetic_detect, board=ShakingBoard(still=0), gate_source="energy"
    )
    s.set_energy_gate(threshold=20.0)  # var(|a|) = 9 is below it
    snaps = feed(s, seconds=2.0)
    assert not any(sn.gate for sn in snaps)
    assert s._energy.last_value == pytest.approx(SHAKE_MS2**2, rel=0.01)
    s.set_energy_gate(threshold=5.0, on_windows=1, off_windows=4)
    snaps = feed(s, seconds=1.0, start=60)
    assert snaps[-1].gate is True
    with pytest.raises(ValueError):
        s.set_energy_gate(on_windows=0)
    with pytest.raises(ValueError):
        s.set_energy_gate(threshold=-1.0)
    assert s._energy.threshold == 5.0  # a rejected call changes nothing


def test_set_exercise_resets_counters_and_keeps_one_log():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    log = s.log
    feed(s, seconds=9.0)
    assert {k: v["reps"] for k, v in s.pipeline.arm_totals().items()} == {"l": 3, "r": 3}
    s.set_exercise("squat")
    snap = feed(s, seconds=1.0, start=270)[-1]
    assert s.log is log and s.pipeline.log is log
    assert set(snap.totals) == {"all"} and snap.totals["all"]["reps"] == 0
    assert set(snap.counters) == {"all"}
    assert snap.exercise == "squat" == s.exercise
    assert kinds(s).count("rep") == 6
    exercise_events = [e for e in log.events if e.kind == "exercise"]
    assert len(exercise_events) == 1 and exercise_events[0].payload == {"exercise": "squat"}
    assert exercise_events[0].t == pytest.approx(269 / FPS)
    # the gate never closed, so the new pipeline must not open it a second time
    assert kinds(s).count("gate_open") == 1 and "gate_close" not in kinds(s)
    assert log.frames_total == 300
    result = s.finish()
    assert result.summary["exercise"] == "squat" and result.summary["frames_total"] == 300
    with pytest.raises(ValueError):
        s.set_exercise("deadlift")


def test_reset_counters_keeps_the_exercise_and_the_log():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    feed(s, seconds=9.0)
    s.reset_counters()
    snap = feed(s, seconds=0.1, start=270)[-1]
    assert snap.totals == {
        k: {"reps": 0, "good": 0, "flagged": 0, "faults_by_code": {}} for k in ("l", "r")
    }
    assert snap.exercise == "curl"
    assert kinds(s).count("rep") == 6 and kinds(s).count("counters_reset") == 1


def test_idle_has_no_counters_but_still_reports_landmarks():
    s = LiveSession(exercise="idle", detect=synthetic_detect)
    assert s.pipeline is None
    snap = feed(s, seconds=3.0)[-1]
    assert snap.counters == {} and snap.totals == {} and snap.completed == []
    assert snap.pose_ran is True and snap.image_landmarks.shape == (33, 4)
    assert snap.exercise == "idle"
    result = s.finish()
    assert result.reps == [] and result.frames_total == 90 and result.frames_processed == 90
    assert result.summary["exercise"] == "idle" and result.summary["reps"] == 0
    assert result.summary["arms"] == {} and result.summary["live"] is True
    assert result.summary["gated_seconds"] == pytest.approx(89 / FPS, abs=0.01)


def test_idle_logs_gate_changes_and_skips_pose_while_closed():
    s = LiveSession(exercise="curl", detect=must_not_be_called, gate_source="manual")
    s.set_exercise("idle")
    feed(s, seconds=0.2)
    s.detect = synthetic_detect
    s.set_manual_gate(True)
    feed(s, seconds=0.2, start=6)
    s.set_manual_gate(False)
    feed(s, seconds=0.2, start=12)
    assert kinds(s) == ["exercise", "gate_open", "gate_close"]
    assert (s.log.frames_total, s.log.frames_processed) == (18, 6)
    s.set_exercise("curl")  # back to counting: a fresh pipeline on the same log
    assert s.pipeline is not None and s.pipeline.log is s.log


def test_unknown_gate_source_raises_valueerror():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    with pytest.raises(ValueError):
        s.set_gate_source("laptop")
    assert s.gate_source == "always_on"
    with pytest.raises(ValueError):
        LiveSession(exercise="curl", detect=synthetic_detect, gate_source="open")


def test_set_gate_source_emits_and_takes_effect_on_the_next_frame():
    s = LiveSession(exercise="curl", detect=synthetic_detect, board=FakeBoard(gate=False))
    assert feed(s, seconds=0.1)[-1].gate is False  # device flag False
    s.set_gate_source("always_on")
    snap = feed(s, seconds=0.1, start=3)[-1]
    assert snap.gate is True and snap.gate_source == "always_on"
    assert [e.payload for e in s.log.events if e.kind == "gate_source"] == [{"source": "always_on"}]
    assert s.finish().summary["gate"] == "always_on"


def test_set_board_none_closes_device_and_energy_sources_and_emits_link():
    s = LiveSession(exercise="curl", detect=synthetic_detect, board=FakeBoard(gate=True))
    assert feed(s, seconds=0.1)[-1].gate is True
    s.set_board(None)
    snap = feed(s, seconds=0.1, start=3)[-1]
    assert snap.gate is False and snap.gate_source == "device"  # the source is kept
    assert snap.gates == {"device": None, "energy": None} and snap.link_state == "none"
    board = FakeBoard(gate=True)
    s.set_board(board)
    assert feed(s, seconds=0.1, start=6)[-1].gate is True
    links = [e.payload for e in s.log.events if e.kind == "link"]
    assert links == [{"state": "none", "name": None}, {"state": "ok", "name": "FormCoach-TEST"}]
    assert s.finish().summary["board"] == "FormCoach-TEST"


def _short_rep_elbow(t: float) -> float:
    """Curls (2.5 s) for 7.5 s, rest at 150 deg, then a 0.3 s dip to 50 deg at t = 8.5 s."""
    if t < 7.5:
        return curl_deg(t)
    if 8.5 <= t <= 8.8:
        return 150.0 - 100.0 * math.sin(math.pi * (t - 8.5) / 0.3)
    return 150.0


def _gap_hidden(t: float) -> bool:
    """Six frames (at 30 fps) hidden around the top of the third curl (t = 6.25 s)."""
    return abs(t - 6.25) < 0.1


def _dropped(session: LiveSession) -> list[dict]:
    return [e.payload for e in session.log.events if e.kind == "rep_dropped"]


def test_rep_dropped_events_for_a_short_rep_and_for_a_tracking_gap():
    s = LiveSession(exercise="curl", detect=scripted_detect(_short_rep_elbow))
    feed(s, seconds=10.0)
    short = _dropped(s)
    assert len(short) == 2 and {d["arm"] for d in short} == {"l", "r"}
    for d in short:
        assert d["reason"] == "too_short"
        assert 8.4 <= d["t_start"] < d["t_end"] <= 9.0 and d["t_end"] - d["t_start"] < 0.4
    assert set(d) == {"arm", "t_start", "t_end", "reason"}

    s = LiveSession(exercise="curl", detect=scripted_detect(curl_deg, hidden=_gap_hidden))
    feed(s, seconds=10.0)
    gaps = _dropped(s)
    assert {d["arm"] for d in gaps} == {"l", "r"}
    for d in gaps:
        assert d["reason"] == "tracking_gap" and d["t_start"] < 6.25 < d["t_end"]
    assert len(gaps) == 2
    # the dropped rep is not counted: curls at 2.5 s finish 4 reps by t = 10 s, one was dropped
    assert {k: v["reps"] for k, v in s.pipeline.arm_totals().items()} == {"l": 3, "r": 3}


def test_report_dropped_false_emits_nothing_new():
    for detect in (
        scripted_detect(_short_rep_elbow),
        scripted_detect(curl_deg, hidden=_gap_hidden),
    ):
        s = LiveSession(exercise="curl", detect=detect, report_dropped=False)
        feed(s, seconds=10.0)
        assert set(kinds(s)) <= {"gate_open", "gate_close", "rep", "fault"}


def test_rep_context_is_recorded():
    s = LiveSession(
        exercise="curl", detect=synthetic_detect, rules=RuleEngine.from_yaml(WEBCAM_RULES)
    )
    reps = [r for snap in feed(s, seconds=9.0) for r in snap.completed]
    assert len(reps) == 6
    for rep in reps:
        assert isinstance(rep.context, frozenset)
        assert {"two_arm", "frontal_view"} <= rep.context
    # without rules the tags are still recorded (re-judging later needs them)
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    reps = [r for snap in feed(s, seconds=9.0) for r in snap.completed]
    assert reps and all("two_arm" in r.context for r in reps)


def test_setters_run_from_another_thread_through_call():
    s = LiveSession(exercise="curl", detect=synthetic_detect)
    s.call(lambda sess: sess.set_exercise("squat"))
    snap = s.step(0, 0.0, BLANK)
    assert snap.exercise == "squat" and set(snap.counters) == {"all"}
