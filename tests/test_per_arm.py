"""Per-arm rep tracking (live-demo plan Task 4): one segmenter per arm for curl/press/raise,
per-arm metrics, visibility floor, two-arm rule pairing, ``arm_totals()``, buffer trimming.
"""

from __future__ import annotations

from formcoach.app.pipeline import Pipeline
from formcoach.rules.engine import RuleEngine

from .synthetic_pose import curl_deg, pose_frame

FPS = 30.0


def _run(left_fn, right_fn, seconds: float = 25.0, **frame_kw):
    """Feed ``pose_frame`` at 30 fps into a per-arm curl ``Pipeline`` with the gate open;
    returns ``(pipe, events)``."""
    events = []
    pipe = Pipeline(
        exercise="curl", per_arm=True, rules=RuleEngine.from_yaml(), on_event=events.append
    )
    pipe.set_gate(True, 0.0)
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        fr = pose_frame(i, t, left_fn(t), right_fn(t), **frame_kw)
        pipe.on_frame(fr)
    return pipe, events


def _faults(events, code=None):
    out = [e for e in events if e.kind == "fault"]
    if code is not None:
        out = [e for e in out if e.payload.get("code") == code]
    return out


def test_only_left_arm_swings_counts_left_only():
    pipe, events = _run(curl_deg, lambda t: 150.0)
    totals = pipe.arm_totals()
    assert 8 <= totals["l"]["reps"] <= 10
    assert totals["r"]["reps"] == 0
    assert _faults(events, "CURL_ASYMMETRY") == []


def test_both_arms_together_count_both_and_pair_two_arm_rules():
    pipe, events = _run(curl_deg, lambda t: curl_deg(t, top=90.0))
    totals = pipe.arm_totals()
    assert 8 <= totals["l"]["reps"] <= 10
    assert 8 <= totals["r"]["reps"] <= 10
    arms_with_asymmetry = {e.payload["arm"] for e in _faults(events, "CURL_ASYMMETRY")}
    assert arms_with_asymmetry == {"l", "r"}


def test_alternating_arms_count_each_without_two_arm_rules():
    pipe, events = _run(curl_deg, lambda t: curl_deg(t, top=90.0, phase_s=1.25))
    totals = pipe.arm_totals()
    assert 8 <= totals["l"]["reps"] <= 10
    assert 8 <= totals["r"]["reps"] <= 10
    assert _faults(events, "CURL_ASYMMETRY") == []


def test_arm_below_visibility_floor_counts_nothing():
    pipe, _events = _run(curl_deg, curl_deg, vis_r=0.3)
    totals = pipe.arm_totals()
    assert totals["r"]["reps"] == 0
    assert 8 <= totals["l"]["reps"] <= 10


def test_left_rep_metrics_ignore_the_right_arm():
    _pipe, events = _run(curl_deg, lambda t: 150.0)
    left_reps = [e for e in events if e.kind == "rep" and e.payload["arm"] == "l"]
    assert left_reps
    for e in left_reps:
        assert e.payload["metrics"]["elbow_min"] < 60


def test_events_carry_arm_and_per_arm_rep_id():
    _pipe, events = _run(curl_deg, lambda t: curl_deg(t, top=90.0))
    for e in events:
        if e.kind in ("rep", "fault"):
            assert e.payload["arm"] in ("l", "r")
    left_ids = [e.payload["rep_id"] for e in events if e.kind == "rep" and e.payload["arm"] == "l"]
    right_ids = [e.payload["rep_id"] for e in events if e.kind == "rep" and e.payload["arm"] == "r"]
    assert left_ids == list(range(len(left_ids)))
    assert right_ids == list(range(len(right_ids)))


def test_replay_style_pipeline_has_no_arm_key():
    events = []
    pipe = Pipeline(exercise="curl", rules=RuleEngine.from_yaml(), on_event=events.append)
    pipe.set_gate(True, 0.0)
    for i in range(int(25.0 * FPS)):
        t = i / FPS
        pipe.on_frame(pose_frame(i, t, curl_deg(t), curl_deg(t)))
    assert events
    for e in events:
        assert "arm" not in e.payload


def test_squat_uses_one_track_with_null_arm():
    events = []
    pipe = Pipeline(
        exercise="squat", per_arm=True, rules=RuleEngine.from_yaml(), on_event=events.append
    )
    pipe.set_gate(True, 0.0)
    for i in range(int(25.0 * FPS)):
        t = i / FPS
        knee = curl_deg(t, top=100.0, bottom=175.0)
        pipe.on_frame(pose_frame(i, t, 150.0, 150.0, knee_deg=knee))
    assert set(pipe.arm_totals()) == {"all"}
    rep_events = [e for e in events if e.kind == "rep"]
    assert rep_events
    assert all(e.payload["arm"] is None for e in rep_events)


def test_arm_totals_split_good_and_flagged():
    # left under-ranges (CURL_PARTIAL_ROM). Right is a full rep: bottom=170 (not the default
    # 150) so the *measured* elbow_max -- the adaptive segmenter's rep window edge sits at
    # exit = hi - frac*range, well inset from the true extreme -- still clears the
    # CURL_PARTIAL_ROM threshold (140). Right is also phase-shifted so the two never pair
    # (requires: two_arm), isolating the ROM fault to the left arm only.
    left_fn = lambda t: curl_deg(t, top=90.0)  # noqa: E731
    right_fn = lambda t: curl_deg(t, bottom=170.0, phase_s=1.1)  # noqa: E731
    pipe, _events = _run(left_fn, right_fn)
    totals = pipe.arm_totals()
    assert totals["l"]["flagged"] == totals["l"]["reps"]
    assert totals["l"]["faults_by_code"]["CURL_PARTIAL_ROM"] == totals["l"]["reps"]
    assert totals["r"]["good"] == totals["r"]["reps"]
    result = pipe.finish(compute_imu=False)
    assert result.summary["arms"] == totals


def test_buffers_stay_bounded():
    pipe = Pipeline(
        exercise="curl", per_arm=True, max_buffer_frames=300, rules=RuleEngine.from_yaml()
    )
    pipe.set_gate(True, 0.0)
    for i in range(int(120.0 * FPS)):
        t = i / FPS
        pipe.on_frame(pose_frame(i, t, curl_deg(t), curl_deg(t)))
    assert pipe.buffered_frames <= 600
    totals = pipe.arm_totals()
    assert totals["l"]["reps"] >= 40
    assert totals["r"]["reps"] >= 40


def test_hidden_arm_gap_does_not_crash_and_counting_resumes():
    events = []
    pipe = Pipeline(
        exercise="curl", per_arm=True, rules=RuleEngine.from_yaml(), on_event=events.append
    )
    pipe.set_gate(True, 0.0)
    for i in range(int(25.0 * FPS)):
        t = i / FPS
        vis_r = 0.0 if 8.0 <= t < 14.0 else 1.0
        pipe.on_frame(pose_frame(i, t, curl_deg(t), curl_deg(t), vis_r=vis_r))
    totals = pipe.arm_totals()
    assert totals["r"]["reps"] >= 4


def test_a_hidden_arm_stuck_mid_rep_does_not_pair_with_the_other_arm():
    """The right arm drops out mid-rep (its segmenter stays 'active' on NaN frames); the left
    arm's later reps must not be treated as two-arm reps, so CURL_ASYMMETRY never fires."""
    from formcoach.rules.engine import RuleEngine

    pipe = Pipeline(exercise="curl", per_arm=True, rules=RuleEngine.from_yaml())
    pipe.set_gate(True, 0.0)
    fps = 30.0
    for i in range(int(30 * fps)):
        t = i / fps
        if t < 6.9:
            pipe.on_frame(pose_frame(i, t, curl_deg(t), curl_deg(t)))
        else:  # right arm hidden with a wild reading, left keeps curling
            pipe.on_frame(pose_frame(i, t, curl_deg(t), 170.0, vis_r=0.2))
    result = pipe.finish(compute_imu=False)
    left = [r for r in result.reps if r.arm == "l" and r.t_end > 8.0]
    assert len(left) >= 6
    assert all("CURL_ASYMMETRY" not in {f["code"] for f in r.faults} for r in left)
