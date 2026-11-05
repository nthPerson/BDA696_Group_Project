"""Per-arm rep tracking (live-demo plan Task 4): one segmenter per arm for curl/press/raise,
per-arm metrics, visibility floor, two-arm rule pairing, ``arm_totals()``, buffer trimming.
"""

from __future__ import annotations

import pandas as pd
import pytest

from formcoach.app import pipeline as pipeline_mod
from formcoach.app.live import REP_PAD_S, pose_frame_from_detection
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


# --- review fixes: live-mode behaviour that the tests above did not reach ---------------------


def _rest_curl(t: float, offset: float = 0.0, top: float = 50.0) -> float:
    """A curl with a 1 s rest between 4 s reps (5 s cadence), starting ``offset`` s late. The rest
    matters: with back-to-back reps the staggered arms look like one continuous overlap."""
    if t < offset:
        return 150.0
    u = (t - offset) % 5.0
    return curl_deg(u, period_s=4.0, top=top) if u < 4.0 else 150.0


def _drive(
    left_fn, right_fn, seconds, *, per_arm=True, gate_windows=(), vis_r_fn=None, lost_fn=None
):
    """Like ``_run`` but with the live metrics-window padding, optional gate-closed windows
    ``[(t0, t1), ...]``, a right-arm visibility function and ``lost_fn(t)`` (no person found).
    ``per_arm=False`` is the replay-style pipeline. Returns ``(pipe, events)``."""
    events = []
    pipe = Pipeline(
        exercise="curl",
        per_arm=per_arm,
        rules=RuleEngine.from_yaml(),
        on_event=events.append,
        pad_rep_start_s=REP_PAD_S if per_arm else 0.0,
    )
    pipe.set_gate(True, 0.0)
    state = True
    for i in range(int(seconds * FPS)):
        t = i / FPS
        want = not any(a <= t < b for a, b in gate_windows)
        if want != state:
            pipe.set_gate(want, t)
            state = want
        if lost_fn is not None and lost_fn(t):
            pipe.on_frame(pose_frame_from_detection(i, t, None, None))
            continue
        vis_r = vis_r_fn(t) if vis_r_fn else 1.0
        pipe.on_frame(pose_frame(i, t, left_fn(t), right_fn(t), vis_r=vis_r))
    return pipe, events


def _reps(events, arm):
    return [e.payload for e in events if e.kind == "rep" and e.payload.get("arm") == arm]


def _asymmetry_by_arm(events):
    return {
        arm: sum(1 for e in _faults(events, "CURL_ASYMMETRY") if e.payload["arm"] == arm)
        for arm in ("l", "r")
    }


@pytest.mark.parametrize("lag_s", [2.0, 3.0])
def test_staggered_arms_with_the_same_range_never_flag_asymmetry(lag_s):
    """Both arms curl the full range, one ``lag_s`` behind the other: the reps barely overlap
    (they are not a two-arm set), so neither arm may be told to curl the same distance."""
    pipe, events = _drive(_rest_curl, lambda t: _rest_curl(t, lag_s), 22.0)
    assert min(pipe.arm_totals()[a]["reps"] for a in "lr") >= 3
    assert _asymmetry_by_arm(events) == {"l": 0, "r": 0}


def test_arms_half_a_second_apart_with_different_ranges_still_pair():
    """The pairing must not become too strict: arms 0.5 s apart are one two-arm set, so a
    shallow right curl is flagged on both arms (the leading arm completes while the other is
    still mid-rep)."""
    _pipe, events = _drive(_rest_curl, lambda t: _rest_curl(t, 0.5, top=90.0), 22.0)
    by_arm = _asymmetry_by_arm(events)
    assert by_arm["l"] >= 2
    assert by_arm["r"] >= 2


def test_rep_numbers_keep_counting_across_a_gate_close_and_reopen():
    """Live (per-arm): the wearable's gate closes after a set and opens for the next one. The
    rep numbers must carry on (the counter on screen does), with no duplicate (arm, rep_id)."""
    pipe, events = _drive(
        lambda t: curl_deg(t, period_s=3.0),
        lambda t: curl_deg(t, period_s=3.0),
        40.0,
        gate_windows=[(12.0, 16.0)],
    )
    for arm in ("l", "r"):
        ids = [p["rep_id"] for p in _reps(events, arm)]
        assert len(ids) == pipe.arm_totals()[arm]["reps"] >= 8
        assert ids == list(range(len(ids)))
    keys = [(e.payload["arm"], e.payload["rep_id"]) for e in events if e.kind == "rep"]
    assert len(keys) == len(set(keys))
    fault_keys = [(e.payload["arm"], e.payload["rep_id"]) for e in _faults(events)]
    rep_keys = set(keys)
    assert all(k in rep_keys for k in fault_keys)


def test_replay_style_rep_numbers_restart_after_a_gate_close():
    """Replay (not per-arm) keeps its original numbering: every gate-open segment starts at 0.
    The golden replay output pins this only for an always-open gate."""
    _pipe, events = _drive(curl_deg, curl_deg, 40.0, per_arm=False, gate_windows=[(12.0, 16.0)])
    closes = [e.t for e in events if e.kind == "gate_close"]
    assert len(closes) == 1
    before = [e.payload["rep_id"] for e in events if e.kind == "rep" and e.t < closes[0]]
    after = [e.payload["rep_id"] for e in events if e.kind == "rep" and e.t > closes[0]]
    assert before == list(range(len(before))) and len(before) >= 3
    assert after == list(range(len(after))) and len(after) >= 3


def test_a_long_tracking_gap_inside_a_rep_drops_that_rep():
    """The right arm is hidden for 2.25 s in the middle of a rep (docs/02 4.3: gaps over 3
    frames are invalid): that rep must not be counted or judged on interpolated angles, the
    left arm is unaffected and the right arm counts again afterwards."""
    hide = (6.25, 8.5)
    base, base_events = _drive(_rest_curl, _rest_curl, 22.0)
    pipe, events = _drive(
        _rest_curl, _rest_curl, 22.0, vis_r_fn=lambda t: 0.0 if hide[0] <= t < hide[1] else 1.0
    )

    def touches_gap(p):
        return p["t_end"] > hide[0] and p["t_start"] < hide[1]

    assert any(touches_gap(p) for p in _reps(base_events, "r")), "probe must hit a rep"
    assert not any(touches_gap(p) for p in _reps(events, "r"))
    assert any(p["t_start"] > hide[1] for p in _reps(events, "r"))
    assert pipe.arm_totals()["l"]["reps"] == base.arm_totals()["l"]["reps"]
    assert pipe.arm_totals()["r"]["reps"] == base.arm_totals()["r"]["reps"] - 1
    ids = [p["rep_id"] for p in _reps(events, "r")]
    assert ids == list(range(len(ids)))  # a dropped rep does not leave a hole in the numbering


def test_a_three_frame_tracking_gap_inside_a_rep_is_bridged_and_counted():
    base, _ = _drive(_rest_curl, _rest_curl, 22.0)
    pipe, _events = _drive(
        _rest_curl,
        _rest_curl,
        22.0,
        vis_r_fn=lambda t: 0.0 if 6.999 <= t < 7.099 else 1.0,  # exactly 3 frames
    )
    assert pipe.arm_totals()["r"]["reps"] == base.arm_totals()["r"]["reps"]


def test_replay_style_pipeline_bridges_three_frames_and_invalidates_the_rest():
    """Replay keeps docs/02 4.3 bookkeeping: of a 10-frame gap the first 3 frames reuse the
    last good angles and the other 7 are invalid (NaN)."""
    pipe, _events = _drive(
        curl_deg, curl_deg, 20.0, per_arm=False, lost_fn=lambda t: 6.999 <= t < 7.332
    )
    summary = pipe.finish(compute_imu=False, end_t=20.0).summary
    assert (summary["frames_bridged"], summary["frames_invalid"]) == (3, 7)


def test_replay_style_pipeline_still_counts_a_rep_across_a_long_gap():
    """The per-arm 'drop a rep with a long gap' rule is live-only: replay output must not
    change, so a rep with a long gap inside is still counted there (interpolated, as on main)."""
    hide = (6.25, 8.5)
    base, _ = _drive(_rest_curl, _rest_curl, 22.0, per_arm=False)
    pipe, _events = _drive(
        _rest_curl, _rest_curl, 22.0, per_arm=False, lost_fn=lambda t: hide[0] <= t < hide[1]
    )
    assert pipe.arm_totals()["all"]["reps"] == base.arm_totals()["all"]["reps"]
    assert pipeline_mod.MAX_GAP_FRAMES == 3


def _press_run(left_fn, right_fn, seconds: float = 30.0):
    """A live-style (per-arm, padded) press ``Pipeline`` at 30 fps; returns ``(pipe, events)``."""
    events = []
    pipe = Pipeline(
        exercise="press",
        per_arm=True,
        rules=RuleEngine.from_yaml(),
        on_event=events.append,
        pad_rep_start_s=REP_PAD_S,
    )
    pipe.set_gate(True, 0.0)
    for i in range(int(seconds * FPS)):
        t = i / FPS
        pipe.on_frame(pose_frame(i, t, left_fn(t), right_fn(t)))
    return pipe, events


def test_a_one_arm_press_is_not_flagged_for_asymmetry_by_the_idle_arm():
    """``elbow_top_asymmetry`` always compares both arms, so with the left arm pressing 80-170 deg
    and the right arm held bent at 100 deg it read ~70 deg on every rep. PRESS_ASYMMETRY is a
    two-arm rule (``requires: two_arm``), like CURL_ASYMMETRY, so a one-arm set has none."""
    pipe, events = _press_run(lambda t: curl_deg(t, top=80.0, bottom=170.0), lambda t: 100.0)
    totals = pipe.arm_totals()
    assert totals["l"]["reps"] >= 10
    assert totals["l"]["good"] == totals["l"]["reps"]
    assert totals["l"]["faults_by_code"] == {}
    assert totals["r"]["reps"] == 0
    assert _faults(events, "PRESS_ASYMMETRY") == []


def test_two_arms_pressing_together_still_flag_press_asymmetry_on_both_arms():
    """The rule still fires for a real two-arm set: the right arm tops out at 130 deg while the
    left locks out at 170 deg."""
    pipe, events = _press_run(
        lambda t: curl_deg(t, top=80.0, bottom=170.0),
        lambda t: curl_deg(t, top=80.0, bottom=130.0),
    )
    totals = pipe.arm_totals()
    assert totals["l"]["reps"] >= 10 and totals["r"]["reps"] >= 10
    for arm in ("l", "r"):
        flagged = [e for e in _faults(events, "PRESS_ASYMMETRY") if e.payload["arm"] == arm]
        assert len(flagged) == totals[arm]["reps"], (arm, len(flagged), totals[arm])


def test_finish_can_be_called_twice_with_the_same_result():
    """``finish`` is a snapshot: the open gate segment is closed in the result only, so a second
    call does not count it twice."""
    pipe, _events = _drive(curl_deg, curl_deg, 2.0)
    first = pipe.finish(compute_imu=False, end_t=2.0)
    second = pipe.finish(compute_imu=False, end_t=2.0)
    assert first.summary["gated_seconds"] == second.summary["gated_seconds"] == 2.0
    assert first.summary == second.summary


# --- live diagnostics: the counter state behind every frame goes to frames.parquet -------------

_DIAG_COLS = {"ang", "enter", "exit", "armed"}


def _frames_of(pipe, tmp_path):
    """Write the pipeline's log and read ``frames.parquet`` back."""
    pipe.log.write(tmp_path, {})
    return pd.read_parquet(tmp_path / "frames.parquet")


def test_live_frames_log_each_arms_segmenter_state(tmp_path):
    """A right-arm-solo run that missed reps could not be explained from the old log (it held no
    angles). Per counter: the angle the segmenter saw, its enter/exit thresholds in degrees and
    whether it was armed; per arm the lowest landmark visibility."""
    pipe, _events = _drive(
        curl_deg,
        lambda t: curl_deg(t, top=126.0, bottom=150.0),  # 24 deg: below the 25 deg floor
        25.0,
        gate_windows=[(10.0, 12.0)],
        vis_r_fn=lambda t: 0.3 if 18.0 <= t < 20.0 else 1.0,
    )
    fr = _frames_of(pipe, tmp_path)
    for k in ("l", "r"):
        for name in _DIAG_COLS:
            assert f"{name}_{k}" in fr.columns, (name, k)
        assert f"vis_{k}" in fr.columns
    assert not [c for c in fr.columns if c.endswith("_all")]
    assert all(
        str(fr[c].dtype) == "float64" for c in fr.columns if c not in ("processed", "gate_state")
    )

    on = fr[fr["processed"]]
    last = on.iloc[-1]
    # left arm: full curl, so it arms; thresholds sit inside the 50-150 deg swing, enter < exit
    assert last["armed_l"] == 1.0
    assert 50.0 < last["enter_l"] < last["exit_l"] < 150.0
    assert on["ang_l"].min() >= 49.0 and on["ang_l"].max() <= 151.0
    # right arm: 24 deg swing (10th-90th percentile range ~23 deg) never reaches min_range 25 deg
    assert (on["armed_r"] == 0.0).all()
    assert pipe.arm_totals()["r"]["reps"] == 0 and pipe.arm_totals()["l"]["reps"] >= 4
    assert on["ang_r"].min() >= 125.0 and on["ang_r"].max() <= 151.0
    # visibility is logged as given, and a long low-visibility stretch leaves no angle
    low = on[(on["t"] >= 18.5) & (on["t"] < 19.5)]
    assert len(low) > 0
    assert ((low["vis_r"] - 0.3).abs() < 1e-9).all() and low["ang_r"].isna().all()
    assert ((low["vis_l"] - 1.0).abs() < 1e-9).all() and low["ang_l"].notna().all()
    # gate-closed frames carry no diagnostics at all
    off = fr[~fr["processed"]]
    assert len(off) > 0 and off["t"].between(10.0, 12.0).all()
    diag = [c for c in fr.columns if c not in ("t", "processed", "gate_state")]
    assert off[diag].isna().all().all()


def test_a_curl_that_barely_moves_never_arms_the_counter_and_a_bigger_one_does(tmp_path):
    """The 10th-90th percentile of the last 6 s must span ``min_range`` before anything counts, so
    a small movement is silent. The log shows it as ``armed == 0`` with the angle still moving."""
    small, _ = _drive(curl_deg, lambda t: curl_deg(t, top=126.0, bottom=150.0), 20.0)
    big, _ = _drive(curl_deg, lambda t: curl_deg(t, top=100.0, bottom=150.0), 20.0)
    small_fr = _frames_of(small, tmp_path / "small")
    big_fr = _frames_of(big, tmp_path / "big")
    assert (small_fr["armed_r"] == 0.0).all() and small.arm_totals()["r"]["reps"] == 0
    assert small_fr["ang_r"].max() - small_fr["ang_r"].min() > 20.0  # it did move
    assert big_fr["armed_r"].iloc[-1] == 1.0 and big.arm_totals()["r"]["reps"] >= 5


def test_replay_style_frames_keep_exactly_the_original_columns(tmp_path):
    """Replay output must not change: no diagnostics for the single-track, non-per-arm pipeline."""
    pipe, _events = _drive(curl_deg, curl_deg, 10.0, per_arm=False)
    fr = _frames_of(pipe, tmp_path)
    assert list(fr.columns) == ["t", "processed", "gate_state"]
    assert len(fr) == int(10.0 * FPS)


def test_a_fixed_threshold_segmenter_is_always_armed():
    """``armed`` only means something for adaptive thresholds."""
    from formcoach.pose.reps import RepSegmenter

    assert RepSegmenter(enter=60.0, exit=150.0).armed is True
    adaptive = RepSegmenter(enter=60.0, exit=150.0, adaptive=True)
    assert adaptive.armed is False
    for i in range(60):
        adaptive.push(i / FPS, 100.0 + 0.1 * (i % 2))  # flat: stays unarmed
    assert adaptive.armed is False
    for i in range(60, 240):
        adaptive.push(i / FPS, curl_deg(i / FPS))
    assert adaptive.armed is True
