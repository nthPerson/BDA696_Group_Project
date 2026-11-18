"""Tuning (Bench plan Task 5): the parameter registry, the overrides set over a rules file, and
live re-judging of a session's reps from their stored metrics. Fake frames and boards only."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import yaml

from formcoach.app.pipeline import Pipeline, RepMetrics
from formcoach.app.session import LiveSession
from formcoach.app.tuning import Overrides, Param, rejudge
from formcoach.rules.engine import DEFAULT_RULES, WEBCAM_RULES, RuleEngine

from .session_fakes import FakeBoard, feed, synthetic_detect

SWING = "rule.curl.CURL_SWING.upper_arm_trunk_drift"
ROM_TOP = "rule.curl.CURL_PARTIAL_ROM.elbow_max"  # elbow_max < 128.079 fires
WEBCAM_NAME = "rules.mediapipe-webcam.yaml"


def webcam() -> RuleEngine:
    return RuleEngine.from_yaml(WEBCAM_RULES)


def overrides() -> Overrides:
    return Overrides(webcam(), WEBCAM_NAME)


def curl_rep(rep_id: int, *, drift: float, asymmetry: float = 0.0, context=("frontal_view",)):
    """A curl rep whose metrics pass every webcam curl rule except, maybe, the drift one."""
    metrics = {
        "duration_s": 2.0,
        "elbow_min": 55.0,
        "elbow_max": 150.0,
        "upper_arm_trunk_drift": float(drift),
        "elbow_drift_max": 0.1,
        "elbow_asymmetry": float(asymmetry),
    }
    return RepMetrics(
        "curl",
        rep_id,
        2.0 * rep_id,
        2.0 * rep_id + 2.0,
        2.0,
        "pose",
        {"elbow_l": np.zeros(30)},
        metrics,
        arm="l",
        context=frozenset(context),
    )


def codes(faults: list[dict]) -> list[str]:
    return [f["code"] for f in faults]


def test_params_list_one_slider_per_rule_condition_with_ranges():
    params = overrides().params("curl")
    p = {x.key: x for x in params}
    assert len(p) == len(params)  # keys are unique
    swing = p[SWING]
    assert isinstance(swing, Param)
    assert (swing.default, swing.min, swing.max, swing.step) == (35.0, 0.0, 87.5, 1.0)
    assert swing.unit == "deg" and swing.group == "rules"
    assert "CURL_SWING" in swing.label and "upper_arm_trunk_drift" in swing.label
    assert {
        "gate.energy_threshold",
        "gate.on_windows",
        "gate.off_windows",
        "seg.min_range_deg",
    } <= set(p)
    # one slider per condition of every curl rule, nothing from the other exercises
    rule_keys = sorted(k for k in p if k.startswith("rule."))
    assert rule_keys == sorted(
        [
            "rule.curl.CURL_PARTIAL_ROM.elbow_min",
            ROM_TOP,
            SWING,
            "rule.curl.CURL_SWING.elbow_drift_max",
            "rule.curl.CURL_TOO_FAST.duration_s",
            "rule.curl.CURL_ASYMMETRY.elbow_asymmetry",
        ]
    )
    drift = p["rule.curl.CURL_SWING.elbow_drift_max"]  # 0.35 torso units
    assert (drift.min, drift.max, drift.step, drift.unit) == (0.0, 1.35, 0.01, "")
    fast = p["rule.curl.CURL_TOO_FAST.duration_s"]  # 1.0 s
    assert (fast.min, fast.max, fast.step, fast.unit) == (0.0, 2.5, 0.1, "s")
    gate = p["gate.energy_threshold"]
    assert (gate.min, gate.max, gate.step, gate.default) == (0.5, 30.0, 0.5, 3.0)
    assert (p["gate.on_windows"].min, p["gate.on_windows"].max, p["gate.on_windows"].default) == (
        1,
        10,
        2,
    )
    assert (
        p["gate.off_windows"].min,
        p["gate.off_windows"].max,
        p["gate.off_windows"].default,
    ) == (1, 40, 12)
    seg = p["seg.min_range_deg"]
    assert (seg.min, seg.max, seg.step, seg.default, seg.unit) == (10.0, 60.0, 1.0, 25.0, "deg")
    # a negative limit keeps its default inside the range
    valgus = {x.key: x for x in overrides().params("squat")}[
        "rule.squat.SQUAT_KNEE_VALGUS.knee_track_at_bottom"
    ]
    assert valgus.min <= valgus.default == -0.10 <= valgus.max
    with pytest.raises(KeyError):
        overrides().params("deadlift")


def test_set_clamps_and_engine_uses_the_new_threshold():
    base = webcam()
    before = copy.deepcopy(base.doc)
    o = Overrides(base, WEBCAM_NAME)
    assert o.engine() is base  # nothing overridden yet
    assert o.set(SWING, 1000.0) == 87.5
    assert o.set(SWING, -5) == 0.0
    assert o.set(SWING, 45) == 45.0
    assert o.set("gate.on_windows", 2.6) == 3  # window counts are whole numbers
    swing = next(r for r in o.engine().rules_for("curl") if r.code == "CURL_SWING")
    assert [c.threshold for c in swing.when] == [45.0, 0.35]
    assert o.engine().evaluate(curl_rep(0, drift=41.0), context={"frontal_view"}) == []
    assert codes(
        [f.as_dict() for f in base.evaluate(curl_rep(0, drift=41.0), context={"frontal_view"})]
    ) == ["CURL_SWING"]
    assert base.doc == before  # the base engine's document is never touched
    doc = o.document()
    assert doc["gate"]["on_windows"] == 3
    doc["exercises"]["curl"]["rules"].clear()  # a deep copy: changing it changes nothing
    assert o.document()["exercises"]["curl"]["rules"]


def test_unknown_key_raises_keyerror_and_nan_raises_valueerror():
    o = overrides()
    o.set(SWING, 40.0)
    for key in (
        "rule.curl.CURL_SWING.nope",
        "rule.press.CURL_SWING.upper_arm_trunk_drift",
        "gate.x",
    ):
        with pytest.raises(KeyError):
            o.set(key, 1.0)
    for bad in (float("nan"), "45", None, True, [45.0]):
        with pytest.raises(ValueError):
            o.set(SWING, bad)
    assert o.values == {SWING: 40.0}  # nothing changed


def test_values_holds_only_changes_and_reset_clears_them():
    o = overrides()
    assert o.values == {}
    o.set(SWING, 45.0)
    o.set("seg.min_range_deg", 30)
    assert o.values == {SWING: 45.0, "seg.min_range_deg": 30.0}
    o.set(SWING, 35.0)  # back to the default: no longer a change
    assert o.values == {"seg.min_range_deg": 30.0}
    assert o.document()["rep_segmentation"]["min_range_deg"] == 30.0
    o.reset()
    assert o.values == {}
    assert o.document() == webcam().doc


def test_rejudge_changes_faults_without_touching_metrics():
    reps = [curl_rep(0, drift=41.0), curl_rep(1, drift=41.0), curl_rep(2, drift=20.0)]
    metrics = [dict(r.metrics) for r in reps]
    o = overrides()
    strict = rejudge(reps, o.engine())
    assert [codes(f) for f in strict] == [["CURL_SWING"], ["CURL_SWING"], []]
    assert strict[0][0]["value"] == 41.0 and strict[0][0]["threshold"] == 35.0
    assert sum(1 for f in strict if f) == 2  # flagged 2, good 1
    o.set(SWING, 45.0)
    loose = rejudge(reps, o.engine())
    assert loose == [[], [], []]  # flagged 0, good 3
    assert [r.metrics for r in reps] == metrics
    assert all(r.faults == [] for r in reps)  # rejudge returns faults; it does not assign them


def test_rejudge_respects_each_reps_context():
    one_arm = curl_rep(0, drift=0.0, asymmetry=60.0, context=("frontal_view",))
    two_arm = curl_rep(1, drift=0.0, asymmetry=60.0, context=("frontal_view", "two_arm"))
    o = overrides()
    o.set("rule.curl.CURL_ASYMMETRY.elbow_asymmetry", 0.0)
    faults = rejudge([one_arm, two_arm], o.engine())
    assert codes(faults[0]) == []  # never gains a two-arm fault
    assert codes(faults[1]) == ["CURL_ASYMMETRY"]


def _session(**kw) -> LiveSession:
    return LiveSession(
        exercise="curl", detect=synthetic_detect, rules=webcam(), rules_name=WEBCAM_NAME, **kw
    )


def _payloads(session: LiveSession, kind: str) -> list[dict]:
    return [e.payload for e in session.log.events if e.kind == kind]


def test_session_set_param_emits_params_changed_and_rejudged_with_new_totals():
    s = _session()
    seen = []
    s.subscribe(on_event=seen.append)
    feed(s, seconds=9.0)  # 3 clean curls per arm (elbow_max reads about 150)
    clean = {"reps": 3, "good": 3, "flagged": 0, "faults_by_code": {}}
    assert s.pipeline.arm_totals() == {"l": clean, "r": clean}
    assert s.overrides is not None and s.overrides.base_name == WEBCAM_NAME

    s.set_param(ROM_TOP, 155.0)  # every rep's elbow_max is now below the limit
    flagged = {"reps": 3, "good": 0, "flagged": 3, "faults_by_code": {"CURL_PARTIAL_ROM": 3}}
    assert s.pipeline.arm_totals() == {"l": flagged, "r": flagged}
    assert all(codes(r.faults) == ["CURL_PARTIAL_ROM"] for t in s.pipeline.tracks for r in t.reps)
    changed = _payloads(s, "params_changed")
    assert changed == [{"key": ROM_TOP, "value": 155.0, "values": {ROM_TOP: 155.0}}]
    (rejudged,) = _payloads(s, "rejudged")
    assert rejudged["totals"] == {"l": flagged, "r": flagged}
    assert sorted((r["arm"], r["rep_id"]) for r in rejudged["reps"]) == [
        (arm, i) for arm in ("l", "r") for i in range(3)
    ]
    assert all(codes(r["faults"]) == ["CURL_PARTIAL_ROM"] for r in rejudged["reps"])
    assert [e.kind for e in seen[-2:]] == ["params_changed", "rejudged"]
    assert _payloads(s, "rep")[0]["faults"] == []  # the log's rep events are not rewritten

    # reps counted from now on are judged with the override too
    snaps = feed(s, seconds=3.0, start=270)
    new = [r for sn in snaps for r in sn.completed]
    assert new and all(codes(r.faults) == ["CURL_PARTIAL_ROM"] for r in new)

    s.set_param("gate.energy_threshold", 12.5)  # not a rule: no re-judging
    assert len(_payloads(s, "rejudged")) == 1
    assert _payloads(s, "params_changed")[-1]["values"] == {
        ROM_TOP: 155.0,
        "gate.energy_threshold": 12.5,
    }

    s.reset_params()
    assert s.overrides.values == {} and s.pipeline.rules is s.rules
    assert _payloads(s, "params_changed")[-1]["values"] == {}
    assert _payloads(s, "rejudged")[-1]["totals"] == s.pipeline.arm_totals()
    assert all(t["flagged"] == 0 for t in s.pipeline.arm_totals().values())

    with pytest.raises(KeyError):
        s.set_param("rule.curl.NOPE.elbow_max", 1.0)
    with pytest.raises(ValueError):
        s.set_param(ROM_TOP, math.nan)
    assert s.overrides.values == {}

    bare = LiveSession(exercise="curl", detect=synthetic_detect)
    assert bare.overrides is None
    with pytest.raises(RuntimeError):
        bare.set_param(ROM_TOP, 155.0)
    with pytest.raises(RuntimeError):
        bare.reset_params()
    unnamed = LiveSession(exercise="curl", detect=synthetic_detect, rules=webcam())
    assert unnamed.overrides.base_name == "rules"


def test_overrides_survive_an_exercise_switch_and_a_counter_reset():
    s = _session()
    s.set_param(ROM_TOP, 155.0)
    s.set_param("seg.min_range_deg", 40.0)
    s.set_exercise("press")
    s.set_exercise("curl")
    s.reset_counters()
    assert s.pipeline.rules is s.overrides.engine() and s.pipeline.rules is not s.rules
    assert all(t.seg.min_range == 40.0 for t in s.pipeline.tracks)
    reps = [r for sn in feed(s, seconds=9.0) for r in sn.completed]
    assert len(reps) == 6 and all(codes(r.faults) == ["CURL_PARTIAL_ROM"] for r in reps)


class _ShakingBoard(FakeBoard):
    """Gate flag off; |a| alternates 9.81 +/- 3 m/s² from the first sample (var = 9 (m/s²)²)."""

    def __init__(self):
        super().__init__(gate=False, samples_per_poll=10)

    def accel(self, seq):
        return 0.0, 0.0, 9.81 + (3.0 if seq % 2 else -3.0)


def test_gate_params_drive_the_energy_gate_and_reset_restores_it():
    s = _session(board=_ShakingBoard(), gate_source="energy")
    s.set_param("gate.energy_threshold", 20.0)  # above var(|a|) = 9
    assert not any(sn.gate for sn in feed(s, seconds=2.0))
    s.reset_params()  # back to 3.0
    assert feed(s, seconds=1.0, start=60)[-1].gate is True


def test_save_writes_a_loadable_rules_file_and_refuses_the_package_dir(tmp_path):
    o = overrides()
    o.set(SWING, 45.0)
    o.set("gate.off_windows", 20)
    o.set("seg.min_range_deg", 30)
    out = o.save(tmp_path / "tuned" / "curl-tuned.yaml")
    assert out == tmp_path / "tuned" / "curl-tuned.yaml"
    text = out.read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert text.startswith("#") and WEBCAM_NAME in header[0]
    assert any(SWING in line and "45" in line for line in header)
    assert any("gate.off_windows" in line for line in header)
    loaded = RuleEngine.from_yaml(out)
    swing = next(r for r in loaded.rules_for("curl") if r.code == "CURL_SWING")
    assert swing.when[0].threshold == 45.0
    assert loaded.gate["off_windows"] == 20 and loaded.segmentation["min_range_deg"] == 30.0
    assert yaml.safe_load(text) == o.document()
    for target in (WEBCAM_RULES, DEFAULT_RULES.parent / "tuned.yaml"):
        before = target.read_text(encoding="utf-8") if target.exists() else None
        with pytest.raises(ValueError):
            o.save(target)
        assert (target.read_text(encoding="utf-8") if target.exists() else None) == before


def test_min_range_change_applies_to_live_segmenters():
    pipe = Pipeline(exercise="curl", per_arm=True)
    pipe.set_min_range(40.0)
    assert pipe.cfg["min_range"] == 40.0
    assert [t.seg.min_range for t in pipe.tracks] == [40.0, 40.0]
    pipe.set_gate(True, 0.0)
    pipe.set_gate(False, 1.0)  # a gate close builds fresh segmenters from the config
    assert [t.seg.min_range for t in pipe.tracks] == [40.0, 40.0]
    with pytest.raises(ValueError):
        pipe.set_min_range(float("nan"))

    s = _session()
    feed(s, seconds=3.0)
    s.set_param("seg.min_range_deg", 200.0)  # clamped to 60
    assert [t.seg.min_range for t in s.pipeline.tracks] == [60.0, 60.0]
    assert _payloads(s, "params_changed")[-1] == {
        "key": "seg.min_range_deg",
        "value": 60.0,
        "values": {"seg.min_range_deg": 60.0},
    }
    s.reset_params()
    assert [t.seg.min_range for t in s.pipeline.tracks] == [25.0, 25.0]
