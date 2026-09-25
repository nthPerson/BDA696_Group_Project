"""rules.yaml + engine: every §7 rule fires on the right perturbation and not on a correct rep."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.app.pipeline import REP_STEPS, RepMetrics
from formcoach.rules import engine

CORRECT = {
    "curl": {
        "duration_s": 1.8,
        "elbow_min": 45.0,
        "elbow_max": 165.0,
        "elbow_min_l": 45.0,
        "elbow_min_r": 47.0,
        "elbow_max_l": 165.0,
        "elbow_max_r": 164.0,
        "elbow_asymmetry": 2.0,
        "elbow_top_asymmetry": 1.0,
        "upper_arm_trunk_drift": 5.0,
        "elbow_drift_max": 0.03,
        "trunk_incl_max": 5.0,
        "abd_peak": 20.0,
        "elbow_at_abd_peak": 100.0,
        "knee_min": 175.0,
        "trunk_incl_at_bottom": 5.0,
        "knee_track_at_bottom": 0.02,
        "hip_knee_height_at_bottom": 0.9,
    },
    "press": {
        "duration_s": 1.6,
        "elbow_min": 70.0,
        "elbow_max": 172.0,
        "elbow_min_l": 70.0,
        "elbow_min_r": 72.0,
        "elbow_max_l": 172.0,
        "elbow_max_r": 170.0,
        "elbow_asymmetry": 2.0,
        "elbow_top_asymmetry": 2.0,
        "upper_arm_trunk_drift": 60.0,
        "elbow_drift_max": 0.2,
        "trunk_incl_max": 6.0,
        "abd_peak": 150.0,
        "elbow_at_abd_peak": 170.0,
        "knee_min": 175.0,
        "trunk_incl_at_bottom": 6.0,
        "knee_track_at_bottom": 0.0,
        "hip_knee_height_at_bottom": 0.9,
    },
    "raise": {
        "duration_s": 2.0,
        "elbow_min": 150.0,
        "elbow_max": 172.0,
        "elbow_min_l": 150.0,
        "elbow_min_r": 152.0,
        "elbow_max_l": 172.0,
        "elbow_max_r": 171.0,
        "elbow_asymmetry": 2.0,
        "elbow_top_asymmetry": 1.0,
        "upper_arm_trunk_drift": 70.0,
        "elbow_drift_max": 0.4,
        "trunk_incl_max": 4.0,
        "abd_peak": 88.0,
        "elbow_at_abd_peak": 160.0,
        "knee_min": 175.0,
        "trunk_incl_at_bottom": 4.0,
        "knee_track_at_bottom": 0.0,
        "hip_knee_height_at_bottom": 0.9,
    },
    "squat": {
        "duration_s": 2.5,
        "elbow_min": 160.0,
        "elbow_max": 175.0,
        "elbow_min_l": 160.0,
        "elbow_min_r": 160.0,
        "elbow_max_l": 175.0,
        "elbow_max_r": 175.0,
        "elbow_asymmetry": 0.0,
        "elbow_top_asymmetry": 0.0,
        "upper_arm_trunk_drift": 10.0,
        "elbow_drift_max": 0.1,
        "trunk_incl_max": 30.0,
        "abd_peak": 30.0,
        "elbow_at_abd_peak": 170.0,
        "knee_min": 85.0,
        "trunk_incl_at_bottom": 30.0,
        "knee_track_at_bottom": 0.05,
        "hip_knee_height_at_bottom": -0.05,
    },
}

PERTURBATIONS = [
    ("curl", {"elbow_min": 80.0}, "CURL_PARTIAL_ROM"),
    ("curl", {"elbow_max": 130.0}, "CURL_PARTIAL_ROM"),
    ("curl", {"upper_arm_trunk_drift": 25.0}, "CURL_SWING"),
    ("curl", {"elbow_drift_max": 0.2}, "CURL_SWING"),
    ("curl", {"duration_s": 0.8}, "CURL_TOO_FAST"),
    ("curl", {"elbow_asymmetry": 25.0}, "CURL_ASYMMETRY"),
    ("press", {"elbow_max": 150.0}, "PRESS_NO_LOCKOUT"),
    ("press", {"trunk_incl_max": 20.0}, "PRESS_TRUNK_LEAN"),
    ("press", {"elbow_top_asymmetry": 25.0}, "PRESS_ASYMMETRY"),
    ("press", {"duration_s": 0.7}, "PRESS_TOO_FAST"),
    ("raise", {"abd_peak": 105.0}, "RAISE_OVER"),
    ("raise", {"abd_peak": 60.0}, "RAISE_PARTIAL"),
    ("raise", {"elbow_at_abd_peak": 120.0}, "RAISE_BENT_ELBOW"),
    ("raise", {"duration_s": 0.9}, "RAISE_TOO_FAST"),
    ("squat", {"knee_min": 115.0}, "SQUAT_SHALLOW"),
    ("squat", {"hip_knee_height_at_bottom": 0.1}, "SQUAT_SHALLOW"),
    ("squat", {"trunk_incl_at_bottom": 50.0}, "SQUAT_FORWARD_LEAN"),
    ("squat", {"knee_track_at_bottom": -0.15}, "SQUAT_KNEE_VALGUS"),
    ("squat", {"duration_s": 1.0}, "SQUAT_TOO_FAST"),
]


def _rep(exercise: str, metrics: dict) -> RepMetrics:
    angles = {k: np.linspace(0, 1, REP_STEPS) for k in ("elbow_l", "elbow_r")}
    return RepMetrics(
        exercise, 0, 0.0, metrics["duration_s"], metrics["duration_s"], "pose", angles, metrics
    )


@pytest.fixture(scope="module")
def eng() -> engine.RuleEngine:
    return engine.RuleEngine.from_yaml()


def test_yaml_covers_the_catalog(eng):
    codes = {r.code for ex in ("curl", "press", "raise", "squat") for r in eng.rules_for(ex)}
    assert codes == {c for _, _, c in PERTURBATIONS}
    for ex in ("curl", "press", "raise", "squat"):
        for r in eng.rules_for(ex):
            assert r.message and r.severity in ("info", "warn", "error"), r.code
    assert eng.gate["on_windows"] == 2 and eng.gate["off_windows"] == 12
    assert eng.gate["energy_threshold_ms2sq"] == 3.0
    assert eng.rep_config("curl")["angle"] == "elbow"


@pytest.mark.parametrize("exercise", ["curl", "press", "raise", "squat"])
def test_correct_rep_has_no_faults(eng, exercise):
    assert eng.evaluate(_rep(exercise, CORRECT[exercise])) == []


@pytest.mark.parametrize(("exercise", "change", "code"), PERTURBATIONS)
def test_each_perturbation_fires_exactly_its_rule(eng, exercise, change, code):
    m = {**CORRECT[exercise], **change}
    faults = eng.evaluate(_rep(exercise, m))
    assert [f.code for f in faults] == [code], faults
    f = faults[0]
    assert f.metric in change and f.value == pytest.approx(change[f.metric])
    assert f.threshold is not None and f.message
    d = f.as_dict()
    assert set(d) >= {"code", "severity", "metric", "value", "threshold", "message"}


def test_missing_metric_is_skipped_not_raised(eng):
    m = dict(CORRECT["curl"])
    del m["upper_arm_trunk_drift"]
    m["elbow_drift_max"] = float("nan")
    assert eng.evaluate(_rep("curl", m)) == []


def test_unknown_exercise_raises(eng):
    with pytest.raises(KeyError):
        eng.rules_for("deadlift")


def test_engine_plugs_into_the_replay_pipeline(tmp_path):
    from formcoach.app import pipeline

    e = engine.RuleEngine.from_yaml()
    res = pipeline.run_replay(gate="always_on", headless=True, rules=e, out_dir=tmp_path / "r")
    reps = [x for x in res.events if x.kind == "rep"]
    assert len(reps) >= 8
    # MM-Fit lifted pose reads the elbow at ~95° min: with docs/02 defaults every rep is PARTIAL_ROM
    faults = [x for x in res.events if x.kind == "fault"]
    assert faults and all("code" in x.payload for x in faults)
    assert res.summary["faults"] == len(faults)
