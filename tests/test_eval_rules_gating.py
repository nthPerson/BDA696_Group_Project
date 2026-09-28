"""eval rules on MM-Fit pose (fixture), eval gating on replay sessions, train gate --model rf."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from formcoach.data import convert
from formcoach.data.fixtures import FIXTURE_ROOT
from formcoach.eval import gating, rules_eval
from formcoach.io.replay import FIXTURE_SESSION

MMFIT_FIXTURE = FIXTURE_ROOT / "mmfit" / "mm-fit"


def test_eval_rules_writes_pass_rates_and_perturbation_table(tmp_path):
    out = tmp_path / "reports" / "rules_validation.md"
    result = rules_eval.evaluate(MMFIT_FIXTURE, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Rules validation")
    for section in (
        "## Rep segmentation",
        "## Pass rate per rule",
        "## Perturbation detection",
        "## Metric distributions",
    ):
        assert section in text, section
    assert "CURL_PARTIAL_ROM" in text and "SQUAT_SHALLOW" in text
    assert result["reps"]["curl"] >= 8 and result["reps"]["squat"] >= 6
    pert = result["perturbation"]
    assert pert[("curl", "speed_x2", "CURL_TOO_FAST")] >= 0.9
    assert pert[("squat", "squat_speed_x2", "SQUAT_TOO_FAST")] >= 0.9
    # +25° trunk lean only crosses the 45° design threshold when the base lean is > 20°:
    # the detection rate is a finding the report shows, not a pass condition
    assert 0.0 <= pert[("squat", "trunk_lean_25", "SQUAT_FORWARD_LEAN")] <= 1.0
    assert pert[("curl", "rom_x0.7", "CURL_PARTIAL_ROM")] >= 0.9
    assert (tmp_path / "reports" / "figures").exists()


def test_eval_rules_calibrate_writes_percentile_thresholds(tmp_path):
    out = tmp_path / "rules_validation.md"
    cal = tmp_path / "rules.calibrated.yaml"
    rules_eval.evaluate(MMFIT_FIXTURE, out, calibrate_out=cal)
    import yaml

    doc = yaml.safe_load(cal.read_text(encoding="utf-8"))
    assert doc["calibrated_from"]["pose_source"] == "mmfit-pose3d"
    curl = {r["code"]: r for r in doc["exercises"]["curl"]["rules"]}
    assert curl["CURL_PARTIAL_ROM"]["when"][0]["threshold"] != 75.0  # moved to the 95th percentile


def test_train_gate_rf_writes_joblib_and_laptop_gate_loads_it(tmp_path):
    from formcoach.app.gate import LaptopGate
    from formcoach.models import train
    from formcoach.signal import build

    processed = tmp_path / "processed"
    convert.convert_dataset("mmfit", MMFIT_FIXTURE, processed)
    build.build_features(processed)
    path = train.train_rf_gate(processed, datasets=("mmfit",), out=tmp_path / "gate_rf.joblib")
    assert path.exists()
    g = LaptopGate(model_path=path)
    assert g.name == "laptop"


def test_eval_gating_compares_gates_on_replay_sessions(tmp_path):
    out = tmp_path / "reports" / "gating.md"
    table = gating.evaluate([FIXTURE_SESSION], gates=("always_on", "energy"), out=out)
    assert set(table["gate"]) == {"always_on", "energy"}
    on = table[table["gate"] == "always_on"].iloc[0]
    en = table[table["gate"] == "energy"].iloc[0]
    assert on["frames_processed_pct"] == 100.0 or on["frames_processed_pct"] > 99
    assert en["frames_processed_pct"] < on["frames_processed_pct"]
    assert en["reps_match"] and en["faults_match"]
    text = out.read_text(encoding="utf-8")
    assert "# Gating benchmark" in text and "| energy |" in text
    assert isinstance(table, pd.DataFrame) and Path(out).exists()
