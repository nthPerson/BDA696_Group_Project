"""Replay output is pinned to a golden file written before the Pipeline refactor.

``tests/golden/replay_mmfit_w00_curls.json`` holds the events and summary of
``run_replay`` on the committed MM-Fit fixture for ``always_on`` and ``energy`` gates (the energy
run covers gate events and the IMU cross-check). The live demo work must never change them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formcoach.app import pipeline
from formcoach.app.events import _json_default
from formcoach.data.fixtures import FIXTURE_ROOT
from formcoach.rules.engine import RuleEngine

SESSION = FIXTURE_ROOT / "replay" / "mmfit_w00_curls"
GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "replay_mmfit_w00_curls.json").read_text(encoding="utf-8")
)


def _close(actual, expected, path="$"):
    """Recursive equality: floats within 1e-6 relative (NaN equals NaN), everything else exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for k in expected:
            _close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9, nan_ok=True), path
    else:
        assert actual == expected, path


@pytest.mark.parametrize("gate", ["always_on", "energy"])
def test_replay_matches_golden(gate):
    res = pipeline.run_replay(SESSION, gate=gate, headless=True, rules=RuleEngine.from_yaml())
    summary = {k: v for k, v in res.summary.items() if k != "session"}
    events = [{"kind": e.kind, "t": e.t, "payload": e.payload} for e in res.events]
    actual = json.loads(json.dumps({"summary": summary, "events": events}, default=_json_default))
    expected = GOLDEN[gate]
    assert [e["kind"] for e in actual["events"]] == [e["kind"] for e in expected["events"]]
    _close(actual["summary"], expected["summary"], "$.summary")
    _close(actual["events"], expected["events"], "$.events")


def test_golden_covers_gate_events_and_the_imu_cross_check():
    kinds = {e["kind"] for e in GOLDEN["energy"]["events"]}
    assert {"gate_open", "rep"} <= kinds
    summary = GOLDEN["energy"]["summary"]
    assert summary["reps_imu"] > 0 and summary["frames_processed"] < summary["frames_total"]
    assert "arm" not in GOLDEN["always_on"]["events"][0]["payload"]
