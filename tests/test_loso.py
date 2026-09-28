"""LOSO split utility, energy + RF baselines, and the LOSO report writer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formcoach.eval import loso
from formcoach.models import energy, rf


def _windows(n_subjects: int = 4, per_subject: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic windows: idle = small noise, curl = 1 Hz sine, squat = 0.5 Hz sine (x column)."""
    rng = np.random.default_rng(seed)
    rows = []
    t = np.arange(100) / 50.0
    for s in range(n_subjects):
        for i in range(per_subject):
            label = ("idle", "curl", "squat")[i % 3]
            x = rng.normal(0, 0.1, (100, 6))
            x[:, 2] += 9.81
            if label == "curl":
                x[:, 0] += 3 * np.sin(2 * np.pi * 1.0 * t)
            elif label == "squat":
                x[:, 0] += 4 * np.sin(2 * np.pi * 0.5 * t)
            rows.append(
                {
                    "dataset": "syn",
                    "subject": f"S{s}",
                    "session": "a",
                    "device": "d",
                    "window_id": i,
                    "t_start": float(i),
                    "t_end": float(i + 2),
                    "label_exercise": label,
                    "label_active": int(label != "idle"),
                    "label_purity": 1.0,
                    "units": "si",
                    "x": x.astype(np.float32).reshape(-1),
                }
            )
    return pd.DataFrame(rows)


def test_loso_splits_leave_each_subject_out_once():
    groups = np.array(["A", "A", "B", "C", "C", "C"])
    splits = loso.loso_splits(groups)
    assert [g for _, _, g in splits] == ["A", "B", "C"]
    for train, test, g in splits:
        assert set(groups[test]) == {g} and g not in set(groups[train])
        assert len(train) + len(test) == len(groups)


def test_group_kfold_splits_keep_subjects_together():
    groups = np.array([f"S{i}" for i in range(10) for _ in range(5)])
    splits = loso.group_kfold_splits(groups, n_folds=3, seed=1)
    assert len(splits) == 3
    seen = set()
    for train, test, _name in splits:
        assert not (set(groups[train]) & set(groups[test]))
        seen |= set(groups[test])
    assert seen == set(groups)


def test_energy_gate_separates_active_from_idle():
    w = _windows()
    X = loso.energy_features(w)
    assert X.shape == (len(w), 1)
    gate = energy.EnergyGate().fit(X, w["label_active"].to_numpy())
    assert gate.threshold_ > 0
    assert (gate.predict(X) == w["label_active"].to_numpy()).mean() > 0.95


def test_run_loso_rf_on_features_reports_per_fold_and_pooled(tmp_path):
    w = _windows()
    from formcoach.signal.features import featurize_windows

    w = featurize_windows(w, 50)
    X = loso.feature_matrix(w)
    y = w["label_exercise"].to_numpy()
    res = loso.run_loso(X, y, w["subject"].to_numpy(), lambda: rf.make_rf(seed=0, n_estimators=50))
    assert list(res.per_fold["fold"]) == ["S0", "S1", "S2", "S3"]
    assert res.per_fold["macro_f1"].min() > 0.9
    assert res.pooled["accuracy"] > 0.9 and 0 < res.pooled["macro_f1"] <= 1
    assert res.confusion.shape == (3, 3) and res.labels == ["curl", "idle", "squat"]
    out = tmp_path / "loso_rf.md"
    loso.write_loso_report(res, out, title="LOSO rf", command="formcoach eval loso", meta={"k": 1})
    text = out.read_text(encoding="utf-8")
    assert "# LOSO rf" in text and "| fold |" in text and "Pooled" in text and "seed" in text
    assert "| curl |" in text  # confusion matrix rows
    assert (tmp_path / "figures").exists()


def test_run_loso_rejects_mixed_units():
    w = _windows()
    w.loc[w.index[:10], "units"] = "normalized"
    with pytest.raises(ValueError, match="units"):
        loso.check_units(w)
