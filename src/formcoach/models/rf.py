"""Random-forest baseline on the hand-crafted window features (docs/02 §6.2)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def make_rf(seed: int, n_estimators: int = 200, n_jobs: int = -1) -> RandomForestClassifier:
    """Balanced random forest with a fixed seed (logged in every report)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=seed,
    )
