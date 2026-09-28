"""Motion-energy gate: one threshold on the variance of |a| over a window (docs/02 §6.2)."""

from __future__ import annotations

import numpy as np


class EnergyGate:
    """``active`` iff ``var(|a|) > threshold``. ``fit`` picks the threshold maximising balanced
    accuracy on the training windows (scanned over quantiles of the feature); sklearn-like API
    so it drops into the LOSO harness. Input ``X`` is ``(n, 1)``: the energy feature."""

    def __init__(self, n_candidates: int = 200):
        self.n_candidates = n_candidates
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnergyGate:
        e = np.asarray(X, dtype=np.float64).reshape(-1)
        y = np.asarray(y).astype(int)
        cands = np.unique(np.quantile(e, np.linspace(0.0, 1.0, self.n_candidates)))
        best, best_thr = -1.0, float(np.median(e))
        pos, neg = y == 1, y == 0
        for thr in cands:
            pred = e > thr
            tpr = pred[pos].mean() if pos.any() else 0.0
            tnr = (~pred[neg]).mean() if neg.any() else 0.0
            bal = 0.5 * (tpr + tnr)
            if bal > best:
                best, best_thr = bal, float(thr)
        self.threshold_ = best_thr
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        e = np.asarray(X, dtype=np.float64).reshape(-1)
        return (e > self.threshold_).astype(int)

    def get_params(self, deep: bool = False) -> dict:  # sklearn compatibility
        return {"n_candidates": self.n_candidates}
