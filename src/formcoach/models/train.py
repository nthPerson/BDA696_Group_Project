"""``formcoach train gate --model rf``: fit the random-forest gate on every window of the given
datasets and save it for the laptop-side gate (``models/gate_rf.joblib``, gitignored)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from formcoach.data import convert
from formcoach.eval import loso, report
from formcoach.models.rf import make_rf
from formcoach.signal import build

DEFAULT_OUT = report.REPO_ROOT / "models" / "gate_rf.joblib"


def train_rf_gate(
    processed_root: Path = convert.PROCESSED_ROOT,
    datasets: tuple[str, ...] = ("recofit", "mmfit"),
    *,
    task: str = "active",
    min_purity: float = 0.8,
    out: Path = DEFAULT_OUT,
    seed: int = report.SEED,
) -> Path:
    parts = []
    for ds in datasets:
        w = build.load_windows(processed_root, ds, min_purity=min_purity)
        loso.check_units(w)
        if (w["units"] != "si").any():
            raise ValueError(f"{ds} is not in SI units; the laptop gate is trained on SI data only")
        parts.append(w)
    w = pd.concat(parts, ignore_index=True)
    X = loso.feature_matrix(w)
    y = w["label_active"].to_numpy() if task == "active" else w["label_exercise"].to_numpy()
    model = make_rf(seed=seed)
    model.fit(X, y)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    meta = {
        "task": task,
        "datasets": list(datasets),
        "windows": len(w),
        "subjects": int(w["subject"].nunique()),
        "seed": seed,
        "features": list(loso.FEATURE_NAMES),
        "class_balance": {
            str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=True)
        },
    }
    import json

    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
