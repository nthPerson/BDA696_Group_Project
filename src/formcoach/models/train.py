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


FIRMWARE_MODEL_DIR = report.REPO_ROOT / "firmware" / "model"


def train_cnn_gate(
    processed_root: Path = convert.PROCESSED_ROOT,
    datasets: tuple[str, ...] = ("recofit", "mmfit"),
    *,
    min_purity: float = 0.8,
    epochs: int = 30,
    seed: int = report.SEED,
    export_int8: bool = True,
    model_dir: Path = FIRMWARE_MODEL_DIR,
    log=print,
) -> dict:
    """Train the 6-class CNN on all windows of ``datasets`` (subject-held-out early stopping),
    export int8 TFLite + ``gate_model_data.cc`` + ``preprocess.{json,h}`` into
    ``firmware/model/`` and write ``reports/gate_cnn_export.md`` (float vs int8 on the held-out
    subjects, sizes). Returns the summary dict."""
    from formcoach.models import cnn, export
    from formcoach.signal.windows import windows_to_array

    parts = []
    for ds in datasets:
        w = build.load_windows(processed_root, ds, min_purity=min_purity)
        loso.check_units(w)
        parts.append(w)
    w = pd.concat(parts, ignore_index=True)
    X = windows_to_array(w)
    y = w["label_exercise"].to_numpy()
    groups = w["subject"].to_numpy()
    log(f"training CNN on {len(w):,} windows from {w['subject'].nunique()} subjects")
    res = cnn.train(X, y, groups, classes=cnn.CLASSES6, seed=seed, epochs=epochs, verbose=0)
    val_mask = np.isin(groups, res.val_subjects)
    pred_f = cnn.predict(res.model, res.preprocess, X[val_mask])
    y_val = np.array([cnn.CLASSES6.index(v) for v in y[val_mask]])
    from sklearn.metrics import accuracy_score, f1_score

    summary = {
        "windows": len(w),
        "subjects": int(w["subject"].nunique()),
        "val_subjects": len(res.val_subjects),
        "epochs_run": len(res.history.get("loss", [])),
        "params": int(res.model.count_params()),
        "val_accuracy_float": float(accuracy_score(y_val, pred_f)),
        "val_macro_f1_float": float(f1_score(y_val, pred_f, average="macro")),
        "val_active_f1_float": float(f1_score(y_val != 0, pred_f != 0)),
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    res.model.save(model_dir / "gate_model.keras")
    res.preprocess.save(model_dir / "preprocess.json", cnn.CLASSES6)
    if export_int8:
        rng = np.random.default_rng(seed)
        rep_idx = rng.choice(
            np.flatnonzero(~val_mask), size=min(500, int((~val_mask).sum())), replace=False
        )
        tfl = export.to_tflite_int8(
            res.model, res.preprocess.apply(X[rep_idx]), model_dir / "gate_model.tflite"
        )
        pred_q = export.tflite_predict(tfl, res.preprocess.apply(X[val_mask]))
        summary.update(
            {
                "tflite_bytes": tfl.stat().st_size,
                "val_accuracy_int8": float(accuracy_score(y_val, pred_q)),
                "val_macro_f1_int8": float(f1_score(y_val, pred_q, average="macro")),
                "val_active_f1_int8": float(f1_score(y_val != 0, pred_q != 0)),
                "float_to_int8_macro_f1_drop": float(
                    summary["val_macro_f1_float"] - f1_score(y_val, pred_q, average="macro")
                ),
            }
        )
        export.write_cc(tfl, model_dir / "gate_model_data.cc")
        export.write_preprocess_header(
            res.preprocess, cnn.CLASSES6, model_dir / "preprocess.h", tfl
        )
        export.to_tflite_float(res.model, model_dir / "gate_model_float.tflite")
    rows = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    out = report.REPORTS_DIR / "gate_cnn_export.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        report.header("CNN gate: training and int8 export", "formcoach train gate --model cnn",
                      extra={"datasets": ", ".join(datasets), "classes": ", ".join(cnn.CLASSES6),
                             "held-out subjects": ", ".join(res.val_subjects)}),
        report.md_table(rows, floatfmt=".4f"), "",
        "Held-out numbers are on the early-stopping subjects; the unseen-subject headline is "
        "`formcoach eval loso --model cnn` / `--model cnn-int8` (reports/loso_cnn_*.md).", "",
    ]), encoding="utf-8")  # fmt: skip
    summary["report"] = str(out)
    return summary
