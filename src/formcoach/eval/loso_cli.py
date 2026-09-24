"""Glue between ``formcoach eval loso`` and :mod:`formcoach.eval.loso`: assembles X/y/groups
for a (model, dataset, task), runs the folds, writes ``reports/loso_<model>_<task>_<dataset>.md``
(or ``transfer_<model>_<task>_<train>-to-<test>.md`` for cross-dataset runs)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from formcoach.data import convert
from formcoach.eval import loso, report
from formcoach.signal import build

MODELS = ("energy", "rf", "cnn", "cnn-int8")


def _xy(w: pd.DataFrame, model: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "active":
        y = w["label_active"].to_numpy()
    elif task == "exercise":
        y = w["label_exercise"].to_numpy()
    else:
        raise ValueError("task must be 'active' or 'exercise'")
    if model == "energy":
        if task != "active":
            raise ValueError("the energy gate is binary: use --task active")
        X = loso.energy_features(w)
    elif model == "rf":
        X = loso.feature_matrix(w)
    else:
        raise ValueError(f"model {model!r} is not available in this PR (CNN lands in PR 6)")
    return X, y


def _factory(model: str) -> Callable[[], object]:
    if model == "energy":
        from formcoach.models.energy import EnergyGate

        return EnergyGate
    if model == "rf":
        from formcoach.models.rf import make_rf

        return lambda: make_rf(seed=report.SEED)
    raise ValueError(model)


def _load(dataset: str, task: str, min_purity: float) -> pd.DataFrame:
    w = build.load_windows(convert.PROCESSED_ROOT, dataset, min_purity=min_purity)
    loso.check_units(w)
    if task == "exercise":
        w = w[w["label_active"] == 1].reset_index(drop=True)
    return w


def run(
    *,
    model: str,
    dataset: str,
    task: str,
    folds: str = "loso",
    min_purity: float = 0.8,
    test_dataset: str | None = None,
    out: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    w = _load(dataset, task, min_purity)
    X, y = _xy(w, model, task)
    meta = {
        "model": model,
        "task": task,
        "dataset": dataset,
        "windows": f"{len(w):,} (label purity ≥ {min_purity})",
        "subjects": int(w["subject"].nunique()),
        "class balance": ", ".join(f"{k}: {v:,}" for k, v in pd.Series(y).value_counts().items()),
    }
    if test_dataset is None:
        folds_arg: str | int = "loso" if folds == "loso" else int(folds)
        meta["folds"] = (
            "leave-one-subject-out" if folds_arg == "loso" else f"{folds_arg} subject-grouped folds"
        )
        res = loso.run_loso(X, y, w["subject"].to_numpy(), _factory(model), folds=folds_arg,
                            progress=lambda f: log(f"  fold {f}"))  # fmt: skip
        path = out or report.REPORTS_DIR / f"loso_{model}_{task}_{dataset}.md"
        title = f"LOSO · {model} · {task} · {dataset}"
        command = (
            f"formcoach eval loso --model {model} --dataset {dataset} --task {task} --folds {folds}"
        )
    else:
        wt = _load(test_dataset, task, min_purity)
        if wt["units"].iloc[0] != w["units"].iloc[0]:
            raise ValueError("cannot transfer between si and normalized datasets (ADR-0014)")
        Xt, yt = _xy(wt, model, task)
        m = _factory(model)()
        m.fit(X, y)
        pred = np.asarray(m.predict(Xt))
        labels = sorted(set(y.tolist()) | set(yt.tolist()))
        per = []
        subjects = wt["subject"].to_numpy()
        for s in sorted(set(subjects)):
            mk = subjects == s
            per.append(
                {
                    "fold": s,
                    "n_test": int(mk.sum()),
                    "accuracy": float(accuracy_score(yt[mk], pred[mk])),
                    "macro_f1": float(
                        f1_score(yt[mk], pred[mk], labels=labels, average="macro", zero_division=0)
                    ),
                }
            )
        pooled = {
            "accuracy": float(accuracy_score(yt, pred)),
            "macro_f1": float(f1_score(yt, pred, labels=labels, average="macro", zero_division=0)),
            "n": len(yt),
            "folds": len(per),
        }
        res = loso.LosoResult(
            pd.DataFrame(per),
            pooled,
            confusion_matrix(yt, pred, labels=labels),
            [str(lab) for lab in labels],
            yt,
            pred,
            subjects,
        )
        meta["test dataset"] = (
            f"{test_dataset} ({len(wt):,} windows, {wt['subject'].nunique()} subjects)"
        )
        meta["folds"] = "train on all of --dataset, test per subject of --test-dataset"
        path = out or report.REPORTS_DIR / f"transfer_{model}_{task}_{dataset}-to-{test_dataset}.md"
        title = f"Cross-dataset · {model} · {task} · {dataset} → {test_dataset}"
        command = (f"formcoach eval loso --model {model} --dataset {dataset} --task {task} "
                   f"--test-dataset {test_dataset}")  # fmt: skip
    return loso.write_loso_report(res, path, title=title, command=command, meta=meta)
