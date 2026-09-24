"""Leave-one-subject-out evaluation and its report (docs/02 §8, `formcoach eval loso`).

Headline numbers are unseen-subject: every fold's test subjects never appear in its training
set. ``loso_splits`` is strict leave-one-subject-out; ``group_kfold_splits`` groups subjects
into ``n`` folds (still unseen-subject, cheaper for models that train slowly). Reports carry
the seed, the fold list, per-fold accuracy / macro-F1, pooled metrics and the pooled confusion
matrix (table + PNG).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from formcoach.eval import report
from formcoach.signal.features import FEATURE_NAMES
from formcoach.signal.windows import windows_to_array

SEED = report.SEED


@dataclass
class LosoResult:
    per_fold: pd.DataFrame  # fold, n_test, accuracy, macro_f1
    pooled: dict  # accuracy, macro_f1, n
    confusion: np.ndarray
    labels: list[str]
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    groups: np.ndarray = field(repr=False)


def loso_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """``[(train_idx, test_idx, subject), …]`` — one fold per unique subject, sorted."""
    groups = np.asarray(groups)
    out = []
    for g in sorted(set(groups.tolist())):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        out.append((train, test, str(g)))
    return out


def group_kfold_splits(
    groups: np.ndarray, n_folds: int, seed: int = SEED
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Subjects shuffled (seeded) and dealt into ``n_folds`` folds; one test fold per subject."""
    groups = np.asarray(groups)
    subjects = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    out = []
    for k in range(n_folds):
        held = set(subjects[k::n_folds])
        mask = np.isin(groups, list(held))
        out.append((np.flatnonzero(~mask), np.flatnonzero(mask), f"fold{k}"))
    return out


def check_units(w: pd.DataFrame) -> None:
    """Refuse to train on a mix of SI and normalised windows (ADR-0014)."""
    units = set(w["units"].unique())
    if len(units) > 1:
        raise ValueError(f"windows mix units {units}; never combine normalized with si data")


def feature_matrix(w: pd.DataFrame) -> np.ndarray:
    return w[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)


def energy_features(w: pd.DataFrame) -> np.ndarray:
    """``(n, 1)``: variance of |a| per window, computed from the raw ``x`` column."""
    x = windows_to_array(w)
    amag = np.linalg.norm(x[:, :, :3], axis=2)
    return amag.var(axis=1, ddof=0)[:, None]


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    make_model: Callable[[], object],
    folds: str | int = "loso",
    seed: int = SEED,
    progress: Callable[[str], None] | None = None,
) -> LosoResult:
    """Fit a fresh ``make_model()`` per fold; returns per-fold and pooled metrics."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    splits = (
        loso_splits(groups) if folds == "loso" else group_kfold_splits(groups, int(folds), seed)
    )
    labels = sorted(set(y.tolist()))
    y_pred = np.empty_like(y)
    rows = []
    for train, test, name in splits:
        if progress:
            progress(name)
        model = make_model()
        model.fit(X[train], y[train])
        pred = np.asarray(model.predict(X[test]))
        y_pred[test] = pred
        rows.append(
            {
                "fold": name,
                "n_test": len(test),
                "accuracy": float(accuracy_score(y[test], pred)),
                "macro_f1": float(
                    f1_score(y[test], pred, labels=labels, average="macro", zero_division=0)
                ),
            }
        )
    per_fold = pd.DataFrame(rows)
    pooled = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)),
        "n": len(y),
        "folds": len(splits),
    }
    cm = confusion_matrix(y, y_pred, labels=labels)
    return LosoResult(per_fold, pooled, cm, [str(lab) for lab in labels], y, y_pred, groups)


def _fig_confusion(res: LosoResult, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = res.confusion.astype(float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(res.labels), 1.0 + 0.9 * len(res.labels)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(res.labels)), res.labels, rotation=45, ha="right")
    ax.set_yticks(range(len(res.labels)), res.labels)
    for i in range(len(res.labels)):
        for j in range(len(res.labels)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")  # fmt: skip
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_loso_report(
    res: LosoResult, path: Path, *, title: str, command: str, meta: dict | None = None
) -> Path:
    """Markdown report + confusion PNG under ``<path.parent>/figures/``."""
    fig = path.parent / "figures" / f"{path.stem}_confusion.png"
    _fig_confusion(res, fig, title)
    cm = pd.DataFrame(res.confusion, index=res.labels, columns=res.labels)
    cm.index.name = "true \\ predicted"
    parts = [
        report.header(title, command, extra=meta),
        "## Pooled (unseen-subject)",
        "",
        f"- accuracy **{res.pooled['accuracy']:.3f}** · macro-F1 "
        f"**{res.pooled['macro_f1']:.3f}** · windows {res.pooled['n']:,} · "
        f"folds {res.pooled['folds']}",
        f"- per-fold macro-F1: mean {res.per_fold['macro_f1'].mean():.3f}, "
        f"median {res.per_fold['macro_f1'].median():.3f}, min {res.per_fold['macro_f1'].min():.3f}",
        "",
        "## Confusion matrix (pooled over folds)",
        "",
        report.md_table(cm, index=True),
        "",
        report.relative_figure(fig, path),
        "",
        "## Per fold",
        "",
        report.md_table(res.per_fold),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
