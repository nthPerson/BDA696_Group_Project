"""Keras 1D-CNN gate / exercise model (docs/02 §6.1–6.2). Needs the ``train`` extra.

Input: 2 s windows at 50 Hz, 6 channels (ax..gz in SI), per-channel standardised with
constants from the training set (:class:`Preprocess`, saved to ``preprocess.json`` and compiled
into the firmware). Architecture: Conv1D(16,5)-ReLU-MaxPool(2)-Conv1D(32,5)-ReLU-MaxPool(2)-
Conv1D(32,3)-ReLU-GlobalAvgPool-Dense(n_classes), ~10 k parameters. One multi-class model over
``idle, curl, press, raise, squat, other`` serves both the gate (``active = argmax != idle``) and
exercise recognition. Augmentation: random 3-D rotation of the accel/gyro triplets (±30°),
time-warp (±10 %), jitter (σ = 0.05 g), magnitude scaling (±10 %).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WINDOW, CHANNELS, FS = 100, 6, 50
CLASSES6 = ("idle", "curl", "press", "raise", "squat", "other")
G = 9.80665


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("the CNN gate needs TensorFlow: `uv sync --extra train`") from exc
    return tf


@dataclass
class Preprocess:
    mean: np.ndarray
    std: np.ndarray
    window: int = WINDOW
    channels: int = CHANNELS
    fs: int = FS

    @classmethod
    def fit(cls, X: np.ndarray) -> Preprocess:
        X = np.asarray(X, dtype=np.float32)
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
        return cls(mean.astype(np.float32), std.astype(np.float32), X.shape[1], X.shape[2])

    def apply(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path, classes: tuple[str, ...]) -> Path:
        doc = {
            "window": self.window,
            "channels": self.channels,
            "fs": self.fs,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "classes": list(classes),
            "units": "ax..az m/s^2, gx..gz rad/s (SI); standardised = (x - mean) / std",
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> Preprocess:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(doc["mean"], np.float32), np.array(doc["std"], np.float32),
                   doc["window"], doc["channels"], doc.get("fs", FS))  # fmt: skip


def build_model(n_classes: int, window: int = WINDOW, channels: int = CHANNELS, seed: int = 0):
    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    L = tf.keras.layers
    return tf.keras.Sequential(
        [
            L.Input(shape=(window, channels)),
            L.Conv1D(16, 5, padding="same", activation="relu"),
            L.MaxPool1D(2),
            L.Conv1D(32, 5, padding="same", activation="relu"),
            L.MaxPool1D(2),
            L.Conv1D(32, 3, padding="same", activation="relu"),
            L.GlobalAveragePooling1D(),
            L.Dense(n_classes, activation="softmax"),
        ],
        name="formcoach_gate_cnn",
    )


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def augment(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    rotate_deg: float = 30.0,
    time_warp: float = 0.1,
    jitter_g: float = 0.05,
    scale: float = 0.1,
) -> np.ndarray:
    """Augment raw SI windows ``(n, window, 6)``; returns a new array."""
    X = np.asarray(X, dtype=np.float32)
    n, w, _ = X.shape
    out = np.empty_like(X)
    grid = np.linspace(0, 1, w)
    for i in range(n):
        x = X[i]
        if rotate_deg > 0:
            R = _rotation(rng, rotate_deg).astype(np.float32)
            x = np.concatenate([x[:, :3] @ R.T, x[:, 3:] @ R.T], axis=1)
        if time_warp > 0:
            f = 1.0 + rng.uniform(-time_warp, time_warp)
            src = np.clip(grid * f, 0, 1)
            x = np.column_stack([np.interp(src, grid, x[:, c]) for c in range(x.shape[1])])
        if scale > 0:
            x = x * (1.0 + rng.uniform(-scale, scale))
        if jitter_g > 0:
            x = x + np.column_stack(
                [rng.normal(0, jitter_g * G, (w, 3)), rng.normal(0, jitter_g, (w, 3))]
            )
        out[i] = x
    return out


@dataclass
class TrainResult:
    model: object
    preprocess: Preprocess
    classes: tuple[str, ...]
    history: dict = field(default_factory=dict)
    val_subjects: list[str] = field(default_factory=list)


def _class_weights(y_idx: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    w = counts.sum() / (n_classes * counts)
    return {i: float(w[i]) for i in range(n_classes)}


def train(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    classes: tuple[str, ...] = CLASSES6,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 256,
    augment: bool = True,
    val_frac: float = 0.15,
    patience: int = 5,
    verbose: int = 0,
) -> TrainResult:
    """Train with early stopping on a held-out subject subset (random windows if ``groups`` is
    None); class weights balance the labels. ``y`` holds class names."""
    tf = _tf()
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([idx[v] for v in y], dtype=np.int64)
    n = len(X)
    if groups is not None:
        subjects = sorted(set(np.asarray(groups).tolist()))
        rng.shuffle(subjects)
        n_val = max(1, round(len(subjects) * val_frac))
        val_subjects = set(subjects[:n_val])
        val_mask = np.isin(groups, list(val_subjects))
    else:
        val_subjects = set()
        val_mask = rng.random(n) < val_frac
    if val_mask.all() or not val_mask.any():
        val_mask = np.zeros(n, bool)
        val_mask[rng.choice(n, max(1, n // 10), replace=False)] = True
    pre = Preprocess.fit(X[~val_mask])
    X_tr = X[~val_mask]
    y_tr = y_idx[~val_mask]
    if augment:
        X_tr = np.concatenate([X_tr, globals()["augment"](X_tr, rng)])
        y_tr = np.concatenate([y_tr, y_tr])
    model = build_model(len(classes), X.shape[1], X.shape[2], seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])  # fmt: skip
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    hist = model.fit(
        pre.apply(X_tr), y_tr,
        validation_data=(pre.apply(X[val_mask]), y_idx[val_mask]),
        epochs=epochs, batch_size=batch_size, class_weight=_class_weights(y_tr, len(classes)),
        callbacks=[es], verbose=verbose, shuffle=True,
    )  # fmt: skip
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return TrainResult(model, pre, tuple(classes), history, sorted(val_subjects))


def predict(model, pre: Preprocess, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Class indices (argmax) for raw SI windows."""
    p = model.predict(pre.apply(X), batch_size=batch_size, verbose=0)
    return np.argmax(p, axis=1)


class CnnClassifier:
    """sklearn-like wrapper so :func:`formcoach.eval.loso.run_loso` can fit a CNN per fold.

    ``int8=True`` quantises after training and predicts through the TFLite interpreter;
    ``int8="both"`` predicts float but also quantises and stores the int8 predictions of every
    ``predict`` call in ``int8_predictions`` (a list shared across folds when passed in), so one
    training per fold yields both the float and the int8 LOSO tables."""

    def __init__(
        self, classes, seed=0, epochs=30, batch_size=256, augment=True, int8=False,
        int8_predictions: list | None = None,
    ):  # fmt: skip
        self.classes, self.seed, self.epochs = tuple(classes), seed, epochs
        self.batch_size, self.augment, self.int8 = batch_size, augment, int8
        self.int8_predictions = int8_predictions
        self.result: TrainResult | None = None
        self.tflite: Path | None = None

    def fit(self, X, y):
        self.result = train(X, y, None, classes=self.classes, seed=self.seed, epochs=self.epochs,
                            batch_size=self.batch_size, augment=self.augment)  # fmt: skip
        if self.int8:
            import tempfile

            from formcoach.models import export

            rep = self.result.preprocess.apply(X[: min(len(X), 500)])
            self.tflite = export.to_tflite_int8(
                self.result.model, rep, Path(tempfile.mkdtemp()) / "fold.tflite"
            )
        return self

    def predict(self, X):
        assert self.result is not None
        from formcoach.models import export

        z = self.result.preprocess.apply(X)
        if self.int8 == "both":
            idx8 = export.tflite_predict(self.tflite, z)
            if self.int8_predictions is not None:
                self.int8_predictions.append(np.array(self.classes)[idx8])
            idx = predict(self.result.model, self.result.preprocess, X)
        elif self.int8 and self.tflite is not None:
            idx = export.tflite_predict(self.tflite, z)
        else:
            idx = predict(self.result.model, self.result.preprocess, X)
        return np.array(self.classes)[idx]


def loso_factory(
    classes, seed=0, epochs=30, batch_size=256, augment=True, int8=False, int8_predictions=None
):
    return lambda: CnnClassifier(classes, seed, epochs, batch_size, augment, int8, int8_predictions)
