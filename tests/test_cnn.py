"""Keras 1D-CNN gate (docs/02 §6.2), preprocessing, augmentation, int8 export (§6.3).
Skipped without the `train` extra (TensorFlow)."""

from __future__ import annotations

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from formcoach.models import cnn, export  # noqa: E402

CLASSES = ("idle", "curl", "squat")


def _data(n_per: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(100) / 50.0
    X, y, g = [], [], []
    for label in CLASSES:
        for i in range(n_per):
            x = rng.normal(0, 0.1, (100, 6))
            x[:, 2] += 9.81
            if label == "curl":
                x[:, 0] += 3 * np.sin(2 * np.pi * 1.0 * t)
            elif label == "squat":
                x[:, 0] += 4 * np.sin(2 * np.pi * 0.5 * t)
                x[:, 4] += 1.5 * np.cos(2 * np.pi * 0.5 * t)
            X.append(x.astype(np.float32))
            y.append(label)
            g.append(f"S{i % 4}")
    return np.stack(X), np.array(y), np.array(g)


def test_model_is_small_and_has_the_design_shape():
    m = cnn.build_model(n_classes=6, seed=0)
    assert m.count_params() < 20_000
    assert m.output_shape == (None, 6) and m.input_shape == (None, 100, 6)
    layer_types = [type(layer).__name__ for layer in m.layers]
    assert layer_types.count("Conv1D") == 3 and "GlobalAveragePooling1D" in layer_types


def test_preprocess_fit_apply_and_json_round_trip(tmp_path):
    X, _, _ = _data(20)
    pre = cnn.Preprocess.fit(X)
    Z = pre.apply(X)
    assert Z.shape == X.shape and abs(Z.mean()) < 0.05 and abs(Z.std() - 1) < 0.1
    p = tmp_path / "preprocess.json"
    pre.save(p, classes=CLASSES)
    doc = json.loads(p.read_text())
    assert doc["window"] == 100 and doc["channels"] == 6 and doc["classes"] == list(CLASSES)
    assert len(doc["mean"]) == 6 and len(doc["std"]) == 6 and doc["fs"] == 50
    pre2 = cnn.Preprocess.load(p)
    assert np.allclose(pre2.apply(X), Z)


def test_augment_keeps_shapes_and_acceleration_magnitude():
    X, _, _ = _data(10)
    rng = np.random.default_rng(1)
    A = cnn.augment(X, rng, rotate_deg=30.0, time_warp=0.0, jitter_g=0.0, scale=0.0)
    assert A.shape == X.shape
    assert np.allclose(
        np.linalg.norm(A[:, :, :3], axis=2), np.linalg.norm(X[:, :, :3], axis=2), atol=1e-3
    )
    assert not np.allclose(A, X)
    B = cnn.augment(X, rng)
    assert B.shape == X.shape and np.isfinite(B).all()


def test_train_predict_and_int8_export_agree(tmp_path):
    X, y, groups = _data(80)
    res = cnn.train(X, y, groups, classes=CLASSES, seed=0, epochs=15, batch_size=32, augment=False)
    assert res.history["val_accuracy"][-1] > 0.85
    pred = cnn.predict(res.model, res.preprocess, X)
    acc = (np.array(CLASSES)[pred] == y).mean()
    assert acc > 0.9
    tfl = export.to_tflite_int8(res.model, res.preprocess.apply(X[:200]), tmp_path / "m.tflite")
    assert tfl.exists() and tfl.stat().st_size < 60_000
    pred_q = export.tflite_predict(tfl, res.preprocess.apply(X))
    assert (pred_q == pred).mean() > 0.9
    cc = export.write_cc(tfl, tmp_path / "gate_model_data.cc")
    text = cc.read_text()
    assert "extern const unsigned char gate_model_data[] = {" in text
    assert "extern const size_t gate_model_data_len" in text
    assert "alignas(16)" in text
    h = export.write_preprocess_header(res.preprocess, CLASSES, tmp_path / "preprocess.h", tfl)
    ht = h.read_text()
    assert "FC_PRE_MEAN" in ht and "FC_PRE_STD" in ht and "FC_CLASS_IDLE" in ht
    assert "FC_INPUT_SCALE" in ht and "FC_INPUT_ZERO_POINT" in ht


def test_cnn_loso_factory_runs_two_grouped_folds():
    from formcoach.eval import loso

    X, y, groups = _data(60)
    store: list = []
    make = cnn.loso_factory(
        classes=CLASSES, seed=0, epochs=12, batch_size=32, augment=False, int8="both",
        int8_predictions=store,
    )  # fmt: skip
    res = loso.run_loso(X, y, groups, make, folds=2)
    assert len(res.per_fold) == 2 and res.pooled["accuracy"] > 0.5  # plumbing, not accuracy
    assert set(res.labels) == set(CLASSES)
    assert len(store) == 2  # one int8 prediction array per fold
    res8 = loso.result_from_predictions(y, groups, store, folds=2)
    assert res8.pooled["n"] == len(y)
    assert abs(res8.pooled["accuracy"] - res.pooled["accuracy"]) < 0.3
