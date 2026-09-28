# How to: train the CNN gate, export it to the firmware, and evaluate float vs int8

**Owner:** Christian Byars · **Built by:** Claude Code with Robert on 2026-09-27 · **PR:** #8
**Checkpoint:** 5 (`docs/00-START-HERE.md`) · **State:** trained on real data (GPU), exported, firmware compiles; **on-device inference not measured** (no board)

## 1. Run it (verified 2026-09-27 on Linux/WSL2 with an RTX 3070 eGPU)

Install the training extra (CPU) or the GPU variant (Linux + NVIDIA):

```bash
uv sync --group dev --extra train          # CPU: works everywhere, ~1 core, slow (hours)
uv sync --group dev --extra train-gpu      # Linux + NVIDIA: TensorFlow with CUDA wheels
```

On WSL2 the pip nvJitLink library must be found before the system one, or TensorFlow
silently falls back to CPU (`GPUS []`):

```bash
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Train on RecoFit + MM-Fit windows (needs `make convert && make features` first), export int8:

```bash
make train-gate MODEL=cnn        # = uv run formcoach train gate --model cnn
```
```
training CNN on 371,929 windows from 104 subjects
  windows: 371929  subjects: 104  val_subjects: 16  epochs_run: 30  params: 6390
  val_macro_f1_float: 0.7076  tflite_bytes: 16760  val_macro_f1_int8: 0.7299
  float_to_int8_macro_f1_drop: -0.0224
  report: reports/gate_cnn_export.md            (23 min on the RTX 3070)
```

Unseen-subject evaluation (5 subject-grouped folds, 10 epochs per fold; writes both the float
and the int8 table from one training per fold):

```bash
uv run formcoach eval loso --model cnn --dataset recofit --task active --folds 5 --epochs 10
uv run formcoach eval loso --model rf --dataset recofit --task active --folds 5 --out reports/loso_rf_active_recofit_5fold.md
```

Build the firmware with the model (TFLite Micro, Chirale library):

```bash
make fw-build
```
```
RAM:   [==        ]  22.3% (used 73020 bytes from 327680 bytes)   # includes the 40 KB arena
Flash: [==        ]  19.2% (used 642745 bytes from 3342336 bytes)
```

## 2. Where things live

| Path | What it is |
|---|---|
| `src/formcoach/models/cnn.py` | `build_model`, `Preprocess` (fit/apply/save/load), `augment`, `train(X, y, groups, ...)`, `predict`, `CnnClassifier` (LOSO wrapper, float / int8 / both), `CLASSES6` |
| `src/formcoach/models/export.py` | `to_tflite_int8(model, representative, out)`, `to_tflite_float`, `tflite_predict(path, X)`, `quant_params`, `write_cc`, `write_preprocess_header` |
| `src/formcoach/models/train.py` | `train_cnn_gate(...)` (what `train gate --model cnn` runs), `train_rf_gate` |
| `src/formcoach/eval/loso_cli.py`, `eval/loso.py` | `--model cnn|cnn-int8`; `result_from_predictions` builds the int8 table from the same folds |
| `src/formcoach/eval/device.py` | `eval device --log <serial log>` → `reports/device.md` (inference ms, arena, flash) |
| `firmware/model/` | `gate_model.tflite` (int8, 16,760 B), `gate_model_float.tflite`, `gate_model_data.cc`, `preprocess.h`, `preprocess.json` — generated, committed |
| `firmware/src/app/gate.{h,cpp}`, `model_data.cpp` | `Gate::begin()` allocates the TFLM interpreter (40 KB arena); `decideModel()` standardises + quantises the window and runs it; falls back to the energy rule if the model fails to load |
| `firmware/platformio.ini` | env `xiao_esp32s3` (CNN) and `xiao_esp32s3_energy` (`-DFC_GATE_ENERGY=1`, no TFLM) |
| `reports/gate_cnn_export.md`, `reports/loso_cnn*_active_recofit.md`, `reports/loso_rf_active_recofit_5fold.md` | the numbers |
| `tests/test_cnn.py` | model shape/size, preprocess round trip, augmentation invariants, train→int8 agreement, `.cc`/`.h` writers, LOSO plumbing (skipped without TensorFlow) |

## 3. How it works

One 6-class model (`idle, curl, press, raise, squat, other`) over 2 s / 50 Hz / 6-channel
windows serves both the gate (`active = argmax != idle`) and exercise recognition. Windows are
standardised per channel with training-set constants (`preprocess.json`); the firmware applies
the same constants to its LSB samples (`preprocess.h`: mean/std in SI, then the int8
quantisation scale/zero-point read from the TFLite model). Training uses class weights, Adam,
early stopping on held-out *subjects*, and augmentation (random 3-D rotation ±30° of the
accel and gyro triplets, time-warp ±10 %, jitter 0.05 g, scaling ±10 %). Post-training int8
quantisation uses 500 representative windows; `tflite_predict` runs the quantised model on the
laptop so the float→int8 drop is reported without a board. `write_cc` emits the flatbuffer as
an `extern const` array (a plain `const` would be file-local in C++ and fail to link).

## 4. Tests

`uv run pytest tests/test_cnn.py` (5 tests; ~1 min on CPU; auto-skipped without TensorFlow).
`make fw-build` proves the model links and fits. Change `CLASSES6`, the window layout or the
preprocess constants only together with a retrain, a re-export and a firmware rebuild.

## 5. Could not verify / open questions

- On-device inference time, arena head-room and the op resolver (Conv2D/MaxPool/FullyConnected/
  Reshape/Softmax/Mean/ExpandDims/Quantize) are compile-checked only; the first board run prints
  `# gate: tflm ok arena=…` and `# infer us=…` lines → `formcoach eval device --log`.
- The full-data model ran all 30 epochs (still improving); the LOSO folds use 10 epochs for
  GPU time. Longer training is a cheap experiment now that the GPU works.
- GPU results are not bit-reproducible run to run (cuDNN); the seed fixes the data split.

## 6. Next steps for Christian

1. Flash `xiao_esp32s3`, capture the serial log for a minute of curls, run `eval device`.
2. Compare `reports/loso_cnn_active_recofit.md` / `loso_cnn-int8_…` with
   `loso_rf_active_recofit_5fold.md` in the report; decide whether the CNN or the RF gate ships
   on the device (ADR).
3. Exercise-class CNN (`--task exercise`) and the MM-Fit-only LOSO are one command each.
