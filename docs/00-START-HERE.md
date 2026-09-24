# 00 — Start here: bootstrap plan for the first sessions

Today is late September 2026. Parts are ordered and arrive in roughly one or two days. Nothing below waits on them. Work through the checkpoints in order; each one ends with something runnable and a PR. Create `docs/STATUS.md` in your first session and keep it updated.

## Checkpoint 0 — Repository skeleton (first session)

- Initialize the layout in `CLAUDE.md`. `uv init`, Python 3.11, package `formcoach` with a `typer` CLI (`formcoach --help`).
- `Makefile` with `setup`, `lint`, `test`, `data`, `features`, `train-gate`, `eval`, `demo`, `record` targets (stubs are fine; each must print what it will do).
- `ruff`, `pytest`, `pre-commit`, GitHub Actions CI (lint + tests on Ubuntu; a second job runs `pio run` in `firmware/`).
- `README.md` quickstart: clone → `make setup` → `make demo` (replay mode) in under five minutes on Linux, macOS and Windows. Windows notes: use PowerShell or Git Bash; `uv` handles the interpreter; serial ports are `COMx`.
- `data/MANIFEST.md` template, `.gitignore` for `data/external`, `data/processed`, `data/team`, `*.mp4`, `*.npy`, `*.mat`, `*.parquet` (fixtures excepted).
- `docs/STATUS.md`, `docs/DECISIONS.md` (first entries: toolchain choices from `CLAUDE.md`).

## Checkpoint 1 — Public data loads (Phase 1, week 1)

- `formcoach data fetch --dataset mmfit|recofit|recgym` downloads into `data/external/<name>/`, verifies SHA-256, records the entry in `data/MANIFEST.md`. MM-Fit sensor/pose data first; **do not** download the 40.8 GB Zenodo video by default (`--with-video` flag, per-session).
- Loaders in `src/formcoach/data/` return the common schema in `docs/02-system-design.md` §5. Each loader has a `describe()` that prints subjects, sessions, sampling rates, label coverage, and gaps, and a test on a tiny committed fixture (`data/fixtures/`).
- `formcoach data profile` writes `reports/data_profile.md` (tables + a few matplotlib figures): per-dataset subjects, minutes, exercise distribution, sampling-rate checks, missing/duplicate timestamps.
- Update `docs/04-datasets.md` with anything the real files contradict (file names, columns, units). Treat that file's format notes as *expected, verify*.

## Checkpoint 2 — Signal pipeline and baselines (Phase 1, week 2)

- `formcoach features build`: resample to 50 Hz, low-pass, gravity separation, 2 s windows with 50 % overlap, per-window features, Parquet per subject-session under `data/processed/`.
- Leave-one-subject-out (LOSO) split utility; `formcoach eval loso --model rf` trains a random-forest baseline on RecoFit for exercise-vs-idle and exercise class; writes `reports/baseline_rf.md` with accuracy, macro-F1, confusion matrix.
- Peak-detection rep counter on MM-Fit smartwatch streams; `reports/baseline_repcount.md` with MAE per exercise.
- These two reports are the first real results; they go into the Written Report Outline.

## Checkpoint 3 — Pose pipeline in replay mode (Phase 1, week 2)

- `formcoach pose extract --video <file>`: OpenCV chunked read → MediaPipe PoseLandmarker (Lite) → `pose.parquet` (frame, t, 33 × (x, y, z, visibility) image coords + 33 × world coords). Checkpointed per session; re-runs skip finished sessions.
- Joint-angle module (`pose/angles.py`) with unit tests against hand-computed angles.
- Rep segmentation from the primary joint angle; cross-check against MM-Fit rep counts on the sessions that have video (download two or three sessions only).
- `formcoach demo --source replay --session <id>`: replays IMU + pose from files through the whole pipeline with the overlay window (or `--headless` printing rep/fault events). This is the demo that must always work.

## Checkpoint 4 — Firmware v0 (Phase 2, as soon as boards arrive)

- PlatformIO project builds in CI before any hardware exists. Target `seeed_xiao_esp32s3`, Arduino framework, NimBLE-Arduino.
- v0 firmware: read BMI160 at 50 Hz, print CSV over USB serial; `formcoach record --source serial --port <port>` logs to `data/team/`.
- v1 firmware: BLE GATT service from `docs/02-system-design.md` §3; `formcoach record --source ble` and `formcoach demo --source ble`.
- Wiring guide `docs/wiring-guide.md` with a Mermaid pin map and photo placeholders; Robert fills in photos after the build session.

## Checkpoint 5 — Gate model on device (Phase 2)

- Keras 1D-CNN gate (§6) trained on RecoFit + MM-Fit windows; LOSO metrics vs. the RF baseline; int8 post-training quantization with a representative dataset; accuracy drop reported.
- Export to `firmware/model/gate_model_data.cc`; on-device inference at 2 Hz (one window per 0.5 s); measure latency, RAM, flash; hysteresis (§6.4).
- `formcoach demo --source ble` now opens/closes the camera from the gate flag; log frames processed vs. available.

## Checkpoint 6 — Rules, evaluation harness, freeze (Phase 3)

- `rules/rules.yaml` with per-exercise thresholds (§7 defaults); `formcoach eval rules` runs the correct-form pass-rate and synthetic-perturbation tests on MM-Fit pose sequences.
- `formcoach eval gating`: gated vs always-on on identical recorded sessions (frames processed, CPU %, wall time; power via OS counters where available).
- Pose-model ablation (Lite/Full/Heavy, YOLO11n-pose optional): accuracy of rep segmentation, FPS, CPU.
- Team validation recordings (about one hour total; protocol in `docs/04-datasets.md` §6) → sensor-transfer report.
- Feature freeze; `reports/` regenerated by `make eval`; report and slide drafting begins.

## What "done" looks like

See `docs/05-roadmap.md` §5. In one sentence: a fresh clone runs `make setup && make demo` on public data, five wearables run the same firmware and stream to the laptop app, every claim in the final report is produced by a script in this repo, and the whole thing is explained in a README a teammate can follow.
