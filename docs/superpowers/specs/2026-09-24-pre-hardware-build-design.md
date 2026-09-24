# Pre-hardware build plan (design note, 2026-09-24)

Written by Claude Code with Robert. Answers two questions: *where do we start the software
build*, and *exactly how much of FormCoach can be built before the boards and case exist*.
The build specification stays `docs/02-system-design.md`; this note only sequences it.

## 1. Understanding (correct me here)

- **Intent:** start implementing code, firmware and software now, in advance of the parts
  (arriving 2026-09-25) and the case, so that hardware never blocks software (rule 1).
- **Constraints:** frozen v1 scope (four dumbbell exercises, rule-based feedback); one task,
  one branch, one PR; everything runs from a fresh clone on Linux/macOS/Windows; public data
  first; headline metrics are leave-one-subject-out.
- **Success for this stretch of work:** `make demo` (replay, headless) runs the real pipeline
  on public data; the Checkpoint 1–2 reports exist so the Written Report Outline can cite
  them; firmware v1 compiles in CI and is ready to flash the day a board is wired.

## 2. What can be built without hardware

| Checkpoint | Buildable now (no hardware) | Needs a board / team recordings |
|---|---|---|
| 1 Data | `data fetch` for MM-Fit, RecoFit, RecGym with SHA-256 + manifest rows; the three loaders → `IMUStream` schema; `describe()`; fixtures; `data profile` → `reports/data_profile.md` | nothing |
| 2 Signal + baselines | resample/filter/gravity/windows/features; LOSO split; energy + RF baselines; `eval loso`; peak rep counter; `eval repcount`; `reports/baseline_*.md` | nothing |
| 3 Pose replay | `pose/angles.py` (pure NumPy, unit-tested); `pose/reps.py`; `ReplaySource`; `demo --source replay --headless` on a 30 s fixture; integration test; `pose extract` wrapper (lazy MediaPipe import; exercised on MM-Fit video only if a session is downloaded) | webcam path is additive and untested until someone runs it |
| 4 Firmware + sources | firmware v0/v1 **written and compiling in CI**: BMI160 driver, 50 Hz sampling, ring buffer, CSV over USB serial, BLE service (IMU notify, control write, status), button/LED behaviour; Python `SerialSource`, `BLESource`, `SessionRecorder`, `record`, `session check` (tested against fakes that emit `protocol.py` packets) | flashing, I2C address, driver choice **verified**, BLE end-to-end with `bleak`, MTU behaviour, LED polarity, `docs/devices.md` rows |
| 5 Gate on device | Keras 1D-CNN gate, LOSO vs RF, int8 post-training quantization, float→int8 drop, `firmware/model/gate_model_data.cc`, `preprocess.json` | on-device inference ms / arena / flash; camera open/close from the device flag |
| 6 Rules + eval | `rules.yaml` + engine + perturbation unit tests; `eval rules` on MM-Fit `pose_3d`; `eval gating` for `always_on / energy / laptop` on replayed sessions | `device` gate benchmark; team validation recordings; `eval transfer`, `eval latency` (live), battery test |

So: **everything in Phase 1, all of the code in Phase 2, and roughly half of Phase 3**
can be built now. What remains for hardware day is verification and measurement, not code.

## 3. Where to start, and in what order

Start with **Checkpoint 1** because every later stage consumes the `IMUStream` schema and the
subject ids that LOSO needs; firmware v1 can be written in parallel since it only has to
compile. Each row below is one branch and one PR, stacked in order (each PR targets the
previous branch; GitHub retargets to `main` as they merge).

| # | Branch | Delivers | Tests / reports |
|---|---|---|---|
| 1 | `feat/checkpoint-1-data` | `data/fetch.py` (registry, resumable download, SHA-256, zip extract, manifest rows), `schema.py`, `manifest.py`, `mmfit.py`, `recofit.py`, `recgym.py`, fixtures, `data profile` | loader tests on fixtures; `reports/data_profile.md` |
| 2 | `feat/checkpoint-2-signal` | `signal/{resample,filters,gravity,windows,features,reps}.py`, `eval/loso.py`, `models/{energy,rf}.py`, `eval/repcount.py` | windowing/angle math unit tests; `reports/baseline_rf.md`, `reports/baseline_repcount.md` |
| 3 | `feat/checkpoint-3-replay` | `pose/{angles,normalize,reps,landmarker}.py`, `io/replay.py`, `app/pipeline.py` (headless), 30 s fixture, `demo --source replay` | hand-computed angle tests; integration test = the CI smoke test |
| 4 | `fw/checkpoint-4-firmware-v1` | firmware v1 (BMI160 + CSV + BLE), `io/{serial,ble,recorder}.py`, `record`, `session check`, `fw-upload PORT=` | `pio run` in CI; packet round-trips through fake sources |
| 5 | `feat/checkpoint-6-rules` | `rules/rules.yaml`, `rules/engine.py`, `eval rules` | perturbation tests; `reports/rules_validation.md` |
| 6 | `feat/checkpoint-5-cnn-gate` | `models/{cnn,export}.py`, `train gate`, int8 export, `.cc` | LOSO table float vs int8; `reports/loso_cnn.md` |

Rules (5) is placed before the CNN (6) because it needs no heavy extra and unblocks the
vision go/no-go evidence earlier.

## 4. Facts established this session (fold into `docs/04-datasets.md` in PR 1)

- **MM-Fit sensor+pose zip:** `https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip`,
  1,742,309,258 bytes (1.74 GB). Downloaded to `data/external/mmfit/mm-fit.zip` (in
  progress when this note was written; the fetch command must verify and, if needed, resume).
- **MM-Fit video (Zenodo 7607736):** one file per workout and modality, named `wXX_rgb.mp4`
  and `wXX_depth.mp4`, at `https://zenodo.org/api/records/7607736/files/<name>/content`;
  the record API returns an **md5** per file, so `--with-video` can verify without a
  hand-entered checksum. RGB files are 0.4–3.1 GB each; `w00_rgb.mp4` is 2.17 GB.
- **RecoFit:** the two `.mat` files are **Git LFS objects** (135-byte pointers via the raw
  URL). Fetch from `https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/<file>`
  (multionly 1,571,881,721 B; singleonly 1,562,300,284 B). The text files (`readme.html`,
  `load_exercise_data.m`, `LICENSE`, `README.md`) are *not* LFS and must come from
  `raw.githubusercontent.com`. Both `.mat` files and the text files are on disk.
- **RecoFit format (from `load_exercise_data.m`):** `subject_data` is a cell matrix, rows =
  subjects, columns = exercise (single-activity file, names in
  `exerciseConstants.activities`) or one column (multi-activity file). Each cell is a struct
  array of recordings with `activityName`, `data.accelDataMatrix` (`[t_s, x, y, z]`, **g**),
  `data.gyroDataMatrix` (`[t_s, x, y, z]`, **dps**), and in the multi file
  `activityStartMatrix` rows `[name, start_s, end_s, notes, reps]` with a `non-exercise`
  class. Subject 52 and "Two-arm Dumbbell Curl (both arms, not alternating)" are a known
  non-empty combination for a smoke test. Whether the files are MATLAB v7 (scipy) or v7.3
  (HDF5, needs h5py) is still **(verify)**.
- **RecGym:** UCI zip is
  `https://archive.ics.uci.edu/static/public/1128/recgym:+gym+workouts+recognition+dataset+with+imu+and+capacitive+sensor-7.zip`
  (108,325,995 B). Downloaded; contents not yet inspected.
- **MM-Fit starter repo** has `utils/` (a package, not `utils.py`) and `EDA.ipynb`; the
  workout→subject mapping is not on the website text and must be found in the repo or paper.

## 5. Tooling finding (fixed, ADR-0010)

`.claude/settings.json` ran both hooks as `uv run --no-sync python .claude/hooks/<name>.py`.
The path was relative to the *shell's current directory*, so once a Bash call `cd`ed into a
subdirectory (this session: `data/external/recofit`), every later Bash call was blocked by
the guard failing to find its own script (Python exits 2, which Claude Code treats as a
block). Fixed on 2026-09-24: both commands now use
`"$CLAUDE_PROJECT_DIR/.claude/hooks/<name>.py"`. Hooks are read at session start, so the fix
applies from the next restart. Still good practice: never `cd` in a Bash call; use absolute
paths, because the persistent shell keeps whatever directory the last command left it in.
