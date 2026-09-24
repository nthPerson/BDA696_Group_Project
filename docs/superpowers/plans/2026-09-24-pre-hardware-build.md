# Pre-hardware build — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (Robert chose inline,
> autonomous execution for this run; no per-task review gate). Steps use `- [ ]` checkboxes.

**Goal:** build everything in the design note's §2 table that needs no wearable: Checkpoints 1–3,
the Phase 2 code (firmware v1 compiling in CI, rules v1 + engine, Keras CNN gate + int8 export)
and the hardware-free half of Phase 3 (`eval rules`, `eval gating` for always_on/energy/laptop).

**Architecture:** raw downloads → `IMUStream` Parquet (one file per subject-session under
`data/processed/<dataset>/streams/`) → windows + features Parquet → models/eval reports. The
replay demo reads a committed 30 s session folder (`imu.parquet`, `pose.parquet`, `meta.json`)
in exactly the layout `SessionRecorder` will write for team recordings, so the pipeline is
source-agnostic from day one. `formcoach.signal` never imports `formcoach.pose` (rule 10).

**Tech stack:** Python 3.11, uv, typer, NumPy, pandas 3, pyarrow, SciPy, scikit-learn; Keras/TF
(extra `train`); MediaPipe/OpenCV (extra `vision`, never required by tests); PlatformIO +
Arduino + NimBLE for the firmware.

**Spec:** `docs/superpowers/specs/2026-09-24-pre-hardware-build-design.md` (sequencing) and
`docs/02-system-design.md` (build spec). Both travel with this plan.

## Global constraints (from CLAUDE.md and the spec)

- One checkpoint = one branch = one PR, stacked on the previous branch; never merge; never `cd`.
- `make lint test` green in every PR; `pio run` green for firmware PRs; CI smoke test is
  `formcoach demo --source replay --headless` and becomes the real pipeline in PR 3.
- Parquet columns follow `docs/02-system-design.md` §5 exactly; add columns, never rename;
  `t` float seconds monotonic per stream; accel m/s², gyro rad/s.
- 19-byte sample / 99-byte batch (ADR-0004) is frozen for this run.
- Heavy packages import lazily with an install hint (ADR-0002). Fixtures < 2 MB each.
- Headline metrics are unseen-subject (LOSO by subject); seeds fixed (`SEED = 20260924`) and
  logged in every report header. Reports under `reports/` are written only by `formcoach eval`.
- Every PR: tests, STATUS entry, DECISIONS entries for every choice or **(verify)** that
  differed, `docs/howto/<topic>.md` for the owning teammate, formcoach-reviewer run first.

## Facts established before coding (2026-09-24; each becomes a docs/04 correction + ADR)

| Item | Docs said | Real file |
|---|---|---|
| MM-Fit sensor arrays | `(n, 2+ch)` | `(n, 5)`: frame, **epoch ms**, x, y, z; watch acc already m/s², gyro rad/s; 100 Hz; ~450 duplicate timestamps per workout |
| MM-Fit pose | 2D 17 joints, 3D 16 | **2D 18 (COCO), 3D 17 (Human3.6M)**, arrays `(dims, frames, 1+joints)`, col 0 = frame |
| MM-Fit subjects | "verify" | notebook mapping: w01/03/06/08/10/14→0, w02/04/07/09/11/15→1, w00/05→2, w12→3, w13→4, w16→5, w17→6, w18→7, w19→8, w20→9 |
| MM-Fit phone files | `sp_l_acc` | `sp_l_acc_0/_5/_7` suffix variants; `sp_l_*` missing in 15 workouts |
| RecoFit files | v7 or v7.3 | MATLAB 5.0 (v7) → `scipy.io.loadmat` works, ~14 s per file |
| RecoFit subjects | 200+ | **94** rows / 94 unique `subjectID`, 126 visits (multi), 4687 recordings (single), 75 activity labels |
| RecoFit `activityStartMatrix` | 5 columns | **7**: name, start_s, end_s, notes, reps, start_s again, sequence-number struct |
| RecoFit license | unclear | **CDLA-Permissive-2.0** (`LICENSE`) |
| RecoFit sensor | forearm/arm | `masterToken == 'rightArm'` for every recording; a `slave*` device exists (missing in 356 recordings) |
| RecGym UCI zip | 103 MB zip → CSV | first 60 MB of the UCI file are zero bytes on the server (same SHA on re-download); **Kaggle** mirror works unauthenticated |
| RecGym columns | `Object, Workout, Position` + 7 signals | `Subject, Position, Session, A_x..G_z, C_1, Workout`; 4,703,320 rows; **all signals min-max normalised to [0,1]**, no timestamps, no physical units |

## Review focus (inputs the spec implies but no task's tests exercise; each gets a test below)

1. A stream with duplicate or non-monotonic timestamps must resample without NaN (Task 2.1).
2. A workout missing a modality (`sp_l_*`) must load with `None`, not crash (Task 1.3).
3. A rep shorter than the window or a window with mixed labels must be labelled by majority and
   flagged, never silently mislabelled (Task 2.4).
4. A pose sequence with `visibility < 0.5` gaps longer than 3 frames must mark the segment
   invalid and the rules engine must skip it (Task 3.2 / 5.2).
5. A truncated BLE packet or a CSV line with the wrong field count must be dropped and counted,
   never raise out of the source loop (Task 4.3).

---

## PR 1 · `feat/checkpoint-1-data` (Ruby)

### Task 1.1 — schema + manifest
Files: `src/formcoach/data/schema.py`, `src/formcoach/data/manifest.py`,
`tests/test_schema.py`, `tests/test_manifest.py`.
Interfaces produced: `IMU_COLUMNS`, `WINDOW_COLUMNS`, `POSE_COLUMNS`, `REP_COLUMNS`,
`CANONICAL_EXERCISES = ("curl","press","raise","squat","other","idle")`,
`validate_imu_stream(df) -> None` (raises `SchemaError`), `empty_imu_stream() -> DataFrame`,
`manifest.upsert_row(path, section, key, row: dict)`, `manifest.read_rows(path, section)`.
- [ ] tests: columns present/order, dtypes, monotonic t, unit sanity (|a| within 0..200 m/s²)
- [ ] manifest round-trip on a temp copy of `data/MANIFEST.md`; idempotent upsert
- [ ] commit `feat: IMUStream schema + manifest writer`

### Task 1.2 — `data fetch`
Files: `src/formcoach/data/fetch.py`, `tests/test_fetch.py`, `cli.py` (replace stub).
Registry: dataclass `RemoteFile(dataset, name, url, size, sha256, license, extract)` with the
URLs/sizes/SHA-256 above; Zenodo API for `--with-video` (md5 from the record). Download with
`requests` streaming + `Range` resume + tqdm; verify size then hash; extract zips (skip when
`.extracted` marker exists); write manifest rows. RecGym: UCI URL kept as `alt_url`, primary is
the Kaggle API redirect; a zip whose first member fails to open raises a clear error naming the
mirror.
- [ ] tests with a local `http.server` fixture: fresh download, resume from a partial file,
      bad hash → error and file kept as `.partial`, manifest row written once
- [ ] run against the real files (verify only, no re-download) → manifest rows
- [ ] commit `feat: data fetch with resume, hash verification and manifest rows`

### Task 1.3 — loaders
Files: `data/mmfit.py`, `data/recofit.py`, `data/recgym.py`, `data/labels.py` (exercise maps),
fixtures under `data/fixtures/{mmfit,recofit,recgym}/` (tiny synthetic raw files in the real
formats, built by `tests/make_fixtures.py`… no: by `scripts/`? → by `formcoach data
make-fixtures` hidden command so it is reproducible), tests per loader.
Interfaces: every loader exposes `available(root) -> bool`, `list_sessions(root) ->
list[SessionKey]`, `load_stream(root, key) -> DataFrame[IMUStream]`, `describe(root) -> dict`
and `iter_streams(root)`. Extra columns: MM-Fit `frame`; RecoFit `label_raw`; all `units`
(`si` | `normalized`). MM-Fit also `load_sets(root, workout) -> DataFrame` (set_id, frames,
reps, exercise) and `load_pose3d(root, workout) -> (frame_idx, xyz[frames,17,3])`.
- [ ] tests on fixtures: schema valid, units, label assignment at set boundaries, missing
      modality → `None`, RecoFit g→m/s² and dps→rad/s, RecGym `units == "normalized"`,
      `t` restarts per session and is strictly increasing
- [ ] commit per loader

### Task 1.4 — `data convert` + `data profile`
Files: `data/convert.py` (raw → `data/processed/<dataset>/streams/<subject>-<session>.parquet`),
`data/profile.py` (→ `reports/data_profile.md` + `reports/figures/data_profile_*.png`), CLI.
- [ ] run on the real data; commit the report; test `profile` on fixtures writes a file with the
      expected sections
- [ ] docs/04 corrections, ADRs, `docs/howto/datasets.md`, STATUS; reviewer; PR

## PR 2 · `feat/checkpoint-2-signal` (Christian)

### Task 2.1 resample/filters/gravity — `signal/resample.py: resample_uniform(t, x, fs) ->
(t_u, x_u)` (dedupe t by mean, linear interp, no NaN); `signal/filters.py: lowpass(x, fs, fc,
order=4)`, `bandpass(x, fs, lo, hi)`; `signal/gravity.py: split_gravity(acc, fs, fc=0.3) ->
(gravity, linear)`. Tests: synthetic sine recovered, duplicate timestamps, constant offset.
### Task 2.2 windows/features — `signal/windows.py: make_windows(stream, fs=50, win_s=2.0,
stride_s=1.0) -> DataFrame[Window]` (x as fixed-shape list column 100×6 float32, label by
majority with `label_purity` column added, `label_active`), `signal/features.py:
window_features(x, fs) -> dict[str,float]` (~30 features per §6.2), `features build` CLI writing
`data/processed/<dataset>/windows/<subject>.parquet`.
### Task 2.3 LOSO + baselines — `eval/loso.py: loso_splits(groups)`, `run_loso(...)`,
`write_loso_report(...)`; `models/energy.py: EnergyGate` (threshold on var|a|, fit on train);
`models/rf.py: make_rf(seed)`; `eval loso --model energy|rf --dataset recofit|mmfit
[--task active|exercise] [--folds loso|N]`; reports `reports/loso_<model>.md` + confusion PNG.
### Task 2.4 rep counter — `signal/reps.py: count_reps(acc, fs, ...) -> list[Rep]` (§4.4
defaults); `eval repcount --source peaks` on MM-Fit sets → `reports/baseline_repcount.md`.
- [ ] run `features build`, `eval loso --model energy`, `eval loso --model rf`, `eval repcount`
      on real data; commit reports; howto `docs/howto/signal-baselines.md`; ADRs; PR

## PR 3 · `feat/checkpoint-3-replay` (Rochelle + Bryce)

### Task 3.1 angles — `pose/skeletons.py` (joint-name → index maps for `mediapipe33` and
`h36m17`), `pose/angles.py` (pure NumPy: `angle_deg(a,b,c)`, `joint_angles(xyz, skeleton, up)
-> dict[str, float]` for elbow_l/r, shoulder_abd_l/r, hip_l/r, knee_l/r, trunk_incl,
knee_track_l/r), hand-computed tests (90°, 180°, collinear guards).
### Task 3.2 normalize + pose reps — `pose/normalize.py` (hip-centre, torso units,
visibility gate, gap interpolation ≤ 3 frames, `valid` flag, median+Savitzky–Golay smoothing,
`estimate_up_vector`), `pose/reps.py: segment_reps(angle, t, start_thr, return_thr) ->
list[Rep]`, `pose/landmarker.py` (lazy MediaPipe; `extract_video`), `pose extract` CLI.
### Task 3.3 replay + pipeline — `io/source.py` (`IMUSource` protocol, `SIsample`),
`io/replay.py: ReplaySource(session_dir, speed)`, `app/gate.py` (always_on/energy/laptop/device
with §4.2 hysteresis), `app/pipeline.py: run_pipeline(imu_source, pose_source, gate, exercise,
rules=None, sink)`, `app/events.py` (event records, `events.parquet`, `frames.parquet`),
fixture session `data/fixtures/replay/mmfit_w00_curls/` (30 s), `demo --source replay --headless`,
`tests/test_integration_replay.py` (≥ 8 reps found on the fixture), CI smoke test real.
- [ ] howto `docs/howto/replay-demo.md`; ADRs (up-vector, fixture choice); PR

## PR 4 · `fw/checkpoint-4-firmware-v1` (Robert)

### Task 4.1 library compile tests — `platformio.ini` envs `xiao_esp32s3` (release) plus
`lib_bmi160_hanyazou`, `lib_bmi160_dfrobot`, `lib_tflm_chirale`, `lib_tflm_tanaka`; `pio run -e`
each locally; keep the winners in the main env; CI builds all envs; ADRs.
### Task 4.2 firmware v1 — `src/main.cpp`, `src/imu.{h,cpp}`, `src/ble.{h,cpp}`, `src/gate.{h,cpp}`
(energy gate until the model lands, same hysteresis constants), `src/ui.{h,cpp}` (button/LED),
CSV over serial, 5-sample batches, control/status characteristics, NVS calibration.
### Task 4.3 Python sources — `io/serial.py`, `io/ble.py` (lazy extras), `io/recorder.py`
(`SessionRecorder`), `record`, `session check`; tests with `FakeSource` emitting `protocol.py`
packets, truncated packets counted not raised; `Makefile fw-upload PORT=`.
- [ ] howto `docs/howto/flashing-firmware.md` (with §4 battery text); PR

## PR 5 · `feat/checkpoint-6-rules` (Rochelle)
`rules/rules.yaml` (§7 thresholds + messages + hysteresis constants), `rules/engine.py:
RuleEngine.evaluate(rep: RepMetrics) -> list[Fault]`, perturbation tests (one per rule),
`eval/rules.py` (`eval rules` on MM-Fit pose_3d → `reports/rules_validation.md`),
`eval/gating.py` (`eval gating --gate always_on|energy|laptop` on replay sessions →
`reports/gating.md`), pipeline wiring; howto `docs/howto/rules.md`; PR.

## PR 6 · `feat/checkpoint-5-cnn-gate` (Christian)
`models/cnn.py` (Keras 1D-CNN §6.2, lazy TF), `models/export.py` (int8 PTQ with representative
set, `.tflite` → `firmware/model/gate_model_data.cc`, `firmware/model/preprocess.json`),
`train gate --model cnn`, `eval loso --model cnn|cnn-int8`, `reports/loso_cnn.md`; firmware
`gate.cpp` runs TFLM on the exported model (compile-tested); howto `docs/howto/cnn-gate.md`; PR.

## Self-review
Spec coverage: every row of design-note §2 "buildable now" maps to a task above; §3 PR order
kept. Placeholders: none intended — where a detail is decided during implementation it is
recorded in DECISIONS, not left "TBD". Type consistency: `IMUStream` DataFrame is the only
inter-PR data contract; `Rep`/`RepMetrics` (PR 3) is consumed by PR 5; `Window` (PR 2) by PR 6.
