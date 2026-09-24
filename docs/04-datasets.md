# 04 — Datasets

Public datasets are the primary training and evaluation data. Team recordings are a small secondary validation set. Everything downloaded goes under `data/external/<dataset>/`, is gitignored, and is recorded in `data/MANIFEST.md` with URL, SHA-256, size, license and date. Loaders live in `src/formcoach/data/` and emit the common schema in `docs/02-system-design.md` §5. Items marked **(verify)** are expectations to check against the real files on first load; update this document when reality differs.

## 1. MM-Fit — primary (IMU + video + pose + rep counts)

- **Site / download:** https://mmfit.github.io/ (sensor + pose zip). Video (RGB + depth, 40.8 GB total, 20 sessions): https://doi.org/10.5281/zenodo.7607736 — download **only selected sessions** (`w00`, `w01`, `w09` are enough for pose-pipeline development) with `--with-video`.
- **Code / starter:** https://github.com/KDMStromback/mm-fit (MIT). Paper: Strömbäck, Huang & Radu, *MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices*, IMWUT 4(4), 2020, https://doi.org/10.1145/3432701.
- **License:** sensor/pose data and code MIT; Zenodo video CC BY 4.0. Cite the paper.
- **Scale:** 10 subjects, 21 workouts (`w00`–`w20`), 809 minutes; 10 exercises × 3 sets × 10 reps per workout: squats, lunges, bicep_curls, situps, pushups, tricep_extensions, dumbbell_rows, jumping_jacks, dumbbell_shoulder_press, lateral_shoulder_raises, plus `non_activity`. Devices: 2 smartwatches (TicWatch Pro, left and right wrist, 100 Hz accel/gyro, 1 Hz HR), 2 phones (Samsung S7 210 Hz, Huawei P20 500 Hz; acc/gyr/mag), eSense earbud (90 Hz), Orbbec Astra Pro RGB-D at 30 fps, 2-D and 3-D pose from the video.
- **File layout (from the starter code — verify on disk):** one directory per workout `wXX/` containing `wXX_labels.csv` and `.npy` files whose names contain the modality key: `sw_l_acc, sw_l_gyr, sw_l_hr, sw_r_acc, sw_r_gyr, sw_r_hr, sp_l_acc, sp_l_gyr, sp_l_mag, sp_r_acc, sp_r_gyr, sp_r_mag, eb_l_acc, eb_l_gyr, pose_2d, pose_3d`.
  - Sensor arrays: shape `(n, 2 + channels)`; **column 0 = video frame index, column 1 = timestamp in ms, columns 2.. = values** (3 for acc/gyr/mag, 1 for HR).
  - Pose arrays: shape `(dims, frames, 1 + joints)`; `[:, i, 0]` = frame index, `[:, i, 1:]` = joint coordinates; `pose_2d` has 17 joints, `pose_3d` has 16 (joint order per the MM-Fit site **(verify)**).
  - `wXX_labels.csv` rows: `start_frame, end_frame, repetition_count, activity` per exercise set (no header).
  - Time alignment is via the video frame index; sensor timestamps are in ms.
- **Splits used by the authors:** train `01 02 03 04 06 07 08 16 17 18`; val `14 15 19`; seen-subject test `09 10 11`; unseen-subject test `00 05 12 13 20`. Use the unseen split for headline numbers and also run LOSO by subject (map workouts → subjects from the site's metadata **(verify)**; two subjects contributed six workouts each).
- **Our use:** smartwatch streams (`sw_l`, `sw_r`) for the gate, exercise recognition and rep counting (the closest public analogue to our wrist unit); `pose_3d` and video for angle-feature development and rule calibration; rep counts as ground truth for `eval repcount`. Map MM-Fit classes to ours: `bicep_curls → curl`, `dumbbell_shoulder_press → press`, `lateral_shoulder_raises → raise`, `squats → squat`; the other six become `other` for the gate model and are still used for the `active` label.
- **Known issues:** only 10 subjects; rep-boundary annotation precision ≈ 60–100 ms; sensor axes are in the watch's frame (orientation differs from our wearable); heart-rate columns are sparse; some sessions lack a modality (loader must return `None`/skip, as the starter code does).

## 2. RecoFit — primary (large IMU cohort)

- **Download:** https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors (archived, read-only since June 2026; files still downloadable). ~2.5 GB in two MATLAB files: `exercise_data.50.0000_multionly.mat` and `exercise_data.50.0000_singleonly.mat`, plus `readme.html` and `load_exercise_data.m`.
- **Paper:** Morris, Saponas, Guillory & Kelner, *RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises*, CHI 2014, https://doi.org/10.1145/2556288.2557116. Microsoft Research page: https://www.microsoft.com/en-us/research/publication/recofit-using-wearable-sensor-find-recognize-count-repetitive-exercises/
- **License:** the repo has a LICENSE file — read it and record the terms in the manifest; the readme asks for citation of the paper / attribution to Microsoft. Treat as non-commercial academic use.
- **Scale:** 200+ participants; accelerometer + gyroscope at 50 Hz from a forearm/arm-worn sensor; many gym exercise classes with labeled segments (exact class list and sensor placement per `readme.html` **(verify)**).
- **Loading:** `scipy.io.loadmat(..., struct_as_record=False, squeeze_me=True)`; follow `load_exercise_data.m` to reconstruct per-subject, per-exercise segments. Expect nested structs/cell arrays; write `recofit.describe()` first and check it against the readme's counts. Convert once to Parquet under `data/processed/recofit/` (optionally with Dask if RAM is tight).
- **Our use:** the main training set for the gate (`active` vs `idle`) and exercise recognition; LOSO by participant is the headline evaluation; map RecoFit classes to `curl / press / raise / squat / other` and keep the original label for the fallback project.
- **Known issues:** MATLAB format; no explicit license text in the readme; sensor placement/orientation differs from a wrist watch and from our unit; class imbalance and long idle periods; some segments may be short — apply a minimum-length filter and report it.

## 3. RecGym — supplementary

- **Download:** https://archive.ics.uci.edu/dataset/1128 (UCI ML Repository; 103 MB zip → `RecGym.csv`, 453 MB). CC BY 4.0.
- **Scale:** 10 participants, 4,432,070 rows × 11 columns: `A_x, A_y, A_z, G_x, G_y, G_z, C_1 (body capacitance), Object (subject), Workout, Position`; 20 Hz; 12 activities: Adductor, ArmCurl, BenchPress, LegCurl, LegPress, Riding, RopeSkipping, Running, Squat, StairsClimber, Walking, Null.
- **Our use:** robustness / second-domain test for `ArmCurl` and `Squat` recognition; a different sampling rate (20 Hz) forces the resampling code to be correct. Drop `C_1`.

## 4. Optional / related (not required)

- **Fitness-AQA** (ECCV 2022, Parmar et al.) — gym video with fine-grained posture-error labels for back squat, overhead press, barbell row; access by request form, non-commercial: https://github.com/ParitoshParmar/Fitness-AQA. Only if access is granted early and only for a transfer experiment of the rules on real faulty reps.
- **FLEX** (2025, under review) — multi-view video + sEMG + error labels; watch for release: https://haoyin116.github.io/FLEX_Dataset/.
- Physical-therapy datasets (KIMORE, UI-PRMD) are relevant only to the PT stretch goal.

## 5. Why the data satisfy the course's scale guideline

> 10 M labeled IMU samples from 220+ subjects across three cohorts (tabular guideline: 25 k × 15), synchronized with 40.8 GB of RGB-D video (gridded guideline: 5 GB) and ≈ 1.4 M frames with 3-D pose. The modeling difficulty is cross-subject and cross-device generalization, which is why LOSO and cross-dataset tests are the headline evaluations.

## 6. Team validation recordings (secondary; ≈ 1 hour total; Phase 3)

Purpose: confirm that public-data models transfer to our wearable, and validate the live pipeline and rules. **Not a training campaign.** Protocol (put this in `docs/recording-protocol.md` and in the `formcoach record` prompts):

- Subjects coded `S1`–`S5`; no names or faces stored outside each teammate's machine; only `imu.parquet`, `pose.parquet`, `events.parquet`, `meta.json` are shared; `video.mp4` stays local unless the owner agrees.
- Per subject: 4 exercises × 2 sets × 8–10 reps of correct form, plus 1 set per exercise with one scripted fault (e.g. partial curl ROM, shallow squat, bent-elbow raise, no lockout press), light dumbbells or none.
- Camera: laptop webcam at 2–3 m, frontal for curl/press/raise, frontal or 45° for squat, whole body in frame; note camera id and laptop model in `meta.json`.
- Wearable on the **left wrist**, case arrow toward the elbow; button press at set start/end; note anything unusual.
- Quality check immediately: `formcoach session check <dir>` reports sample drops, gate timeline, frames processed, reps counted vs expected.
