<!--
Team notes (not part of the submission; dropped when exported to PDF):
- Source of requirements: Canvas "Final Project Paper Outline" + the 2026-10-14 lecture. Where they
  differ we follow the lecture: technical/professional style, one bullet per member in Team
  Contributions, APA or MLA, public GitHub link in the appendix, section structure is ours to choose.
- Due Wed 2026-10-28, 11:59 pm PT, one PDF on Canvas, submitted by the group leader (Robert).
- Fill every [TODO] before export. Numbers in the "Progress to date" box come from reports/ and
  must be re-checked against the regenerated files on the day of submission.
- Export: `pandoc docs/report/outline.md -o outline.pdf` (or print from GitHub's rendered view).
-->

# FormCoach: Wrist-IMU-Gated Pose Estimation for Per-Repetition Exercise Form Feedback

*Final Project Paper Outline — working draft*

| | |
|---|---|
| **Working title** | FormCoach: Wrist-IMU-Gated Pose Estimation for Per-Repetition Exercise Form Feedback |
| **Alternative title** | Wake the Camera Only When It Matters: Sensor-Gated Computer Vision for Real-Time Workout Feedback on a Laptop CPU |
| **Group name** | FormCoach [TODO: confirm the group name used on Canvas] |
| **Authors** | Robert Ashe (group leader) — [TODO]@sdsu.edu · Ruby Rios Ramirez — [TODO]@sdsu.edu · Rochelle Reyes — [TODO]@sdsu.edu · Christian Byars — [TODO]@sdsu.edu · Bryce Hall — [TODO]@sdsu.edu |
| **Course** | BDA 696 — [TODO: course title], Big Data Analytics program, San Diego State University, Fall 2026 |
| **Instructor** | [TODO: instructor name] |
| **Repository** | https://github.com/nthPerson/BDA696_Group_Project (public) |

---

## Working Abstract

People who train alone at home get no feedback on their form, and poor form on simple dumbbell
movements is a common cause of wasted effort and of injury. Camera-based pose estimation can
judge form, but running it continuously is costly on an ordinary laptop and records video
nobody needs. FormCoach tests whether a cheap wrist-worn motion sensor can decide *when* to
look. A wearable that our team assembles (Seeed XIAO ESP32-S3 microcontroller with a Bosch
BMI160 six-axis inertial measurement unit) runs a small quantized neural network that detects
"exercise in progress" and streams that signal over Bluetooth Low Energy to a Python
application. Only while that gate is open does the laptop run MediaPipe pose estimation on the
webcam. It computes joint angles, segments repetitions, and applies per-exercise
biomechanical rules to flag faults such as elbow swing on a curl or a shallow squat. Scope is
fixed to four dumbbell exercises: bicep curl, overhead press, lateral raise and squat.

The learned components are trained and evaluated on three public wearable-exercise datasets:
RecoFit (94 subjects, about 79 hours of 50 Hz forearm IMU data), MM-Fit (10 subjects,
synchronized smartwatch IMU, RGB-D video and 3-D pose, with repetition counts) and RecGym (10
subjects, 4.7 million rows). A short recording session from our own wearables checks transfer
to our device. The processing workflow converts four heterogeneous formats (MATLAB structs,
NumPy arrays, CSV, video) into one partitioned Parquet schema in SI units. It then resamples
to 50 Hz, cuts labelled two-second windows, engineers 63 time- and frequency-domain features,
and extracts joint-angle time series from pose landmarks. We compare a motion-energy threshold
baseline against a random forest and a one-dimensional convolutional network, both in floating
point and as an 8-bit integer model deployed on the microcontroller. The headline evaluation is
leave-one-subject-out, reporting macro-F1 for gate and exercise recognition, mean absolute
error for repetition counting, and rule pass and detection rates under synthetic faults. For
the system-level claim we run gated and always-on vision on identical recordings and compare
frames processed, CPU load, latency, and whether the per-repetition feedback is the same. The
intended contribution is a reproducible, open-source reference design showing how much vision
computation a sensor gate can save without changing the feedback. That pattern carries over
to other battery- and privacy-sensitive edge-AI applications.

---

## 1. Introduction: Problem, Background and Objectives

**Practical problem and motivation**
- Home and gym-floor strength training is mostly unsupervised; the main form faults are
  visible (partial range of motion, momentum swing, trunk lean, knee valgus, rushed tempo) but
  the person training cannot see them.
- Vision-based coaching apps exist, but continuous pose estimation is the dominant cost on a
  CPU-only laptop, and an always-on camera captures far more video than the task needs
  (privacy).
- Wrist IMUs (smartwatches) already detect and count exercise well but cannot judge posture;
  cameras judge posture but do not know when an exercise starts.

**Background to review (prior work)**
- IMU-based exercise recognition and counting: RecoFit (Morris et al., 2014); MM-Fit
  multimodal logging across devices (Strömbäck et al., 2020); capacitive + IMU gym recognition
  (Bian et al., 2022; RecGym dataset).
- Deep learning for sensor-based activity recognition: survey (Wang et al., 2019); 1-D
  convolutional / recurrent models on raw windows (Ordóñez & Roggen, 2016).
- On-device pose estimation: BlazePose / MediaPipe (Bazarevsky et al., 2020).
- TinyML deployment and integer quantization: TensorFlow Lite Micro (David et al., 2021);
  integer-arithmetic-only inference (Jacob et al., 2018).
- Learned form assessment and why we do not train one: Fitness-AQA (Parmar et al., 2022)
  shows that form-fault labels exist only for a few barbell lifts and are access-restricted.

**Intended users and application**
- People training alone with light dumbbells and a laptop; secondarily physical-therapy and
  rehabilitation settings where range-of-motion tracking matters.
- Engineering audience: anyone building battery- or privacy-sensitive vision systems that can
  use a cheap sensor to decide when the expensive model runs.

**Objectives**
- Primary objective: build and evaluate an end-to-end system in which a wrist-worn gate model
  wakes a pose pipeline that gives per-repetition form feedback on four exercises.
- Hypothesis under test: *sensor-gated vision produces the same per-repetition feedback as
  always-on vision while processing a fraction of the camera frames on a laptop CPU.*
- Supporting objectives: (1) a gate / exercise recognizer that generalizes to unseen people;
  (2) accurate repetition segmentation; (3) form rules that pass on correct reps and fire on
  faulty ones; (4) an int8 model that fits and runs in real time on the microcontroller.
- Expected outputs: the `formcoach` Python package and CLI, firmware, five assembled
  wearables, trained models, an evaluation harness that regenerates every table and figure,
  and the written report.

**Overall approach (one paragraph in the paper)**
- Public data first: every learned component is trained and evaluated on public datasets, and
  the full pipeline runs on recorded sessions without hardware (`--source replay`).
- The learned part is the gate; form judgement is rule-based on top of a pretrained pose model
  because no public dataset labels form faults for our four exercises.
- A small team recording set (about one hour, subjects coded S1–S5) checks transfer to our
  sensor; it is a validation set, not a training campaign.

---

## 2. Data Sources and Acquisition

**2.1 MM-Fit (primary for pose, rep counts, cross-dataset tests)**
- Source: https://mmfit.github.io/ (sensor + pose archive); video on Zenodo,
  https://doi.org/10.5281/zenodo.7607736. Paper: Strömbäck et al. (2020).
- Contents: 10 subjects, 21 workouts, 809 minutes; 10 exercises × 3 sets × 10 reps per workout
  (4 map to ours: squats, bicep curls, dumbbell shoulder press, lateral shoulder raises).
- Modalities: left/right smartwatch accelerometer + gyroscope (100 Hz), two phones, an earbud,
  RGB-D video (30 fps), 2-D (18 COCO joints) and 3-D pose (17 Human3.6M joints, mm).
- Labels: per-set start/end frame, activity and repetition count (616 sets) — the ground truth
  for rep counting.
- Format: per-workout NumPy `.npy` arrays (column 0 frame index, column 1 Unix ms, then values)
  and a headerless labels CSV; 1.74 GB archive → 3.6 GB extracted; video 40.8 GB (we download
  only selected sessions).
- License: MIT (sensor, pose, code); CC BY 4.0 (video).
- Quality issues found: left-phone streams absent in 15 of 21 workouts; duplicated sensor
  timestamps (~21–28 k per watch stream); video starts about 2 minutes before the sensors;
  rep-boundary precision about 60–100 ms; only 10 subjects.

**2.2 RecoFit (primary for gate and exercise-recognition training)**
- Source: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors (Git LFS;
  archived, read-only). Paper: Morris et al. (2014).
- Contents (verified on load): 94 subjects, 126 lab visits, 79.4 h of multi-exercise
  recordings plus 4,687 single-exercise recordings (78 h); 75 activity labels; 50 Hz forearm
  accelerometer (g) and gyroscope (deg/s); repetition counts on many segments.
- Format: two MATLAB v5 files, ~1.57 GB each, nested structs and cell arrays; ~3 GB RAM each
  to load.
- License: CDLA-Permissive-2.0; the README asks for the CHI 2014 citation.
- Quality issues found: the README claims "over 200 participants" but the release holds 94;
  severe class imbalance (2,642 idle minutes vs 23 minutes of lateral raise); 78 minutes of
  junk labels (device taps, arm-band adjustment); forearm placement differs from a wrist watch.

**2.3 RecGym (supplementary robustness check)**
- Source: UCI ML Repository, https://doi.org/10.24432/C5PW4K (Bian & Lukowicz, 2025); the UCI
  zip is served corrupt, so we fetch the authors' Kaggle mirror (same file, checksummed).
- Contents: 10 subjects × 5 sessions × 3 sensor positions; 4,703,320 rows × 11 columns
  (6 IMU channels, a capacitance channel, labels); 20 Hz; 12 workouts including arm curl and
  squat; 65 h.
- Format: one 475 MB CSV. License: CC BY 4.0.
- Quality issues found: every signal column is min–max normalized to [0, 1] and there are no
  timestamps, so physical units and gravity are unrecoverable — RecGym is evaluated only
  within itself and never mixed into training with the SI-unit datasets.

**2.4 Team validation recordings (secondary, Phase 3)**
- About one hour total from five team members (S1–S5) wearing our device: 4 exercises × 2 sets
  of correct form plus one set per exercise with one scripted fault.
- Captured by `formcoach record`: IMU Parquet, pose Parquet, event log, metadata.
- Privacy: webcam video never leaves the owner's laptop; only derived pose and IMU files are
  shared; subjects are coded, never named.

**2.5 Acquisition process**
- One command per dataset (`make data DATASET=…`) downloads, verifies SHA-256 (MD5 for Zenodo
  video) and records URL, checksum, size, license and date in `data/MANIFEST.md`.
- Raw data is never committed (license and size); a 30-second MM-Fit fixture is committed for
  tests.

**2.6 Scale and why the data fit the project**
- About 29 million time-stamped six-axis IMU samples from 114 public subjects
  (RecoFit ≈ 14.3 M, MM-Fit smartwatches ≈ 10.2 M, RecGym 4.7 M rows).
- After windowing: 276,080 labelled RecoFit windows and 95,849 MM-Fit windows, each with 63
  engineered features and a raw 100 × 6 signal tensor — well above the 25,000 × 15 tabular
  guideline.
- Synchronized RGB-D video (40.8 GB, above the 5 GB guideline for array data) and ≈ 1.4 million
  frames of 3-D pose.
- Appropriateness: wrist/forearm IMU matches our sensor; MM-Fit is the only public set with
  synchronized wrist IMU, video, pose and rep counts for our exercises; the many-subject
  RecoFit cohort makes unseen-subject evaluation meaningful.
- Integration plan: harmonize all labels to one vocabulary (`curl, press, raise, squat, other,
  idle`); train the gate on RecoFit + MM-Fit; hold RecGym out as a separate check.

---

## 3. Data Processing and Preparation Workflow

**3.1 Ingest and conversion (raw → common schema)**
- Loaders for each format (`scipy.io.loadmat` for MATLAB, NumPy for `.npy`, pandas for CSV)
  emit one `IMUStream` schema: dataset, subject, session, device, placement, time (s),
  ax/ay/az (m/s²), gx/gy/gz (rad/s), exercise, rep id, set id.
- Unit conversion (g → m/s², deg/s → rad/s); RecGym tagged `units = normalized`.
- Time alignment: MM-Fit streams and pose share one clock fitted per workout from video frame
  index and timestamp; duplicated timestamps dropped.
- Label harmonization: 75 RecoFit, 10 MM-Fit and 12 RecGym labels mapped to six canonical
  classes; junk labels dropped; original label kept for traceability.
- Output: one Parquet file per visit/workout under `data/processed/` (columnar, compressed,
  partitioned so no step has to hold a whole dataset in memory).

**3.2 Data profiling and validation**
- `formcoach data profile` writes a generated report of subjects, sessions, minutes per label,
  sampling rate per device, missing modalities and duplicate timestamps.
- Schema checks in unit tests; every report stamps the command, commit and random seed.

**3.3 Signal processing and windowing**
- Resample all IMU streams to 50 Hz (the device rate) with anti-alias filtering.
- Two-second windows (100 samples × 6 channels) every 1 s (50 % overlap); majority label plus
  a label-purity score; evaluation keeps windows with purity ≥ 0.8.
- Per-channel standardization constants computed from training data and exported to
  `preprocess.json`, then compiled into the firmware so device and laptop preprocess
  identically.

**3.4 Feature engineering**
- 9 statistics × 7 channels (ax, ay, az, gx, gy, gz, |a|) = 63 features: mean, standard
  deviation, min, max, energy, dominant frequency (0.3–8 Hz), spectral entropy, autocorrelation
  peak lag and height (0.3–3 s).
- The raw window is stored alongside as a fixed-shape float32 column, so tree models and the
  CNN read the same files.
- Augmentation for the CNN (training only): random 3-D rotation (±30°), time-warp (±10 %),
  jitter, magnitude scaling — targets sensor-orientation and placement differences between
  datasets.

**3.5 Pose and video processing**
- Video frames → MediaPipe PoseLandmarker (33 landmarks, image and metric world coordinates);
  MM-Fit's own 3-D pose used for calibration.
- Normalization: hip-centred world coordinates scaled by torso length so thresholds are
  comparable across people and camera distances.
- Cleaning: drop frames where required landmarks have visibility < 0.5; interpolate gaps of
  ≤ 3 frames, otherwise mark the segment invalid; 5-frame median then Savitzky–Golay smoothing.
- Joint angles (elbow, shoulder abduction/flexion, hip, knee, trunk inclination, knee
  tracking) in degrees; each repetition resampled to 30 time steps.

**3.6 Scalability strategy and tools**
- Tools: NumPy, pandas, PyArrow/Parquet, SciPy (MATLAB I/O, filtering, peak detection),
  DuckDB for queries over Parquet, OpenCV for video, `concurrent.futures` process pools.
- Out-of-core by partition: each visit/workout is processed independently, so memory stays
  bounded (~3 GB peak for a RecoFit file) and the feature build parallelizes across cores
  (~35 s for all datasets).
- Video is streamed frame by frame and only needed sessions are downloaded.
- Dask, PySpark and Polars were considered; at ~7 GB of raw sensor data, partitioned Parquet
  plus process-level parallelism is sufficient, and the paper will justify that choice.
- Reproducibility: `uv` lockfile, fixed seed (20260924), one Makefile target per step.

---

## 4. Exploratory Data Analysis

(Summaries and visual checks only; no modeling.)

- **Structure:** subjects, sessions, hours and windows per dataset (table); sampling rate and
  timestamp-jitter distributions per device.
- **Class distributions:** minutes per canonical label per dataset (bar chart) to document the
  imbalance that drives the choice of macro-F1 and class weights.
- **Per-subject coverage:** heat map of which subjects performed which target exercises in
  RecoFit — explains why per-fold scores vary and why pooled LOSO is the headline.
- **Missingness:** missing modalities per MM-Fit workout; fraction of windows below the label-
  purity threshold; pose frames dropped for low landmark visibility.
- **Signal examples:** one set of each exercise from each dataset (acceleration and
  gyroscope traces with rep boundaries) to show the periodic structure the models use.
- **Cross-dataset domain shift:** gravity-direction and |a| distributions per dataset and
  placement (forearm vs wrist vs normalized); 2-D PCA of the 63 features colored by dataset
  and by class.
- **Feature relationships:** correlation matrix of the engineered features; distributions of
  dominant frequency and autocorrelation peak by exercise.
- **Repetition kinematics:** rep-duration distributions per exercise; joint-angle percentiles
  (5th / 50th / 95th) on correct-form MM-Fit reps — the evidence used to set rule thresholds.
- **Outliers:** sessions with abnormal sampling gaps, implausible rep durations, or pose jumps.

---

## 5. Methodology and System Architecture

**5.1 Architecture overview** (planned Figure 1: block diagram)
- Wearable: BMI160 IMU at 50 Hz → 2 s ring buffer → int8 1D-CNN gate every 0.5 s (TensorFlow
  Lite Micro) → hysteresis → BLE notifications (IMU samples + gate flag); button for sessions
  and calibration; LED for feedback codes.
- Laptop (`formcoach` Python package, one process): IMU source (BLE, USB serial or replay) →
  gate logic → webcam capture only while the gate is open → MediaPipe pose → normalization and
  joint angles → repetition segmentation → rules engine → on-screen overlay, event log, and an
  LED code back to the wearable.
- Every stage reads and writes the same Parquet schemas, so the full pipeline runs on
  recorded sessions with no hardware.

**5.2 Gate and exercise-recognition models** (the machine-learning core)
- Input: 2 s × 6-channel window (raw tensor for the CNN; 63 features for the random forest).
- Outputs: `active` (binary: exercise in progress) and `exercise` (multi-class: curl, press,
  raise, squat, other, idle).
- Model A, baseline: motion-energy threshold on the variance of |a| (one parameter).
- Model B: balanced random forest, 200 trees, on the 63 hand-crafted features (scikit-learn).
- Model C: 1D-CNN in Keras/TensorFlow — three Conv1D blocks, global average pooling, dense
  output; about 6.4 k parameters; class weights, Adam, early stopping.
- Model C-int8: post-training full-integer quantization → TFLite → C array compiled into the
  firmware (TFLite Micro on the ESP32-S3).
- Hysteresis: gate opens after 2 consecutive active windows (1 s), closes after 6 s idle.

**5.3 Pose estimation and repetition segmentation**
- MediaPipe PoseLandmarker (Lite first; Full and Heavy as variants), CPU only.
- IMU-primary segmentation: band-pass filter + peak detection; pose cross-check: excursions of
  the primary joint angle with adaptive thresholds; fusion counts a rep when IMU and pose
  intervals overlap ≥ 50 %.

**5.4 Form rules engine**
- Configuration-driven (`rules.yaml`), multi-label per repetition; each rule returns code,
  severity, measured value, threshold, and a one-sentence coaching message.
- 16 rules across the four exercises, e.g. `CURL_SWING`, `PRESS_NO_LOCKOUT`, `RAISE_OVER`,
  `SQUAT_SHALLOW`, `SQUAT_KNEE_VALGUS`, and tempo rules.
- Thresholds calibrated at the 5th/95th percentile of correct-form MM-Fit reps, per pose
  source.

**5.5 Why this methodology**
- No public data labels form faults for our exercises, so a learned fault classifier cannot
  be trained or evaluated honestly; rules on a pretrained pose model are transparent and
  testable.
- The gate is a well-posed supervised problem with large labelled public data, and it must fit
  in ~40 KB of microcontroller memory — hence a tiny CNN compared against a strong
  feature-based model.

**5.6 Libraries and technologies**
- Python 3.11; scikit-learn, TensorFlow/Keras, Optuna (tuning), MediaPipe, OpenCV, SciPy,
  pandas, PyArrow, DuckDB, `bleak` (BLE), `pyserial`, `typer` (CLI), Streamlit (dashboard).
- Firmware: PlatformIO, Arduino framework, NimBLE-Arduino, TensorFlow Lite Micro (Chirale
  port), in-tree BMI160 driver.

---

## 6. Baselines, Comparisons and Refinement Strategy

**6.1 Comparisons (what is compared, and against which baseline)**
- **Gate / exercise recognition:** energy threshold (baseline) vs random forest vs 1D-CNN vs
  1D-CNN int8 — at least two ML methods against a baseline, on identical unseen-subject folds.
- **Repetition counting:** peak-detection baseline vs pose-only vs IMU + pose fusion.
- **System architecture:** always-on vision (baseline) vs energy gate vs laptop-side model
  gate vs on-device model gate, on identical recordings.
- **Pose model:** MediaPipe Lite vs Full vs Heavy (YOLO11n-pose optional) for rep agreement,
  frames per second and CPU.
- **Rules:** design-default thresholds vs data-calibrated thresholds.

**6.2 Why the comparisons are meaningful**
- The energy baseline is what a gate "for free" would look like; beating it shows the model
  earns its microcontroller memory.
- The random forest vs CNN comparison weighs interpretable hand-crafted features against
  learned features that can be deployed as an int8 model.
- Gated vs always-on isolates the effect of the gate on cost while holding feedback fixed.

**6.3 Refinement plan**
- Hyperparameter tuning (Optuna) for the random forest and CNN within training folds only.
- Feature ablations: adding the |a| channel; feature-group importance for the forest.
- Augmentation ablation (rotation, time-warp, jitter, scaling) for cross-device robustness.
- Quantization: if float → int8 loses more than 2 macro-F1 points, quantization-aware training.
- Hysteresis sweep (open/close windows) to trade frames saved against missed rep starts.
- Rule threshold calibration on MediaPipe landmarks once the webcam path is measured.
- Transfer to our sensor: evaluate public-trained models on team recordings, before and after
  last-layer fine-tuning on one subject.

---

## 7. Evaluation Methodology

**7.1 Protocol**
- Headline numbers are on **unseen subjects**: leave-one-subject-out on RecoFit (94 folds) and
  MM-Fit (10 folds); 5 subject-grouped folds for the CNN (cost), with the baselines re-run on
  the same folds so comparisons are like-for-like.
- Cross-dataset test: train on RecoFit, test on MM-Fit (different device and placement).
- Seen-subject accuracy is never the headline.

**7.2 Measures, what they evaluate, and why**

| Measure | Evaluates | Why it fits the objective |
|---|---|---|
| Macro-F1 (primary), accuracy, per-class precision/recall, confusion matrix | Gate and exercise recognition | Classes are highly imbalanced (idle ≫ lateral raise); macro-F1 weights rare exercises equally |
| Per-fold score spread (min / median) | Robustness across people | Shows whether a new user is likely to be served well |
| Rep-count MAE, % sets exactly right, % within ±1 | Repetition segmentation | Feedback is per rep; a missed or extra rep misattributes feedback |
| Rule pass rate on correct-form reps | Specificity of each rule | A rule that fires on good reps erodes trust |
| Detection rate under synthetic perturbations (ROM ×0.7, +25° trunk lean, 2× speed, elbow shift) | Sensitivity of each rule | No public fault labels, so faults are injected in a controlled way |
| Frames processed (%), process CPU (%), wall time | Computational efficiency of gating | Directly tests the "fraction of the frames" claim |
| Agreement of rep/fault events with always-on | Correctness of gating | The gate must not change the feedback |
| Rep-end → feedback latency (p50 / p95) | Usability | Target < 500 ms so feedback arrives before the next rep |
| On-device inference time, RAM arena, flash, battery hours | Deployability | Must fit and run in real time on a $16 wearable |
| Float → int8 macro-F1 drop | Quantization cost | Target ≤ 2 points |
| BLE sample-drop rate | System reliability | Gaps in the stream corrupt windows |

**7.3 Interpretation**
- The gating claim holds if gated runs reproduce the always-on rep and fault events while
  processing materially fewer frames and less CPU; any mismatch is reported and analyzed.
- Model choice for the device is the best int8 model within memory and latency budgets, not
  simply the best float model.
- Go/no-go (~Nov 3): if rep segmentation or rule validity is unacceptable on webcam pose, the
  deliverable falls back to an IMU-only coach on the same wearable, and the evaluation keeps
  the gate, rep-count and on-device measures.

---

## 8. Planned Results and Technical Outputs

**Planned tables** (all generated by `make eval` into `reports/`; none hand-edited)
- T1 Dataset summary: subjects, hours, windows, classes, licenses.
- T2 Gate (`active`) LOSO: energy / RF / CNN / CNN-int8 — accuracy, macro-F1, per-fold spread.
- T3 Exercise recognition LOSO, per class.
- T4 Cross-dataset (RecoFit → MM-Fit) and RecGym within-dataset results.
- T5 Rep-count error per exercise: peaks / pose / fused.
- T6 Rule pass rates and perturbation detection rates.
- T7 Gated vs always-on: frames processed, CPU, wall time, event agreement.
- T8 Pose-model ablation: rep agreement, FPS, CPU.
- T9 On-device metrics: inference time, memory, flash, battery; float → int8 drop.
- T10 Transfer to team recordings, before/after fine-tuning.

**Planned figures**
- F1 System architecture diagram; F2 data-processing workflow diagram.
- F3 Minutes per label per dataset; F4 example IMU traces with rep boundaries.
- F5 Confusion matrices for each model; F6 per-subject score distribution.
- F7 Gating timeline for one session (gate state, frames processed, reps, faults).
- F8 Joint-angle curves for a correct vs perturbed rep, with rule thresholds.
- F9 Latency distribution; F10 photos of the assembled wearable and case.

**Example system outputs**
- Annotated overlay screenshots (skeleton, rep counter, fault message) from public-data replay;
  a 30-second demo video.

**Questions the results will answer**
- Does a learned gate beat a motion-energy threshold on people it has never seen?
- How much does a ~6 k-parameter int8 network lose against a random forest and its own float
  version?
- How many frames and how much CPU does gating save, and does it change any feedback?
- Do public-trained models transfer to our own sensor and placement?

> **Progress to date (preliminary; generated by scripts on public data, will be regenerated
> for the final report).** Data pipeline, baselines, CNN, int8 export, rules engine, replay
> demo and firmware are built; the first wearable runs the int8 gate. Unseen-subject gate
> macro-F1 on RecoFit (5 subject-grouped folds): energy 0.67, random forest 0.87, CNN 0.84,
> CNN-int8 0.83 (`reports/loso_*_recofit*.md`); on-device inference 13.4 ms with a 6.2 KB
> tensor arena (`reports/device.md`). Webcam pose, team recordings, on-device gating
> benchmarks and latency are still to be measured.

---

## 9. Discussion, Applications and Limitations

**Interpretation (planned)**
- Relate each result back to the hypothesis: recognition quality of the gate, cost saved by
  gating, and whether the feedback stayed the same.
- Error analysis: which subjects and exercises fail (e.g. press vs lateral raise confusion,
  forearm vs wrist placement), and why.

**Applications beyond the project**
- Home fitness and tele-coaching on commodity hardware.
- Physical therapy and rehabilitation: range-of-motion and tempo tracking between visits.
- General edge-AI pattern: a cheap always-on sensor gates an expensive model (smart cameras,
  wildlife or industrial monitoring, wearables) to save energy and limit recording.

**Limitations to discuss**
- Data: few subjects with video (10 in MM-Fit); public subjects skew young and healthy;
  forearm vs wrist placement and sensor-orientation differences; RecGym normalized units.
- Labels: no real form-fault labels — rule sensitivity is measured with synthetic
  perturbations, which may not match real faulty movement.
- Pose: monocular webcam, single frontal view; MM-Fit's lifted 3-D pose compresses joint
  angles, so thresholds must be calibrated per pose source.
- Team validation set is small (~1 hour, 5 subjects who are also the developers).
- System: CPU-only laptops, BLE reliability, battery life of a 300 mAh cell.
- Generalizability: four dumbbell exercises; rules are hand-written per exercise.

**Ethics, privacy and responsible use**
- Gating reduces camera-on time; team video stays on each owner's machine; subjects are coded.
- FormCoach is a training aid, not a medical device; incorrect feedback could encourage a
  wrong movement, so messages are conservative and thresholds are documented.
- Hardware safety: lithium-polymer cell handling rules are included in every hardware
  document.
- All data used under their licenses and cited; no personal data is published.

---

## 10. Conclusion and Future Work

**Planned conclusion**
- Restate the problem (unsupervised training, costly always-on vision) and the system built.
- Answer the hypothesis with the gated-vs-always-on comparison and the unseen-subject gate
  results.
- State the contribution: an open, reproducible reference design — hardware, firmware, data
  pipeline, models and evaluation harness — for sensor-gated form feedback.

**Future work**
- Learned per-rep fault classifier if a labelled dataset (e.g. Fitness-AQA) becomes available.
- Per-user personalization by fine-tuning the last layer on a few labelled reps.
- More exercises and camera views; side-view robustness with a second camera.
- Natural-language coaching from fault codes with a local language model.
- Physical-therapy framing with clinical range-of-motion targets.

---

## 11. Team Contributions and Responsibilities

- **Robert Ashe (group leader):** system architecture, wearable hardware and firmware, case
  design, GitHub repository and reproducibility; drafts Sections 1 and 5 and the appendix.
- **Ruby Rios Ramirez:** data acquisition, loaders, profiling and Parquet pipeline; drafts
  Sections 2–4 (data, processing, EDA figures).
- **Rochelle Reyes:** pose pipeline, joint angles, repetition segmentation and form rules;
  drafts the pose and rules parts of Sections 5–8.
- **Christian Byars:** gate models, quantization and the evaluation harness (LOSO, gating,
  on-device metrics); drafts Sections 6–7 and the results tables.
- **Bryce Hall:** live application and dashboard, demo video, report assembly and editing,
  presentation slides; drafts Sections 9–10.
- **Shared:** team validation recordings, EDA review, proofreading and the final presentation.

---

## 12. References (APA 7)

Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020).
*BlazePose: On-device real-time body pose tracking* (arXiv:2006.10204). arXiv.
https://doi.org/10.48550/arXiv.2006.10204

Bian, S., & Lukowicz, P. (2025). *RecGym: Gym workouts recognition dataset with IMU and
capacitive sensor* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PW4K

Bian, S., Rey, V. F., Yuan, S., & Lukowicz, P. (2022). *The contribution of human body
capacitance/body-area electric field to individual and collaborative activity recognition*
(arXiv:2210.14794). arXiv. https://doi.org/10.48550/arXiv.2210.14794

David, R., Duke, J., Jain, A., Janapa Reddi, V., Jeffries, N., Li, J., Kreeger, N., Nappier,
I., Natraj, M., Wang, T., Warden, P., & Rhodes, R. (2021). TensorFlow Lite Micro: Embedded
machine learning for TinyML systems. *Proceedings of Machine Learning and Systems, 3*, 800–811.

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D.
(2018). Quantization and training of neural networks for efficient integer-arithmetic-only
inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
(pp. 2704–2713). https://doi.org/10.1109/CVPR.2018.00286

Morris, D., Saponas, T. S., Guillory, A., & Kelner, I. (2014). RecoFit: Using a wearable
sensor to find, recognize, and count repetitive exercises. In *Proceedings of the SIGCHI
Conference on Human Factors in Computing Systems* (pp. 3225–3234). ACM.
https://doi.org/10.1145/2556288.2557116

Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks for
multimodal wearable activity recognition. *Sensors, 16*(1), 115.
https://doi.org/10.3390/s16010115

Parmar, P., Gharat, A., & Rhodin, H. (2022). Domain knowledge-informed self-supervised
representations for workout form assessment. In *Computer Vision – ECCV 2022* (pp. 105–123).
Springer. https://doi.org/10.1007/978-3-031-19839-7_7

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay,
É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research,
12*, 2825–2830.

Strömbäck, D., Huang, S., & Radu, V. (2020). MM-Fit: Multimodal deep learning for automatic
exercise logging across sensing devices. *Proceedings of the ACM on Interactive, Mobile,
Wearable and Ubiquitous Technologies, 4*(4), 1–22. https://doi.org/10.1145/3432701

Wang, J., Chen, Y., Hao, S., Peng, X., & Hu, L. (2019). Deep learning for sensor-based activity
recognition: A survey. *Pattern Recognition Letters, 119*, 3–11.
https://doi.org/10.1016/j.patrec.2018.02.010

Software documentation to cite in the final report: MediaPipe Pose Landmarker
(Google AI Edge), TensorFlow Lite for Microcontrollers, Seeed Studio XIAO ESP32-S3 wiki, Bosch
BMI160 datasheet.

---

## 13. Appendix and Reproducibility Plan

- **Public GitHub repository:** https://github.com/nthPerson/BDA696_Group_Project — all code,
  firmware, case files and documentation.
- **Reproduction:** `make setup` (locked `uv` environment) → `make data` (download + checksum)
  → `make features` → `make train-gate` → `make eval` (regenerates every table and figure) →
  `make demo` (full pipeline on public data, no hardware); tested on Linux, macOS and Windows in
  continuous integration.
- **Configuration:** fixed seed and commit hash stamped into every report; `rules.yaml`,
  `preprocess.json`, `platformio.ini`, `uv.lock`.
- **Documentation:** README quickstart, system design specification, dataset notes, numbered
  design decisions (ADRs), how-to guides per component, data manifest with licenses.
- **Hardware:** bill of materials, wiring tables and photos, flashing guide, printable case
  files, lithium-polymer safety rules.
- **Additional material:** full per-fold tables, model specifications, extra figures.
- **Excluded by design:** raw datasets (license and size; fetched by script), team video, any
  credentials or keys.

---

## Requirement Coverage Map

| Assignment component | Where it is addressed |
|---|---|
| Title and team information | Header table |
| Working abstract | Working Abstract |
| Problem, background, objectives | §1 |
| Data sources and acquisition | §2 |
| Data processing and preparation | §3 |
| Exploratory data analysis | §4 |
| Methodology and model / system architecture | §5 |
| Baseline, comparison and refinement | §6 |
| Evaluation methodology | §7 |
| Planned results and technical outputs | §8 |
| Discussion, applications and limitations | §9 |
| Conclusion and future work | §10 |
| Team contributions | §11 |
| References (≥ 3) | §12 |
| Appendix and reproducibility | §13 |
