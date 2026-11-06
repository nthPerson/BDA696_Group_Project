# DECISIONS

Architecture decision records, one paragraph each, newest last. Add an entry whenever you choose
between alternatives; reference it from `docs/STATUS.md`. Format: **ADR-NNNN · title · date ·
status** then context → decision → consequences.

---

## ADR-0001 · Python toolchain · 2026-09-24 · accepted
Context: five teammates on Linux/macOS/Windows with mixed experience must run everything from
a fresh clone with one command. Decision: `uv` manages the interpreter (pinned to 3.11 via
`.python-version` and `requires-python`), the virtualenv and a committed universal `uv.lock`;
`src/` layout with package `formcoach` built by hatchling; `ruff` for lint + format (line
length 100); `pytest`; `pre-commit` hooks installed by `make setup`. Consequences: no one needs
a system Python; CI uses `uv sync --locked` so the lock is the single source of truth and a
drifted lock fails CI. Python 3.12+ is deliberately excluded until MediaPipe and TensorFlow
wheels are verified there.

## ADR-0002 · Heavy dependencies are optional extras · 2026-09-24 · accepted
Context: `make setup && make demo` must finish in under five minutes, but TensorFlow, PyTorch,
MediaPipe and OpenCV together exceed a gigabyte of downloads, and the replay demo does not need
any of them (pose is pre-extracted to Parquet). Decision: core dependencies cover the replay
pipeline and public-data processing; `vision` (mediapipe, opencv), `device` (bleak, pyserial),
`train` (tensorflow, torch, optuna) and `app` (streamlit, psutil) are extras; `all` bundles
them. PyTorch resolves from the official CPU-only index on every OS because nobody has a
discrete GPU and the default Linux wheel pulls ~2.5 GB of CUDA packages. Consequences: modules
under `pose/`, `io/ble.py`, `io/serial.py` and `models/cnn.py` must import their heavy
dependency lazily and fail with a clear "install with `uv sync --extra vision`" message. The
extras must still resolve in the universal lock, so they are locked (not installed) in CI.

## ADR-0003 · Firmware toolchain · 2026-09-24 · accepted
Context: firmware must compile in CI before hardware exists and be flashable by teammates.
Decision: PlatformIO with `platform = espressif32@^6.9.0` (pinned major/minor for
reproducibility; arduino-esp32 2.x core), board `seeed_xiao_esp32s3`, Arduino framework,
`h2zero/NimBLE-Arduino@^2.1.0`. The BMI160 driver (`hanyazou/BMI160-Arduino` vs
`DFRobot_BMI160`) and the TFLite Micro library (`Chirale_TensorFlowLite` vs
`TensorFlowLite_ESP32` vs Edge Impulse export) are **not** chosen yet; Checkpoint 4/5 will
compile-test the candidates in CI and record the winner here. Consequences: the bring-up
sketch only uses `Wire` and NimBLE; if the espressif32 6.x core turns out to lack something we
need, moving to the pioarduino fork (arduino-esp32 3.x) is a one-line change here.

## ADR-0004 · BLE sample is 19 bytes, batch is 99 bytes · 2026-09-24 · proposed (confirm before firmware v1)
Context: `docs/02-system-design.md` §3 lists a per-sample `seq` (uint16) for drop detection but
sizes the batch as "5 × 17 + 4 = 89", which is the size *without* `seq`. Decision: keep `seq`
(drop detection per sample is worth 2 bytes at 50 Hz) → packed sample = 4 + 12 + 1 + 2 = 19
bytes, header 4 bytes, batch 99 bytes, still well under the 185-byte MTU. The layout exists
once as packed structs with `static_assert`s in `firmware/include/protocol.h` and once as
`struct.Struct` formats in `src/formcoach/io/protocol.py`; `tests/test_protocol.py` pins both
sizes and checks that constants, UUIDs and command codes agree. Consequences: any protocol
change touches both files and the test in the same PR. Per working rule 7, the protocol must
not change after firmware is deployed to teammates' devices without asking Robert.

## ADR-0005 · CI shape · 2026-09-24 · accepted
Context: teammates run three operating systems; the course grades reproducibility. Decision:
GitHub Actions with three jobs: `lint` (ruff check + format check, Ubuntu), `test` (pytest and
a `formcoach demo --source replay --headless` smoke test on a Ubuntu/macOS/Windows matrix) and
`firmware` (`pio run` with the PlatformIO cache). The spec asked for Ubuntu only; the matrix is
additive and is the cheapest way to honour "runnable by teammates on all three OSes".
Consequences: a change that only works on Linux fails the PR; CI minutes are free for a public
repo. The replay smoke test is currently a stub and becomes the real integration test at
Checkpoint 3.

## ADR-0006 · Data policy in git · 2026-09-24 · accepted
Context: raw datasets are gigabytes, RecoFit's license terms are unclear, and team video is
private. Decision: `data/external/*`, `data/processed/*`, `data/team/*` and every `*.mp4 *.npy
*.mat *.parquet *.zip` are gitignored; only `data/fixtures/**` (small, < 2 MB per file,
enforced by the `check-added-large-files` hook) and `data/MANIFEST.md` are committed. Every raw
file is recorded in the manifest with URL, SHA-256, size, license and date; `formcoach data
fetch` will write those rows. Team subjects are coded S1–S5 and `video.mp4` never leaves the
recording laptop. Consequences: `tests/test_repo_contract.py` fails if the ignore patterns are
removed; anyone who needs the data runs `make data`.

## ADR-0007 · Repository root is the project root · 2026-09-24 · accepted
Context: the kickoff bundle arrived as `formcoach-kickoff/` inside an otherwise empty GitHub
repo named `BDA696_Group_Project`, and its layout diagram showed a `formcoach/` top-level
folder. Decision: the repo root *is* the project root (`pyproject.toml`, `src/`, `firmware/`,
`docs/` at top level); the kickoff docs moved to `docs/` unchanged and the kickoff `CLAUDE.md`
was replaced by the top-level `CLAUDE.md`. Consequences: one less directory level for every
path in the docs; the README clones the repo into a folder called `formcoach` so commands read
naturally. Renaming the GitHub repository is Robert's call.

## ADR-0008 · CLI structure and stub convention · 2026-09-24 · accepted
Context: Checkpoint 0 asks for Makefile targets that print what they will do, and Windows
teammates may not have `make`. Decision: every Makefile workflow target delegates to a `typer`
command (`formcoach data|features|pose|train|eval|session|demo|record`), so `uv run formcoach
...` is always the direct equivalent. Unimplemented commands call `cli.stub(command,
checkpoint, what)`, which prints a yellow `STUB` banner with the checkpoint that delivers it,
and exit 0 so `make` chains and CI smoke tests pass. Consequences: a teammate implementing a
command keeps its signature, replaces the stub body and removes the `"STUB" in output`
assertion in `tests/test_cli.py`; option names in the CLI are the vocabulary used throughout
the docs (`--source replay|serial|ble`, `--gate always_on|energy|laptop|device`).

## ADR-0009 · Claude Code configuration lives in the repository · 2026-09-24 · accepted
Context: five teammates will run Claude Code on three operating systems, and the working
rules with the highest cost of violation (committing raw data or teammate video, deleting
data, skipping the STATUS read) are advisory prose in `CLAUDE.md`. Decision: hooks, skills and
subagents live under `.claude/` and are committed; `.claude/settings.local.json` is gitignored
for personal overrides. Hooks are stdlib-only Python invoked as `uv run --no-sync python
.claude/hooks/<name>.py` rather than shell scripts, because `uv` is the one tool every clone
already needs, `--no-sync` keeps the hook fast and works before `make setup`, and the same
command runs under bash and PowerShell; each hook is covered by `tests/test_claude_hooks.py`
and linted by `ruff check .` in CI. The data guard only ever *denies*; anything it does not
recognise falls through to the normal permission flow. Skills are written test-first
(baseline subagent without the skill, then with it) per the writing-skills process.
Consequences: a teammate who never uses Claude Code is unaffected; one who does inherits the
guard, the session banner, the two skills and the two agents on `git clone`. Hook paths are
relative to the repository root, so Claude Code must be started from the repo (or a
subdirectory of it). *Amended by ADR-0010: hook paths are now absolute.*

## ADR-0010 · Hook commands resolve through `$CLAUDE_PROJECT_DIR` · 2026-09-24 · accepted
Context: the ADR-0009 hook commands used a path relative to the shell's current directory
(`.claude/hooks/data_guard.py`). Claude Code's Bash tool keeps one persistent shell, so a
single `cd data/external/recofit && ...` left every later hook invocation looking for
`data/external/recofit/.claude/hooks/data_guard.py`; Python exits 2 when it cannot open the
script, and Claude Code treats exit 2 from a PreToolUse hook as a block. The whole session
lost shell access with no in-session recovery. Decision: both hook commands are now
`uv run --no-sync python "$CLAUDE_PROJECT_DIR/.claude/hooks/<name>.py"`. Claude Code sets
`CLAUDE_PROJECT_DIR` to the launch directory for every hook and runs hook commands through a
POSIX shell on all three operating systems (Git Bash on Windows), so the quoted variable
expands everywhere; `session_start.py` already finds the repo root via `git rev-parse`, so it
needs no change. Alternatives rejected: a `cd "$CLAUDE_PROJECT_DIR" &&` prefix (same effect,
more to get wrong), and making the guard tolerate a missing script (it would silently stop
guarding). Consequences: hooks work from any cwd inside the repo; settings are read at
session start, so the change takes effect on the next restart; the ADR-0009 sentence about
starting Claude Code from the repo root still holds for `CLAUDE_PROJECT_DIR` itself.

## ADR-0011 · Raw → IMUStream Parquet conversion is an explicit step · 2026-09-24 · accepted
Context: RecoFit's `.mat` files take 14 s and 3 GB each to load, MM-Fit is 324 `.npy` files
and RecGym one 475 MB CSV; every later stage (windows, LOSO, replay) needs the same rows.
Decision: `formcoach data convert` writes one Parquet per stream in the §5 `IMUStream` schema
under `data/processed/<dataset>/streams/<subject>-<session>[-<device>].parquet` (gitignored);
`features build`, `eval …` and the replay source read only those files. Loaders keep a
`describe()` for the profile report but are otherwise not called in hot paths. Consequences:
a fresh clone runs `make data && make convert` once (about a minute); the raw formats are
isolated in `src/formcoach/data/{mmfit,recofit,recgym}.py`.

## ADR-0012 · MM-Fit session clock is fitted from frame index and timestamp · 2026-09-24 · accepted
Context: MM-Fit sensor rows carry a video frame index and a Unix-ms timestamp; pose arrays
carry only frame indices; sensors start ~135 s after the video. Decision: per workout,
`mmfit.session_clock()` least-squares-fits `timestamp = t0 + ms_per_frame × frame` on the
watch accelerometer (≈ 33.4 ms/frame) and every stream's `t` is seconds since video frame 0;
pose frames map to `t` through the same fit. Alternatives: `t` relative to the first sensor
sample (breaks pose alignment across devices) or raw Unix seconds (breaks the "starts near 0"
convention). Consequences: MM-Fit `t` starts at ~135 s, which is correct, and pose/IMU
alignment error is below one frame.

## ADR-0013 · RecGym is fetched from the authors' Kaggle mirror · 2026-09-24 · accepted
Context: the UCI zip for dataset 1128 is served with its first 60 MB zeroed (same SHA-256
`9713a9ca…` on three downloads and on a byte-range probe, 2026-09-24); no other UCI endpoint
serves the CSV. The authors' page points to Kaggle, whose v1 download endpoint redirects to a
signed URL without credentials. Decision: the registry's primary RecGym source is the Kaggle
archive (116,851,172 B, SHA-256 `0abc140f…`), the UCI URL is kept as `alt_urls` for the record,
and the manifest row says why. Consequences: if Kaggle changes the redirect, the fetch fails
loudly with both URLs in the message; the CSV inside has one extra trailing byte compared with
the UCI listing but the same content.

## ADR-0014 · RecGym is kept in normalised units and never mixed into SI training · 2026-09-24 · accepted
Context: every RecGym signal column is min-max scaled to [0, 1] with no timestamps; the
implied full scale is implausible (~76 g) so gravity and physical units cannot be recovered.
Decision: `recgym.load_stream` centres the values (`x − 0.5`), builds a synthetic 20 Hz clock
and sets a new `IMUStream` column `units = "normalized"` (SI streams carry `"si"`);
`schema.validate_imu_stream` skips magnitude checks for normalised streams, and windowing /
training refuse to combine the two. RecGym is used only for within-dataset checks (20 Hz
resampling, curl/squat recognition). Consequences: the "second domain" claim in
`docs/04-datasets.md` §3 is weaker than planned; cross-dataset transfer is RecoFit ↔ MM-Fit.

## ADR-0015 · Canonical label vocabulary and what counts as idle · 2026-09-24 · accepted
Context: three label sets (75 RecoFit, 10 MM-Fit, 12 RecGym) must map onto the gate's
`active` flag and the four v1 exercises. Decision (`src/formcoach/data/labels.py`): canonical
labels are `curl, press, raise, squat, other, idle`; `active = label != idle`. Idle =
Non-Exercise, Device on Table, Rest, `<Initial Activity>` and **Walk** (the camera must stay
off while walking between stations); static holds (plank, wall squat) and machines are
`other` (the athlete is exercising). RecoFit junk labels (device taps, arm-band adjustment,
notes, "Invalid", "Unlisted Exercise") stay `idle` in the stream with the raw name in
`label_raw`, and windowing drops any window that overlaps them. Curl includes the band curl
and the alternating curl; squat includes goblet / hands-behind-head / arms-forward / dumbbell
variants but not squat jumps or wall squats; press includes the rack shoulder press.
Consequences: label choices are one table, testable (`test_every_recofit_label_is_mapped`),
and changing them is a one-file PR plus a rerun of `make eval`.

## ADR-0016 · Placement vocabulary extended · 2026-09-24 · accepted
Context: §5 lists `wrist_l, wrist_r, upper_arm, pocket, ear`; RecoFit's sensor is on the right
forearm and RecGym's wrist side is unknown, its third position is the calf. Decision: add
`forearm_r`, `wrist` (side unknown) and `calf` to `schema.PLACEMENTS`; nothing is renamed.
Consequences: filters that select "wrist-like" placements must include `wrist`, `wrist_l`,
`wrist_r` and `forearm_r`.

## ADR-0017 · Windowing, features and LOSO conventions for the baselines · 2026-09-24 · accepted
Context: docs/02 §5–§6 fix 2 s windows at 50 Hz and "~30 hand-crafted features" but leave the
labelling of mixed windows, the treatment of RecoFit junk labels, the feature list and the
fold scheme open. Decision: windows are cut every 1 s (50 % overlap) and carry the majority
canonical label plus `label_purity`; evaluation keeps windows with purity ≥ 0.8; any window
overlapping a RecoFit junk label (device taps, arm-band adjustment, notes) is dropped
(ADR-0015). Features are 9 statistics × 7 channels (ax..gz and |a|): mean, std, min, max,
energy, dominant frequency in 0.3–8 Hz, spectral entropy, autocorrelation peak lag and height
in 0.3–3 s = 63 `feat_*` columns; the raw window is kept as a flat float32 list column `x` so
the CNN reads the same files. The RF baseline is a balanced random forest (200 trees,
`min_samples_leaf=2`, seed 20260924). `eval loso` defaults to strict leave-one-subject-out
(94 folds on RecoFit, 10 on MM-Fit) and offers `--folds N` subject-grouped folds for slow
models; both are unseen-subject, and every report states which it used. RecGym is evaluated
only within RecGym (ADR-0014). Consequences: `make features` takes ~35 s; RecoFit RF LOSO
takes 15–60 min on 16 cores; per-fold numbers on RecoFit vary widely because many subjects
performed only one or two of the four target exercises, so the pooled table is the headline.

## ADR-0018 · Pose rep segmentation uses adaptive thresholds · 2026-09-24 · accepted
Context: docs/02 §4.4 defines a rep as one excursion of the primary joint angle beyond fixed
thresholds (curl: elbow < 60° then > 150°). On MM-Fit's lifted 3-D pose (Human3.6M joints
from video) a two-arm curl reads as an elbow swing of 95–140° and a press tops out at ~108°,
so the fixed thresholds never fire; MediaPipe world landmarks will have yet another scale.
Decision: `pose.reps.RepSegmenter(adaptive=True)` recomputes `enter`/`exit` from the 10th/90th
percentiles of the last 6 s of the smoothed primary angle (`enter = lo + 0.3·range`,
`exit = hi − 0.3·range`, mirrored for "above" modes) and counts nothing until the observed
range exceeds 25°; the docs/02 values remain the initial thresholds and the fixed mode is kept
for the rules-engine tests. Rules (§7) keep absolute thresholds — segmentation and form
judgement are separate concerns. Consequences: on MM-Fit the pipeline finds 586/599 curl,
557/559 raise and 637/639 squat reps (press 540/598 is the weak case); the first rep of a set
is detected only once the range has built up (~1 rep of latency at set start); `min_range`
may need to be per exercise.

## ADR-0019 · Replay session layout and the laptop energy-gate threshold · 2026-09-24 · accepted
Context: the replay demo must exercise the same code path as a live session, and the
energy gate needs one number. Decision: a *session directory* (`imu.parquet` in the
IMUStream schema, optional `pose.parquet` in the Pose schema with a `skeleton` column, and
`meta.json`) is the contract shared by `ReplaySource`, `SessionRecorder` and `session check`;
the committed 30 s fixture `data/fixtures/replay/mmfit_w00_curls` is MM-Fit w00's left watch
plus 3-D pose around its first curl set (MIT). The energy gate threshold is var(|a|) = 3.0
(m/s²)², the balanced-accuracy optimum on all RecoFit windows (MM-Fit alone gives 1.05); it
lives in `protocol.h`/`protocol.py`/`rules.yaml` so laptop and firmware agree. The vertical
for angle computation is estimated per session from the median hip→shoulder direction rather
than assumed from the pose source's axes. Consequences: any recorded team session replays
through `formcoach demo --source replay --session <dir>` unchanged; the demo of record finds
10 of 10 reps with always-on and energy gating (83 % of frames processed).

## ADR-0020 · BMI160 driver is in-tree · 2026-09-24 · accepted
Context: docs/02 §2.2 named `hanyazou/BMI160-Arduino` and `DFRobot_BMI160`. Compile-tested
both for `seeed_xiao_esp32s3` with espressif32 6.x (`firmware/platformio.ini`
`libtest_bmi160_*` envs, `make fw-libtest`): DFRobot's header includes `arduino.h` in lower
case, which fails on Linux CI (case-sensitive), and hanyazou's fork carries Intel Curie SPI
sources that do not build for the ESP32. Decision: a 120-line register driver
(`firmware/src/app/bmi160.{h,cpp}`: chip-id check, soft reset, ODR 100 Hz, ±8 g, ±1000 dps,
burst read of 12 data bytes) with the same LSB scaling as `protocol.h`. Both library envs are
kept so hardware day can switch with one `lib_deps` line if the register sequence misbehaves.
Consequences: no third-party IMU code; the register values are from the datasheet and are
**(verify)** on the first board.

## ADR-0021 · TFLite Micro library: Chirale_TensorFlowLite · 2026-09-24 · accepted
Context: docs/02 §2.2 listed `Chirale_TensorFlowLite`, `tanakamasayuki/TensorFlowLite_ESP32`
and an Edge Impulse export. Compile tests: Chirale (git, current TFLM API) builds in 3 m 49 s
(RAM 10.8 %, flash 9.3 % for an empty interpreter); tanaka's library builds only with the
older `ErrorReporter` constructor; TensorFlow's `tflite-micro-arduino-examples` fails with
"unsupported board". Decision: `https://github.com/spaziochirale/Chirale_TensorFlowLite.git`
(not in the PlatformIO registry under that name). CI builds the app env and the Chirale
libtest env. Consequences: the model export (PR 8) targets this library's op resolver.

## ADR-0022 · Rule thresholds are calibrated per pose source; v1 ships both files · 2026-09-24 · accepted
Context: docs/02 §7 gives absolute thresholds and asks that each be set at the 5th/95th
percentile of correct-form MM-Fit reps. `eval rules` on 2,320 MM-Fit reps shows the design
values fire on nearly every rep for the range-of-motion rules (CURL_PARTIAL_ROM pass rate
0.00, PRESS_NO_LOCKOUT 0.00, RAISE_OVER 0.04) because the lifted 3-D pose compresses joint
angles, while tempo/asymmetry/valgus rules behave (SQUAT_KNEE_VALGUS 0.93, PRESS_ASYMMETRY
0.96). Decision: `rules.yaml` keeps the design values and states `pose_source:
design-defaults`; `formcoach eval rules --calibrate-out` writes `rules.mmfit-pose3d.yaml`
with `gt` thresholds at the 95th and `lt` thresholds at the 5th percentile for that pose
source, and both files ship. The engine, pipeline and demo take `--rules <file>`. Which file
is v1 for the webcam path is decided after the first MediaPipe run (Rochelle). Consequences:
the report's pass-rate and perturbation tables are the evidence for the go/no-go on the
vision path; every threshold change is a YAML diff plus a rerun of `eval rules`.

## ADR-0023 · CNN evaluation folds, GPU extra, and one-training float/int8 tables · 2026-09-27 · accepted
Context: strict 94-fold LOSO of the CNN would train 94 networks per task; on the CPU the full
model took an hour per 30 epochs and did not finish, on the RTX 3070 eGPU 23 minutes.
Decision: `eval loso --model cnn` defaults to subject-grouped folds (`--folds 5` used for the
committed reports, 10 epochs per fold) — still unseen-subject — and writes the `cnn-int8`
table from the same trained folds (quantised after training, scored with the TFLite
interpreter) instead of training twice. The RF and energy baselines are re-run on exactly those
folds (`reports/*_5fold.md`) so the comparison table is like-for-like. Training needs the
`train` extra; a new `train-gpu` extra adds the CUDA wheels on Linux (WSL2 additionally needs
`LD_LIBRARY_PATH` pointing at the pip `nvjitlink` library, or TensorFlow falls back to CPU).
Consequences: CNN numbers are 5-fold, RF headline numbers remain 94-fold (both stated in the
reports); GPU runs are not bit-reproducible, so the seed pins the split, not the weights.

## ADR-0024 · First-board findings: antenna, advertising layout, LED polarity, NVS · 2026-10-08 · accepted
Context: firmware v1.1.0 ran on a bare XIAO ESP32-S3 for the first time (kit K1, Robert's
bench, via usbipd from WSL2). Findings and decisions: (1) the XIAO has **no onboard antenna**;
without the clip-on u.FL antenna it advertises at −90 to −99 dBm and laptops cannot connect,
with it −54 dBm at desk range — `docs/assembly-day.md` now starts with the antenna. (2) A
128-bit service UUID plus the device name exceed the 31-byte advertisement (NimBLE: "Cannot
add UUID, data length exceeded"), so the UUID goes in the advertisement and the name in the
scan response; `bleak` on Windows then sees both, and connects with MTU 185 as designed.
(3) The user LED is **active-low** (verified by eye: mostly off, brief flash once a second with
`LED_ACTIVE_LOW = true`). (4) `Preferences.begin(ns, true)` on a fresh board logs
`nvs_open failed`; the namespace is opened read-write once at boot. (5) TFLite Micro allocates
**6,196 B** of arena for the int8 model (budget 40 KB); the firmware uses 643,168 B of flash.
(6) The board enumerates as USB-JTAG/serial `303a:1001` before and after flashing, so one
`usbipd bind` covers both. Still open until an IMU is wired: BMI160 register sequence, I2C
address, 50 Hz notify rate and serial+BLE drops, on-device inference time per window.

## ADR-0025 · BMI160 is configured after power-up, and the firmware verifies it · 2026-10-08 · accepted
Context: with the IMU wired (GY-BMI160 at 0x69), the first self-test read |a| = 4.0 g at rest
and a gyro of exactly zero. A register read-back showed the ACC_RANGE/GYR_RANGE writes issued
right after the soft reset (before the accelerometer/gyroscope power-mode commands) were
dropped: ERR_REG = 0x04 and the ranges read as invalid values, leaving the ±2 g default
(16384 LSB/g) — four times the scale the protocol assumes. Decision: `Bmi160::begin()` sends
the two PMU commands first, then writes CONF/RANGE, then reads ERR_REG, PMU_STATUS and the
four config registers back and only reports `ok` when they match (one `# bmi160 regs …` banner
line; `MISCONFIGURED` otherwise, and `begin()` fails). Verified: err=0, |a| = 1.007 g at rest,
gyro bias 0.6 dps, 50.1 Hz, 0 drops, inference 13.4 ms per window. Consequences: the in-tree
driver (ADR-0020) stands; `device check` would have passed the first time with a wrong scale
if gravity were not checked — keep the gravity check in the runbook.

## ADR-0026 · Browser BLE viewer: Web Bluetooth + vendored uPlot, no build step, GitHub Pages · 2026-10-15 · accepted
Context: the team needs to see a kit's live IMU stream for the prototype demo and for per-kit
bring-up, on macOS and Windows laptops (WSL2 has no Bluetooth), without installing anything.
Options: (a) a Python GUI on bleak + matplotlib/Streamlit, which needs `--extra device`, runs on
the Windows side for WSL users, and is slow at 50 Hz × 7 traces; (b) a web page with Chart.js
or Plotly from a CDN, which stutters with streaming traces and needs internet at the demo; (c)
plain ES modules + Web Bluetooth + uPlot vendored in the repo. Decision: (c), under
`web/ble-viewer/`, with no npm build. It is served by `formcoach viewer` on localhost (Web
Bluetooth needs a secure context; the server sets explicit JS MIME types because Windows can map
`.js` to text/plain) and deployed to GitHub Pages by `.github/workflows/pages.yml`.
`protocol.js` is a third mirror of the protocol, pinned against `protocol.py` by
`tests/test_web_protocol.py` (constants and golden packets). The Node unit tests run locally
via `make test` and are not in CI (Robert, 2026-10-15). Consequences: Chrome/Edge only (no Safari,
Firefox or iOS), and the page has a simulator for everyone else. uPlot is bumped by hand
(`web/ble-viewer/vendor/README.md`).

## ADR-0027 · mediapipe 1.1.1, excluding 1.0.x (Apple Silicon abort) · 2026-10-21 · accepted
Context: the first real MediaPipe run (Christian's M-series Mac, Python 3.11) aborted the process
when `PoseLandmarker` started: `graph_service.h:139 Check failed: service_ Service is
unavailable` in `DrishtiMetalHelper` under `TensorsToDetectionsCalculator::Open`. The lockfile
had resolved `mediapipe>=0.10.14` to 1.0.1, and the vision path had never run anywhere.
`BaseOptions(delegate=CPU)` does not help: the delegate only places inference, and the Mac build
of the detection post-processing opens a Metal helper regardless. Evidence: the same probe aborts
on 1.0.1 in a clean venv and in the repo venv, and runs on 0.10.21 (142 fps), 0.10.35 (178 fps)
and 1.1.1 (221 fps), Lite model, 960×640 still image. A third-party report shows the same abort
with HandLandmarker on 1.0.1 (gladsjr/super-ta#411). Options: (a) pin 0.10.21, the last release
with Intel-Mac (x86_64/universal2) wheels; (b) pin 0.10.35; (c) take 1.1.1 and exclude 1.0.x.
Decision: (c), `mediapipe>=0.10.14,!=1.0.*` in the `vision` extra, locked at 1.1.1 (only mediapipe
changed in `uv.lock`). `tests/test_landmarker.py::test_real_landmarker_starts_on_this_machine`
starts the real graph in a subprocess and fails on 1.0.1; it skips without the extra or
`models/pose_landmarker_lite.task`, so CI does not exercise it. Consequences: mediapipe ≥ 0.10.30
ships macOS wheels for arm64 only, so an Intel Mac cannot install the `vision` extra (true of the
previous lock too); if a teammate has one, revisit (a). Linux x86_64/aarch64 and Windows
amd64/arm64 have 1.1.1 wheels (verify on Robert's WSL2 and a Windows laptop).

## ADR-0028 · Stateful `Pipeline`, one `_Track` per arm, `arm` key only in per-arm mode · 2026-11-04 · accepted
Context: `run_pipeline` (`app/pipeline.py`) was IMU-driven — it looped IMU samples and pulled pose
frames as a side effect, with all angle/rep/rules logic as closures inside that one function.
The live demo has no IMU loop to drive it (webcam-only has no IMU at all) and needs a per-frame
entry point; it also needs each arm counted and judged independently (curl/press/raise with both
arms together, alternating, or one arm), which the existing `side="both"` cannot do because
`primary_angle` takes the mean of the two elbows and halves a one-arm or alternating swing.
Decision: the closures became a `Pipeline` class (`on_imu`/`set_gate`/`on_frame`/`skip_frame`/
`finish`) holding one `_Track` per side; `per_arm=True` builds two tracks (`side="l"/"r"`,
`arm="l"/"r"`) for curl/press/raise and one track (`arm=None`) for squat, while the default
`per_arm=False` keeps the single `side="both"` track replay has always used. `run_pipeline` keeps
its exact signature and is now a thin wrapper: build a `Pipeline`, feed it, call `finish()`. `rep`
and `fault` event payloads gain an `"arm"` key (`"l"`/`"r"`/`None`) only when `per_arm=True`;
replay events (`per_arm=False`) have no `arm` key at all. `tests/test_pipeline_equivalence.py`
pins replay's events and summary against `tests/golden/replay_mmfit_w00_curls.json`, generated
from the unmodified code before the refactor, so the refactor could not silently change replay
output. Two-arm rules (`requires: two_arm`, e.g. `CURL_ASYMMETRY`) only fire for an arm track when
the other arm's current or last rep overlaps this rep by ≥ 50% of the shorter rep's duration
(`Pipeline._paired`), so an alternating or one-arm curl never trips them. Consequences:
`formcoach demo --source replay --headless` is byte-identical (`reps 10 (pose) / 17 (imu) / 10
(fused)`); any change to `Pipeline` must keep the golden test green; a live `Pipeline` never
computes the IMU cross-check (`finish(compute_imu=False)`) because the board is on one wrist and
cannot validate two arms.

## ADR-0029 · Live-demo defaults; the padded metrics window fixes `CURL_PARTIAL_ROM` firing on every rep · 2026-11-04 · accepted
Context: the design (`docs/superpowers/specs/2026-11-02-live-demo-design.md`) asked for one
command, webcam-only by default with the wearable opt-in, reps logged per arm as good or flagged,
and no IMU cross-check live. While building the live loop, feeding a synthetic clean curl through
the shared `rules.yaml` showed the adaptive `RepSegmenter` ends a rep where the angle crosses the
*exit* threshold (about 127-135° for a 35-165° curl), so a rep's `elbow_max` metric reads near
that exit threshold even for a perfect full-range curl; with `CURL_PARTIAL_ROM` defined as
`elbow_min > 75 or elbow_max < 140`, it fired on **every** live curl and no rep was ever "good"
(same root cause as ADR-0022's 0.00 MM-Fit pass rate). Decision: (1) webcam-only
(`AlwaysOnGate`) is the default, `--board ble|serial` is opt-in; (2) with a board, the gate is the
device's gate flag **and** the board having sent a sample within the last 2 s
(`BoardLink.alive`/`gate_open`); (3) good = a rep with no fault of any severity, flagged = ≥ 1
fault; (4) logging is on by default to `runs/live/<timestamp>/` (`--out` moves it, `--no-log`
disables it), writing only `events.parquet`/`frames.parquet`/`session.json`, never camera frames
or video; (5) live `Pipeline` never computes `reps_imu`/`reps_fused` (`finish(compute_imu=False)`
— the board can't cross-check two arms and it's an eval metric, not a demo feature); (6) the
2026-10-21 spike's scale-free swing counter is **not** adopted — the adaptive `RepSegmenter`
stays the only counter; (7) instead of tuning the shared `rules.yaml` or adding a second counter,
`Pipeline` gained `pad_rep_start_s`: a completed rep's *metrics* window (not its count, not
`t_start`/`t_end`/`duration_s`) is extended back up to 1.0 s toward the rest position before the
segmenter's exit-crossing point, never into the previous rep (`Pipeline._padded_start`); live mode
sets `pad_rep_start_s=1.0` (`REP_PAD_S` in `app/live.py`), replay keeps `0.0`, so replay's golden
output is unchanged. Evidence, synthetic curls only: clean 50-150°, 35-165°, slow (4 s) and fast
(1.6 s) curls on both arms all come out good (0 faults, `elbow_max` reads the true top of the
curl, e.g. 150.0°); 90-130° and 40-100° curls are correctly flagged `CURL_PARTIAL_ROM`.
Consequences: **no rep has been scored "good" on a real webcam clip, because none exists yet** —
this fix is unvalidated against real MediaPipe angle noise, occlusion or a non-frontal view.
Rochelle ratifies it (and decides whether a separate `rules.mediapipe-webcam.yaml` is still needed
on top of it) once Christian films a clip; the shared `rules.yaml` itself was not changed and
ADR-0022 still governs per-pose-source calibration.
*Update 2026-11-04 (later):* real webcam runs now exist and reps were scored good on them, but only
under the provisional `rules.mediapipe-webcam.yaml` (ADR-0033, 11 of 33 good); nothing here is
validated against labeled reps.

## ADR-0030 · BLE scan in 2 s rounds with no backoff; `scan_timeout` renamed `scan_round_s` · 2026-11-04 · accepted
Context: `BLESource._find` ran `BleakScanner.discover(timeout=10)` on every attempt, with the
reconnect backoff (1 → 30 s) applied between scans too, so a wearable switched on partway through
a live session could take up to ~40 s to be found — too slow once the demo window is already open
and coaching. Decision: `_find` now loops `BleakScanner.find_device_by_filter` (name prefix
`FormCoach`) in `scan_round_s` (default 2.0 s) rounds with no backoff between rounds, returning
the matched `BLEDevice` so `BleakClient` skips its own re-scan; the backoff (1 → 30 s) is kept
only for reconnecting after a connection error once a device has been found. The constructor
parameter is renamed `scan_timeout` → `scan_round_s` to describe what it now controls. Unit-tested
with a fake `BleakScanner.find_device_by_filter` (`tests/test_ble_find.py`) asserting every round
passes `timeout=2.0` and that the test does not sleep. Consequences: a board powered on late is
found within about one round instead of up to ~40 s; `BLESource(scan_timeout=...)` call sites must
use `scan_round_s` (no other caller in the repo besides the default).

## ADR-0031 · Live per-arm fixes from the pre-PR review: pairing, rep numbers, tracking gaps · 2026-11-04 · accepted
Context: an independent review of `feat/vision-first-run` found live-only behaviours that replay
never exercises. (1) Two arms curling the same range but about two seconds apart got a false
`CURL_ASYMMETRY` on one arm: `Pipeline._paired` compared a rep against the *elapsed* part of the
other arm's rep in progress, so a rep that had only just begun counted as a full overlap.
(2) Each `_Track.reset()` (gate close) starts a new segmenter, so with `--board` the rep numbers
restarted at 0 after every gate re-open and `(arm, rep_id)` repeated in `events.parquet`.
(3) A rep with a long tracking gap inside (arm hidden or out of frame) was still counted and
judged on interpolated angles, against docs/02 §4.3 ("mark the segment invalid"). Decision:
(1) the in-progress branch now measures the overlap against *this* rep's duration (≥ 50%, the
same fraction as the completed-rep branch); (2) in per-arm mode `rep_id` is the arm's running
count (`len(track.reps)`), continuous across gate cycles, and (3) per-arm mode drops a rep whose
window contains a frame with no usable angle, i.e. a gap longer than `MAX_GAP_FRAMES` (3);
replay (`per_arm=False`) keeps the segmenter's ids, which restart on every gate-open segment, and
keeps interpolating, because both are in its golden output. Also: `Pipeline.finish()` closes the
open gate segment in its result only (calling it twice no longer doubles `gated_seconds`); a
`--rules` file is validated up front (malformed, or missing the chosen exercise, is a `ValueError`
before the pose model is built); the live CLI rejects `--gate`/`--speed`/`--session`, `--port`
without `--board`, and `--camera` with `--video`. Consequences: replay output is byte-identical
(`tests/test_pipeline_equivalence.py` plus the replay-style tests in `tests/test_per_arm.py`, which
pin gate-cycle numbering and gap bridging that the golden file does not); arms about half a rep or
more apart no longer count as a two-arm set; a dropped rep leaves no event, only its lost frames
in `frames_invalid`. Unvalidated on a real webcam, like ADR-0029; `rules.yaml` is unchanged.

## ADR-0032 · `PRESS_ASYMMETRY` is a two-arm rule; the live HUD keeps one banner per counter and sends one LED command per frame · 2026-11-04 · accepted
Context: review of PR #14 found two live-only problems. (1) `elbow_top_asymmetry` always compares
both arms, and `PRESS_ASYMMETRY` had no `requires` tag (`CURL_ASYMMETRY` has `two_arm`), so a
one-arm press with the idle arm held at 100° was flagged on every rep (12 reps, 0 good in a 30 s
synthetic set). (2) In a two-arm set both arms finish on the same frame and `run_live` kept one
`banner` and sent one LED command per rep: the arm processed second replaced the first arm's
fault text on screen, and `LED_GOOD_REP` replaced `LED_FAULT` on the device (the firmware's
800 ms fault flash is cancelled by any newer flash). Decision: (1) add `requires: two_arm` to
`PRESS_ASYMMETRY` in `rules.yaml` and `rules.mmfit-pose3d.yaml`. Replay and `eval rules` already
pass `two_arm` (`run_pipeline(side="both")`, `MMFIT_CONTEXT`), so the golden file and
`reports/rules_validation.md` are unchanged; only live per-arm reps are affected, where the tag is
satisfied by `Pipeline._paired`. (2) `HudState.banner` becomes `HudState.banners`, a dict with one
banner per counter (`"l"`, `"r"`, `"all"`) that expires on its own after 3 s or when that counter's
next rep replaces it, drawn stacked (L, R, all). `run_live` sends one LED command per frame
(`LED_FAULT` if any rep completed on that frame has a fault, else `LED_GOOD_REP`) and does not
send `LED_GOOD_REP` within `FAULT_FLASH_S` = 0.8 s (wall clock) of an `LED_FAULT`. Consequences:
a one-arm press is no longer told to lock out both arms; arms finishing a frame or two apart keep
both banners and the fault flash, at the cost that the later arm's good rep gets no LED flash of
its own; `HudState` callers must use `banners`. Regression tests in `tests/test_per_arm.py`,
`tests/test_live.py` and `tests/test_hud.py`. Not changed: protocol, firmware, replay output.

## ADR-0033 · The live webcam demo judges with provisional MediaPipe thresholds (`rules.mediapipe-webcam.yaml`) · 2026-11-04 · accepted (provisional)
Context: Christian's first real webcam curl runs (3 runs, 54 reps, one person, no labels:
`runs/live/20261104-{215742,220059,220224}`) scored **0 good reps**: `CURL_PARTIAL_ROM` fired on 51
and `CURL_SWING` on 51. ADR-0022 had already shown that the design range values fire on the
project's own correct-form MM-Fit reps (the golden replay: `CURL_PARTIAL_ROM` on 10 of 10) because
lifted 3-D pose compresses joint angles, and ADR-0022/ADR-0029 left "which file is v1 for the
webcam path" until the first MediaPipe run. That run shows webcam curls reading `elbow_min` about
88 (82-112) and `elbow_max` about 150, and drifts far above the MM-Fit baseline
(`upper_arm_trunk_drift` median 32° vs 3°, `elbow_drift_max` median 0.25 vs 0.03). Decision: a new
packaged file `rules.mediapipe-webcam.yaml` is `rules.yaml` with two curl rules changed, and
`formcoach demo --camera/--video` uses it by default (`--rules` still overrides; replay keeps
`rules.yaml`, so the golden file and `reports/` are unchanged; `session.json` now records `rules`).
`CURL_PARTIAL_ROM` becomes `elbow_min > 103.869` / `elbow_max < 128.079`, the 5th/95th-percentile
values already in `rules.mmfit-pose3d.yaml`; 8 of the 54 reps are still flagged (the ones that
barely bend the elbow, `elbow_min` 124-137). `CURL_SWING` becomes `> 35°` / `> 0.30`, about the
70th percentile of the 54 reps, on the assumption (Christian: "some of the reps my elbows were
moving forward") that roughly a quarter really swung; it flags 17 of 54 and leaves 28 good.
Consequences: **the swing numbers are a guess, not a calibration.** They come from 54 unlabeled reps
of one person and pass about 70% of those reps by construction, so "more good reps on the same
data" is not evidence. The check that counts uses labeled reps: strict curls (back and upper arms
against a wall) must come out good, deliberately swung curls must be flagged; if they are not, move
the thresholds. Press, raise and squat keep the design defaults and are unvalidated on a webcam
(their range rules will very likely need the same treatment). Rochelle ratifies this, or replaces
it from labeled S1-S5 webcam reps, as ADR-0022/ADR-0029 already say.
*Update 2026-11-04 (first labeled test, round 2).* Christian ran a strict set (back and upper arms
against a wall, `runs/live/20261104-221856`, 22 reps) and then a swingy set with a mix of one-arm and
two-arm reps (`222027`, 9 reps); the labels are his, per run, not per rep. Under the round-1 numbers
the strict set had only 15 good of 22: 6 reps flagged `CURL_PARTIAL_ROM` (right arm
`elbow_min` 104-107, limit 103.869) and 1 `CURL_SWING` (`elbow_drift_max` 0.31, limit 0.30); the
left arm was 8 of 8 good. Strict reps read `upper_arm_trunk_drift` 25-33°, so 35° held; the swingy set
had 4 of 9 reps flagged for swing, all by arm drift (37-44°), but with mixed arms the number of reps
actually swung is unknown, so this gives no detection rate. Revised, still provisional:
`CURL_PARTIAL_ROM` `elbow_min > 115` (above the strict set's 106.8, below the barely-bending reps at
124 and 137) and `CURL_SWING` `elbow_drift_max > 0.35`. These were set using the same two runs, so
they are **unscored**: the pass criteria (strict good with no swing fault; every-rep-swung flagged;
half-range flagged) must be met on new labeled runs before this counts as calibrated.
*Update 2026-11-04 (second labeled round).* On three fresh runs with both arms together, the round-2
numbers gave: strict 16 of 16 good (`222614`), every-rep-swung 15 of 16 flagged for swing
(`222655`), half-range 6 of 25 flagged for range (`222733`). The half-range set was not clearly
half-range (Christian: "a bit relaxed"; 9 of 25 reps read within the strict range), so the 115°
limit is neither confirmed nor refuted and was not changed. Swing margins are thin (strict max 34°,
lowest flagged swung rep 37°, limit 35°). Still provisional: one person, one session per set.
*Update 2026-11-04 (missed reps, logging only).* Christian reported that on a right-arm-solo set
"more than half" of his reps were not counted. The logged run (`223036`) shows the arm was tracked
(27 invalid frames of 3091) and that the counted reps span only 23-52 degrees, close to the
counter's 25 degree `min_range_deg`, but the old `frames.parquet` held no angles, so the cause is
not confirmed. Decision: change no threshold yet; live runs now log each counter's angle,
thresholds, armed flag and arm visibility per frame (extra `frames.parquet` columns, live only,
replay output byte-identical). Whether `min_range_deg`/`window_s` in this file should change waits
for a re-run that shows the cause.
