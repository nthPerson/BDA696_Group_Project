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
