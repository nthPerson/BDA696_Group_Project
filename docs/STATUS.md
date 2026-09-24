# STATUS

Newest entry first. Every work session ends with an entry: what changed, what's next, open
questions, blockers. This is how teammates and future Claude Code sessions pick up context.
Phase checklist: `docs/05-roadmap.md` §2. Checkpoint definitions: `docs/00-START-HERE.md`.

**Current phase:** Phase 1 (Sep 23 – Oct 6) · **Current checkpoint:** 1 done (PR #3 open) → 2 next
**Hardware:** parts arriving 2026-09-25; case not yet designed · **Go/no-go on vision path:** ~Nov 3

---

## 2026-09-24 — Checkpoint 1: public data loads (Claude Code, autonomous run for Robert)

### What changed
- **PR #3 `feat/checkpoint-1-data`** (stacked base for the pre-hardware series; plan in
  `docs/superpowers/plans/2026-09-24-pre-hardware-build.md`).
- `formcoach data fetch`: registry of every raw file (URL, exact size, SHA-256, license),
  resumable downloads, checksum verification, zip extraction, manifest rows. All eight raw
  files verified on disk; `data/MANIFEST.md` rows written.
- `formcoach data convert` → 314 IMUStream Parquet files (MM-Fit 42, RecoFit 126, RecGym 146)
  under `data/processed/<dataset>/streams/` in 41 s. `formcoach data profile` →
  `reports/data_profile.md` + 2 figures (committed).
- Loaders `mmfit.py`, `recofit.py`, `recgym.py`, label map `labels.py`, schema `schema.py`,
  manifest writer, fixture builder (`data/fixtures/`, raw formats, < 400 KB each), 51 tests.
- `docs/04-datasets.md` corrected from the real files (MM-Fit units/joints/subject mapping,
  RecoFit 94 subjects / CDLA license / 7-column labels, RecGym columns + normalised units +
  corrupt UCI zip); ADR-0011 … ADR-0016.
- `Ruby`: read `docs/howto/datasets.md` first, then `src/formcoach/data/schema.py` and
  `src/formcoach/data/recgym.py` (the smallest loader).

### Facts that differed from the docs (all now in docs/04 and DECISIONS)
- RecoFit has **94 subjects**, not 200+; files are MATLAB v5 (scipy); license CDLA-Permissive-2.0.
- MM-Fit smartwatch data is already m/s² and rad/s; pose_3d has 17 joints (H3.6M), pose_2d 18 (COCO).
- RecGym: UCI zip served corrupt → Kaggle mirror; signals min-max normalised (no units);
  columns `Subject, Position, Session, …`; 4,703,320 rows.

### Next
- PR 2 `feat/checkpoint-2-signal` (targets this branch): resample/filters/gravity/windows/
  features, LOSO, energy + RF baselines, peak rep counter, `reports/loso_*.md`,
  `reports/baseline_repcount.md`.

### Open questions
- Should RecoFit's junk-labelled minutes (device taps etc.) be used as extra idle negatives
  instead of dropped? Currently dropped (ADR-0015).

### Blockers
- None.

---

## 2026-09-24 — Pre-hardware build: assessment, plan, data downloads (Claude Code with Robert)

### What changed
- `docs/superpowers/specs/2026-09-24-pre-hardware-build-design.md`: what can be built before
  the boards exist (all of Phase 1, all Phase 2 code, about half of Phase 3), the stacked PR
  sequence (Checkpoint 1 → 2 → 3 → firmware v1 → rules → CNN), and the dataset facts found
  today (URLs, sizes, RecoFit is Git LFS, RecoFit units g/dps/s, Zenodo per-file md5).
- Datasets downloaded to `data/external/` (gitignored): RecoFit `.mat` ×2 + text files,
  RecGym zip; MM-Fit `mm-fit.zip` was still downloading when the session stalled — confirm
  its size is 1,742,309,258 bytes before loading it.
- Branch `feat/checkpoint-1-data` created; no code yet.

### Next
1. Robert: fix the hook path (below), restart Claude Code from the repo root, resume PR 1.
2. PR 1 (Checkpoint 1): `data fetch` verifies/resumes the downloads above and writes manifest
   rows; loaders + `describe()` + fixtures; `data profile`; corrections to `docs/04-datasets.md`.

### Open questions
- RecoFit `.mat` version (v7 → scipy, v7.3 → h5py) is still **(verify)**.
- MM-Fit workout→subject mapping is not on the website; look in the starter repo / paper.

### Blockers
- **Resolved, pending restart (ADR-0010):** `.claude/settings.json` ran the hooks with a path
  relative to the shell cwd; after a Bash call moved the shell into `data/external/recofit`,
  every Bash call was denied by the guard failing to find its own script. Both hook commands
  now use `"$CLAUDE_PROJECT_DIR/.claude/hooks/<name>.py"`. Takes effect on the next Claude
  Code start; this session's changes (design note, STATUS, DECISIONS, howto, settings) are
  uncommitted on `feat/checkpoint-1-data` and should go into PR 1 or a small `chore:` PR.

---

## 2026-09-24 — Claude Code project configuration (Claude Code with Robert)

### What changed
- `.claude/settings.json` (committed, so every clone gets it) registers two hooks:
  - **SessionStart** → `.claude/hooks/session_start.py` prints the branch (warns on `main`),
    the uncommitted-change count and the top `docs/STATUS.md` entry into the session context.
  - **PreToolUse on Bash** → `.claude/hooks/data_guard.py` denies `git add`/`git commit` of
    anything under `data/external|processed|team/` or any `*.mp4 *.npy *.mat *.parquet *.zip`
    outside `data/fixtures/`, `rm`/`find -delete` under `data/`, and `git clean -x` (rules 4, 7).
  - Both are stdlib-only Python run via `uv run --no-sync python`, so they work on a fresh
    clone on Linux, macOS and Windows before `make setup`. `tests/test_claude_hooks.py` (27
    tests) pins the behaviour; `ruff check .` covers `.claude/`.
- Skills: `.claude/skills/firmware-bringup/` (XIAO ESP32-S3 flashing from Windows or WSL2 via
  usbipd-win, expected serial output, IMU wiring check, LiPo text) and
  `.claude/skills/howto-doc/` (the exact shape of a `docs/howto/<topic>.md` hand-over).
- Subagents: `.claude/agents/verify-claim.md` (checks one **(verify)** claim against the
  real file/package/board and reports a doc change) and `.claude/agents/formcoach-reviewer.md`
  (pre-PR review against CLAUDE.md rules and the §5 schema contracts).
- Each skill was written test-first: a subagent did the task without the skill (baseline:
  invented a polars API in the howto; a 367-line hardware procedure with no battery-safety
  text), then again with it. `docs/howto/claude-code-config.md` is the howto produced under
  the skill, kept as the first howto (Bryce: read it first, then the two hook scripts).
- `.gitignore`: `.claude/settings.local.json` (personal overrides) stays local.
- `verify-claim` trial run settled two firmware **(verify)** comments: `LED_BUILTIN` is
  GPIO21 and active-low (installed variant header + vendor page); `Wire` defaults are
  SDA=D4/GPIO5, SCL=D5/GPIO6 (`pins_arduino.h`). The comments in `main.cpp` can drop the
  markers in the next firmware PR; polarity still worth a glance on the real board.

### Next
- Unchanged: Checkpoint 1 (Ruby) and hardware bring-up (Robert, use the `firmware-bringup`
  skill; the Makefile `fw-upload` target still needs a `PORT` argument — add it in that PR).
- Robert: after this PR merges, open `/hooks` once or restart Claude Code so the hooks load.
- Not implemented (offered, declined for now): ruff-on-save, protocol-drift and STATUS-nudge
  hooks; loader/eval-report/session-wrapup skills; PR template, CODEOWNERS, `make ci`.

### Open questions
- CI's `windows-latest` job now runs both hook scripts through pytest (first run failed with
  `UnicodeEncodeError: 'charmap'` because Windows consoles default to cp1252 and STATUS.md
  contains `→`; fixed by forcing UTF-8 stdout in the hooks). A live Claude Code session on
  Windows has not been tried yet; first Windows teammate should confirm the session banner.

### Blockers
- None.

---

## 2026-09-24 — Checkpoint 0: repository skeleton (Claude Code with Robert)

### What changed
- Moved the kickoff docs from `formcoach-kickoff/` into `docs/` and replaced the kickoff
  `CLAUDE.md` with the top-level `CLAUDE.md` (persistent agent directive).
- `pyproject.toml` (uv, Python 3.11, `src/` layout, hatchling), `uv.lock`, `.python-version`.
  Core deps are light; MediaPipe/OpenCV, BLE/serial, TensorFlow/PyTorch (CPU index) and
  Streamlit are optional extras (ADR-0002).
- `src/formcoach/` package with all eight sub-packages documented by docstring, a typer CLI
  (`formcoach --help`) whose commands are labelled stubs that print what they will do, and
  `io/protocol.py`, the Python mirror of the BLE packet contract.
- `firmware/` PlatformIO project (espressif32 6.x, `seeed_xiao_esp32s3`, NimBLE-Arduino 2.x)
  with `include/protocol.h` (packed structs + `static_assert` sizes) and a bring-up
  `main.cpp` that prints chip info, scans I2C for the BMI160 at 0x68/0x69, debounces the
  button, blinks the LED and advertises the FormCoach BLE service. It compiles in CI.
- `Makefile` with `setup lint format test data features train-gate eval demo record fw-*`.
- `.github/workflows/ci.yml`: ruff on Ubuntu; pytest + replay-demo smoke test on Ubuntu,
  macOS and Windows; `pio run` for firmware. `.pre-commit-config.yaml` (ruff, hygiene hooks,
  block direct commits to `main`).
- `README.md` quickstart for Linux/macOS/Windows; `data/MANIFEST.md` template; `.gitignore`
  for raw/processed/team data with the fixtures exception; `docs/devices.md`, `case/README.md`.
- Tests: protocol round-trip and size pins (Python ↔ C header constants), CLI smoke tests,
  repo-contract tests (Makefile targets, gitignore patterns, layout).
- `docs/DECISIONS.md` with ADR-0001 … ADR-0008.

### Next (Checkpoint 1, owner: Ruby with Claude Code)
1. `formcoach data fetch --dataset mmfit` (sensor + pose zip only), SHA-256 verify, manifest row.
2. `data/mmfit.py` loader → common `IMUStream` schema; `describe()`; tiny fixture in `data/fixtures/`.
3. Same for RecoFit (`.mat` via scipy, follow `load_exercise_data.m`) and RecGym (CSV, 20 Hz).
4. `formcoach data profile` → `reports/data_profile.md`.
5. Update `docs/04-datasets.md` wherever the real files differ from the expectations.

### Next (hardware, owner: Robert) — as soon as parts arrive 2026-09-25
1. Flash the bring-up firmware to one bare XIAO: `make fw-upload`, then `make fw-monitor`.
   Confirm the version banner, the LED blink, and that the board shows up as `FormCoach-XXXX`
   in a BLE scanner. Note the I2C scan result before wiring the IMU (should find nothing).
2. Wire one IMU on a breadboard/jumpers, re-run: confirm 0x68 or 0x69. Record in
   `docs/DECISIONS.md` which BMI160 Arduino library builds cleanly (Checkpoint 4).
3. Caliper the real parts and pick the band-width family for the case (`case/README.md`).

### Resolved versions worth knowing (from `uv.lock`, 2026-09-24)
The resolver picked major versions newer than the design doc assumed. Treat the doc's API
notes as **(verify)** against these: `mediapipe 1.0.1` (Tasks API may have moved since 0.10),
`opencv-python 5.0`, `pandas 3.0` (copy-on-write and string dtype defaults changed),
`bleak 3.0`, `tensorflow 2.21`, `torch 2.14+cpu`, `scikit-learn 1.9`, `typer 0.27`, `ruff 0.16`.
Pin tighter in `pyproject.toml` if any of them bites.

### Open questions
- **Packet size:** `docs/02-system-design.md` §3 lists a per-sample `seq` field but computes
  the batch as 5 × 17 + 4 = 89 bytes; with `seq` a sample is 19 bytes and a batch is 99.
  Implemented 19/99 (ADR-0004). Confirm or drop `seq` before firmware v1 ships to devices.
- **License** for the repository (MIT is the natural choice given MM-Fit's code license).
- **Course due dates** for the Written Report Outline / First Draft / Slides / Report are
  still blank in `docs/05-roadmap.md` §1.
- **OneDrive:** Robert's clone lives under OneDrive on a 9p mount from WSL. `.venv` inside a
  synced folder is slow and OneDrive will try to sync thousands of files. Recommend cloning to
  the Linux filesystem (`~/src/formcoach`) or excluding `.venv/` from OneDrive sync.
- Repository is named `BDA696_Group_Project`; README clones it as `formcoach`. Rename is
  Robert's call (GitHub redirects old URLs).

### Blockers
- None. Hardware is not required for Checkpoints 1–3.
