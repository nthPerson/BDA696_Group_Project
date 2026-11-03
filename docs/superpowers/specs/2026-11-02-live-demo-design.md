# Live demo: webcam coaching with an optional wearable (design)

**Date:** 2026-11-02 · **Brainstormed with:** Christian (Claude Code) · **Branch:** `feat/vision-first-run`
**Status:** Christian approved the shape in chat on 2026-11-02 (window layout "B", webcam-only
default with an optional board, each arm counted on its own). A fresh-builder check has been
folded in; what is still a judgment call is listed in §11.
**Next step:** a plan (`superpowers:writing-plans`) in a fresh session, then build.

## 1. Intent

One command gives a teammate live form coaching from the computer-vision pipeline: webcam in,
skeleton drawn on the live video, rep count, elbow angle, and after every rep a coaching message
from `rules.yaml`. The wearable is optional. With it, the board's gate flag wakes the pose model
and the board's LED flashes on reps; without it, the demo runs webcam-only. Each arm is counted
on its own, automatically (both arms together, alternating, or one at a time), and every rep is
logged per arm as good or flagged.

Christian's words for success: *"Them just being able to run the command and get the live
coaching on the CV model."* He will hand this to teammates (Rochelle: vision and rules; Bryce:
app and demo), so it has to run from a fresh clone and be easy to take over.

Why now: the vision path has only ever run on a photo and in a throwaway script, and everything
Christian builds next (a Streamlit dashboard, a Swift app, a RAG coaching bot) needs a live loop
to attach to. The vision go/no-go is ~Nov 3 and feature freeze ~Nov 10. This demo is how the team
first sees pose reps and rules on a real webcam; it does **not** replace the measured go/no-go
(`docs/05-roadmap.md` Phase 3).

## 2. What exists today (verified 2026-11-02, `main` @ 23f192b)

- `src/formcoach/app/pipeline.py`: `run_pipeline(imu, frames, …)` is **IMU-driven**: it loops
  `for s in imu.iter_samples()` and pulls frames from a prebuilt iterator while
  `pending.t <= s.t`. All pose logic (angles → `RepSegmenter` → `compute_rep_metrics` → rules) is
  closures inside that function. There is no per-sample or per-frame entry point. `run_replay`
  wraps it; `make demo` finds 10/10 reps on the fixture.
- `PoseFrame` (`io/replay.py`): `frame, t, world (J,3), visibility, valid, skeleton`. It has **no
  image-plane landmarks**; the existing `Overlay` (`app/overlay.py`) draws `world[:, :2]` on a
  black canvas, not on video.
- `pose/landmarker.py`: `make_landmarker(variant)` returns `detect(frame_bgr, t_ms) -> (img, world)`
  where `img` is (33,4) normalised x, y, z, visibility (or `None`) and `world` is (33,3) metres.
  VIDEO mode: `t_ms` must strictly increase. Only video-file reading exists (`open_video`); there
  is no webcam code in `src/`. `model_path` raises `FileNotFoundError` with a curl hint.
- `cli.py`: `demo --source replay|serial|ble|fake`; anything but `replay` prints a stub message.
  `record --camera` prints "not wired yet". `_open_source` already builds Serial/BLE/Fake IMU sources.
- `io/ble.py` `BLESource`: background thread + queue, `iter_samples()`, `send(Command)`,
  `status`, `stats`. Its `_find` runs `BleakScanner.discover(timeout=10)` on every attempt, and
  backoff between failed scans grows 1 → 30 s, so reconnects can be ~40 s late.
- `app/gate.py`: `make_gate("always_on"|"energy"|"laptop"|"device")`; `DeviceGate` follows the
  sample's `GATE_STATE` flag. `io/protocol.py`: `Command.LED_GOOD_REP = 0x01`, `LED_FAULT = 0x02`.
- `rules/rules.yaml` v1: four exercises; each fault has `code`, `severity` (`warn`|`info`) and an
  ASCII `message`. Rules run on a **finished** rep, so feedback always arrives after the rep.
- `docs/howto/replay-demo.md` is the model for the hand-off doc; it still says the webcam path is
  "written but not run".

## 3. The window (mockup "B", chosen by Christian)

An OpenCV window over the live, mirrored (selfie) video:

1. **Top-left:** the exercise and one counter per arm, in the person's own left/right, e.g.
   `CURL  L 7 (6 good)  R 7 (5 good)`. An arm that has not moved shows 0. Squat shows one counter.
   (The mockup showed a single `reps 7`; this replaces it.)
2. **Top-right status pill:**
   - board connected and gate on: `<device name> | gate ON | pose running`;
   - gate off: `gate OFF | pose paused`, and the video is dimmed (the spike used ~35%) so it is
     obvious the camera is on but pose is idle;
   - board connected before but no sample for >2 s: `board silent - reconnecting`, treated as gate OFF;
   - no board: `webcam only | gate always on`.
3. **Elbow angle** in degrees beside each tracked elbow (the person's own left/right).
4. **Angle trace** along the lower third: the last ~6-10 s of the primary angle, with the rep
   enter/exit thresholds drawn as lines or a band.
5. **Banner after each rep**, prefixed with the arm (`L rep 7:`), visible ~3 s: the fault message (amber for `warn`, grey-blue for
   `info`; if several faults fire show the most severe plus "+N more"), or `Good rep` (green).
   Word-wrapped to the window width.
6. Small `fps` text in a bottom corner.

OpenCV's `putText` (Hershey fonts) is **ASCII-only**: no `°`, `·`, `✓`, `⚠`. Write `deg`, `|`, `-`.
The browser mockup used those glyphs for sketching only.

## 4. Design

Items marked **(builder's call)** are suggestions; the constraints in §6 are not.

### 4.1 Command surface
- `formcoach demo --camera N` runs the live mode, webcam-only. `--camera` is what selects live mode.
- `--board ble` (or `--board serial --port COMx`) adds the wearable: IMU source + `DeviceGate`.
  Default is no board → `AlwaysOnGate`.
- `--exercise curl|press|raise|squat` (default `curl`), `--model lite|full|heavy` (default `lite`),
  `--rules YAML`, `--headless` (no window; print rep/fault events), `--out DIR` (session log, 4.6).
- **No `--side` flag, and no separate commands for one arm or two.** Each arm is tracked on its own
  (4.3). This is Christian's decision.
- `--video PATH` reads frames from a file instead of the camera. It lets a developer or an agent
  exercise the whole live path on a clip, and lets a teammate without a webcam try the demo.
- `--source replay` (the default today) is unchanged: `formcoach demo --source replay --headless`
  keeps working and keeps its output. `--source ble|serial` without `--camera` may stay a stub;
  update its message to point at `--camera`/`--board`.
- Makefile: `make live CAMERA=0 [BOARD=ble]`.

### 4.2 One pipeline, not two
The live path must reuse the existing angles → `RepSegmenter` → `compute_rep_metrics` → rules code,
not fork it (the spike used its own simpler counter on purpose; the repo version must not).
Suggested **(builder's call)**: turn `run_pipeline`'s body into a stateful class with
`on_imu(sample)`, `on_frame(PoseFrame)` and `finish()`, and make `run_pipeline` a thin wrapper.
Replay must stay identical: same `events` and same `summary` on the fixture, pinned by a test
written before the refactor.

Time base: replay uses one session clock. Live uses the laptop's `time.monotonic()` for frames and
`SISample.t_arrival` for IMU samples. Webcam-only has no IMU samples at all, so frames are
processed whenever the gate is on (always). The live demo does **not** compute the IMU cross-check
(`reps_imu`, `reps_fused`): the board is on one wrist, so it cannot validate both arms, and the
cross-check is an evaluation metric, not a demo feature. Replay keeps it. (my call)

Threading **(builder's call)**: main thread owns camera → pose → draw. `BLESource` already has its
own thread and queue; drain it non-blockingly each frame and call `on_imu`, so the pipeline stays
single-threaded. The camera loop must never block on BLE.

### 4.3 Rep counting
Default to the repo's adaptive `RepSegmenter` (`rep_segmentation` in `rules.yaml`: 6 s window,
25 deg minimum range, 30% fraction); it has never run on a live webcam. The 2026-10-21 spike found
fixed elbow thresholds (below 80 then above 140 deg) miss front-camera curls: MediaPipe reads
~70-90 deg at the top and 120-155 deg at the bottom. A scale-free swing counter (elbow closes ≥35 deg
from its running max, then reopens ≥35 deg from its running min) counted 4/4 and 2/2 curls where the
fixed rule counted 1 and 0. Before choosing, run both on a real recording (see "The clip" below).
If the adaptive one misses, tune its numbers in a **new per-pose-source rules file, not in the
shared `rules.yaml`**: that is the project's rule (ADR-0022: thresholds are calibrated per pose
source, `rules.mmfit-pose3d.yaml` is the precedent, `--rules <file>` selects one). The shared
`rules.yaml` is what the pinned replay test and `make demo` run, so it stays byte-for-byte as is.
If a file is not enough, put the swing rule inside `RepSegmenter`, gated so replay output does not
change. Record the choice as an ADR. **Do not ship two counters.**

Rules on a real webcam: ADR-0022 found the design-default range-of-motion rules fire on nearly
every MM-Fit rep, and the spike's curls read ~70-90 deg at the top and 120-155 at the bottom, so
`CURL_PARTIAL_ROM` (elbow_min > 75 or elbow_max < 140) may fire on most live reps and the banner
would nag constantly. Check which rules fire on a clean-form clip. If the range-of-motion rules
fire on everything, write a webcam rules file (e.g. `rules.mediapipe-webcam.yaml`), make it the
live default, and say in the howto and STATUS that Rochelle ratifies it as v1 for the webcam path
(ADR-0022 already leaves that choice to her after the first MediaPipe run).

**The clip.** The counter and rules decisions need a real filmed curl clip, and none exists
(§8 item 7). Christian films it; this is already his open task in the 2026-10-21 STATUS entry
(front view, whole body, ~10 clean curls and ~10 sloppy, with a both-arms-together set, an alternating set and a one-arm set, see "Arms"),
saved under `data/team/` (gitignored; video is never pushed). It is a prerequisite for the
counter/rules decision only, **not** for the rest of the build: build and test everything with the
adaptive segmenter and the default rules first, then run the comparison when the clip exists. If
it does not exist by then, ship the adaptive default and list the open comparison in STATUS. **Do
not wire `formcoach record --camera` to get the clip**; any camera app will do, and `--video`
plus `pose extract` consume the file.

**Arms (decided by Christian): each arm is counted dynamically, with one command.** Curling both
arms together, alternating, or one at a time must all just work, with no flag, and every rep is
logged per arm. (The spike already counted 7 curls per arm live with one counter per arm.) The
repo's `side="both"` would not do: `primary_angle` takes the **mean of the two elbows**, which
halves the swing for a one-arm or alternating curl. So for the three arm exercises (curl, press,
raise) run **one segmenter, one angle buffer and one smoother per side**, each on its own side's
primary angle (`primary_angle(..., side="l")` and `"r"`). An arm that does not swing at least
`min_range_deg` (25 deg) never produces a rep, so an idle arm stays at 0 on its own. This forces
three things:

1. **Per-arm metrics.** `metrics_from_angles` computes the metrics the rules read (`elbow_min`,
   `elbow_max`, `upper_arm_trunk_drift`, `abd_peak`, …) from `0.5 * (left + right)`, so a rep from
   one arm judged that way would be diluted by the idle arm. Give it a `side` argument (default
   `both`, so replay output is identical) and compute a left rep's metrics from the left series
   and a right rep's from the right. Rules then run per arm and each fault belongs to an arm.
2. **Per-arm visibility.** A frame is valid for an arm only if its shoulder, elbow and wrist all
   have visibility >= 0.5 (the spike's `MIN_VIS`). An arm that is out of frame or occluded counts
   nothing instead of being guessed. Bridge gaps of <= 3 frames per arm (docs/02 §4.3).
3. **Two-arm rules.** `requires: two_arm` rules (`CURL_ASYMMETRY` compares left and right over the
   rep) only make sense when both arms are working; with one arm idle they would always fire, and
   `run_pipeline` today sets the `two_arm` context only for `side == "both"`. Evaluate them only
   when the left and right reps overlap by >= 50% of the shorter one (the `_fuse` overlap test),
   otherwise skip them. If pairing proves messy, defer these rules in v1 and say so in the howto
   and STATUS; never let them fire on a one-arm curl. (my call)

Squat is bilateral, so it keeps a single counter on the mean knee angle (`arm` is `null`). (my call)

### 4.4 Overlay
Add an optional `image: np.ndarray | None = None` field to `PoseFrame` (image-plane landmarks:
normalised x, y, visibility) so the overlay can draw on the video. Add, never rename. The drawing
function should return the finished BGR frame so tests can inspect it without a window; showing it
is a thin wrapper. Keep the old `Overlay` working for `--source replay` without `--headless`, or
replace it. **(builder's call.)**

### 4.5 Wearable link
- Port the spike's `FastBLESource._find` into `io/ble.py`: scan in 2 s rounds with
  `BleakScanner.find_device_by_filter` (name prefix `FormCoach`), return the `BLEDevice` so
  `BleakClient` skips its own re-scan, no backoff while scanning. Unit test with a fake scanner.
- Gate with a board = the sample's gate flag **and** the board alive (a sample within 2 s).
- LED: `LED_GOOD_REP` after a rep with no faults, `LED_FAULT` after a rep with faults, from either arm. **No
  protocol or firmware change.**

### 4.6 Session log, per arm
Christian wants reps logged, good and bad, with counts, for each arm.
- Every live run writes `events.parquet`, `frames.parquet` and `session.json` through the existing
  `EventLog.write` to `runs/live/<YYYYMMDD-HHMMSS>/`. `--out DIR` overrides the location and
  `--no-log` turns logging off. On by default so the dashboard and coaching bot have data without
  extra flags. (my call)
- `rep` and `fault` events gain a payload key `arm`: `"l"`, `"r"`, or `null` for squat. Add keys,
  never rename: `rep_id`, `metrics`, `code`, `severity`, `message` stay as they are, because the
  dashboard and bot will read them later. `rep_id` counts per arm. (builder's call)
- `session.json` gets an `arms` object, in live runs only, e.g.
  `{"l": {"reps": 7, "good": 6, "flagged": 1, "faults_by_code": {"CURL_TOO_FAST": 1}}, "r": {…}}`.
- **Good** = a rep with no fault of any severity; **flagged** = at least one fault. Events keep
  `severity`, so a dashboard can relax "good" later. (my call)
- The same per-arm totals print when the demo exits.
- **The demo never writes camera frames or video to disk.**

### 4.7 Teammate experience
- `make setup` installs no extras. Add `make setup-live`: `uv sync` with `vision` + `device`, and
  fetch `models/pose_landmarker_lite.task` if missing, in Python, not `curl`, so it works on Windows.
  **(my call; see §9.)**
- Friendly errors, one line each with the fix: camera cannot open (the macOS "allow your terminal
  under Privacy & Security > Camera" text from the spike); `vision` extra missing; model missing.
- On Christian's Mac camera 0 is the iPhone Continuity Camera and shows black; the spike used
  `--camera 1`. If the first ~30 frames are black (mean < 2), print: `camera N looks black (iPhone
  Continuity Camera?) - try --camera N+1`.
- Docs: `docs/howto/live-demo.md` (use the `howto-doc` skill; owners Bryce and Rochelle), README
  command table, `docs/STATUS.md` entry naming what teammates read first, ADRs from 0028 (§8 step 2).

## 5. Goals, non-goals, anti-goals

**Goals**
- `formcoach demo --camera N` works on macOS (Apple Silicon), and is written to work on Windows and
  Linux, from a fresh clone, with no board.
- Each arm is counted and judged on its own, automatically (both arms together, alternating, or
  one at a time), and logged per arm as good or flagged.
- The same command with `--board ble` uses the wearable to gate pose and flash the LED.
- Curl is the validated exercise; the other three run through the same code via `--exercise`.
- The BLE reconnect fix lands with a test.

**Non-goals (this work)**
- Streamlit dashboard, Swift app, RAG/LLM coaching bot (Christian's next steps; see §10).
- Mid-rep cues (rules judge finished reps), new rules, re-calibrated thresholds, new exercises.
- Firmware, BLE protocol, or pin-map changes. Fixing board 8428's dropouts.
- Recording sessions inside the demo (`formcoach record` exists), training, evaluation tables.
- Per-leg squat counting; the IMU cross-check (`reps_imu`, `reps_fused`) in live mode.
- More than one person in frame.

**Anti-goals (a failure even if it technically works)**
- A demo that needs the board, a GPU, a network, or a paid API.
- Any change to `formcoach demo --source replay --headless` output, to the shared `rules.yaml`, or
  a red `main`.
- `opencv`/`mediapipe` imported at module top level in anything the core install imports
  (they are optional extras; import lazily with an install hint).
- Any import of `formcoach.pose` from `formcoach.signal` (keeps the IMU-only fallback buildable).
- Video of a teammate written to disk or pushed anywhere.

## 6. Constraints that bite (from `CLAUDE.md`)

Branch per task, conventional commits, `make lint test` green before a PR, PR description a
teammate can follow, then a `docs/STATUS.md` entry and `docs/DECISIONS.md` ADRs for every choice
between alternatives. `uv` with the committed `uv.lock`. Paths via `pathlib`; no shell tricks in
Python; serial ports `/dev/ttyACM0`, `/dev/cu.usbmodem*`, `COMx`. Public APIs get docstrings with
units (m/s², rad/s, degrees, seconds). CI has neither the `vision` extra nor the pose model, so new
tests must run without them (inject a fake detector and synthetic `PoseFrame`s) or
`pytest.importorskip`. Work is on a branch and goes through a PR; ask Robert before expanding
scope (see §10).

## 7. Known risks

- **Board FormCoach-8428 (fw 1.1.0, on battery) stopped streaming 5 times**, 12-25 s after
  switch-on, mid-curl, and stayed off until switched back on (2026-10-21). Suspected hardware
  (cell, wire, switch). Unresolved as far as `docs/STATUS.md` on `main` shows; fw 1.1.1 fixes the
  BMI160 range bug, a different problem. The live loop must survive a silent board, and the
  webcam-only path is the safety net.
- **WSL2 cannot see a webcam or BLE adapter** by default; Robert would run the live demo from
  Windows Python. (verify)
- **Intel Macs cannot install the `vision` extra** (mediapipe ≥ 0.10.30 ships arm64-only macOS
  wheels; ADR-0027 in the stash).
- MediaPipe VIDEO mode needs strictly increasing millisecond timestamps (the spike used
  `max(int((now - t0) * 1000), last_ms + 1)`).

## 8. Starting state (do these first)

1. **Branch.** `feat/vision-first-run` points at `eb53756`: 0 commits of its own, 11 behind `main`.
   `git merge --ff-only main`.
2. **Christian's 2026-10-21 work is in `git stash@{0}`, not in the tree.** It holds
   `pyproject.toml` (`vision = ["mediapipe>=0.10.14,!=1.0.*", …]`), `uv.lock` (mediapipe 1.1.1),
   `tests/test_landmarker.py::test_real_landmarker_starts_on_this_machine` (starts the real graph in
   a subprocess; skips without the extra or the model), `docs/DECISIONS.md` (**ADR-0027**) and a
   `docs/STATUS.md` 2026-10-21 entry. Without it, `uv sync --locked --extra vision` on Apple Silicon
   installs mediapipe 1.0.1, which aborts at graph start. `git stash pop` after the fast-forward;
   `docs/STATUS.md` and `docs/DECISIONS.md` changed on `main` meanwhile, so expect conflicts there:
   keep both entries, newest first. Commit it first as its own `fix:` commit, ideally its own PR
   (teammates on Macs need it regardless). Do not drop the stash until it is committed.
3. `uv sync --locked --extra vision --extra device`, then `make lint test` for a baseline (192
   passed, 1 skipped on 2026-10-21; `main` has added tests since).
4. `models/pose_landmarker_lite.task` is present locally and gitignored.
5. **Reference spike:** `docs/superpowers/specs/2026-11-02-live-demo-spike-reference.py.txt`. The
   original lived in `/private/tmp` and was wiped; this is rebuilt from the session transcript
   (9 of 24 edits did not re-apply; duplicates removed by hand; compiles; **never re-run**). Use it
   to see the intent: the board gate with a silence timeout, the swing counter, skeleton drawing on
   the frame, mirrored display, macOS capture flags (`CAP_AVFOUNDATION`, 1280x720), LED on rep. It
   is not product code: do not import it, copy it wholesale, or ship it.
6. **Untracked clutter:** `SESSION-SUMMARY.md` (Christian's notes; do not commit unless asked);
   `runs/S4/…` (a `source: fake` test recording). `runs/` is not gitignored although the docs use
   `runs/<name>` for `--out`; add `runs/` to `.gitignore`.
7. **Real data for testing:** `data/team/S4/20261021-172411/` is a 6.5 s palm-held BLE recording
   (IMU only; the board died at 12 s uptime). There is **no** filmed curl or squat video yet;
   Christian films the clip described under "The clip" in §4.3.

## 9. How to know it works

**Automated**
- `make lint test` green. `formcoach demo --source replay --headless` still finds 10/10 reps on the
  fixture with an identical summary (a test pinned before any refactor).
- New tests: pipeline equivalence (class vs wrapper); live loop fed injected frames and a fake
  detector yields the expected reps and faults; banner wrapping; overlay returns a drawn frame
  (needs OpenCV: `importorskip`); fake BLE scanner for the 2 s rounds; board-silent → gate off.
- Per-arm tests with synthetic frames: only the left arm swings → L = n, R = 0; both together →
  n and n; alternating → n and n with no overlap and no `two_arm` rules; an arm below the
  visibility floor counts nothing; a left rep's metrics ignore the right arm.
- Smoke on a clip: `formcoach demo --camera 0 --video <clip> --headless` prints rep and fault
  events. An agent can run this without a camera, with any short video of a person doing curls.

**Manual (Christian, on the Mac)**
1. `uv run formcoach demo --camera 1`, one command for all three: (a) 10 curls with both arms
   together give L 10 and R 10 (±1); (b) 10 alternating curls give about 10 per arm; (c) 10 curls
   with one arm give 10 on that arm and **0** on the other. The window opens, the skeleton sits on
   the video, a banner names the arm after each rep, and it runs at roughly 20+ fps on the Lite
   model (the spike saw ~30; default, calibrate). The totals printed on exit and in
   `runs/live/…/session.json` match what you counted.
2. With `--board ble`: the pill shows the board name; pose runs while curling and pauses (dimmed)
   when still; the LED flashes on a good rep; switching the board off turns the pill to "board
   silent" and the window keeps running.
3. A second machine (a teammate): clone → `make setup-live` → `make live`.

Record the numbers in `docs/STATUS.md`.

## 10. Later, and a flag for Robert

After a working minor demo, Christian moves on to a dashboard, a Swift app and a RAG bot that
summarises outputs and gives more detailed coaching. None of that is in this work; §4.6 is the only
seam left for it. Per `CLAUDE.md`: the Streamlit dashboard is in the roadmap (Phase 3, Bryce);
LLM coaching sentences are a stretch goal, off-limits until feature freeze (~Nov 10), local
Ollama or a key Robert provides, nothing paid in the core path; the Swift app is in no doc. Working
rule 7 says to ask Robert before expanding scope beyond frozen v1.

## 11. Calls I made (Christian: please check these)

- Flag spelling: `--camera`, `--board`, `--video`, `make live`. Christian approved "something like
  `formcoach demo --camera 0`"; the rest is mine.
- Webcam-only is the default and the board is opt-in, so a teammate without a kit gets everything.
- `--video PATH` exists so the live path can be tested and demoed without a camera.
- LED: good rep → good, any fault → fault (the spike only flashed good reps).
- The pill text, the ~2 s silence timeout, the ~3 s banner, the black-frame hint.
- `make setup-live` with a Python model fetch.
- Curl validated first; the other three exercises reachable via `--exercise` but untested live.
- Adaptive `RepSegmenter` is the default candidate, the swing counter the fallback, decided by a
  real recording.
- Landing the mediapipe pin as its own commit/PR before the live work.
- Attaching the reconstructed spike as a `.txt` reference inside `docs/` (outside ruff's reach).
- Live-only tuning goes in a separate per-pose-source rules file (ADR-0022); the shared
  `rules.yaml` and replay output never change. Rochelle ratifies the webcam file.
- The filmed clip is Christian's and gates only the counter/rules decision, not the build;
  `record --camera` stays unwired.
- Per-arm tracking covers curl, press and raise; squat keeps one counter. Two-arm rules run only on
  overlapping left/right reps and may be deferred. No IMU cross-check in live mode.
- "Good" = no fault of any severity. Logging is on by default to `runs/live/<timestamp>/`
  (`--out` to move it, `--no-log` to stop it). Events gain an `arm` key and `rep_id` counts per arm.

## 12. Left to the builder

Class vs functions for the pipeline; module and file names; thread vs poll; colours, fonts and
layout numbers; trace window length; the exact rep-counter outcome; how webcam-only reports the IMU
cross-check; wording of messages; ADR numbering and count; test structure; how left and right reps are paired
for the two-arm rules; whether `rep_id` is per arm or global in the log.
