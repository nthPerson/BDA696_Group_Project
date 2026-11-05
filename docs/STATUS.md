# STATUS

Newest entry first. Every work session ends with an entry: what changed, what's next, open
questions, blockers. This is how teammates and future Claude Code sessions pick up context.
Phase checklist: `docs/05-roadmap.md` §2. Checkpoint definitions: `docs/00-START-HERE.md`.

**Current phase:** Phase 3 (Oct 28 – Nov 17) · **Current checkpoint:** 1–3 + software of 4–6 merged → flash and test the assembled kits
**Hardware:** K1 fully verified 2026-10-08 (IMU, BLE, gate); more kits assembled, flashing next with fw 1.1.1; see `docs/assembly-day.md` · **Go/no-go on vision path:** ~Nov 3

---

## 2026-11-04 — Live webcam rules: provisional MediaPipe thresholds for curl (Claude Code with Christian)

### What changed
- Christian's first real webcam curl runs scored 0 good reps out of 54 (`CURL_PARTIAL_ROM` 51,
  `CURL_SWING` 51). New packaged `src/formcoach/rules/rules.mediapipe-webcam.yaml` (ADR-0033): a
  copy of `rules.yaml` with `CURL_PARTIAL_ROM` at `elbow_min > 103.869` / `elbow_max < 128.079`
  (the MM-Fit-calibrated values) and `CURL_SWING` at `> 35°` / `> 0.30` (round 1; revised after the
  first labeled test, see below). `formcoach demo
  --camera/--video` uses it by default; `--rules` overrides; replay still uses `rules.yaml`.
  `session.json` records which file judged the run (`rules`).
- Tests: `tests/test_rules.py` (the file differs from the defaults in exactly those two rules;
  webcam-range and swinging reps against both files), `tests/test_live.py` (default file and the
  `rules` key; a 88-150° curl is good under the webcam rules and flagged by the defaults; a
  barely-bending curl is still flagged).
- `docs/howto/live-demo.md` and `docs/02` §7 point at the new file.

### Verified
- Against the 54 logged reps: `CURL_PARTIAL_ROM` 51 → 8 flagged, `CURL_SWING` 51 → 17 flagged,
  good reps 0 → 28 (before two-arm asymmetry). **This is by construction** (the swing threshold is
  the ~70th percentile of the same reps), so it shows the thresholds are in range, not that they
  are right.
- `ruff check` and `ruff format --check` clean; **370 passed, 1 skipped** (TensorFlow not
  installed; 361 before this entry). Mutation check: replacing the webcam file with the defaults
  fails 4 tests, and making the live demo ignore the file fails the default-file test.
- Replay is unchanged: no diff in `tests/golden/` or `reports/`, and `rules.yaml` is untouched.

### Christian's test of the new rules (reported 2026-11-04)
Christian: he ran it once for both the strict and the swingy test, "it works better", but "the
media pipe outline is kinda shaky". He gave no counts; these are read from
`runs/live/20261104-221325/session.json` (curl, `rules.mediapipe-webcam.yaml`, `lite`, 141 s,
29.6 fps mean): **L 16 reps (5 good), R 17 reps (6 good)**, `CURL_SWING` 19, `CURL_PARTIAL_ROM` 5,
`CURL_TOO_FAST` 5, `CURL_ASYMMETRY` 2; earlier runs under the design defaults were 0 good of 54.
The log does not say which reps were strict and which swingy, so this is **not** the labeled
check; the 11 good reps sit at 28-34° / 0.19-0.30, right under the 35° / 0.30 swing limits.
- Shaky skeleton outline: **recorded, not fixed** (Christian: "maybe we can just put it in the
  comments"). Cause unconfirmed. Known: the HUD draws raw per-frame landmarks, the landmarker is
  `VIDEO` mode, `lite` is the least stable size. Untested guess: the same noise inflates the
  drift metrics. Options for later, none done: smooth only what is drawn (cosmetic, replay-safe),
  try `--model full`, or smooth the angle series before the metrics (changes results, needs its
  own ADR and a replay check). Owner: Christian/Rochelle to decide. Also in `docs/howto/live-demo.md` §6.

### First labeled test and round-2 numbers (Christian, 2026-11-04)
Christian ran the two tests I asked for, in order: strict (back and upper arms against a wall)
= `runs/live/20261104-221856`, then swingy = `222027`, where he did "a mixture of both arms and 1
arm", so the swing label is per run, not per rep. He also said the first (strict) test "went
well"; the log agrees on swing. Both runs were judged by `rules.mediapipe-webcam.yaml`.
- **Strict, 22 reps (L 8, R 14): 15 good.** Left arm 8 of 8 good. Swing clean on 21 of 22 (arm
  drift 25-33°; the one hit was `elbow_drift_max` 0.31 vs 0.30). The 6 other flags were
  `CURL_PARTIAL_ROM` on the right arm, `elbow_min` 104-107 vs the 103.869 limit.
- **Swingy, 9 reps (L 4, R 5): 2 good.** 4 flagged for swing, all right arm, all by arm drift
  (37-44°); the 4 left-arm reps read 28-33° like strict, but with mixed arms we do not know they
  were swung, so this is **not** a detection rate. Tempo (`CURL_TOO_FAST`) hit 4 of 9.
- **Revised (still provisional):** `CURL_PARTIAL_ROM` `elbow_min > 115`, `CURL_SWING`
  `elbow_drift_max > 0.35`; arm drift 35° and `elbow_max < 128.079` unchanged. They remove exactly
  the 7 strict misses and keep the earlier barely-bending reps (124, 137) flagged. **Unscored:** set
  using the same two runs, so they need a new labeled round. Pass criteria: strict (both arms) good
  with no swing fault; every-rep-swung (both arms) flagged; half-range flagged.
- Checks that ran: rules file confirmed in both `session.json`; per-arm rep counts read;
  `CURL_ASYMMETRY` fired 0 times (not tested by this). Did not run: partial-range labels, a
  per-rep swing label, a held-out strict set.

### Round 3: re-test of the round-2 numbers (Christian, 2026-11-04, three fresh runs, both arms together)
Order as asked: strict, every-rep-swung, half-range. Christian: "first two went well, last one seemed
a bit relaxed" (the log cannot say whether that means his half-range curls or the rule). All three
`session.json` name `rules.mediapipe-webcam.yaml`.
- **Strict** `20261104-222614`: 16 reps, **16 good**. `upper_arm_trunk_drift` 30-34 (limit 35),
  `elbow_drift_max` up to 0.31 (limit 0.35), `elbow_min` 88-93. Pass.
- **Every-rep-swung** `222655`: 16 reps, 1 good, **15 of 16 flagged `CURL_SWING`** (left 7 of 8,
  right 8 of 8; flagged reps read 37-73°). The unflagged one read 34° / 0.27. Pass.
- **Half-range** `222733`: 25 reps, 16 good; only **6 of 25 flagged `CURL_PARTIAL_ROM`**
  (`elbow_min > 115`), plus `CURL_TOO_FAST` on 5. `elbow_min` median 111 (strict 91), but 9 of 25
  read 106.8 or less, the strict maximum seen. **Inconclusive** for the 115 limit: that set was not
  clearly half-range, so it is not used to move the limit. Not changed.
- Margins are thin: strict arm drift tops out at 34 against the 35 limit, and the lowest flagged
  swung rep is 37. The same strict curl read `elbow_min` 88-93 in one session and 104-107 on the
  right arm in another, so absolute `elbow_min` limits move with camera setup (one person, one
  session each).
- Checks that ran: labeled strict and every-rep-swung sets on fresh runs, rules file confirmed.
  Did not run: a clean half-range set, a second person, a different camera position.

### Round 4: right-arm solo missed reps (Christian, 2026-11-04) - open
Christian: "when I did right arm solo, more than half of my reps didn't get registered". He gave no
count of what he did. From `runs/live/20261104-223036` (104 s, `rules.mediapipe-webcam.yaml`, 29.8 fps,
3091 frames, `frames_invalid` 27): L 10 reps (4 good), R 8 reps (0 good). Right-arm reps start at
4.6, 7.5, 10.4, 15.1, 18.1 s, then **none between 18 s and 66 s**, then 66.4, 70.2, 87.9 s; left reps
run 4.8-48.1 s. The log cannot say which stretches were solo or how many reps he did.
- Ruled out as the main cause: tracking. `frames_invalid` is 27 of 3091 frames (max over the two
  arms), so the right arm was visible nearly all the time.
- Suspect, **unproven**: the adaptive counter (`RepSegmenter`) does not count anything until the last
  6 s of angle spans `min_range_deg` 25 (10th to 90th percentile). The counted right-arm reps span
  only 23-52 deg (elbow max minus min; most 31-37), so small or relaxed reps near 25 deg, or one
  rep after a pause, can be silently ignored. The old log kept no angles, so this could not be checked.
- **Changed (logging only, no behaviour or threshold change):** live runs now write the counter state
  per frame to `frames.parquet` (`ang_*`, `enter_*`, `exit_*`, `armed_*`, `vis_l/vis_r`; documented in
  docs/02 §4.6 and the live-demo howto §6, §9). `RepSegmenter.armed` exposes the flag.
- Verified: `make lint test` clean, **377 passed, 1 skipped** (TensorFlow not installed) plus the 21
  viewer tests. New tests in `tests/test_per_arm.py` (columns and values per arm, NaN on gate-closed
  frames, small vs large movement arms or not, replay-style columns stay `t, processed, gate_state`,
  `armed` flag) and `tests/test_live.py` (a real `run_live` writes them). Mutation checks: dropping the
  diagnostics fails 2 tests, logging them for replay-style fails 1, forcing `armed` true fails 3.
  Replay: `formcoach demo --source replay --headless` on the working tree vs the previous commit
  (`a295cdf`, detached worktree): `events.parquet` and `frames.parquet` byte-identical, `session.json`
  differs only in the fixture's absolute path; no change under `tests/golden/` or `reports/`.
  Real webcam run: Christian reported "appears to be working good. I did one or two good reps just
  to test" (no counts; a smoke test, **not** the right-arm-solo re-run). The newest log
  (`20261104-233200`, 66.7 s, 29.8 fps, `rules.mediapipe-webcam.yaml`; L 5 reps, R 11) has more reps
  than that, so which run he meant is unconfirmed. In it all 12 new columns are present with sensible
  values (`ang` 92-165 deg, `armed` 0/1, `vis` 0-1). Right arm armed on 61% of frames, left 24%. The
  right arm was unarmed in three 2.7-6 s stretches (20.5-26.6, 39.0-41.7, 58.9-63.0 s) in which its
  angle moved 31-36 deg with no rep counted; that fits the suspected cause but **does not prove it**,
  because we do not know what Christian was doing in them. Not verified: the baseline for the
  replay comparison is the branch's previous commit, not `origin/main`.
- **Next, needs Christian:** one right-arm-solo set (and ideally a left-arm one) with this logging,
  counting your reps out loud or noting the number done. Then read `armed_r`, `vis_r`, `ang_r` for
  the uncounted stretches. Only if `armed_r` is 0 while curling is `rep_segmentation.min_range_deg` /
  `window_s` in `rules.mediapipe-webcam.yaml` worth changing (tradeoff: more jitter "reps"), and that
  needs its own labeled re-test.

### Known, not changed
- **The webcam curl thresholds are provisional**: swing now passed a fresh labeled test (strict
  16 of 16 good, swung 15 of 16 flagged) but with thin margins and one person; the range limit is
  untested (half-range set inconclusive). Needs a clean half-range set (forearm horizontal on
  every rep, both arms) and more subjects. Owner: Rochelle ratifies or replaces
  (ADR-0022/0029), with Christian recording the labeled runs.
- Press, raise and squat still use the design defaults, which will very likely misfire on a webcam
  the same way (their range rules). No webcam data for them yet. Owner: Christian/Rochelle.
- Several logged "reps" last 0.5-0.7 s and barely bend the elbow (jitter counted as a rep by the
  segmenter). Not addressed. Owner: Rochelle.
- `frames.parquet` of runs recorded before this change stores no angles, so those runs (every one up
  to and including `20261104-223036`) cannot be re-judged under new thresholds (only the per-rep
  metrics in `events.parquet` can). `20261104-233200` is the first with the columns. New live runs
  log the counter state per frame (see "Round 4"). Replay's `frames.parquet` is unchanged.
- Right-arm-solo missed reps: **open, cause not confirmed** (see "Round 4"). Owner: Christian re-runs;
  Rochelle decides any `rep_segmentation` change.

## 2026-11-04 — Two fixes from Robert's PR #14 review (Claude Code with Christian)

### What changed
- **One-arm press no longer flagged `PRESS_ASYMMETRY` on every rep.** The rule compares both arms
  but had no `requires: two_arm` tag (`CURL_ASYMMETRY` has one). Added it in `rules.yaml` and
  `rules.mmfit-pose3d.yaml`; `docs/02` §7 now says "two-arm presses only". Replay and
  `eval rules` already run with `two_arm`, so their output is unchanged.
- **Two-arm set: both banners and the fault LED survive.** `HudState.banner` is now `banners`
  (one per counter, own 3 s expiry, drawn stacked L over R). `run_live` sends one LED command per
  frame (`LED_FAULT` if any rep that frame has a fault) and holds `LED_GOOD_REP` back for 0.8 s
  after an `LED_FAULT` (`live.FAULT_FLASH_S`, the firmware flash length).
- Tests: `tests/test_per_arm.py` (one-arm press all good; two-arm press still flagged on both
  arms), `tests/test_live.py` (left fault + right clean on the same frame, one frame later and one
  frame earlier; good-rep LED returns after the flash), `tests/test_hud.py` (stacked render).
  `test_led_commands_follow_rep_quality` now passes `clock=` because the guard is in wall seconds.
- ADR-0032 in `docs/DECISIONS.md`; `docs/howto/live-demo.md` §8 describes the LED/banner rules.

### Verified
- Reproduced first: Robert's one-arm press script gave `l`: 12 reps, 0 good, `PRESS_ASYMMETRY` x12.
  After the rule change the same set is 12 reps, all good, and `PRESS_ASYMMETRY` still fires on both
  arms when two arms press 170 deg vs 130 deg.
- `make lint test` green: ruff check and format clean; **361 passed, 1 skipped** (TensorFlow not
  installed) in pytest, 21/21 in the Node viewer tests. Was 354 passed before this entry (+7 new).
- Replay is unchanged: `tests/golden/` and `reports/` have no diff, and the golden replay and
  `test_pipeline_equivalence.py` pass.
- Mutation check on the new live tests: guard off (`FAULT_FLASH_S = 0`) fails the one-frame-late
  case; one LED command per rep fails the same-frame and late cases; one shared banner fails all
  three. The one-arm press test fails (0 good of 12) with the rule line removed.
- Not run on hardware: the LED flash on a real device, and the stacked banners over a real webcam
  frame (rendered once from synthetic data only). The `test-vision` CI job still needs the branch pushed.

### Known, not changed
- Serial reader thread dying silently; `session.json` `reps` counting both arms; the 1 s
  reach-back applying to live only; the minor review items. Owner: Robert, follow-up PRs (his call
  in the review: "not needed in this PR").
- A long wrapped left-arm banner can overlap the bottom of the angle-trace panel when two banners
  are stacked (readable, not moved). Owner: Christian, to look at on a real webcam frame.

## 2026-11-04 — Pre-PR review fixes on `feat/vision-first-run` (Claude Code with Christian)

### What changed
A second independent review of the whole branch (after `e85bcce`) found live-only problems that
replay never reaches; fixed here, each with a regression test (ADR-0031):
- Two arms curling the same range about two seconds apart no longer get a false `CURL_ASYMMETRY`
  on one arm (`Pipeline._paired`, in-progress branch).
- With `--board`, rep numbers keep counting after the gate re-opens (they restarted at 0, so
  `(arm, rep_id)` repeated in `events.parquet`). Replay keeps its restart: the golden output has it.
- A live rep with a tracking gap of more than 3 frames inside it is not counted (docs/02 §4.3)
  instead of being judged on interpolated angles. Replay still interpolates.
- `--rules` file problems (malformed, or missing the chosen exercise) are a clean exit 2 before
  the pose model loads, not a `KeyError` after the first rep.
- Import errors for OpenCV, MediaPipe, bleak and pyserial now include the real cause (e.g. a
  missing `libGL.so.1` on WSL2) after the install hint.
- `Pipeline.finish()` can be called twice; an unopenable `--video` leaves no empty session
  directory; the camera is released before the wearable link shuts down; headless `--board` prints
  `wearable: <status>` lines; the live CLI rejects `--gate`/`--speed`/`--session`, `--port` without
  `--board`, and `--camera` with `--video` (they were silently ignored).
- Tests: the golden replay test does not pin gate cycling or gap bridging (mutating
  `MAX_GAP_FRAMES` or the reset on gate close still passed it), so `tests/test_per_arm.py` now pins
  both for the replay-style pipeline; new tests cover `WindowShow` (q, Esc, closed window),
  `run_live_demo` cleanup on every exit path and the default `BoardLink` clock. CI gets a
  `test-vision` job (Ubuntu, `--extra vision`) because the existing jobs skip every OpenCV test.

### Verified
`make lint test` green (354 passed, 1 skipped: TensorFlow not installed). `formcoach demo --source
replay --headless` compared against `origin/main` (23f192b): `events.parquet` and `frames.parquet`
byte-identical, `session.json` differs only in the absolute fixture path. The new `test-vision`
CI job has **not run yet** (needs the branch pushed).

### Known, not changed
`make live` ignores `POSE_MODEL` (`setup-live` honours it); `run_pipeline(side="l"|"r")` computes
different metrics from main (no caller passes `side`); live squat has no visibility gating (same
as main); `Pipeline`'s docstring misses `per_arm`/`max_buffer_frames`/`pad_rep_start_s`;
`fetch_model` has no integrity check and `landmarker.py`'s header still says `make setup` downloads
no models. Still needing hardware: on-screen L/R against the physical arm, real MediaPipe
`visibility` against the 0.5 floor, window close on Windows/Linux, `--board` with real IMU samples.

---

## 2026-11-04 — Live webcam demo (Claude Code with Christian)

### What changed
- `formcoach demo --camera N` (or `make live CAMERA=N`) is built: one window — mirrored webcam
  video with the MediaPipe skeleton, a rep counter per arm (good/flagged) with current elbow
  angles, an angle trace with the rep threshold band, a status pill, and a ~3 s coaching banner
  after each rep. One command, no `--side` flag: both arms together, alternating, or one arm are
  each counted and judged per arm automatically; squat keeps one whole-body counter. `--board
  ble|serial` (+ `--port`) follows the wearable: gate = the device's gate flag AND a sample in
  the last 2 s, LEDs flash good-rep/fault; no board → gate always open ("webcam only"). Also:
  `--exercise`, `--video FILE`, `--headless`, `--out DIR`/`--no-log`, `--model lite|full|heavy`,
  `--rules FILE`. `formcoach pose fetch-model` downloads the pose model; `make setup-live` =
  `uv sync` with the `vision`+`device` extras + fetch-model.
- Logging is on by default to `runs/live/<YYYYMMDD-HHMMSS>/` (`events.parquet`, `frames.parquet`,
  `session.json`; `runs/` is gitignored; no video or camera frames are ever written). `rep`/
  `fault` events carry an `arm` key (`"l"`/`"r"`/`null`) only in live/per-arm mode — replay events
  are unchanged and have no `arm` key, read it with `.get("arm")`. `session.json` gains `arms`
  (per-arm reps/good/flagged/faults_by_code), `live`, `board`, `fps_mean`, `duration_s`, `source`,
  `model`.
- Found and fixed (`b07e6d0`): the adaptive `RepSegmenter` ends a rep at the exit threshold, so a
  rep's `elbow_max` metric was measured near that threshold and `CURL_PARTIAL_ROM` fired on
  **every** live curl — no rep was ever good. Fixed with `Pipeline(pad_rep_start_s=1.0)` in live
  mode only (`REP_PAD_S` in `app/live.py`): only the rep's *metrics* window is padded back to the
  rest position (never into the previous rep); the rep count and `t_start`/`t_end`/`duration_s`
  are unchanged. Replay keeps `pad_rep_start_s=0`, so its golden output
  (`tests/golden/replay_mmfit_w00_curls.json`) is byte-identical. Measured on synthetic curls
  only (ADR-0029) — **not validated on a real webcam clip.**
- Commits on `feat/vision-first-run` (none pushed, nothing opened as a PR yet): `c1a22c9`
  mediapipe pin (ADR-0027), `f183d45` golden replay test, `2442c11` camera + model fetch,
  `5f8ce50` BLE 2 s scan rounds + `BoardLink` (ADR-0030), `46ab462` HUD, `415cf5f` per-arm
  `Pipeline` (ADR-0028), `34df78b` live loop, `a446dbb` CLI `--camera/--board/--video` + `make
  live`, `b07e6d0` padded metrics window (ADR-0029), `e85bcce` review fixes; this docs change (`docs/howto/live-demo.md`,
  README, ADR-0028–0030, this entry) follows as its own commit.
- An independent `formcoach-reviewer` pass over the whole branch diff found, and `e85bcce` fixes
  (each with a regression test): an arm that drops out mid-rep stayed "in progress" and made every
  later rep of the other arm count as a two-arm rep (false `CURL_ASYMMETRY`, `LED_FAULT` instead of
  good); `session.json` key order differed from main (now byte-identical for both gates, checked
  with `cmp` against main's code); a failing log write could mask the camera error and skip the
  totals; the camera could stay open if the board failed to open; `--model huge`, a missing display
  and live-only flags without `--camera` gave tracebacks or were silently ignored; the BLE "is the
  wearable on?" hint is back (every 10 s); the model download now times out.
- Verified this session: `make lint test` green (301 passed, 1 skipped); `formcoach demo
  --source replay --headless` unchanged (`reps 10 (pose) / 17 (imu) / 10 (fused)`); the live loop
  runs end to end through the real MediaPipe model on a blank video (no person, 0 reps) and,
  separately, on synthetic curl frames (8/8 good reps per arm, `arm` key present).
- Manual test, later the same day (Christian, macOS, `uv run formcoach demo --camera 1`, no
  `--board`): he ran it on a real webcam with a real person and reports it works. **No rep
  counts, flagged reps or fps were recorded**, so this is a smoke test, not a measurement.
- Wearable, same day (Christian's board, USB serial `28:84:85:B3:90:EC`): built and flashed fw
  1.1.1 (`make fw-upload`); it advertises as `FormCoach-8428` and the web BLE viewer connects, but
  the boot banner says `bmi160 NOT FOUND at 0x00` (no I2C reply at 0x68 or 0x69), so the firmware
  sends no samples and the demo's pill says `looking for board...`. The cause is **not known**
  (wiring, a solder joint, no 3V3 at the module, or a bad module); not metered, module not
  swapped. The slide switch is not involved (the IMU is on the 3V3 pin). `--board ble|serial` has
  therefore never run with real samples. No `docs/devices.md` row yet.

### Next
1. Christian: repeat the webcam test with the design spec §9 checklist and record the numbers
   here: `uv run formcoach demo --camera 1` with both arms together, alternating, and one arm.
   Then `--board ble` once a board with a working IMU exists.
1a. Whoever has the bench: find why Christian's board reports `bmi160 NOT FOUND` (swap SDA/SCL
   first, then meter 3.3 V at the module, then try another module; `docs/assembly-day.md`).
   After a fix, reset the board, read the boot banner, then `make device-check PORT=... KIT=...`
   and add its `docs/devices.md` row.
2. A teammate on a fresh clone: `make setup-live` then `make live CAMERA=1`.
3. Christian films a front-view clip (about 10 clean and 10 sloppy curls; both arms, alternating,
   one arm; saved under `data/team/`, never pushed) so Rochelle can ratify the webcam thresholds.
   **Bryce and Rochelle: read `docs/howto/live-demo.md` first.**

### Open questions
- Counter/rules comparison on a real clip — none exists yet.
- Whether the padded metrics window should become the default for replay too (would change the
  goldens and ADR-0022's evidence) — not done.
- Whether to push / open PRs — the mediapipe pin commit `c1a22c9` was meant to be its own PR.
- **Robert:** the BLE name `FormCoach-XXXX` is built from `ESP.getEfuseMac() & 0xFFFF`, which is
  the *first* two MAC bytes (`28:84` → `8428`). K1 (`28:84:85:B3:8F:05`) and Christian's board
  (`28:84:85:B3:90:EC`) both advertise `FormCoach-8428`, so kits from one batch can't be told
  apart in the BLE chooser. Not changed here; `firmware/src/app/main.cpp` `deviceName()` and the
  docs that say "low 16 bits of the MAC" are yours to decide.
- `BoardLink` swallows every exception in its reader thread, so a reader that has died shows the
  same `looking for board...` as a board that isn't there. Not fixed.

### Blockers
- Christian's board has no IMU detected (above), so the wearable path can't be tested end to end
  from his machine. The webcam-only demo is not blocked.

---

## 2026-10-28 — BMI160 fix merged, firmware 1.1.1 (Claude Code with Robert)

### What changed
- `fw/imu-bringup` (written 2026-10-08, never pushed) is now on main. `Bmi160::begin()` powers
  the sensors up before writing CONF/RANGE and reads the registers back (ADR-0025): firmware
  from older main read about 4 g at rest because the ±8 g range write was dropped. The boot log
  now has a `# bmi160 regs … ok` line (`MISCONFIGURED` if the read-back differs).
- Firmware version **1.1.1**, so a fixed kit is distinguishable from a 1.1.0 kit in the
  banner, `device check`, the BLE viewer and `docs/devices.md`.
- `formcoach device check --interactive` walks through short press (session), 3 s hold
  (calibration) and shake (gate); gravity and gyro bias come from the still window before the
  first phase only. New test in `tests/test_selftest.py`.
- `reports/device.md` names its log repo-relative (`report.repo_relative`), and the raw
  device logs under `reports/logs/*.log` are committed so the report regenerates from a clone.
- `docs/assembly-day.md`: 1.1.1 banner and the `# bmi160 regs` line; ≈ 4 g → re-flash.

### Next
1. Robert: flash the newly assembled kits (and re-flash K1, which runs the fix under the old 1.1.0 label):
   `make attach-xiao`, `make fw-upload PORT=/dev/ttyACM0`, `make device-check
   PORT=/dev/ttyACM0 KIT=K2 INTERACTIVE=1`; one row per kit in `docs/devices.md`.
2. Each kit in the BLE viewer from Windows Chrome: |a| ≈ 1.00 g flat, 50 Hz, 0 drops, GATE on
   shake (`docs/howto/ble-viewer.md` §5).

### Open questions
- None new.

### Blockers
- None.

---

## 2026-10-21 — First real MediaPipe run; mediapipe 1.1.1 (Claude Code with Christian)

### What changed
- Branch `feat/vision-first-run`. The team connected a wired board (chips, no case) to the GitHub
  Pages BLE viewer; the traces follow motion. Bench checks (|a| ≈ 1 g, GATE on curls) in progress.
- MediaPipe ran for the first time, on Christian's Mac: `uv sync --extra vision --extra device`,
  `models/pose_landmarker_lite.task` downloaded. The locked mediapipe 1.0.1 aborted at graph start
  on Apple Silicon; 1.1.1 runs (221 fps Lite on a still). `pyproject.toml` excludes 1.0.x;
  `uv.lock` → 1.1.1 (ADR-0027).
- `tests/test_landmarker.py::test_real_landmarker_starts_on_this_machine` starts the real graph in
  a subprocess (skips without the extra or the model, so not in CI).
- Verified on a sample photo: `formcoach pose extract` wrote 60/60 valid `mediapipe33` frames and
  `joint_angles` gave plausible degrees (seated, arms straight: elbows 143–150°).

### Next
1. Christian: film ~10 clean curls, ~10 sloppy curls, ~10 squats (front view, whole body) on the
   Mac; keep the video under `data/team/` (never pushed). Then `pose extract` → replay through the
   pipeline: reps vs truth, which rules fire clean vs sloppy.
2. With the board strapped on and powered from USB: `formcoach record --source ble` alongside the
   video (first hardware run of that command), then the paired replay.
3. Design doc for the live path (`demo --source webcam|ble`, `Pipeline.on_imu/on_frame`).
4. Rochelle: read ADR-0027 and `src/formcoach/pose/landmarker.py`; the vision path is now runnable.

### Open questions
- Does anyone on the team have an Intel Mac? mediapipe ≥ 0.10.30 has no x86_64 macOS wheel.
- 1.1.1 on WSL2/Linux and Windows not yet installed (verify).

### Blockers
- None.

---

## 2026-10-15 — Browser BLE live viewer (Claude Code with Robert)

### What changed
- `web/ble-viewer/`: a static Web Bluetooth page. **Connect** to a kit (Chrome/Edge on Windows,
  macOS, Android) for live accel (g, with |a|) and gyro (dps) charts over the last 10 s, the
  status row (fw, uptime, calibrated, session), GATE/BUTTON/SESSION pills, packet rate, sample
  rate and drops. Buttons send PING, LED good/fault, start/stop session and calibrate. **Record**
  downloads a serial-format CSV. **Simulate** streams synthetic curls through the same decoder
  for anyone without a kit or with Safari/Firefox.
- `formcoach viewer` / `make ble-viewer` serves it on http://localhost:8000/ble-viewer/.
  `.github/workflows/pages.yml` publishes it to
  https://nthperson.github.io/BDA696_Group_Project/ble-viewer/ on merge to main.
- `tests/test_web_protocol.py` pins `protocol.js` against `protocol.py` (constants plus golden
  packets). `tests/test_viewer_server.py` covers the server, its MIME types and busy-port handling. 21 Node tests run
  locally via `make test` when Node ≥ 18 is installed (not in CI). ADR-0026. Spec and plan are in
  `docs/superpowers/`.
- Verified: simulator and a scripted fake Bluetooth device in headless Chromium (10.0 pkt/s,
  50.0 Hz, 0 drops; reconnect without duplicate streams; queued writes; the recording survives a
  disconnect); Windows `curl.exe` reaches the WSL server on localhost.

### Next
1. Robert: Settings → Pages → Source: **GitHub Actions** (one time), then re-run the Pages
   workflow after merge.
2. Robert: connect to K1 from Windows Chrome. Check |a| ≈ 1.00 g flat, gyro, GATE on shake,
   Ping → `# pong`, LEDs, CSV. Note the result in `docs/howto/ble-viewer.md` §5.
3. Finish `fw/imu-bringup` (ADR-0025 BMI160 fix). Until it merges, a kit flashed from main reads
   about 4 g at rest in the viewer.
4. **Bryce: read `docs/howto/ble-viewer.md` first, then `web/ble-viewer/app.js`.** The demo
   script is its "What a healthy kit looks like" paragraph.

### Open questions
- Linux Chrome may need the experimental-web-platform flag (verify).

### Blockers
- None.

---

## 2026-10-14 — Final Project Paper Outline drafted (Claude Code with Robert)

### What changed
- `docs/report/outline.md`: full draft of the **Written Report Outline** (due **Wed 2026-10-28,
  11:59 pm PT**, one PDF on Canvas, submitted by Robert as group leader). Built from the Canvas
  assignment text and the 2026-10-14 lecture (Plaud recording). Where they differ the lecture wins:
  technical/professional style, one bullet per member for contributions, APA or MLA, public GitHub
  link in the appendix, section structure is ours. Ends with a requirement → section map.
- The abstract is written in the future tense and reports no results. §8 has one clearly labelled
  "progress to date" box quoting `reports/` numbers. Re-check those numbers on submission day.
- 11 APA references; every DOI/page range checked against Crossref, arXiv or UCI.
- `docs/04-datasets.md` §5: the "220+ subjects" scale line was wrong (RecoFit releases 94, not
  200+); now ≈ 29 M IMU samples from 114 subjects, with the per-dataset split and window counts.
- `docs/05-roadmap.md` §1: outline due date filled in.

### Next
1. Team fills the `[TODO]`s: SDSU emails, group name as registered on Canvas, course title,
   instructor name. Each member checks their §11 bullet and the sections they will draft.
2. Read the pitch feedback (released Sat 2026-10-17) and fold any scope notes into §1 and §9.
3. Export to PDF (`pandoc docs/report/outline.md -o outline.pdf`, or print GitHub's rendered
   view) and submit by 2026-10-28.

### Open questions
- `reports/device.md` (cited in §8) lives on the unmerged `fw/imu-bringup` branch.

### Blockers
- None.

---

## 2026-10-08 — First board: firmware v1.1 verified on a bare XIAO, BLE end-to-end, self-test tooling (Claude Code with Robert)

### What changed
- **Firmware ran on hardware for the first time** (kit K1, bench, USB via usbipd from WSL2):
  boot banner, TFLite Micro model loads (arena 6,196 B of the 40 KB budget), BLE advertises
  `FormCoach-8428`, serial `P` → `# pong`. Two bugs found and fixed on the board: the service
  UUID did not fit the advertisement next to the name (now adv = UUID, scan response = name) and
  a spurious NVS error on first boot (ADR-0024).
- **BLE verified from Windows** (`firmware/tools/ble-check.py`, standalone bleak script run with
  `uv run`): scan shows name + service UUID at −48 dBm, connect with MTU 185, status read,
  PING/START/STOP/LED commands all acted on (serial shows `# pong`, `# session start id=…`).
  Without the clip-on u.FL antenna the board sat at −90 dBm and could not be connected to —
  the antenna is now step 0 of `docs/assembly-day.md`.
- **LED polarity settled**: active-low as assumed (mostly off, brief flash once a second).
- `formcoach device check --port … --kit K1 [--interactive]` (`make device-check`): the
  assembly-day self-test (banner, IMU, gate, BLE name, ping, 50 Hz, drops, gravity, gyro bias,
  inference, button/session/gate phases), saving `reports/logs/<kit>_<ts>.log/.json`.
- `firmware/tools/attach-xiao.sh` (`make attach-xiao`): usbipd hand-off to WSL2 with auto
  re-attach across flashing resets; documented in `firmware/README.md`.
- `docs/devices.md`: K1 row (MAC 28:84:85:B3:8F:05).

### Next
1. Wire the BMI160 (jumpers are fine for the bench unit) → `make device-check PORT=… KIT=K1
   INTERACTIVE=1` and `ble-check.py` again: I2C address, 50 Hz over serial and BLE, drops,
   gravity, button flags, gate flag, on-device inference time (`# infer us=`).
2. Assembly day per `docs/assembly-day.md`; K2–K6 rows in `docs/devices.md`.
3. Wire `demo --source serial|ble` with a webcam frame source (the live demo).

### Open questions
- Serial CSV + BLE notifications simultaneously at 50 Hz (needs the IMU).

### Blockers
- None.

---

## 2026-09-30 — Understand-Anything knowledge graph added (Claude Code with Christian)

### What changed
- `.ua/` now holds an Understand-Anything knowledge graph of the repo at commit `5214d94`:
  505 nodes / 993 edges, 10 architectural layers, a 15-step guided tour, 63 test→source
  `tested_by` links. Validation: 0 issues. Nothing in `src/`, `firmware/` or `tests/` changed.
- `.ua/.understandignore` excludes tool state (`.serena/`, `.claude/autoharness/`, `.ua/`) and
  binary data (`*.npy`, `*.mat`, `*.parquet`, `*.tflite`). `.gitignore` keeps `.ua/tmp/` and
  `.ua/.trash-*/` local.

### How to use it (teammates)
- Install the `understand-anything` Claude Code plugin, then run `/understand-dashboard` for the
  interactive graph and tour, or `/understand-chat <question>` to query it. `/understand-onboard`
  writes an onboarding guide.
- After merging code changes, run `/understand` — it updates incrementally from
  `.ua/fingerprints.json`. The graph is a snapshot; re-run it before relying on it.

### Next
- Unchanged from the 2026-09-27 entry (assembly day first).

### Open questions
- Whether to commit `.ua/` long term or regenerate locally (graph diffs are noisy JSON).

### Blockers
- None.

## 2026-09-27 — PRs #3–#8 merged; hardware received; assembly-day guide (Claude Code with Robert)

### What changed
- All six pre-hardware PRs are merged into `main` in order (#3 data, #4 signal, #5 replay,
  #6 firmware, #7 rules, #8 CNN gate). `main` is green: 176 tests, `make demo` finds 10/10 reps.
- `docs/assembly-day.md`: the build-session runbook — wiring tables (XIAO pin → part pin, wire
  colour), station roles, the per-kit procedure with the exact serial lines firmware v1.1
  prints at each step, the battery step last, laptop record/check commands, troubleshooting,
  end-of-day checklist. `docs/05-roadmap.md` checkboxes updated.
- Parts have arrived; the team meets soon to assemble. Nothing has run on a board yet.

### Next (in order)
1. **Assembly day** (`docs/assembly-day.md`): 5 + 1 kits flashed and registered in
   `docs/devices.md`; every **(verify)** settled in DECISIONS; boot logs → `eval device`.
2. Wire `demo --source serial|ble` (the live path: `_open_source` exists in `cli.py`, the
   pipeline takes any `IMUSource`; add the webcam frame source) and `eval gating --gate device`.
3. First MediaPipe run (`uv sync --extra vision`, `pose extract` on a webcam clip or one MM-Fit
   video) → rerun `eval rules` and choose the shipped thresholds (ADR-0022).
4. Team validation recordings (docs/04 §6) → `eval transfer`; that is where the RecoFit → MM-Fit
   transfer gap (macro-F1 0.33 for exercise class) gets re-measured on our own sensor.
5. Written Report Outline from `reports/` (Phase 1 deliverable, due date still blank in docs/05 §1).

### Open questions
- CNN (macro-F1 0.837, 5-fold) is behind the RF (0.867) on the gate task with 10 epochs per
  fold; longer training on the eGPU is cheap now — decide which model ships on the device once
  the on-device inference time is known.
- Case v1 is not modeled; kits can be used on a strap for the first recordings.

### Blockers
- None.

---

## 2026-09-27 — Checkpoint 5 (laptop half): CNN gate trained on the eGPU, int8 export, TFLM firmware (Claude Code, autonomous run for Robert)

### What changed
- **PR #8 `feat/checkpoint-5-cnn-gate`** (stacked on PR #7).
- `formcoach train gate --model cnn` (GPU, 23 min): 6-class 1D-CNN, 6,390 parameters, trained
  on 371,929 RecoFit + MM-Fit windows (104 subjects, 16 held out); int8 TFLite 16,760 B;
  held-out macro-F1 0.71 float / 0.73 int8 (no quantisation loss) → `reports/gate_cnn_export.md`,
  `firmware/model/{gate_model.tflite,gate_model_data.cc,preprocess.h,preprocess.json}` committed.
- `formcoach eval loso --model cnn` (5 subject-grouped folds, 10 epochs; writes the `cnn-int8`
  table from the same folds) with RF and energy re-run on the same folds (ADR-0023):
  energy 0.668 · RF 0.867 · **CNN 0.837 float / 0.832 int8** (macro-F1; the RF still leads the
  10-epoch CNN by 3 points; float→int8 costs 0.5 points)
- Firmware v1.1.0: `Gate::decideModel()` runs the model with TFLite Micro (Chirale, 40 KB
  arena; 22.3 % RAM, 19.2 % flash), logs `# infer us=… arena=…`; `xiao_esp32s3_energy` env keeps
  the v1 rule; `formcoach eval device --log` parses those lines.
- `train-gpu` extra (TensorFlow + CUDA wheels on Linux) and the WSL2 `LD_LIBRARY_PATH` note.
- Known cosmetic issue: `reports/gate_cnn_export.md` prints integer counts as `371929.0000`;
  the formatter was fixed after that run and the report was not regenerated (23-minute
  retrain). Numbers are correct; the next `train gate --model cnn` run rewrites it.
- README command table brought up to date. `Christian`: read `docs/howto/cnn-gate.md` first.

### What remains for hardware day (everything below needs a board or a teammate recording)
1. Flash `xiao_esp32s3`; confirm the boot banner (`# bmi160 ok at 0x69`, `# gate: tflm ok arena=…`),
   LED polarity, button on D1, I2C address; register the kit in `docs/devices.md` (skill
   `firmware-bringup`). Settle the **(verify)** items in ADR-0020 in DECISIONS.
2. `formcoach record --source serial --port COMx` and `--source ble` (from Windows Python; WSL2
   has no Bluetooth): MTU 185, 50 Hz notify rate, drop counts via `session check`.
3. `formcoach eval device --log <serial capture>` → inference ms, arena, flash; battery test.
4. `formcoach demo --source ble` opening/closing the camera from `flags.gate_state`
   (`--gate device`); `eval gating --gate device` rows; `eval latency`.
5. MediaPipe path: `uv sync --extra vision`, `pose extract` on one MM-Fit video or a webcam clip;
   rerun `eval rules` on MediaPipe landmarks and choose the shipped thresholds (ADR-0022).
6. Team validation recordings (docs/04 §6) → `eval transfer`; pose-model ablation.

### Open questions
- Exercise-class CNN LOSO and MM-Fit-only CNN LOSO are not run yet (one command each on the GPU).

### Blockers
- None for software. Hardware verification waits for the boards.

---

## 2026-09-24 — Checkpoint 6 (hardware-free half): rules v1, eval rules, eval gating (Claude Code, autonomous run for Robert)

### What changed
- **PR #7 `feat/checkpoint-6-rules`** (stacked on PR #6).
- `rules/rules.yaml` (16 rules, docs/02 §7) + `rules/engine.py`; the demo and pipeline take
  `--rules`; 27 tests (one perturbation per rule).
- `formcoach eval rules [--calibrate-out]` → `reports/rules_validation.md`: rep segmentation
  agreement (curl 586/599, press 540/598, raise 557/559, squat 637/639), pass rate per rule,
  perturbation detection, metric percentiles; `rules.mmfit-pose3d.yaml` generated (ADR-0022).
- `formcoach eval gating` → `reports/gating.md`: on the 30 s fixture the energy gate processes
  82.9 % of frames with identical reps/faults; the RF laptop gate 79.2 % but loses one rep at a
  gate boundary. `formcoach train gate --model rf` → `models/gate_rf.joblib`.
- `formcoach eval all` (`make eval`) regenerates every report (`--quick` for 10-fold RecoFit).
- `Rochelle`/`Christian`: read `docs/howto/rules.md` first.

### Next
- PR 8 (`feat/checkpoint-5-cnn-gate`): Keras CNN, int8 export, TFLM gate in firmware.

### Open questions
- Which thresholds ship for the webcam path (design vs MM-Fit-calibrated) — after the first
  MediaPipe run.

### Blockers
- None.

---

## 2026-09-24 — Checkpoint 4: firmware v1 and the device-side sources (Claude Code, autonomous run for Robert)

### What changed
- **PR #6 `fw/checkpoint-4-firmware-v1`** (stacked on PR #5). Compiles in CI (9.6 % RAM,
  16.6 % flash); **nothing has run on a board**.
- Firmware v1: in-tree BMI160 driver (ADR-0020), 100 Hz reads averaged to 50 Hz, 2 s ring
  buffer, energy gate + hysteresis from `protocol.h`, NimBLE service (IMU notify batches,
  control writes, 12-byte status), serial CSV + one-letter commands, button (short = session,
  long = 3 s calibration to NVS), LED patterns, deep sleep after 10 min idle.
- `protocol.h`/`protocol.py`: `FcStatus` (12 B), serial command letters, shared energy
  threshold; pinned by `tests/test_protocol.py`.
- Python: `io/serial_source.py`, `io/ble.py` (lazy `device` extra), `io/fake.py` (software
  wearable), `io/recorder.py`, `io/session_check.py`; `formcoach record --source
  serial|ble|fake` and `formcoach session check`; `make fw-upload PORT=… fw-monitor PORT=…
  fw-libtest`. Library compile tests decided ADR-0020/0021.
- `Robert`/`Christian`: read `docs/howto/flashing-firmware.md` first.

### Next
- PR 7 (`feat/checkpoint-6-rules`): rules.yaml v1, engine, eval rules, eval gating.

### Open questions / hardware day
- BMI160 register sequence, I2C address, LED polarity, NimBLE MTU/notify rate, deep-sleep
  wake on GPIO2, serial + BLE at 50 Hz simultaneously: all unverified until a board is flashed.
- Camera capture during `record --camera` is not wired.

### Blockers
- None for software; hardware verification waits for the parts.

---

## 2026-09-24 — Checkpoint 3: pose pipeline in replay mode (Claude Code, autonomous run for Robert)

### What changed
- **PR #5 `feat/checkpoint-3-replay`** (stacked on PR #4).
- `formcoach demo --source replay --headless` now runs the real pipeline on the committed 30 s
  MM-Fit session (`data/fixtures/replay/mmfit_w00_curls`): IMU → gate → 3-D pose → joint
  angles → adaptive rep segmentation → rep metrics → events. Output: 10 pose reps / 10 fused
  (expected 10), 83 % of frames processed with the energy gate. It is the CI smoke test.
- `pose/{skeletons,angles,normalize,reps,store,landmarker}.py`, `io/{source,replay}.py`,
  `app/{gate,events,pipeline,overlay}.py`; `formcoach pose extract` (MediaPipe, lazy import,
  not executed here); 33 tests including hand-computed angle cases.
- `Rochelle` and `Bryce`: read `docs/howto/replay-demo.md` first, then
  `src/formcoach/app/pipeline.py`.

### Facts that differed from the docs
- MM-Fit's lifted 3-D pose reads a curl as elbow 95–140°, not 60–150°: fixed rep thresholds
  never fire → adaptive segmentation (ADR-0018); the rules catalogue (§7) needs per-pose-source
  calibration (PR 7).

### Next
- PR 6 (`fw/checkpoint-4-firmware-v1`): firmware v1, serial/BLE sources, recorder.

### Open questions
- The MediaPipe path (`pose extract`, overlay window) is untested: needs `uv sync --extra
  vision` and one MM-Fit video (w00_rgb.mp4, 2.17 GB) or a webcam clip.

### Blockers
- None.

---

## 2026-09-24 — Checkpoint 2: signal pipeline and baselines (Claude Code, autonomous run for Robert)

### What changed
- **PR #4 `feat/checkpoint-2-signal`** (stacked on PR #3).
- `formcoach features build`: resample → 50 Hz, 2 s / 1 s windows with majority label and
  purity, 63 features, one Parquet per stream (314 files, ~35 s with `--jobs 14`).
- `formcoach eval loso --model energy|rf --dataset recofit|mmfit|recgym --task active|exercise
  [--folds N] [--test-dataset mmfit]` → `reports/loso_*.md`, `reports/transfer_*.md` with
  pooled + per-fold tables and confusion matrices; `formcoach eval repcount` →
  `reports/baseline_repcount.md`. All reports committed, seeds logged in each header.
- `signal/{resample,filters,gravity,windows,features,reps,build}.py`, `eval/{loso,loso_cli,
  repcount}.py`, `models/{energy,rf}.py`; 30 tests. `formcoach.signal` imports no pose code.
- `Christian`: read `docs/howto/signal-baselines.md` first, then `src/formcoach/eval/loso.py`.

### Headline numbers (unseen-subject; pooled over folds)
| Task | Dataset | Energy gate | RF |
|---|---|---|---|
| active vs idle | RecoFit (94-fold LOSO) | macro-F1 0.665 | macro-F1 0.869 |
| active vs idle | MM-Fit (10-fold LOSO) | macro-F1 0.648 | macro-F1 0.935 |
| exercise (5 classes) | RecoFit (94-fold LOSO) | — | acc 0.951 · macro-F1 0.797 |
| exercise (5 classes) | MM-Fit (10-fold LOSO) | — | acc 0.970 · macro-F1 0.950 |
| active, RecoFit → MM-Fit transfer | — | — | acc 0.770 · macro-F1 0.741 |
| exercise, RecoFit → MM-Fit transfer | — | — | acc 0.600 · macro-F1 0.333 |
| active vs idle | RecGym (10-fold, normalised units) | — | macro-F1 0.868 |
| rep count (peaks, docs/02 defaults) | MM-Fit 1,232 sets | MAE 2.39 reps (curl 0.69, press 3.66, squat 3.71) | |

### Next
- PR 3 (`feat/checkpoint-3-replay`): pose angles, replay source, headless pipeline, fixture, demo.

### Open questions
- Cross-dataset exercise recognition (RecoFit forearm → MM-Fit wrist) is near chance (macro-F1
  0.33): placement/orientation differences dominate; the sensor-transfer experiment on team
  recordings (Phase 3) is where this must be re-measured with our own wearable.
- The rep counter's design defaults over-count slow exercises; calibrate `min_distance_s` per
  exercise (open for Christian).

### Blockers
- None.

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
