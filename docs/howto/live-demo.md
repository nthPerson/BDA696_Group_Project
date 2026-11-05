# How to: live webcam form coaching (per-arm rep counting, optional wearable)

**Owner:** Bryce Hall (app, demo) · Rochelle Reyes (vision & rules) · **Built by:** Claude Code
with Christian on 2026-11-04 · **PR:** none yet — commits are local on `feat/vision-first-run`
**Checkpoint:** 3/5 (`docs/00-START-HERE.md`) · **State:** partial — the full automated suite
passes (301 passed, 1 skipped) and the loop runs end to end on synthetic frames and on a blank
video with the real MediaPipe model; Christian then ran `demo --camera 1` on a real webcam several
times on 2026-11-04 (his first run recorded no counts; later runs are in `runs/live/`, gitignored, and
summarised in §5), and **the wearable has never streamed real IMU samples into it** (see §5).

## 1. Run it (verified 2026-11-04 on macOS, Apple Silicon)

```bash
make setup-live
```
```
uv sync --group dev --extra vision --extra device
Resolved 140 packages in 3ms
Checked 65 packages in 10ms
uv run pre-commit install
pre-commit installed at .git/hooks/pre-commit
uv run formcoach pose fetch-model --model lite
pose model ready models/pose_landmarker_lite.task
```

The real loop, through the real MediaPipe model, on a 3 s blank clip (no person) instead of a
camera — this is as far as this session could go without a webcam or a human:

```bash
uv run formcoach demo --video blank.mp4 --headless --out runs/smoke-test
```
```
live demo: video blank.mp4, exercise=curl
gate_open t=0.0s
L: 0 reps (0 good, 0 flagged)
R: 0 reps (0 good, 0 flagged)
wrote runs/smoke-test/events.parquet frames.parquet session.json
```

**Not covered by the blank-clip run above:** after it, Christian ran `uv run formcoach demo
--camera 1` on a real webcam with a real person (macOS, 2026-11-04) and reported that it works;
that first run wrote no counts down. Later runs that day were logged, see §5 for the numbers.
`--board ble|serial` has **not** been run with real samples: the one board tried connected over
BLE but its IMU was not detected (§5). Next for Christian/Bryce/Rochelle — see §12 and
`docs/STATUS.md`.

## 2. Where things live

| Path | What it is |
|---|---|
| `src/formcoach/cli.py` | `demo --camera/--video/--board/--model/--no-log`, `pose fetch-model`, `_demo_live` |
| `src/formcoach/app/live.py` | `run_live` (the frame loop), `run_live_demo` (assembles camera/video + model + board + window) |
| `src/formcoach/app/pipeline.py` | `Pipeline` (`per_arm=True`, `pad_rep_start_s=`), `_Track`, `arm_totals()` |
| `src/formcoach/app/hud.py` | `counter_text`, `pill`, `Banner`, `render` (the window), `WindowShow` |
| `src/formcoach/app/board.py` | `BoardLink`: non-blocking wearable drain, `alive()`, `gate_open()`, `state()` |
| `src/formcoach/pose/camera.py` | `open_camera` (friendly errors, macOS `CAP_AVFOUNDATION`) |
| `src/formcoach/pose/landmarker.py` | `fetch_model`, `model_path`, `make_landmarker` |
| `src/formcoach/io/ble.py` | `BLESource._find` (2 s scan rounds, no backoff) |
| `src/formcoach/rules/rules.yaml` | shared rule thresholds (unchanged by this work, ADR-0022) |
| `src/formcoach/rules/rules.mediapipe-webcam.yaml` | what the live demo judges with by default: `rules.yaml` plus two provisional curl thresholds (ADR-0033); `session.json` records it as `rules` |
| `Makefile` | `live` (`CAMERA=`, `EXERCISE=`, `BOARD=`, `PORT=`), `setup-live` (`POSE_MODEL=`) |
| `runs/live/<YYYYMMDD-HHMMSS>/` | `events.parquet`, `frames.parquet` (live runs: plus per-frame counter columns, §9), `session.json` (gitignored; never video) |

## 3. How it works

One `Pipeline(per_arm=True)` holds one `_Track` per arm for curl/press/raise (a single track for
squat); each track has its own angle buffer, `RepSegmenter` and rules context, so both-arms,
alternating and one-arm curling all "just work" with no flag. `run_live` drains an optional
`BoardLink` each frame (gate = device gate flag **and** a sample within 2 s), feeds the frame to
pose only while the gate is open, and renders `hud.render` unless `--headless`. A rep's *metrics*
window (not its count or timing) is padded back up to 1 s to the rest position
(`Pipeline(pad_rep_start_s=1.0)`, `REP_PAD_S` in `live.py`) because the segmenter ends a rep at the
exit threshold, which otherwise makes `elbow_max` read near that threshold and fires
`CURL_PARTIAL_ROM` on every rep (ADR-0029). Replay keeps `pad_rep_start_s=0`, so its golden output
is untouched. Good = a rep with no fault of any severity.

## 4. Tests

```
uv run pytest tests/test_live.py tests/test_live_cli.py tests/test_per_arm.py \
  tests/test_hud.py tests/test_board.py tests/test_ble_find.py tests/test_camera.py \
  tests/test_fetch_model.py tests/test_pipeline_equivalence.py -v
```
`tests/test_pipeline_equivalence.py` pins replay's events/summary against
`tests/golden/replay_mmfit_w00_curls.json`; change it only if you mean to change replay output.
`tests/test_per_arm.py` pins the per-arm/visibility/pairing behavior with synthetic frames
(`tests/synthetic_pose.py`), with no camera or model needed. Update `Pipeline` and these tests
together; update `hud.py`'s pill/banner text and `tests/test_hud.py` together.

## 5. Could not verify / open questions

- **Real camera: Christian ran it several times on 2026-11-04 (macOS, camera 1, `lite`).** His
  first run wrote no counts down. Three curl runs under the design-default `rules.yaml` (54 reps,
  one person, no labels) scored **0 good**: `CURL_PARTIAL_ROM` and `CURL_SWING` each fired on 51
  (ADR-0033). After the provisional `rules.mediapipe-webcam.yaml`, one 141 s curl run
  (`runs/live/20261104-221325`; Christian said he did both the strict and the swingy test in it
  and "it works better") logged **L 16 reps (5 good), R 17 reps (6 good)**: 11 of 33 good;
  `CURL_SWING` on 19, `CURL_PARTIAL_ROM` 5, `CURL_TOO_FAST` 5, `CURL_ASYMMETRY` 2; 29.6 fps mean,
  165 of 4185 frames without a person. The log has no marker for which reps were the strict set
  and which the swingy one, so this is **not** the labeled check ADR-0033 asks for. The 11 good
  reps read 28-34° upper-arm drift and 0.19-0.30 elbow drift: just under the 35° / 0.30 limits.
  Those figures are read from the session logs, not reported by Christian.
- **First labeled test (2026-11-04):** strict set `20261104-221856` (22 reps, 15 good; left 8/8,
  right 7/14 because the right arm's `elbow_min` read 104-107 against a 103.869 limit; swing clean
  on 21 of 22) and a swingy set `222027` (9 reps, mixed one- and two-arm, so no per-rep swing label:
  4 flagged for swing, all by 37-44° arm drift). Thresholds were then moved to `elbow_min > 115` and
  `elbow_drift_max > 0.35` **using those same runs, so they are untested**; the next round needs
  new runs: strict both arms, every-rep-swung both arms, half-range both arms (ADR-0033).
- **Second labeled round (2026-11-04, both arms together):** strict `20261104-222614` 16 of 16 good;
  every-rep-swung `222655` 15 of 16 flagged for swing; half-range `222733` only 6 of 25 flagged for
  range, but Christian said the last set "seemed a bit relaxed" and 9 of its 25 reps read like
  strict ones, so the 115 limit is **untested**, not wrong. Margins on swing are thin (strict
  max 34° vs the 35° limit; lowest flagged swung rep 37°). Still one person.
- **Real wearable: never run with samples.** Christian's board (USB serial `28:84:85:B3:90:EC`,
  fw 1.1.1) booted, advertised as `FormCoach-8428` and connected to the web BLE viewer, but its
  boot banner said `bmi160 NOT FOUND` (no I2C reply at 0x68 or 0x69), so it sent no samples. The
  cause (wiring, a joint, no 3V3 at the module, or a bad module) is not yet known. So the
  `--board ble|serial` gate/LED path is covered only by unit tests with fakes.
- **Rule thresholds on real webcam data: still not validated.** Webcam curls read `elbow_min`
  about 88 and `elbow_max` about 150, so the design range limits flag nearly every rep (ADR-0033).
  `rules.mediapipe-webcam.yaml` is provisional: its `CURL_SWING` numbers are a guess from unlabeled
  reps and need labeled strict-vs-swinging reps (S1-S5) before "good rep" means anything. Press,
  raise and squat are unmeasured on a webcam. Rochelle ratifies or replaces the file.
- **Intel Macs cannot install the `vision` extra** — verified against `pyproject.toml`
  (`vision = ["mediapipe>=0.10.14,!=1.0.*", ...]`) and ADR-0027: mediapipe ≥ 0.10.30 ships macOS
  wheels for arm64 only.
- **WSL2 and a webcam/BLE adapter:** not verified here (no WSL2 machine in this session); the
  design spec (§7) flags it `(verify)`. If `make live` can't open any camera index under WSL2,
  run it from Windows Python instead (`uv run formcoach demo --camera 0 ...` in PowerShell).

## 6. Troubleshooting

- **Camera looks black** (iPhone Continuity Camera is often camera 0 on a Mac): after the first
  30 frames average under 2/255, the console prints `camera N looks black (iPhone Continuity
  Camera?) - try --camera N+1`. Try `--camera 1`, `2`, ...
- **Camera won't open:** `cannot open camera N: no device responded. On macOS: System Settings >
  Privacy & Security > Camera, allow your terminal app, then run again.`
- **Board never found / goes quiet** — the status pill (top-right of the window) says why:
  - no `--board`: `webcam only | gate always on`
  - never heard from it: `looking for board... | pose paused` (video dimmed). This also shows
    when the board is on and connected but its IMU is not detected: the firmware sends samples
    only after it finds the IMU. Check with `make fw-monitor` (or any serial monitor), reset the
    board, and read the boot banner for `bmi160 NOT FOUND`. It prints once, at boot.
  - heard before, now quiet > 2 s: `board silent - reconnecting` (video dimmed)
  - alive, gate open: `<device name> | gate ON | pose running`
  - alive, gate closed: `gate OFF | pose paused` (video dimmed)
- **Skeleton outline looks shaky/jittery** (Christian, 2026-11-04, `lite`, camera 1): observed,
  **cause not confirmed**. What is known from the code: the window draws each frame's raw
  landmarks with no smoothing (`hud._draw_skeleton`), the landmarker runs in `VIDEO` mode, and
  `--model full` is the other size to try. Untested guess: the same per-frame noise also feeds the
  angles behind `CURL_SWING` (the rule metrics use the unsmoothed series), so it may be part of why
  swing reads high. Not changed in this PR; see `docs/STATUS.md`.
- **Reps you did were not counted** (Christian, right arm solo, 2026-11-04: "more than half of my reps didn't
  get registered"): **cause not confirmed**. Read the run's `frames.parquet` for that arm: `armed_r` 0 while you
  were curling means the last 6 s of `ang_r` never spanned 25 degrees (a small movement, or one rep after a
  pause, is silently ignored - the counter has to see the range first); `vis_r` under 0.5 means the arm was
  not tracked; `ang_r` blank means no usable angle; `armed_r` 1 with `ang_r` crossing `enter_r` and then
  `exit_r` but no rep means a rep shorter than 0.4 s or one dropped for a tracking gap (no event for those).
  Only runs recorded after this logging was added have these columns. Do not change `min_range_deg`
  in `rules.mediapipe-webcam.yaml` before a run shows which of these it is.
- **Model missing:** run `uv run formcoach pose fetch-model --model lite` (or `--model full`,
  `--model heavy`); `make setup-live` does this for you.

## 7. Per-arm behavior (one command, no `--side` flag)

| You do | Result |
|---|---|
| Curl both arms together | L and R both count, each judged on its own form |
| Alternate arms | each arm counts its own reps; `CURL_ASYMMETRY` only fires if the two arms' reps overlap ≥ 50% of the shorter one's duration, so alternating reps are exempt |
| Curl one arm only | that arm counts; the idle arm stays at 0 |

Squat has one whole-body counter (key `"all"` in `session.json["arms"]`; the HUD prints
`SQUAT  7 (6 good)`), not per-arm.

## 8. Wearable (optional, `--board ble` / `--board serial --port COMx`)

Adds the device's gate (pose runs only while the gate flag is set **and** a sample arrived in the
last 2 s) and LEDs (`LED_GOOD_REP` / `LED_FAULT` after each rep, either arm). If both arms finish
on the same frame one command is sent: `LED_FAULT` if either rep has a fault. A `LED_GOOD_REP` is
held back for 0.8 s after an `LED_FAULT` (the firmware's fault flash lasts 800 ms and a newer
command cancels it), so an arm finishing a frame or two later cannot wipe the fault flash. On
screen each arm keeps its own post-rep banner (L above R), so one arm's "Good rep" never hides the
other arm's fault. No protocol or firmware change. For anything about the physical kit — wiring, flashing, battery — read
`docs/03-hardware.md` first; this doc does not repeat it.

## 9. `--video`, `--headless`, `--out` / `--no-log`, and the log

`--video FILE` plays a recording through the identical loop instead of a camera (pass one of
`--camera`/`--video`, not both) — useful without a webcam. `--headless` drops the window and prints
`rep`/`fault`/`gate_*` lines instead (no OpenCV window is ever opened); with `--board` it also
prints `wearable: <status>` whenever the status changes, the text the window's pill would show.
`--gate`, `--speed` and `--session` are replay options and are rejected with `--camera`/`--video`;
`--port` needs `--board`. Logging is **on by default** to
`runs/live/<YYYYMMDD-HHMMSS>/`; `--out DIR` moves it, `--no-log` turns it off (the two conflict).
`rep`/`fault` events carry `"arm": "l"|"r"|null`; replay events have no `arm` key at all, so read
it with `.get("arm")`. `session.json` adds `arms`, `live`, `board`, `fps_mean`, `duration_s`,
`source`, `model`, `rules` to the usual summary fields. `frames.parquet` also has one row per frame with the
counter state behind it (live only, NaN while the gate is closed): per counter `ang_l`/`ang_r` (the angle the
rep counter saw, degrees), `enter_*`/`exit_*` (its current thresholds), `armed_*` (1 once the last 6 s of
angle spans 25 degrees; **until then nothing can count as a rep**) and `vis_l`/`vis_r` (the arm's lowest
landmark visibility, 0-1; a rep needs it above 0.5). Runs recorded before this was added (up to
`20261104-223036`) do not have these columns and cannot be re-read. To see where an arm was not being
counted:

```python
import pandas as pd

fr = pd.read_parquet("runs/live/<ts>/frames.parquet")
print(f"right arm armed {(fr['armed_r'] == 1).mean():.0%} of frames")
idle = fr[fr["armed_r"] == 0]  # frames where no right-arm rep could count; look at idle["ang_r"]
```

Reading one rep event (genuine output from this
session's synthetic-frame check, not a real clip):

```python
import json
import pandas as pd

events = pd.read_parquet("runs/live/<ts>/events.parquet")
rep = json.loads(events[events["kind"] == "rep"].iloc[0]["payload"])
rep["arm"]  # "l"
```

## 10. Known limits

- No IMU cross-check live: `reps_imu`/`reps_fused` are absent from the live summary (the board is
  on one wrist and can't validate both arms; replay still computes them).
- Two-arm rules (e.g. `CURL_ASYMMETRY`) only run on reps where both arms overlap in time.
- A rep with a stretch of lost tracking inside it (an arm hidden or out of frame for more than 3
  frames, docs/02 §4.3) is not counted at all rather than judged on interpolated angles; replay
  still interpolates. Rep numbers keep counting across a gate close and re-open.
- Squat is a single whole-body counter; no per-leg tracking.
- The camera must see the whole upper body, front view.
- Rule thresholds are not validated against labeled webcam reps (§5); the curl ones are provisional.
- The drawn skeleton is unsmoothed and looks shaky on a real webcam (§6).
- The rep counter ignores any movement until the last 6 s of angle spans 25 degrees (10th to 90th
  percentile), so small or isolated reps may not count, and nothing is shown when that happens (§6).

## 11. What to read to take this over

`docs/superpowers/specs/2026-11-02-live-demo-design.md` (why, §3-4.6 the contract),
`docs/superpowers/plans/2026-11-02-live-demo.md` (how it was built, task by task),
`docs/DECISIONS.md` ADR-0022, ADR-0028, ADR-0029, ADR-0030, then the docstrings in `app/live.py`,
`app/pipeline.py`, `app/hud.py`, `app/board.py` (more precise than the plan for exact behavior).

## 12. Next steps

1. Christian: the webcam run has been done informally; repeat it with the design spec §9
   checklist (both arms, alternating, one arm) and write the counts in `docs/STATUS.md`. Then
   `--board ble`, once a board with a working IMU exists (his currently has none detected).
2. A teammate on a fresh clone: `make setup-live` then `make live CAMERA=1`.
3. Christian films a front-view curl clip (clean + sloppy, both/alternating/one-arm) under
   `data/team/` (never pushed) so Rochelle can ratify the webcam rule thresholds (§5).
