# How to: live webcam form coaching (per-arm rep counting, optional wearable)

**Owner:** Bryce Hall (app, demo) · Rochelle Reyes (vision & rules) · **Built by:** Claude Code
with Christian on 2026-11-04 · **PR:** none yet — commits are local on `feat/vision-first-run`
**Checkpoint:** 3/5 (`docs/00-START-HERE.md`) · **State:** partial — the full automated suite
passes (301 passed, 1 skipped) and the loop runs end to end on synthetic frames and on a blank
video with the real MediaPipe model; Christian then ran `demo --camera 1` on a real webcam and
reports it works, but **recorded no rep counts**, and **the wearable has never streamed real IMU
samples into it** (see §5).

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
no counts or timings were written down, so treat "works" as a smoke test, not a measurement.
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
| `Makefile` | `live` (`CAMERA=`, `EXERCISE=`, `BOARD=`, `PORT=`), `setup-live` (`POSE_MODEL=`) |
| `runs/live/<YYYYMMDD-HHMMSS>/` | `events.parquet`, `frames.parquet`, `session.json` (gitignored; never video) |

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

- **Real camera: run once, informally, no numbers.** Christian ran `demo --camera 1` with a real
  person on 2026-11-04 and said it works; no rep counts, flagged reps or fps were recorded. Every
  number in this doc is from the automated suite or the synthetic/blank-clip smoke test, never a
  counted real session.
- **Real wearable: never run with samples.** Christian's board (USB serial `28:84:85:B3:90:EC`,
  fw 1.1.1) booted, advertised as `FormCoach-8428` and connected to the web BLE viewer, but its
  boot banner said `bmi160 NOT FOUND` (no I2C reply at 0x68 or 0x69), so it sent no samples. The
  cause (wiring, a joint, no 3V3 at the module, or a bad module) is not yet known. So the
  `--board ble|serial` gate/LED path is covered only by unit tests with fakes.
- **Rule thresholds on real webcam data: not validated.** The padded-window fix was only measured
  on synthetic curls (clean 50-150°/35-165°/slow 4 s/fast 1.6 s all come out "good"; 90-130° and
  40-100° are flagged `CURL_PARTIAL_ROM`, correctly). MediaPipe's real angle ranges on a webcam
  are unmeasured. Rochelle ratifies this (and whether a separate `rules.mediapipe-webcam.yaml` is
  needed) after Christian films a clip — do not treat "good rep" as validated until then.
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
last 2 s) and LEDs (`LED_GOOD_REP` / `LED_FAULT` after each rep, either arm). No protocol or
firmware change. For anything about the physical kit — wiring, flashing, battery — read
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
`source`, `model` to the usual summary fields. Reading one rep event (genuine output from this
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
- Rule thresholds are not yet validated on a real webcam clip (§5).

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
