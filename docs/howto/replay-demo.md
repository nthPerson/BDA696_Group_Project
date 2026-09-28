# How to: run and extend the replay demo (pose → reps pipeline, no hardware)

**Owner:** Rochelle Reyes (pose, angles, reps) · Bryce Hall (app, demo) · **Built by:** Claude Code with Robert on 2026-09-24 · **PR:** #5
**Checkpoint:** 3 (`docs/00-START-HERE.md`) · **State:** works on the committed 30 s MM-Fit fixture; webcam/MediaPipe path written but not run (no `vision` extra installed here)

## 1. Run it (verified 2026-09-24 on Linux/WSL2)

```bash
make demo                                      # = the CI smoke test
uv run formcoach demo --source replay --headless
```
```
replaying …/data/fixtures/replay/mmfit_w00_curls (gate=always_on, exercise=meta)
gate_open t=1697.70s
rep   1  t= 1703.80s  dur=0.80s  elbow 95-134 deg
rep   2  t= 1705.50s  dur=1.04s  elbow 96-134 deg
…
rep  10  t= 1719.50s  dur=1.10s  elbow 97-140 deg
reps 10 (pose) / 17 (imu) / 10 (fused), expected 10 · faults 0 · frames processed 897/898 (99.9%) · gated 29.98s of 29.98s
```

Gated by motion energy instead of always-on, and keep the event log:

```bash
uv run formcoach demo --source replay --headless --gate energy --out runs/demo1
```
```
reps 10 (pose) / 15 (imu) / 10 (fused), expected 10 · faults 0 · frames processed 744/898 (82.9%) · gated 24.87s of 29.98s
wrote runs/demo1/events.parquet frames.parquet session.json
```

Extract pose from a video (needs `uv sync --extra vision` and the model file; the command
prints the download line if it is missing):

```bash
uv run formcoach pose extract --video data/external/mmfit/w00_rgb.mp4 --model lite
```

## 2. Where things live

| Path | What it is |
|---|---|
| `src/formcoach/pose/skeletons.py` | `H36M17`, `MEDIAPIPE33`, `joint(xyz, skeleton, name)` (aliases + derived centres) |
| `src/formcoach/pose/angles.py` | `angle_deg(a, b, c)`, `joint_angles(xyz, sk, up) -> dict`, `ANGLE_NAMES`, `torso_length` |
| `src/formcoach/pose/normalize.py` | `estimate_up_vector`, `normalize_sequence(xyz, sk, visibility, ...)`, `smooth_angles` |
| `src/formcoach/pose/reps.py` | `RepSegmenter(enter, exit, mode, adaptive=...)`, `segment_reps(...)`, `DEFAULT_REP_CONFIG` |
| `src/formcoach/pose/store.py` | `write_pose(...)` / `read_pose(path) -> PoseSequence` (Pose Parquet, docs/02 §5) |
| `src/formcoach/pose/landmarker.py` | `extract_video(video, out, variant)`, `extract_frames(frames, detect)` (MediaPipe, lazy) |
| `src/formcoach/io/source.py`, `io/replay.py` | `SISample`, `IMUSource`, `ReplaySource(session_dir, speed)`, `PoseReplay(pose.parquet)` |
| `src/formcoach/app/gate.py` | `make_gate("always_on"|"energy"|"laptop"|"device")`, `Hysteresis` (firmware constants) |
| `src/formcoach/app/pipeline.py` | `run_pipeline(imu, frames, exercise, gate, rules)`, `run_replay(session_dir, ...)`, `RepMetrics` |
| `src/formcoach/app/events.py`, `app/overlay.py` | `EventLog` → `events.parquet`/`frames.parquet`/`session.json`; OpenCV overlay (lazy) |
| `data/fixtures/replay/mmfit_w00_curls/` | `imu.parquet`, `pose.parquet`, `meta.json`: 30 s of MM-Fit w00 around the first curl set (MIT) |
| `tests/test_pose_math.py test_replay_pipeline.py test_landmarker.py` | the tests (`uv run pytest tests/test_pose_math.py …`) |

## 3. How it works

`run_pipeline` merges IMU samples and pose frames by time. Every IMU sample updates the gate;
a pose frame is *processed* only while the gate is open (else logged as skipped in
`frames.parquet`). Processing = `joint_angles` on the frame's 3-D joints with a running
up-vector → the exercise's primary angle (`DEFAULT_REP_CONFIG`: elbow for curl/press,
shoulder abduction for raise, knee for squat; mean of both sides) → 5-frame median →
`RepSegmenter`. Thresholds are **adaptive** (ADR-0018): 10th/90th percentiles of the last 6 s
give `enter = lo + 0.3·range`, `exit = hi − 0.3·range`, nothing counts until the range exceeds
25°, because MM-Fit's lifted 3-D pose reads a curl as 95–140°, not the 60–150° in docs/02 §7.
When a rep completes, `compute_rep_metrics` resamples all 17 angle series to 30 steps and
derives the scalar metrics rules.yaml consumes (`elbow_min`, `trunk_incl_max`,
`knee_track_at_bottom`, …); the rules engine (PR 5) plugs in via the `rules=` argument. At the
end, IMU peak-detection reps on the gated segments give `reps_imu` and the ≥ 50 % overlap
count `reps_fused` (docs/02 §4.4).

## 4. Tests

`uv run pytest tests/test_pose_math.py tests/test_replay_pipeline.py tests/test_landmarker.py`
(24 tests): hand-computed angles (90°/180°/45°, collinear → NaN, trunk lean 30°, knee valgus),
up-vector and torso normalisation, gap interpolation (≤ 3 frames) vs invalid segments, rep
segmentation fixed and adaptive, gate hysteresis (2 on / 12 off windows), the fixture
producing 8–12 pose reps with always-on **and** energy gating, and the CLI. The CI smoke test
is `formcoach demo --source replay --headless`. If you change `DEFAULT_REP_CONFIG` or the
adaptive constants, rerun `make demo` and check `expected 10` still holds.

## 5. Could not verify / open questions

- MediaPipe path (`pose extract`, `Overlay`) is written against the MediaPipe Tasks API
  (`PoseLandmarker`, VIDEO mode) but **not executed**: the `vision` extra is not installed
  here and no MM-Fit video was downloaded (w00_rgb.mp4 is 2.17 GB). First run: `uv sync
  --extra vision`, download the `.task` file the command prints, then run on any webcam clip.
- Absolute angle thresholds in docs/02 §7 do not match MM-Fit's lifted 3-D pose; rules in
  PR 5 must be calibrated per pose source (the report will say which source it used).
- The IMU peak counter over-counts inside a 30 s clip that includes non-curl movement
  (17 vs 10); it is the cross-check, not the rep source, in this pipeline.
- MM-Fit 3-D pose axes: the estimated up-vector is ≈ (0.03, −0.51, 0.86) in camera frame;
  MediaPipe world landmarks will differ, which is why `up` is estimated rather than assumed.

## 6. Next steps for Rochelle and Bryce

1. Rochelle: run `pose extract` on one MM-Fit video (`formcoach data fetch --dataset mmfit
   --with-video --session w00`) and compare MediaPipe angles with the `pose_3d` angles on the
   same frames (docs/00 Checkpoint 3, "cross-check").
2. Rochelle: `rules.yaml` v1 (PR 5) consumes `RepMetrics.metrics`; the key list is in
   `pipeline.compute_rep_metrics`.
3. Bryce: `formcoach demo --source replay` **without** `--headless` opens the OpenCV overlay
   (`app/overlay.py`) once `uv sync --extra vision` is installed; keyboard shortcuts and the
   Streamlit dashboard are docs/02 §4.6.
