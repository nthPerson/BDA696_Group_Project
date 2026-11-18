# Team validation recordings: protocol

How each teammate records their validation sets (`docs/04-datasets.md` §6). About 45 minutes
per person. The recordings let us set the form-rule thresholds for a webcam, test the
wearable's gate on our own hardware, and make the go/no-go call on the vision path.

> **Status: draft, not yet piloted on hardware.** Robert runs one set end to end before the
> team starts. Every set is recorded in the FormCoach Bench (`docs/howto/bench.md`), which has
> so far run only with its synthetic source, not with a real webcam or watch.

**Privacy:** your video never leaves your laptop and is never committed. You share only
`imu.parquet`, `pose.parquet`, `events.parquet`, `frames.parquet`, `meta.json` and `sets.csv`.
`data/team/` is gitignored. The Bench writes no video at all; `meta.json` records your laptop's
name and operating system.

**Your subject code** (`S1`–`S5`), **the upload location** and **the deadline** are posted by
Robert in the team channel.

## 1. Setup (once)

```
git checkout main
git pull
make setup-live
```

Without `make` (Windows): `uv sync --group dev --extra vision --extra device --extra bench`,
then `uv run formcoach pose fetch-model --model lite`.

Intel Macs cannot install the `vision` extra (`docs/howto/live-demo.md`), so the Bench cannot
run there with a camera. Tell Robert; you can record on a teammate's laptop.

## 2. Camera and space

- Laptop webcam, 2–3 m away, lens at waist-to-chest height, landscape, facing you straight on.
- Your whole body stays in frame for the whole set, including feet and hands overhead.
- Light in front of you, not behind. Only you in frame. Short or fitted sleeves.
- Check the framing in the lab console's video: the skeleton should follow you.
- Keep the same camera position for all your sets, and write the distance and lens height in
  the `notes` of your first set.

## 3. Wearable (if you have a kit)

- Left wrist, snug, case arrow pointing toward your elbow.
- Only one kit switched on in the room: every kit currently advertises the same Bluetooth name.
  If more are on, use **Identify** in the lab console to find yours.
- Windows: run the commands in PowerShell, not WSL (WSL2 has no Bluetooth and no camera).

Battery safety (`docs/03-hardware.md` §4):

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

## 4. What to record: 12 sets, one recording per set

Use light dumbbells or none. Do exactly 10 reps, both arms together, about 3 seconds per rep,
returning fully to the start position each time. Stop if anything hurts.

| Exercise | Sets 1 and 2 | Set 3: one deliberate fault, on every rep | `form` value |
|---|---|---|---|
| curl | clean | stop when the forearm is horizontal | `partial` |
| press | clean | stop with the elbows clearly bent at the top | `no_lockout` |
| raise | clean | elbows bent about 90 degrees | `bent_elbow` |
| squat | clean | quarter squat | `shallow` |

With a kit, add one 2-minute idle take: choose exercise `idle` in the lab console's top bar
(form `clean`), leave "Stop 5 s after the gate closes" unticked, and press **■ Stop** after
2 minutes. Meanwhile walk around, pick up and put down the weights, drink, type. Nothing is
counted. It measures how often the gate wakes when you are not exercising.

## 5. Each set: record in the Bench

1. Start the Bench once for all your sets (Windows without `make`:
   `uv run formcoach bench --camera 0`):

   ```
   make bench ARGS="--camera 0"
   ```

   The lab console opens in your browser. If the video is black, stop it (Ctrl+C) and try
   `--camera 1`.
2. With a kit: in **Watch**, BLE, **Scan**, **Connect** on your kit, **Identify**, and set its
   kit label. **Device health** should show about 50 Hz. Keep the gate source on **Watch**.
   Without a kit, set the gate source to **Open**.
3. Pick the exercise in the top bar. In **Capture** fill in Subject (your code), Form (the
   `form` value from the table), Arm `both`, and Notes. Set **Start after** to 10 s. With a kit,
   tick "Stop 5 s after the gate closes".
4. Press **● Record set** and walk to your mark. On your mark: stand still 5 s, do the 10 reps,
   stand still 5 s. Optional, as a check of the clock: before the reps, clap 3 times (one per
   second, arms wide between claps) in view of the camera.
5. Press **■ Stop** (or let it stop by itself). Read the summary: reps per arm, faults, IMU
   rate and drops.
6. Type **True reps**: what you actually did, per arm (10 for a set of 10 with both arms), not
   what the software counted. Press **Keep take**, or **Discard** for a false start (nothing is
   written until Keep).
7. Read the check under the panel. "✓ Session check: no problems" is what you want. A problem
   line such as `left arm reps 9 != expected 10` means the software counted differently: if it
   is far off, check the framing and redo the set **once**. If it is still off, keep it and say
   so in the `notes` of that line of `sets.csv`: a set the software miscounts is a result we
   need, not a failed recording. Fault sets may count fewer. A rate that is not 50 Hz, or
   dropped samples, means the watch link was poor: move the laptop closer and redo the set.

Keep writes the take to `data/team/<S#>/<YYYYMMDD-HHMMSS>/` and adds its line to
`data/team/<S#>/sets.csv` (`session_id,exercise,form,reps,notes,arm`). You do not need to
write either by hand.

## 6. When you are done

- Copy your whole `data/team/S#` folder to the upload location. There is no video in it.
- If a command fails, post the exact output in the team channel.

## 7. For maintainers: what this protocol relies on

Rewritten 2026-11-19 for the Bench; not yet run on a kit or a real webcam.

**Solved by the Bench** (ADR-0035, ADR-0036):

- IMU and pose are on one clock. Frames are stamped with the laptop's clock; the wearable's
  device time is mapped onto it by a lower-envelope fit (`src/formcoach/io/clock.py`). `t` in
  `imu.parquet` and `pose.parquet` is seconds from the start of the take; the raw times stay in
  `t_device` and `t_arrival`, and the fit and its residual are in `meta.json` under `clock`.
  The claps are only a cross-check of that fit.
- Labels are written for you: `meta.json` holds `labels`, `arm` and `expected_reps`, and
  `sets.csv` gets one line per kept take with an added `arm` column.
- `session check` compares each arm's count with the true count, so a clean two-arm set of 10
  is not flagged.
- `formcoach demo --session <take> --headless` replays a take with the rules, overrides and gate
  it was recorded with.

**Still open:**

- The pilot: Robert records one set end to end on a kit (`docs/howto/bench.md` §9).
- `formcoach eval transfer` is still a stub and there is no `data/team` loader.
- `eval gating` judges a take with `rules.yaml`, not with the take's own rules file.
- No video is kept, so a take cannot be re-extracted with another pose model; choose `--model`
  when starting the Bench. A per-take `video.mp4` option is left for the timeline-review plan.
- The button is not used to mark sets: K2 and K3 failed the button check on 2026-10-28.
