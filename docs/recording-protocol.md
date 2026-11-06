# Team validation recordings: protocol

How each teammate records their validation sets (`docs/04-datasets.md` §6). About 45 minutes
per person. The recordings let us set the form-rule thresholds for a webcam, test the
wearable's gate on our own hardware, and make the go/no-go call on the vision path.

> **Status: draft, not yet piloted on hardware.** Robert runs one set end to end before the
> team starts.

**Privacy:** your video never leaves your laptop and is never committed. You share only
`imu.parquet`, `pose.parquet`, `meta.json` and `sets.csv`. `data/team/` is gitignored.

**Your subject code** (`S1`–`S5`), **the upload location** and **the deadline** are posted by
Robert in the team channel. The target is before the Nov 10 feature freeze.

## 1. Setup (once)

```
git checkout main
git pull
uv sync --group dev --extra vision --extra device
uv run formcoach pose fetch-model
```

Intel Macs cannot install the `vision` extra (`docs/howto/live-demo.md`). Record anyway and
tell Robert; pose extraction then needs another machine and your agreement.

## 2. Camera and space

- Laptop webcam, 2–3 m away, lens at waist-to-chest height, landscape, facing you straight on.
- Your whole body stays in frame for the whole set, including feet and hands overhead.
- Light in front of you, not behind. Only you in frame. Short or fitted sleeves.
- Check the framing with `uv run formcoach demo --camera 0 --exercise curl --no-log`. The
  skeleton should follow you. Press `q` to close it before recording: two programs cannot use
  the camera at once.
- Record with your laptop's camera app at 30 fps (Windows: Camera; Mac: QuickTime Player >
  New Movie Recording).
- Keep the same camera position for all your sets, and write the distance and lens height in
  the `notes` of your first set.

## 3. Wearable (if you have a kit)

- Left wrist, snug, case arrow pointing toward your elbow.
- Only one kit switched on in the room: every kit currently advertises the same Bluetooth name.
- Windows: run the commands in PowerShell, not WSL (WSL2 has no Bluetooth).

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

With a kit, add one 2-minute recording with `--exercise idle` and no video: walk around, pick
up and put down the weights, drink, type. It measures how often the gate wakes when you are
not exercising.

## 5. Each set

1. With a kit, start the wearable recording (change the subject and exercise):

   ```
   uv run formcoach record --source ble --subject S3 --exercise curl
   ```

   Wait for the `samples` lines to appear, and note the folder it prints
   (`data/team/S3/<id>`). Without a kit, create `data/team/S3/<YYYYMMDD-HHMMSS>/` by hand.
2. Start the video recording.
3. On your mark: stand still 5 s, clap 3 times (one per second, arms wide between claps),
   stand still 5 s, do the 10 reps, stand still 5 s. The claps are how the wearable's clock
   and the video's clock get lined up later, so do them in view of the camera.
4. Stop the video. Press Ctrl-C once in the terminal.
5. Move the video into the set's folder as `video.mp4` (or `.mov`).
6. Check and extract (use your own folder and exercise):

   ```
   uv run formcoach session check data/team/S3/<id>
   uv run formcoach pose extract --video data/team/S3/<id>/video.mp4 --out data/team/S3/<id>/pose.parquet
   uv run formcoach demo --video data/team/S3/<id>/video.mp4 --exercise curl --headless --no-log
   ```

   - `session check` (kit only) should show about 50 Hz and `problems none`.
   - The last command prints the reps it counted per arm (one counter for squat). For a clean
     set expect about 10. If it is far off, check the framing and redo the set **once**. If it
     is still off, keep the recording and say so in `notes`: a set the software miscounts is a
     result we need, not a failed recording. Fault sets may count fewer.
7. Add a line to `data/team/S3/sets.csv`:

   ```
   session_id,exercise,form,reps,notes
   20261106-141500,curl,clean,10,camera 2.5 m lens at 1.0 m
   20261106-142200,curl,partial,10,lost count around rep 7
   ```

   `reps` is what you actually did, not what the software printed.

## 6. When you are done

- Copy your `S#` folder **without the video files** to the upload location.
- Keep the videos on your laptop until Dec 9. We may ask you to re-run the extraction with
  another pose model (`--model full` or `heavy`) for the pose comparison.
- If a command fails, post the exact output in the team channel.

## 7. For maintainers: what this protocol works around

Checked against the code on 2026-11-05; none of it has run on a kit yet.

- `formcoach record` writes IMU only. Its `--camera` option prints "not wired yet", so video
  comes from the laptop's camera app.
- The wearable's clock (`t` in `imu.parquet`, device milliseconds) and the video's clock
  (`t = frame / fps` in `pose.parquet`) share no origin, and nothing in the repo aligns them.
  The three claps give a sharp accelerometer spike and a visible hands-together frame. An
  aligner that uses them is not written yet; `eval gating` on team sessions needs it.
- One recording per set keeps each clip short, so a webcam's variable frame rate cannot drift
  far, and makes the label (`sets.csv`) one line per folder.
- The button is not used to mark sets: K2 and K3 failed the button check on 2026-10-28.
- `formcoach eval transfer` is still a stub and there is no `data/team` loader. Threshold
  ratification and the go/no-go need only `pose.parquet` and `sets.csv`.
- `record` accepts `--exercise idle` (a canonical label in `formcoach.data.schema`) although
  its help text lists only the four exercises.
