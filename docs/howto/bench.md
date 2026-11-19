# How to: FormCoach Bench (lab console, coach page, capture)

**Owner:** Bryce Hall (app, demo) · **Built by:** Claude Code with Robert, 2026-11-18/19 · **Branch:** `feat/bench`
**State:** verified only with `--fake` and automated tests (523 pytest, 109 Node). **Not yet run with a real webcam, a real watch, or on Windows or macOS;** Robert runs §9 first.

One command starts a live session and a local web server; two browser pages attach to it: the **lab console**
(`/lab`: watch, signals, tuning, capture) and the **coach page** (`/coach`: big counts and one cue, for the
demo). Design: `docs/superpowers/specs/2026-11-18-bench-ui-design.md`; ADR-0035 and ADR-0036.

## Battery safety

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

## 1. Set up and start

```bash
make setup-live                       # once: dev tools + vision, device, bench extras + pose model
make bench ARGS="--fake"              # no hardware: a synthetic person and a fake watch
make bench ARGS="--camera 0"          # webcam 0; connect the watch from the console
make bench ARGS="--video clip.mp4"    # a video file instead of a camera
```

Windows without `make`: `uv sync --group dev --extra vision --extra device --extra bench`, then
`uv run formcoach pose fetch-model --model lite`, then `uv run formcoach bench --camera 0`. It prints
`FormCoach Bench (camera 0): http://localhost:8010/lab` and opens the lab console; Ctrl+C stops it and
releases the camera. Options (`formcoach bench --help`): `--exercise` (curl, press, raise, squat, idle),
`--model` (lite, full, heavy), `--rules FILE`, `--board ble|serial|fake` with `--port`, `--http-port` (8010),
`--video-width` (640; 1280 for a projector), `--no-open`. A real camera and Bluetooth need native Windows,
macOS or Linux (WSL2 has neither); `--fake` works anywhere, WSL2 included.

## 2. The two pages

**Lab console.** Top: exercise, Reset counters, gate state, Open coach page ↗. Left: watch, device health,
link log. Centre: video with skeleton (Video: on/off; off sends the skeleton only), counts, the last cue with
its value against the limit, and strip charts on one time axis (accel, gyro, gates, joint angle with its
enter/exit band and rep marks, one per-rep metric). Pause (or Space) freezes the charts; the slider scrolls
back up to 5 minutes. Right: tuning and capture.

**Coach page.** Opens in its own window (a popup, not a tab): drag it to the second screen, press F for full
screen. Mirrored video and skeleton, L and R counts with rep dots (one count for squat; hollow dots are reps
counted before the page opened), one cue band (until that arm's next rep, or 8 s), three state words (gate,
watch, recording; CAMERA STOPPED or OFFLINE when the source or the bench stopped). Click the chip once to
enable the two tones (good rep, fault): browsers block sound until a click.

## 3. Connect a watch

Power the kit on, antenna clipped on. In **Watch** pick BLE, USB serial or Fake, press **Scan**, then
**Connect** on a row. Every kit advertises the same name, so press **Identify** (the kit's LED flashes) and
set the row's **kit label** (K1-K6), kept on this laptop (`~/.formcoach/kits.json`) and written into every
take. **Device health**: firmware, rate (about 50 Hz), drops, calibration, uptime; **LED test** and
**Calibrate** (hold the kit still). The watch's own model probability and timing show only over USB serial.

## 4. Tune

**Gate source:** Watch (`device`: the kit's gate flag), Energy (`energy`: the laptop gate on the live IMU
stream), Open (`always_on`) or Hold (`manual`: open or closed by hand). Rule sliders re-judge every rep of
the session ("With these values: … (was …)"). Sliders never edit a packaged file: they are overrides, saved
with every take; **Reset** drops them. **Save as rules file…** writes a complete YAML inside the folder the
bench was started in (suggested `runs/rules/<exercise>-<date>.yaml`; `.yaml`/`.yml` only; it asks before
replacing). Try it with `--rules <file>`; copy it into `src/formcoach/rules/` by hand after review.

## 5. Capture a take

One take is one set. In **Capture**: Subject (S1-S5; letters, digits, `_`, `-`), Form, Arm, Notes, Start after
(0, 5 or 10 s), optional "Stop 5 s after the gate closes" (ignored with Open). The exercise follows the top bar
(choose `idle` there for an idle take). **● Record set**, do the set, **■ Stop**. Read the summary; type **True
reps** (per arm: 10 with both arms means 10 each), then **Keep take** or **Discard**. A take is held in memory
until Keep; Discard and Ctrl+C write nothing. Keep writes `data/team/<S#>/<YYYYMMDD-HHMMSS>/`, appends a line
to `data/team/<S#>/sets.csv` (with an `arm` column) and shows the `session check` result. The watch cannot be
changed while recording.

The folder holds `imu.parquet`, `pose.parquet`, `events.parquet`, `frames.parquet` and `meta.json` (labels,
`clock` fit with `residual_ms` and `segments`, `rules` and `rules_doc`, overrides, gate source, `source` (link
kind), `frame_source`, pose model, camera, firmware, kit, laptop, summary, `latency`). No video is written.
Frames and IMU share one laptop clock: `t` is seconds from the take's start (a restarted watch clock is fitted
per segment; Keep warns). While recording, pose runs on every frame; counting still follows the gate. Cue
delays are recorded only while the coach page is visible, for cues shown as their rep completed.

## 6. Replay a take

`uv run formcoach demo --session data/team/S1/<id> --headless` replays a take as recorded (its stored rules
document, overrides and per-frame gate; per-arm totals): `--gate recorded` is the default,
`--gate always_on|energy|laptop|device` re-gates it, `--no-as-recorded` replays it the old way. `eval gating`
also judges takes with their own rules (unless `--rules`) and skips the IMU gates for a take without a watch.

## 7. Troubleshooting and limits

| Symptom | Cause / fix |
|---|---|
| `cannot serve on 127.0.0.1:8010` | another bench is running: stop it or pass `--http-port 8011` |
| Kit not in the scan list | one BLE client per kit: close the Web Bluetooth viewer, `ble-check.py`, other `formcoach` runs |
| Two identical names listed | Identify, then label; on macOS the identifier is a per-laptop UUID, not the MAC in `docs/devices.md` |
| "disconnected, retrying" on a page | the bench process stopped; the page reconnects by itself when it is back |
| No camera, or a black one | as for `demo` (`docs/howto/live-demo.md` §6): try `--camera 1`; check camera permissions |

Limits: a page opened mid-recording shows no elapsed time; press, raise and squat use design thresholds;
the timeline review page (`/review`) is a placeholder, and it and the per-take `video.mp4` option are a later plan.

## 8. Message schema (for another client)

One WebSocket, `ws://localhost:8010/ws`, schema version 1, built in `src/formcoach/bench/protocol.py`,
pinned by `tests/golden/bench_messages.json`. Only local hosts are answered; other web origins are refused.
The origin check stops other web pages, not other programs: any program or user on this machine can connect
without an Origin header, see the camera frames and send commands.

| Server → page | Main fields |
|---|---|
| `hello`, `state` | `hello` has `v` (1); both: `state`: `exercise, gate_source, manual_gate, params, values, rules, totals, link, capture, video, source` (`"ended"` once the source stops) + option lists |
| `frame` (binary) | 4-byte big-endian header length, JSON header, JPEG (none if `has_image` is false). Header: `i, t, t_mono, gate, gate_source, gates, energy, pose_ran, lm` (33 × `[x, y, vis]`), `counters, totals, fps, exercise, link_state, capture_state, has_image` |
| `imu` | parallel lists `t, ax, ay, az, gx, gy, gz, flags, seq` (s, m/s², rad/s) |
| `event` | `kind, t, payload`; kinds include `rep, fault, rep_dropped, gate_open, gate_close, gate_source, exercise, counters_reset, link, capture, params_changed, rejudged, cue_shown, rules_saved, error`; `exercise` and `counters_reset` start new counts (clear the dots) |
| `link`, `scan`, `log`, `error` | `status`; `kind, devices[{kind, id, name, rssi, kit}]`; `level, text, t`; `text, cmd` |

| Page → server `{"cmd": …}` | Fields |
|---|---|
| watch | `scan {kind}`, `connect {kind, id}`, `disconnect`, `device {action}` (identify, led_good, led_fault, calibrate), `label_kit {id, kit}` |
| session | `set_exercise {exercise}`, `set_gate_source {source}`, `set_manual_gate {open}`, `reset_counters`, `set_video {on}` |
| tuning | `set_param {key, value}`, `reset_params`, `save_rules {path, overwrite?}` |
| capture | `capture_start {labels{subject, exercise, form, arm, notes}, delay_s, auto_stop}`, `capture_stop`, `capture_keep {true_reps}`, `capture_discard`, `cue_shown {arm, rep_id}` |

**Python way in.** Skip the server: `LiveSession` (`src/formcoach/app/session.py`) calls your listeners
(`on_snapshot`, `on_imu`, `on_event`) on its own thread; other threads queue work with `session.call(fn)`:

```python
from formcoach.app.session import LiveSession
from formcoach.pose.synthetic import FakeScene

scene = FakeScene(exercise="curl", duration_s=20)
session = LiveSession(exercise="curl", detect=scene.detect)
session.subscribe(on_event=lambda e: print(e.kind, round(e.t, 1), e.payload.get("arm")))
session.run(scene.frames())  # returns when the frames end, or after session.stop()
```

## 9. Hardware checklist (Robert, native Windows)

- [ ] `uv run formcoach bench --camera 0` in PowerShell: lab console opens; video and skeleton move.
- [ ] BLE: Scan lists the kit with a signal strength; Connect; health shows fw 1.1.1, ~50 Hz, 0 drops.
- [ ] Disconnect and Connect again; switch the kit off and on: the log shows silent, then linked.
- [ ] Identify flashes the right kit's LED; its kit label survives a restart. USB serial connects too.
- [ ] Coach page on the second screen, full screen: counts, cue and state words readable at 2.5 m.
- [ ] Raise your left arm: the L count and the left-arm highlight are on the screen's left on both pages.
- [ ] `uv run python -c "import time; print(time.get_clock_info('monotonic'))"`: note the resolution (may be ~15.6 ms).
- [ ] Keep a 10-rep curl take, coach page open: session check shows no problems; in `meta.json` `clock`,
  `method` is `envelope`: write down `residual_ms` and the summary's cue delay (median).
- [ ] `formcoach demo --session <take> --headless` gives the review summary's per-arm reps and faults.
- [ ] Ctrl+C: the camera light goes off and nothing is left running.
