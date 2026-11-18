# FormCoach Bench: testing, tuning, capture and demo UI — design

Date: 2026-11-18 · Status: awaiting Robert's approval · Decided with Robert in a brainstorming
session the same day · ADR-0035

A clickable mockup with invented data sits next to this file:
`2026-11-18-bench-ui-mockup.html` (open it in Chrome or Edge). Robert on the lab console view:
"I'd be quite happy if the real thing came out just like the mockup."

## 1. Intent

The live demo (`formcoach demo --camera N --board ble`) works end to end, but it is a closed
loop with an OpenCV window: the text cannot be read from the 2–3 m the user stands at, nothing
can be changed while it runs, the watch connection cannot be managed, and a run leaves nothing
that can be replayed or evaluated.

The Bench is one tool, run on the laptop, that the team uses to:

1. **produce numbers for the report** (team-recording transfer, gating on full sessions,
   feedback latency, BLE drop rate, rule rates on labelled webcam reps);
2. **record the team validation dataset** (one or two teammates will do the recording);
3. **run the class demo** in a separate demo mode that is readable from across a room.

Robert builds it. Christian may build a different UI later, so every change to the core must be
usable without this UI.

## 2. Goals, non-goals, anti-goals

**Goals**

- A UI-neutral live session in the core that any front end can attach to.
- A **coach page** (demo mode): count, one cue, state words, video with skeleton.
- A **lab console**: watch connection, device health, link log, IMU and pose charts on one
  time axis, gate source, threshold sliders, capture with labels.
- **Capture** that writes replay-ready session folders with IMU and pose on one clock.
- A **timeline review** page for recorded takes, built after the first two pages work.
- Runs with no hardware (fake watch, video file) for development and demo rehearsal.

**Non-goals (first version)**

- Any firmware or BLE protocol change. Kits are chosen by address; the watch's own model
  probability is shown only when the kit is on USB serial.
- Changing the watch's gate parameters at runtime; battery level.
- Viewing from a phone or tablet over the network (the server binds to 127.0.0.1 only).
- A fine-tuning workflow. Capture produces the data; training on it is a separate decision.
- Fixing the gate closing when the non-watch arm exercises. The gate source switch (§6.1)
  works around it during testing.

**Anti-goals (these make it a failure even if it works)**

- `formcoach demo --source replay --headless` or the golden replay test changes.
- Pose, rep or rule logic re-implemented in JavaScript.
- The core importing anything from the UI or the web server.
- Text drawn over video. Colour used without a word or icon.
- Video of a person saved by default, or anything under `data/team/` or `runs/` published.
- A Node build step, or a new heavy dependency.

## 3. Constraints

- **Dates.** Presentation Dec 2 or Dec 9, 2026; everything due Dec 9. Feature freeze has
  passed; Robert approved this as new scope on 2026-11-18. It competes with the report.
- **Platform.** WSL2 has no webcam and no Bluetooth, so the live path runs on native Windows,
  macOS or Linux. The fake watch and a video file must work in WSL2.
- **One BLE client per kit.** The Python process owns the link. The Web Bluetooth viewer
  cannot be connected to the same kit at the same time.
- **Stack.** CLAUDE.md names OpenCV and Streamlit for UI; ADR-0035 adds a local web server.
  `starlette`, `uvicorn` and `websockets` are already in `uv.lock` (through Streamlit); they
  are declared explicitly in an extra, imported lazily, with an install hint.
- **Team.** One CLI command and one Makefile target on all three operating systems.
- **Boundaries.** `formcoach.signal` never imports `formcoach.pose`. Parquet schemas add
  columns and never rename.
- **Privacy.** No teammate video is pushed anywhere. Pose landmarks may be shared.

## 4. Approaches

- **A. Local Python server with plain browser pages (chosen).** One process runs the live
  session and a small web server. Pages are hand-written HTML and ES modules with no build
  step, like `web/ble-viewer/`. Full control of the coach page's sizing; no new packages. Cost:
  the pages are JavaScript, which Python-only teammates cannot easily extend.
- **B. NiceGUI.** Same shape with panels declared in Python. Rejected: a new 21 MB dependency
  whose current major version is two weeks old, less control over the coach page, and the
  charts and skeleton canvas would be custom JavaScript anyway.
- **C. Larger OpenCV text plus a Streamlit review page.** Rejected: fixes legibility only;
  no comfortable scan list, sliders or label form.

Robert chose A.

## 5. Architecture

```
camera / video file ─┐
watch / fake watch ──┴─> LiveSession  (Python, no UI imports)
                            │  snapshot per frame, IMU batches, events
                            │  ^ commands
          ┌─────────────────┼────────────────────┐
     web bridge        OpenCV window         another UI
   /lab /coach /review  (today's demo)    (Christian, if he wants)
```

### 5.1 `LiveSession` (the core seam)

Today `run_live` (`src/formcoach/app/live.py`) owns the loop and keeps the pipeline, banners
and traces as local variables. That becomes a class that owns the same state and exposes it:

- **Inputs:** a frame source (camera or video), an optional board (BLE, serial, fake), a
  detector, a rules engine, an exercise.
- **Outputs, pushed to subscribers:**
  - a **snapshot per frame**: frame time, the raw unmirrored frame, image and world landmarks
    (whenever pose ran on that frame), every tracked joint angle, each counter's primary angle, enter and
    exit thresholds and armed flag, arm visibility, gate state and source, totals, fps;
  - **IMU batches** as they arrive (time, accel, gyro, flags, sequence), which today are
    discarded;
  - **events**: rep, fault, gate open/close, plus new `rep_dropped` (too short, tracking gap),
    `params_changed`, link state changes and link log lines.
- **Commands, applied between frames:** set exercise, set gate source, set a parameter,
  reset parameters, reset counters, start and stop a capture, send a device command, connect
  and disconnect the watch, stop.
- **No UI knowledge.** It never imports OpenCV windows, the server or anything under `web/`.

`run_live` and `formcoach demo --camera` keep their signatures and behaviour and are
re-implemented as a client of `LiveSession`. That is the proof that the seam serves two front
ends, and it stays as the demo-day fallback.

The replay path's default behaviour and its golden output do not change. Replay gains one
opt-in mode, "as recorded" (§7), so that a captured take is judged by the logic that produced
it.

**When pose runs.** Outside a capture, pose runs only while the selected gate is open, as
today; that is the saving the system exists to show. **During a capture, pose runs on every
frame**, whatever the gate does, and counting still follows the gate. Otherwise a take would
hold landmarks only where the watch gate was open, the always-on baseline of `eval gating`
could not be computed from it, and trying a different gate on the review page would have no
pose to work on.

A slow subscriber must never slow pose estimation: frames go to a latest-value slot (old
frames are dropped), events and IMU batches go to a queue that does not drop.

### 5.2 Watch link

- A connection manager replaces "first name match wins": scan (name, identifier, signal
  strength), connect to a chosen device, disconnect, reconnect, each reported as an event.
  The same panel offers USB serial (port list) and the fake watch.
- **Telling kits apart.** Every kit advertises the same name. The identifier is the MAC
  address on Windows and Linux, and a per-laptop UUID on macOS, so it cannot be matched to
  `docs/devices.md` there. Two things cover both: an **Identify** button that flashes the
  connected kit's LED, and a kit label (K1…K6) the user can attach to a scanned device, which
  is remembered on that laptop and written into each capture's `meta.json`.
- Device status (firmware, uptime, calibrated, session) is read from the status packet that
  `BLESource` already parses; sample rate, drops and bad packets from the decoder's counters.
- The fake watch becomes paced and endless, answers commands, and reports a status, so the
  whole UI can be developed without hardware.
- Over USB serial, the firmware's `# infer … class= p_active=` lines are parsed and shown as
  an extra chart lane. Over BLE that lane is absent.

### 5.3 Web bridge

- `formcoach bench` (and `make bench`) starts the session and the server in one process,
  prints the URL and opens the lab console. Options mirror `demo`: `--camera N` or
  `--video FILE`, `--exercise`, `--model`, plus a watch choice that can also be made from the
  console. Ctrl-C stops cleanly and releases the camera.
- One WebSocket carries everything. The message schema is versioned and documented, because it
  is the second way in for another UI. Each video frame arrives together with its landmarks so
  the skeleton never lags the picture.
- The browser draws the skeleton on a canvas and all text in HTML. Video can be switched off
  (skeleton on a plain background); frames are then not encoded or sent.
- The server binds to 127.0.0.1.

## 6. Pages

### 6.1 Lab console (`/lab`)

As in the mockup:

- **Left rail.** Watch (BLE / USB serial / fake; scan list with address and signal strength;
  connect, disconnect), device health, LED test and calibrate buttons, link log.
- **Centre.** Video or skeleton, a compact count and last cue with the measured value against
  its limit ("drift 41° > limit 35°"), and strip charts on one shared time axis: accel, gyro,
  gate lanes (watch bit, laptop energy gate), primary joint angle per arm with the enter/exit
  band and rep marks coloured good / flagged / dropped, and a per-rep metric against its limit.
- **Pause and scroll back.** The charts keep the last few minutes; pause freezes them so a
  person testing alone can do a set, walk back and inspect it.
- **Right rail, tuning.**
  - *Gate source:* **watch** (today's behaviour), **energy** (the existing laptop-side energy
    gate fed from the live IMU stream, with threshold and hysteresis sliders), **open**, or
    **manual hold**. They are recorded under the names the evaluations already use: `device`,
    `energy`, `always_on`, and a new `manual`. The code's gate called `laptop` is the
    random-forest model; it is not offered until its model file exists on the machine. The
    energy gate is always computed and plotted when a watch is connected, whichever source is
    selected.
  - *Rule threshold sliders* for the selected exercise. Moving one re-judges every rep of the
    session from its stored metrics and shows the new totals next to the old.
  - *Rep counter:* minimum range first; other segmenter parameters later.
- **Right rail, capture.** See §7.
- **Top.** Exercise selector (all four; the console states that only curl has webcam-tuned
  thresholds), reset counters, "Open coach page on second screen".

Slider changes never edit the packaged rules files. They form an overrides set that is shown,
can be reset, and is saved with every capture. "Save as rules file" writes a complete YAML to
a path the user chooses, to be reviewed and committed by hand.

### 6.2 Coach page (`/coach`), the demo mode

- Mirrored video with skeleton on the left; a solid panel with the exercise and the rep
  counts; a full-width cue band; a strip of three state words (gate, watch, recording). No
  controls except full screen and sound.
- **Two big numbers side by side, L and R,** each with its own row of rep dots (Robert's
  choice: always unambiguous, including alternating and one-arm sets). Squat has one number.
- **Sizes** are in viewport-height units so display scaling cannot shrink them: count digits
  about 26% of the height (about 50 mm on a 15.6″ laptop), cue text 9–13%, state words about
  4.5%. Target: readable on a 15.6″ laptop at 2.5 m (about 12 mm letter height for the
  smallest text that must be read).
- The cue stays until that arm's next rep, or 8 s.
- Labels use the person's own left and right, as today.
- Two tones: good rep, fault.
- It opens from the console into its own window, to be dragged to a second screen.

### 6.3 Timeline review (`/review`), built last

Pick a captured take; scrub the skeleton and every lane (watch gate, laptop gate, joint angle
with thresholds, rep blocks, IMU) under one playhead; edit labels; move sliders to re-judge
that take or all takes; see counted against true reps, gate-open share and cue delay. It reads
the session folders of §7 through the same replay code the evaluations use. It supersedes the
roadmap's Streamlit dashboard item.

## 7. Capture

- **One take is one set and one folder:** `data/team/<S#>/<session_id>/`, the layout
  `SessionRecorder` already uses.
- **Labels before recording:** subject, exercise, form (`clean`, `partial`, `no_lockout`,
  `bent_elbow`, `shallow`), working arm (`both`, `l`, `r`), notes. **After:** true rep count.
- **What a rep count means.** Counts are per arm. For `both`, "true reps 10" means ten with
  each arm, as in the recording protocol; for `l` or `r` it is that arm's count. `session
  check` compares per-arm counts with the true count for the labelled arm (today it counts
  every rep event, which would flag each clean two-arm set), and the summary's `reps` total
  stops being the only number.
- **Idle takes.** Exercise `idle` records IMU, and pose if a camera is on, with no counting
  and no rules. It is the recording protocol's two-minute false-wake measurement.
- **Start and stop.** Record starts after a selectable delay (0, 5 or 10 s) so one person can
  reach the mark. It stops on the Stop button, or by itself 5 s after the gate closes when
  that option is on.
- **Files:** `imu.parquet` (with flags), `pose.parquet` (image and world landmarks),
  `events.parquet`, `frames.parquet`, `meta.json` (labels, clock block, overrides, gate
  source, rules file, pose model, camera, firmware, device address, app version), and a line
  appended to `data/team/<S#>/sets.csv` with an added `arm` column.
- **One clock.** Frames and IMU arrivals are stamped with the same laptop monotonic clock.
  Device time is mapped onto it (offset and drift, robust to the five-sample batching), the
  result is the `t` of both Parquet files, and the raw device time is kept in an added column.
  The fit and its residual go in `meta.json`.
- **Replay "as recorded".** When a session's `meta.json` says it was captured by the Bench,
  replay builds the pipeline the way the live run did: one track per arm, the rules file and
  overrides stored in the take, the same metrics window, and the gate as the take recorded it
  frame by frame. The mode is driven by `meta.json` (with a flag to force it on or off). The
  committed fixture has no such marker, so its output is unchanged.
- **Acceptance:** replaying a captured take as recorded gives the same per-arm rep counts and
  faults as the live run, and `eval gating` accepts the folder and computes its always-on
  baseline from it.
- **Keep or discard.** A take is held in memory until the operator presses Keep, so discarding
  deletes nothing that was saved. After Keep, the `session check` result is shown.
- **Video** is not saved. A per-take opt-in writes `video.mp4` into the folder, local only.
- **Latency.** Each rep event records when its frame was captured and when the coach page
  confirmed drawing the cue.
- This replaces the camera-app-and-claps steps of `docs/recording-protocol.md`, which is
  rewritten when capture works. The claps remain as an optional cross-check of the clock fit.

## 8. Build order

Sizes are working days and are estimates.

| # | Step | Size | Useful result |
|---|---|---|---|
| 0 | Measure video encoding and WebSocket cost against the pose loop; fix frame size and rate cap | 0.5 | Go or adjust before building |
| 1 | `LiveSession`; `run_live` on top of it; IMU kept; paced fake watch | 1.5 | Same demo, open core |
| 2 | Server, schema, lab console with watch panel, health, log, video, charts | 2.5 | Signals visible live |
| 3 | Coach page, second-screen button, tones | 1 | Demo mode |
| 4 | Gate source, laptop gate, rule sliders with re-judge, overrides, save-as | 1.5 | Live tuning |
| 5 | Capture: labels, delay, one clock, folder, replay as recorded, check, keep/discard, latency | 3 | Dataset and report numbers |
| 6 | Timeline review | 2–3 | Review and re-judge takes |

Steps 0–5 are about ten days. Each step ends in something usable and is its own PR.

## 9. Testing

- `LiveSession` is unit-tested with the fake watch, synthetic pose frames and an injected
  clock. The existing live and golden replay tests must pass unchanged.
- The WebSocket schema is pinned by a test, as the packet contract is.
- JavaScript modules get Node unit tests run by `make test`, as the BLE viewer's do.
- One headless smoke test starts `bench` with the fake watch and a short video and checks that
  frames, IMU batches and a rep event arrive over the socket.
- Hardware checks that cannot be automated (BLE scan and connect on Windows, the coach page at
  2.5 m, clock-fit residual on a real kit) are a written checklist run by Robert on Windows.
  The live path is not tested on Mac hardware for now (Robert, 2026-11-18); the code stays
  cross-platform and the Identify button and kit label are the workaround for macOS's hidden
  addresses.

## 10. Left to the builder

- Threading layout, the exact wire encoding, where the static files live. Constraints: no
  build step; reuse the vendored uPlot and the viewer's chart code and style tokens rather
  than copying; nothing that needs the server is published as a dead page on GitHub Pages.
- The clock-fit method, within the acceptance test of §7.
- Which per-rep metric lanes to show per exercise.

## 11. Risks

- **Video encoding slows pose.** Unmeasured; step 0 exists to find out before building.
- **Swapping the rules engine mid-run** and reaching frames and landmarks through the current
  seams are untested inferences from reading the code.
- **Windows timer resolution** may add jitter to the clock fit.
- **Time.** Ten days before the timeline page is a large share of what is left. Steps 3 and
  5 are the ones the demo and the report depend on.

## 12. Paperwork that goes with the build

- ADR-0035 (this decision), the UI row of the stack table in CLAUDE.md, a
  `docs/howto/bench.md` that includes the message schema, and the rewrite of
  `docs/recording-protocol.md` once capture works.
