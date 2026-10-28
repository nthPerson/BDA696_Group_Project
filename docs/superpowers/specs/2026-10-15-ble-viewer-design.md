# BLE live viewer — design

**Date:** 2026-10-15 · **Branch:** `feat/ble-viewer` · **Owner on handover:** Bryce (app, docs, demo)
**Status:** approved in conversation with Robert, 2026-10-15

## 1. Purpose

A static web page that connects to a FormCoach wearable over Web Bluetooth and shows its
live IMU stream. Two uses:

1. Robert demos the prototype (K1) to the team: live accelerometer and gyro traces, gate flag.
2. Each teammate uses it as a bring-up and debugging tool for their own kit, on macOS or
   Windows (Windows + WSL users run the browser on the Windows side).

Success: from a fresh clone (or from the Pages URL with no clone), a teammate on Chrome/Edge
connects to their kit in under a minute. They see |a| ≈ 1.00 g when the kit lies flat, gyro
traces respond to rotation, the gate pill lights when they shake the kit, and a recording
downloads as a CSV readable by the existing serial-CSV tooling.

Out of scope: rep counting, pose, rules and gate inference in the browser; multi-device
views; any change to firmware or the BLE protocol.

## 2. Constraints

- Web Bluetooth needs a secure context (https or `http://localhost`) and Chrome/Edge
  (desktop on Windows/macOS, plus Android). Safari, Firefox and iOS have no Web Bluetooth.
- The firmware accepts one BLE client at a time: advertising restarts only on disconnect
  (`firmware/src/app/ble.cpp`).
- No npm, no build step. Teammates must not need a JS toolchain (CLAUDE.md "People").
- The protocol is fixed: `firmware/include/protocol.h` ↔ `src/formcoach/io/protocol.py`.
  The page is a third mirror and is pinned by a test, like `tests/test_protocol.py`.
- Kits flashed from `main` before `fw/imu-bringup` (ADR-0025) merges report accel at 4× scale
  (±2 g default range). The howto warns about this; K1 already runs the fixed firmware.

## 3. Files

```
web/ble-viewer/
  index.html        layout + CSS; loads app.js as an ES module
  protocol.js       UUIDs, scales, sizes, command codes, flag bits, GOLDEN_BATCH_HEX;
                    decodeBatch(), decodeStatus(), encodeBatch()  (no DOM, importable by Node)
  app.js            Web Bluetooth link, ring buffer, uPlot charts, flag pills, controls,
                    CSV recorder, simulator
  vendor/uPlot.iife.min.js, vendor/uPlot.min.css, vendor/LICENSE   (MIT, exact version pinned)
  test/protocol.test.mjs   node --test: golden decode + encode/decode round trip (local only)
src/formcoach/cli.py       + `formcoach viewer [--port 8000] [--no-open]`
Makefile                   + `ble-viewer` target → `formcoach viewer`; `test` also runs the
                             Node test when `node` is on PATH (skipped with a note otherwise)
.github/workflows/pages.yml  push to main touching web/** (or manual dispatch) → deploy web/
tests/test_web_protocol.py   protocol.js constants and golden hex == protocol.py
docs/howto/ble-viewer.md     teammate guide
README.md, docs/DECISIONS.md (ADR-0026), docs/STATUS.md
```

## 4. Page

```
┌ FormCoach live IMU ───────────────────────────────────────────────────────┐
│ [Connect] [Simulate]  FormCoach-8428 · 10.0 pkt/s · 50.0 Hz · drops 0     │
│ fw 1.1.0 · up 312 s · calibrated ✓ · session #3   (GATE) (BUTTON) (SESS)  │
├───────────────────────────────────────────────────────────────────────────┤
│ Accel (g)   ax ay az |a|                                last 10 s, scroll │
├───────────────────────────────────────────────────────────────────────────┤
│ Gyro (dps)  gx gy gz                                                      │
├───────────────────────────────────────────────────────────────────────────┤
│ [Ping] [LED ✓] [LED ✗] [Start/Stop session] [Calibrate]  [● Record → CSV] │
└───────────────────────────────────────────────────────────────────────────┘
```

Light and dark themes follow the OS. Usable at laptop width; the charts stack on narrow
screens (Android).

## 5. Data flow

1. **Connect:** `navigator.bluetooth.requestDevice({filters: [{services: [SERVICE_UUID]}]})`.
   Then connect GATT, subscribe to IMU notifications, read and
   subscribe to status. The control characteristic is looked up and kept for writes.
2. **Decode:** `decodeBatch(DataView)` returns `{sessionId, samples[], error}`. A sample is
   `{t_ms, ax, ay, az, gx, gy, gz, flags, seq}` in raw LSB. A packet shorter than
   `4 + n·19` returns `error` and is counted as a bad packet; it never throws.
3. **Buffer:** a ring of the last 10 s (500 samples at 50 Hz) holds raw values and the derived
   g / dps / |a| (accel ÷ 4096, gyro ÷ 32.8). A `seq` gap (mod 2¹⁶) adds the gap to `drops`.
   Packet and sample rates are measured over a sliding 2 s window.
4. **Render:** a `requestAnimationFrame` loop calls `uplot.setData()` on both charts and
   updates the pills and counters. BLE handling never waits on rendering. The x axis is
   seconds relative to the newest sample (−10 … 0).
5. **Controls:** each button writes one byte to the control characteristic: PING 0x10,
   LED good 0x01, LED fault 0x02, START 0x03 / STOP 0x04 (the button label follows the
   SESSION_ACTIVE flag), CALIBRATE 0x05. Calibrate shows "keep it flat and still for 3 s" first.
6. **Record:** start/stop toggle. Samples are appended raw. Stop downloads
   `<device>_<YYYYMMDD-HHMMSS>.csv` with header `t_ms,ax,ay,az,gx,gy,gz,flags,seq`, which is
   `FC_SERIAL_CSV_HEADER`. A recording survives a disconnect and can still be downloaded.
7. **Simulate:** a synthetic source (gravity on z, a 0.5 Hz curl-like rotation about one
   axis with matching gyro, noise, the gate flag set while "moving", 6 s moving and 4 s still)
   builds samples, packs them with `encodeBatch()` into 5-sample batches every 100 ms and
   feeds the bytes to the same `decodeBatch()` path. No dataset clip is shipped.

## 6. Failure modes

| Situation | Behaviour |
|---|---|
| `navigator.bluetooth` missing | Banner: use Chrome/Edge on Windows, macOS or Android. Connect disabled, Simulate works. |
| User cancels the chooser (`NotFoundError`) | No message. |
| Other connect errors | Message under the Connect button plus the hint: one client at a time, close `ble-check` / `formcoach demo` / other tabs, antenna clipped on. |
| `gattserverdisconnected` | Status grey "disconnected at hh:mm:ss"; **Reconnect** reuses the device without the chooser; charts freeze; the recording is kept. |
| Truncated packet | `bad packets` counter plus an MTU hint (a batch needs MTU ≥ 102). |
| Status/control characteristic missing | Charts work; controls disabled with a tooltip. |

## 7. Testing

1. **CI (pytest), `tests/test_web_protocol.py`:** parse `protocol.js` with regexes and check
   that the UUIDs, `ACCEL_LSB_PER_G`, `GYRO_LSB_PER_DPS`, `SAMPLE_SIZE`, `BATCH_HEADER_SIZE`,
   `BATCH_SAMPLES`, `STATUS_SIZE`, the command codes and the flag bits equal `protocol.py`.
   Also check `GOLDEN_BATCH_HEX == protocol.pack_batch(7, GOLDEN_SAMPLES).hex()` for 5
   fixed samples that cover negative values, every flag bit and a `seq` wrap.
2. **Local (Node 20, `node --test web/ble-viewer/test/`):** decode `GOLDEN_BATCH_HEX` to the
   known samples; round trip encode → decode; truncated packet → `error`; status decode.
   Run by `make test` when `node` is on PATH. Not in CI (Robert, 2026-10-15).
3. **Headless smoke (Playwright, by Claude before the PR):** Simulate for ≥ 5 s; expect about
   10 pkt/s, about 50 Hz, 0 drops, 0 bad packets, |a| near 1 g during the still phase, a CSV
   download with the correct header, and no console errors. Plus a screenshot.
4. **Real kit (Robert, Windows Chrome, K1):** |a| ≈ 1.00 g flat; gyro responds; gate pill on
   shake; PING → `# pong` on serial; LED commands visible; the CSV opens.

## 8. Hosting

- GitHub Pages via Actions (`actions/upload-pages-artifact` + `actions/deploy-pages`),
  publishing `web/` so the page lives at
  `https://nthperson.github.io/BDA696_Group_Project/ble-viewer/`. One-time manual step for
  Robert: Settings → Pages → Source: GitHub Actions.
- Local: `formcoach viewer` serves `web/` with `http.server` bound to `127.0.0.1` (default
  port 8000) and opens `/ble-viewer/` in the default browser. From WSL, open the printed URL
  in Windows Chrome (WSL2 forwards localhost).

## 9. Docs

`docs/howto/ble-viewer.md` (for Bryce and the team) covers opening the page (Pages URL or
`make ble-viewer`), the browser/OS matrix including the macOS Bluetooth permission and the
Linux flag (verify), what each indicator means, the troubleshooting table, the ADR-0025
4× warning, and how to bump the vendored uPlot. Also a README link, ADR-0026 (Web
Bluetooth + vendored uPlot + Pages, no npm build) and a STATUS entry.
