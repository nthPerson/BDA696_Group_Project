# How to: watch a kit's live IMU stream in the browser (BLE viewer)

**Owner:** Bryce Hall (app, demo) · **Built by:** Claude Code with Robert on 2026-10-15 · **PR:** see `docs/STATUS.md` 2026-10-15
**Checkpoint:** 4 (`docs/00-START-HERE.md`) · **State:** works with the simulator and a scripted fake Bluetooth device; first real-kit check (K1) pending

A web page that connects to a FormCoach wearable over Bluetooth and shows live accel/gyro
traces, the gate/button/session flags, the kit's status, buttons for the control commands,
and a CSV recorder. Hosted copy (no clone needed):
**https://nthperson.github.io/BDA696_Group_Project/ble-viewer/**

## Battery safety

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

## 1. Run it (verified 2026-10-15 on Linux/WSL2; page checked in headless Chromium)

Open the hosted URL above in **Chrome or Edge**, or serve your clone locally:

```bash
make ble-viewer                      # Windows without make: uv run formcoach viewer
```
```
BLE viewer: http://localhost:8000/ble-viewer/  (Chrome or Edge; Ctrl+C to stop)
  WSL: open that URL in Chrome/Edge on Windows (WSL2 forwards localhost).
  Hosted copy: https://nthperson.github.io/BDA696_Group_Project/ble-viewer/
```

(`--port 8001` or `make ble-viewer VIEWER_PORT=8001` if 8000 is taken; `--no-open` skips the browser.)
Then **Simulate** (no hardware), or **Connect…** and pick `FormCoach-XXXX` in the chooser. Only kits
advertising the FormCoach service are listed; power the kit on with the antenna clipped on.
**Connect…** always opens the chooser, so you can move on to the next kit. **Reconnect** returns to the last kit
without it. Switching kits (or to Simulate) saves an open recording under the old kit's name and resets the counters.

| Browser | Works? |
|---|---|
| Chrome / Edge on Windows (incl. WSL users: use the Windows browser) | yes |
| Chrome / Edge on macOS | yes; the first time, allow the browser in System Settings → Privacy & Security → Bluetooth |
| Chrome on Android | yes (hosted URL) |
| Chrome on Linux | may need `chrome://flags/#enable-experimental-web-platform-features` (verify) |
| Safari, Firefox, anything on iOS | no Web Bluetooth: Simulate only |

**What a healthy kit looks like:** flat and still gives `|a|` ≈ 1.00 g with az ≈ ±1 g; rotating it moves the
gyro traces; shaking it lights **GATE**; about **10.0 pkt/s** and **50.0 Hz** with **drops 0**; the status row
shows `fw 1.1.0` and `calibrated yes` after a calibration.

| Button | Sends (control char.) | What you should see |
|---|---|---|
| Ping | `0x10` | serial monitor prints `# pong` |
| LED: good rep / fault | `0x01` / `0x02` | the kit's LED flashes |
| Start / Stop session | `0x03` / `0x04` | SESSION pill, `# session start id=N` on serial |
| Calibrate… | `0x05` | keep the kit flat and still 3 s; then `calibrated yes` |
| Record → Stop & download CSV | — | `<kit>_YYYYMMDD-HHMMSS.csv`, header `t_ms,ax,ay,az,gx,gy,gz,flags,seq` |

The CSV holds raw LSB (accel ÷ 4096 = g, gyro ÷ 32.8 = dps), the same format as the serial CSV. It is a
debugging aid, not a team recording: data collection still goes through `formcoach record` (`data/team/`,
MANIFEST), and **a recording of a teammate is never committed**.

**Troubleshooting**

| Symptom | Cause / fix |
|---|---|
| Kit not in the chooser | another app holds it (one client at a time: close `ble-check.py`, `formcoach demo/record`, other tabs); antenna; LED not blinking = not advertising |
| `|a|` ≈ 4 g at rest, gyro flat at 0 | firmware from before ADR-0025 (BMI160 range bug): reflash from `main` once `fw/imu-bringup` is merged |
| "Bad packet … MTU" message | the link negotiated a small MTU; a batch needs ≥ 102 bytes. Disconnect and reconnect |
| Disconnected | press **Reconnect** (no chooser); an unfinished recording is kept, press Stop to save it. A kit that rebooted meanwhile starts a fresh plot |
| Red banner "no Web Bluetooth" | wrong browser (table above), or the page is not on https/localhost |
| Blank page from `make ble-viewer` | hard refresh (Ctrl+Shift+R); check the browser console |

## 2. Where things live

| Path | What it is |
|---|---|
| `web/ble-viewer/index.html` | page layout and CSS (light/dark follow the OS) |
| `web/ble-viewer/protocol.js` | UUIDs, sizes, scales, commands, flags; `decodeBatch`, `encodeBatch`, `decodeStatus`, `encodeStatus` |
| `web/ble-viewer/buffer.js` | `SampleRing` (last 10 s, drops via `seq`, reboot detection), `RateMeter` |
| `web/ble-viewer/ble.js` | `BleLink`: Web Bluetooth connect/reconnect, notifications, queued command writes |
| `web/ble-viewer/sim.js` | `Simulator`: synthetic curls, packed as real 99-byte batches |
| `web/ble-viewer/recorder.js`, `charts.js`, `app.js` | CSV recorder; two uPlot charts; DOM wiring |
| `web/ble-viewer/vendor/` | uPlot 1.6.32 (MIT) and how to bump it (`vendor/README.md`) |
| `src/formcoach/app/viewer.py`, `formcoach viewer` | local server (127.0.0.1, `no-store`, explicit JS MIME types) |
| `.github/workflows/pages.yml` | deploys `web/` to GitHub Pages on merge to `main` |

## 3. How it works

Plain ES modules, no build step and no npm install (ADR-0026). A source (`BleLink` or `Simulator`) hands
raw `DataView`s to `decodeBatch()`. Samples go into a 500-sample ring, and a `requestAnimationFrame` loop
redraws, so BLE never waits on rendering. The wire format is `docs/02-system-design.md` §3 /
`firmware/include/protocol.h`: little-endian, 4-byte header, 19-byte samples, 5 per notification at 50 Hz.

## 4. Tests

- `uv run pytest tests/test_web_protocol.py` (CI) checks every constant in `protocol.js`, plus the golden
  batch/status hex, against `src/formcoach/io/protocol.py`. Change `protocol.h`, `protocol.py` and
  `protocol.js` together; regenerate the golden hex with `protocol.pack_batch(7, samples).hex()`.
- `uv run pytest tests/test_viewer_server.py` (CI) checks the server, the MIME types and the CLI.
- `node --test web/ble-viewer/test/*.test.mjs` (local only; `make test` runs it when Node ≥ 18 is installed,
  21 tests) covers the decoder, the seq wrap, reconnect and reboot handling, the CSV and the simulator.

## 5. Could not verify / open questions

- Not yet run against a real kit from Windows Chrome (no Bluetooth in WSL2). Robert's K1 check is
  pending: |a|, gyro, gate, Ping → `# pong`, LED, CSV.
- Linux Chrome flag (verify). The MTU Chrome negotiates on macOS/Windows is expected to be large;
  the page reports truncated packets if not.
- Windows reaching the WSL server over `localhost` was verified (`curl.exe` → 200, `text/javascript`).

## 6. Next steps for Bryce

1. Open the hosted URL, press **Simulate**, and read §1 until every indicator makes sense.
2. With Robert, connect to K1 and walk through "What a healthy kit looks like". This is the demo script.
3. On assembly day, after each kit's IMU test (`docs/assembly-day.md` §3 step 3), open it here as a visual check before it is registered in `docs/devices.md`.
