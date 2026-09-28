# How to: build, flash and talk to the wearable firmware v1

**Owner:** Robert Ashe (firmware) · Christian Byars (device side of the gate) · **Built by:** Claude Code with Robert on 2026-09-24 · **PR:** #6
**Checkpoint:** 4 (`docs/00-START-HERE.md`) · **State:** partial — compiles in CI and streams through fakes; **never run on a board**

## Battery safety

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

Bring-up uses USB power only; no battery until the IMU is confirmed (`docs/03-hardware.md` §2).

## 1. Run it (verified 2026-09-24 on Linux/WSL2, no hardware)

```bash
make fw-build                      # = uvx platformio run -d firmware -e xiao_esp32s3
```
```
RAM:   [=         ]   9.6% (used 31356 bytes from 327680 bytes)
Flash: [==        ]  16.6% (used 554753 bytes from 3342336 bytes)
========================= [SUCCESS] Took 42.33 seconds =========================
```

A dry run of the recording path with the software wearable (no board, no extras):

```bash
uv run formcoach record --source fake --subject S1 --exercise curl --duration 10 --root runs
uv run formcoach session check runs/S1/<session_id>
```
```
recording curl for S1 from fake -> runs/S1/20260924-165001
  250 samples  t=5.0s  gate=on
  500 samples  t=10.0s  gate=on
wrote runs/S1/20260924-165001/imu.parquet meta.json (500 samples)
subject  S1   exercise curl
samples  500   duration 10.0 s   rate 50.1 Hz
dropped  0   gate open 70.0%
problems none
```

With a board (Windows PowerShell; WSL2 needs usbipd-win, see `.claude/skills/firmware-bringup`):

```powershell
uvx platformio device list                                  # find COMx (VID:PID 2886:0056)
make fw-upload PORT=COM5      # = uvx platformio run -d firmware -e xiao_esp32s3 -t upload --upload-port COM5
make fw-monitor PORT=COM5     # = uvx platformio device monitor -d firmware -b 115200 -p COM5
uv run formcoach record --source serial --port COM5 --subject S1 --exercise curl   # needs: uv sync --extra device
```

## 2. Where things live

| Path | What it is |
|---|---|
| `firmware/platformio.ini` | env `xiao_esp32s3` (the app) + `libtest_*` envs (library compile tests, ADR-0020/0021) |
| `firmware/src/app/main.cpp` | setup/loop: 100 Hz IMU reads → 50 Hz samples, gate, BLE/serial output, button, LED, NVS calibration, deep sleep |
| `firmware/src/app/bmi160.{h,cpp}` | `Bmi160::begin(wire, addr)`, `read(axyz, gxyz)`, `scan(wire)` — register driver, ±8 g / ±1000 dps |
| `firmware/src/app/gate.{h,cpp}` | `Gate::push(axyz, gxyz)` energy rule over the 2 s window + hysteresis (2 on / 12 off) |
| `firmware/src/app/ble.{h,cpp}` | `BleLink::begin(name, handler)`, `queue(sample, session_id)`, `publishStatus(st)` |
| `firmware/src/app/ui.{h,cpp}` | `Button::poll()` (1 = short, 2 = long 2 s), `StatusLed` patterns |
| `firmware/include/protocol.h` ↔ `src/formcoach/io/protocol.py` | packet, status (12 B), commands, serial letters, energy threshold; `tests/test_protocol.py` pins both |
| `src/formcoach/io/serial_source.py`, `io/ble.py`, `io/fake.py` | `SerialSource(port)`, `BLESource(address=None)`, `FakeSource(duration_s)`; all `iter_samples()` / `send(Command)` |
| `src/formcoach/io/recorder.py`, `io/session_check.py` | `SessionRecorder(subject, exercise, root)`, `check(session_dir)` |
| `tests/test_io_sources.py`, `tests/test_protocol.py` | decoders on fake packets, recorder schema, session check, record CLI |

## 3. How it works

The firmware reads the BMI160 at 100 Hz, averages pairs to 50 Hz, pushes each sample into a
100-sample ring; every 25 samples it computes var(|a|) over the ring and applies the same
hysteresis constants as the laptop (`protocol.h`). Samples go out as 19-byte packets in
5-sample BLE notifications and as CSV lines on USB serial (`t_ms,ax,…,flags,seq`, raw LSB).
Flags bit0 = gate, bit1 = session, bit2 = button, bit3 = calibrated. The laptop converts LSB →
m/s² / rad/s once (`protocol.Sample.to_si`) so serial, BLE and replay produce identical
`SISample`s. Short press toggles the session (session_id++); long press stores gravity + gyro
bias in NVS. Serial letters `S X C G F P` mirror the BLE commands.

## 4. Tests

`uv run pytest tests/test_protocol.py tests/test_io_sources.py` (16 tests) and
`make fw-build`. `make fw-libtest` builds every candidate-library env (only the Chirale TFLM
env passes; the two BMI160 libraries and TensorFlow's Arduino library do not build for this
target, which is why the driver is in-tree). Change `protocol.h` and `protocol.py` together.

## 5. Could not verify / open questions (hardware day)

- BMI160 register sequence, I2C address, gyro start-up delay, LED polarity, button on D1:
  all from datasheet/variant files, none exercised. Watch the boot banner (`# bmi160 ok at 0x69`).
- NimBLE 2.x API calls compile; MTU 185, notify rate at 50 Hz and reconnect behaviour are untested.
  WSL2 has no Bluetooth: run `record --source ble` from Windows Python.
- Serial CSV at 50 Hz alongside BLE may cost loop time; measure drops with `session check`.
- Camera capture during `record --camera N` is not wired (prints a notice).
- `esp_sleep_enable_ext0_wakeup` on GPIO2 (an RTC pin on the S3) is assumed to wake the board.

## 6. Next steps for Robert / Christian

1. Robert: flash one bare board, follow `.claude/skills/firmware-bringup` step 4–7, add the
   kit to `docs/devices.md`, settle the **(verify)** items above in DECISIONS.
2. Christian: PR 6 replaces `Gate::decide()` with TFLM inference on the exported int8 model
   (`firmware/model/`), using the Chirale library env that already compiles.
