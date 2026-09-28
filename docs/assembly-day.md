# Assembly day: build, flash and test the five FormCoach wearables

One page for the build session. Print it or keep it open on one laptop per station. Firmware
v1.1.0 (`firmware/`) is compiled in CI but **has never run on a board**; step 3 is where the
**(verify)** items in `docs/DECISIONS.md` (ADR-0020) get settled, so keep a notes file open.
Wiring: `docs/03-hardware.md` §2 (this page repeats it). Bring-up details and the WSL2 USB
path: `.claude/skills/firmware-bringup/SKILL.md`.

## Battery safety (mandatory text in every hardware doc)

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

Batteries are the **last** thing soldered (step 5), after the IMU and button are proven.

## 0. Before the day (each teammate, at home)

1. Clone, `make setup`, `make demo` — must print `reps 10 (pose) … expected 10`.
2. `make fw-build` once (downloads ~1 GB of toolchain; 4 min the first time). On Windows use
   PowerShell: `uvx platformio run -d firmware -e xiao_esp32s3`.
3. `uv sync --extra device` (pyserial + bleak). Linux: `sudo usermod -aG dialout $USER`, re-login.
4. Install **nRF Connect** (phone) to see the BLE name. WSL2 has no Bluetooth: BLE tests run from
   Windows Python or the phone; USB serial works everywhere.
5. Bring: a USB-C **data** cable per station (charge-only cables are the #1 "no port" cause).

Bench kit (Robert): soldering iron + fine tip, leaded or SAC solder, flux, 30 AWG silicone wire
(red, black, blue, yellow, white), wire stripper, tweezers, Kapton tape, 2 mm heat-shrink, a
multimeter, helping hands, a labelled tray per kit (K1–K6), the printed cases if v1 exists.

## 1. Stations and roles

| Station | Kits | People | Does |
|---|---|---|---|
| A: solder | K1–K6 | Robert + 1 | steps 2, 4, 5 |
| B: flash + test | whatever A finishes | 2 laptops (one Windows for BLE) | steps 1, 3, 6, 7 |
| C: scribe | — | 1 | fills `docs/devices.md`, the notes file, photos for the wiring guide |

Budget ~35 min per kit end to end with the stations pipelined; K6 is the spare, build it last.

## 2. Wiring (9 joints per unit; 30 AWG only, thicker wire will not fit)

Pin names are the XIAO ESP32-S3 silkscreen labels; GPIO numbers are what the firmware uses
(`firmware/src/app/main.cpp`). The GY-BMI160 header is usually `VIN 3V3 GND SCL SDA CS SAO`.

**IMU (4 wires, solder first)**

| From (XIAO pin) | GPIO | To (GY-BMI160 pin) | Wire | Notes |
|---|---|---|---|---|
| 3V3 | — | 3V3 (or VIN if the module only has VIN) | red | 3.3 V logic; do not use 5V |
| GND | — | GND | black | one ground joint shared with the button |
| D4 | GPIO5 | SDA | blue | I2C, 400 kHz, `Wire` default pins |
| D5 | GPIO6 | SCL | yellow | |
| — | — | CS, SAO | — | leave unconnected; module pulls SAO high → address **0x69** (0x68 if it reads low) |

**Button (2 wires)**

| From | GPIO | To | Wire | Notes |
|---|---|---|---|---|
| D1 | GPIO2 | button leg 1 | white | firmware uses `INPUT_PULLUP`; pressed = LOW |
| GND | — | button leg 2 (diagonal leg from leg 1) | black | share the IMU ground |

**Power (3 wires, after step 4 passes)**

| From | To | Wire | Notes |
|---|---|---|---|
| XIAO **BAT+** pad (underside) | slide switch **centre** pin | red | |
| slide switch **outer** pin (either) | LiPo **+** (red lead) | red | **meter the JST lead first**: red is not always + |
| XIAO **BAT−** pad (underside) | LiPo **−** (black lead) | black | |

Mount the IMU with its **X axis along the forearm** and the same way up in every kit (the case
pocket has an arrow); a rotated IMU still works but the team recordings will not be comparable.

## 3. Per-kit procedure

Tick each box in the notes file with the kit number. Expected serial lines are what
`firmware/src/app/main.cpp` prints; anything else is a finding.

**Step 1 — bare board flash (station B, no soldering yet)**

```
uvx platformio device list                        # find the port: /dev/ttyACM0, /dev/cu.usbmodem*, COMx
make fw-upload PORT=COM5                          # or: uvx platformio run -d firmware -e xiao_esp32s3 -t upload --upload-port COM5
make fw-monitor PORT=COM5                         # 115200 baud; press RESET after opening the monitor
```
Expected (bare board):
```
# FormCoach fw 1.1.0  chip=ESP32-S3 rev=<n>
# sample=19 B batch=99 B status=12 B rate=50 Hz
# flash app=<bytes>
# bmi160 NOT FOUND at 0x00
# calibration none (long-press to calibrate)
# gate: tflm ok arena=<bytes> in=600 B classes=6      <- or "# gate: motion-energy rule" (see §5)
# gate: int8 CNN (TFLite Micro)
# ble advertising as FormCoach-XXXX
t_ms,ax,ay,az,gx,gy,gz,flags,seq
```
- [ ] port found · [ ] banner as above · [ ] LED blinks ~1 Hz (slow blink = advertising) ·
  [ ] LED polarity: note whether the LED is **on** or **off** between blinks (ADR-0020 verify) ·
  [ ] phone sees `FormCoach-XXXX` in nRF Connect; write XXXX in `docs/devices.md`
- No port → hold **BOOT**, tap **RESET**, release **BOOT**, retry; then try another cable.

**Step 2 — solder the IMU (station A)**, power off. Four wires per the table, ~30 mm long.

**Step 3 — IMU test (station B)**: plug in, open the monitor, press RESET.
```
# bmi160 ok at 0x69
…
12345,-120,35,4090,3,-2,1,0,0        <- one CSV line every 20 ms; flat on the desk: one accel axis ≈ ±4096, gyro within ±50
# infer us=<n> arena=<bytes> class=0 p_active=0.03   <- every 0.5 s while the CNN gate runs
```
- [ ] address (0x68/0x69) · [ ] 50 lines/s (`session check` reports the rate in step 6) ·
  [ ] gravity on the expected axis when flat, then on X when the board is held with X down ·
  [ ] gyro near zero when still · [ ] `# infer us=` value (write it down: this is the on-device
  inference time) · [ ] no `# gate: AllocateTensors failed` message
- `NOT FOUND` → swap SDA/SCL first (most common), then meter 3.3 V at the module, then try
  another module. Values frozen or all zero → cold joint on SDA/SCL.

**Step 4 — solder and test the button (A then B)**: short press →
`# session start id=1`, LED solid; second short press → `# session stop id=1`. Hold 2 s →
`# calibration: keep the device flat and still for 3 s` then `# calibrated g=(…) bias=(…)`;
after RESET the banner says `# calibration loaded from NVS`.
- [ ] short press toggles session · [ ] long press calibrates · [ ] flags column shows bit1
  (value 2) while a session is active (`flags` = 3 when the gate is also open)

**Step 5 — battery (A, last)**: meter the JST lead (+ is the lead that reads +3.7…4.2 V to the
other), solder the switch and cell per the power table, heat-shrink the joints, Kapton over the
BAT pads. Test: USB out, switch ON → LED blinks and the phone still sees the BLE name; switch
OFF → dead. USB in with the switch ON → the XIAO's charge LED lights (note its colour and
whether it turns off when full — **(verify)**). Never leave it charging unattended.
- [ ] polarity metered · [ ] runs on battery · [ ] switch cuts power · [ ] charges over USB

**Step 6 — laptop record + check (B)** with the board on USB:
```
uv run formcoach record --source serial --port COM5 --subject S1 --exercise curl --duration 30 --notes "K1 bench test"
uv run formcoach session check data/team/S1/<session_id>
```
Expected: `samples ≈ 1500   duration 30.0 s   rate 50.0 Hz`, `dropped 0`, and `gate open`
rising when you shake the board for the last 10 s. Then from Windows or macOS:
```
uv run formcoach record --source ble --subject S1 --exercise curl --duration 30
```
- [ ] serial: 0 drops at 50 Hz · [ ] BLE: connects, 0–2 drops per 30 s, `session check` clean ·
  [ ] `flags` bit0 flips when shaken (device gate) — write the observed lag
- Serial drops with BLE connected → note it (serial + BLE at 50 Hz is an open question).

**Step 7 — register and archive (C)**: row in `docs/devices.md` (kit, owner, MAC from nRF
Connect, BLE name, IMU addr, `1.1.0`, calibration date, issues); save the boot log as
`reports/logs/<kit>_boot.log` (copy the monitor output) — `formcoach eval device --log` turns
it into `reports/device.md`; take photos of the wiring for `docs/wiring-guide.md`.

## 4. Case fit (only if cases are printed)

Dry-fit before the battery is soldered: USB-C reachable, switch slot, button under its hole,
IMU pocket arrow along the forearm, nothing pressing on the IMU. Lid last. If v1 cases are not
ready, kits go home in their trays with the IMU wires taped down; the case is not needed for the
recording protocol on a wrist strap.

## 5. Troubleshooting

| Symptom | Try |
|---|---|
| no serial port | data cable; BOOT+RESET; on WSL2 attach the device with usbipd (skill file) |
| garbage in the monitor | 115200 baud; press RESET after opening |
| `# gate: motion-energy rule` instead of TFLM | model failed to load or allocate: note the line before it; the kit still works with the energy gate (`xiao_esp32s3_energy` env is the same firmware without TFLM) |
| `# bmi160 NOT FOUND` | swap SDA/SCL; 3.3 V at the module; SAO floating → 0x69 |
| phone does not see the BLE name | monitor shows `# ble advertising`? If yes, phone BT off/on; if no, the boot stalled before BLE: note the last line |
| `session check` shows drops on serial | different cable/port; close other serial monitors; try BLE |
| LED never changes | note polarity; `LED_ACTIVE_LOW` in `main.cpp` is the one-line fix |

## 6. End of day

- [ ] `docs/devices.md` has one row per built kit · [ ] notes file → DECISIONS entries for
  every **(verify)** settled (I2C address, LED polarity, charge LED, BLE MTU/notify rate, serial +
  BLE drops, TFLM arena and inference time) · [ ] boot logs under `reports/logs/` ·
  [ ] `docs/STATUS.md` entry · [ ] every kit's owner takes it home **switched off**
