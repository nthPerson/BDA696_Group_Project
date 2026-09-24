# 03 — Hardware

Five kits plus one spare electronics set. Each wearable is exactly five electronic parts plus wire. Prices are from late September 2026 (Seeed from the vendor page; Amazon multipacks estimated from listings — confirm in cart).

## 1. Bill of materials (per unit; ordered as multipacks)

| Part | Spec | Pack purchased | ≈ Unit cost | Notes |
|---|---|---|---|---|
| MCU | **Seeed Studio XIAO ESP32-S3** — ESP32-S3R8, 240 MHz dual-core, 8 MB PSRAM, 8 MB flash, WiFi + BLE 5.0, USB-C, onboard LiPo charger (100 mA), 21 × 17.8 mm, deep sleep 14 µA | 2 × 3-pack ($17.91 each) | $5.97 | https://www.seeedstudio.com/Seeed-Studio-XIAO-ESP32S3-3PCS-p-5919.html · wiki: https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/ |
| IMU | **GY-BMI160** module (Bosch BMI160, 3-axis accel ±2–16 g, 3-axis gyro ±125–2000 dps, I2C/SPI, 3–5 V, ≈ 16 × 12 mm) | 2 × 5-pack | ≈ $2.80 | Alternative drop-in: GY-6500 (MPU-6500). Modules usually include an LDO + level shifter; power from 3V3. |
| Battery | **3.7 V 300 mAh LiPo, 402030** (4 × 20 × 30 mm), protected, JST-PH 2.0 lead | singles or 5-pack | ≈ $7 | Alternative: Adafruit #2750 350 mAh (36 × 30 × 5 mm) if the case grows. |
| Switch | **SS12D00** SPDT mini slide switch, 3-pin, 2 mm pitch, ≈ 8.6 × 3.6 × 5 mm | 20-pack | ≈ $0.30 | In series with battery + |
| Button | **6 × 6 × 5 mm tactile** through-hole, 4 legs | 100-pack | ≈ $0.06 | One user input |
| Wire | 30 AWG silicone, 6 colors | 1 kit for the team | — | 28–30 AWG only; thicker wire will not fit |
| Case | 3D-printed body + lid, PLA or PETG, Apple Watch band lugs | printed by Robert | ≈ $0.60 | see §5 |
| Band | Apple Watch sport band (any third-party) | teammates' own or a 4-pack | ≈ $3 | one band-width family for all cases |

Cameras: teammates' laptop webcams. Nothing else is purchased.

## 2. Wiring (9 solder joints per unit)

| From (XIAO) | To | Wire | Notes |
|---|---|---|---|
| 3V3 | BMI160 VCC | red | 3.3 V; module has its own regulator/level shifter |
| GND | BMI160 GND | black | |
| D4 (GPIO5, SDA) | BMI160 SDA | blue | I2C @ 400 kHz; module pull-ups present **(verify)** |
| D5 (GPIO6, SCL) | BMI160 SCL | yellow | leave SA0/CS per module docs → address 0x69 or 0x68; scan |
| BAT+ pad (underside) | slide switch centre pin | red | |
| slide switch outer pin | LiPo + | red | **check polarity with a meter first** — Amazon JST leads are not standardized |
| BAT− pad (underside) | LiPo − | black | |
| D1 (GPIO2) | button pin 1 | white | `INPUT_PULLUP`, pressed = LOW |
| GND | button pin 2 | black | share the IMU ground joint |

Mermaid version for the wiring guide:

```mermaid
graph LR
  XIAO[XIAO ESP32-S3]
  IMU[GY-BMI160]
  SW[SS12D00 switch]
  BAT[LiPo 300 mAh]
  BTN[6mm button]
  XIAO -- 3V3 --> IMU
  XIAO -- GND --> IMU
  XIAO -- D4/SDA --> IMU
  XIAO -- D5/SCL --> IMU
  XIAO -- BAT+ --> SW -- --> BAT
  XIAO -- BAT- --> BAT
  XIAO -- D1 --> BTN -- --> XIAO
```

Assembly order for the build session (write this up with photos in `docs/wiring-guide.md`): (1) flash test firmware to the bare XIAO over USB-C and confirm the serial monitor; (2) solder IMU wires, run the I2C scan sketch, confirm 0x68/0x69 and live readings; (3) solder button; (4) meter the battery lead, solder switch + battery to the BAT pads, insulate with heat-shrink, confirm the charge LED with USB connected and the board running on battery with USB removed; (5) Kapton over the BAT pads; (6) fit into the case; (7) label kit number (K1–K6) and record MAC address in `docs/devices.md`.

## 3. Flashing and serial

- PlatformIO: `pio run -t upload` with the board in normal mode; if upload fails hold BOOT, tap RESET, release BOOT (both are tiny buttons on the XIAO).
- Ports: Linux `/dev/ttyACM0` (add user to `dialout`, or udev rule), macOS `/dev/cu.usbmodem*`, Windows `COMx` (Device Manager). Document all three in the README.
- Serial monitor 115200 baud. Firmware prints one CSV line per sample in dev mode (`t_ms,ax,ay,az,gx,gy,gz,flags,seq`).
- BLE: Windows 10/11 and macOS work with `bleak` out of the box; Linux needs BlueZ ≥ 5.50 and the user in the `bluetooth` group. If BLE is flaky on a teammate's machine, use USB serial; do not spend more than an hour debugging a laptop's Bluetooth stack.

## 4. Battery safety (mandatory text in every hardware doc)

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

## 5. Case (Fusion 360; STEP of the Apple Watch female band connector provided by Robert in `case/`)

- Two candidate internal envelopes: **stacked** ≈ 34 × 26 × 11 mm (LiPo below, XIAO + IMU above) for the 38–42 mm band family; **single layer** ≈ 46 × 24 × 8 mm for the 44–49 mm family. Decide one band width for all five cases.
- Features: USB-C cutout ≈ 9.5 × 3.5 mm in the side wall (charge/flash without opening); slide-switch slot 1.5 × 4.5 mm with two ribs to trap the body; 3.5 mm round hole or printed keycap over the button (button on the outer face so it can be pressed mid-set); IMU pocket with a printed orientation arrow so all kits mount the sensor the same way (X along the forearm); ≥ 1.6 mm wall around the band channel; snap-fit lid (0.2 mm lip) with M2 heat-set inserts as fallback; embossed kit number.
- Keep the IMU away from the switch and button so pressing them does not flex the sensor board.
- Print orientation: lugs down or with tree supports; test the band fit with one print before printing five.
- Component measurements in the BOM sheet are nominal; caliper the real parts before finalizing.

## 6. Device registry

Maintain `docs/devices.md`: kit number, MAC, IMU address, firmware version, owner, calibration date, known issues. The `SessionRecorder` reads the kit number from the device name and stores it in `meta.json`.
