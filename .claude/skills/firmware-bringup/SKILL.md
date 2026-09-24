---
name: firmware-bringup
description: Use when flashing, monitoring or bringing up the FormCoach wearable (Seeed XIAO ESP32-S3 + GY-BMI160) — first power-up of a board, `make fw-upload` / `make fw-monitor`, "port not found", COMx vs /dev/ttyACM0, WSL2 USB, BOOT/RESET, I2C scan, BLE name check, registering a kit in docs/devices.md, or writing any hardware doc or howto.
---

# Firmware bring-up: XIAO ESP32-S3 + BMI160

Facts below were verified on 2026-09-24 against `~/.platformio` (espressif32 6.13.0, arduino
core 2.0.17): board JSON sets `ARDUINO_USB_CDC_ON_BOOT=1` and `ARDUINO_USB_MODE=1` (native
USB serial, port re-enumerates on reset); variant pins `LED_BUILTIN=21`, `D1=2`, `SDA=D4=5`,
`SCL=D5=6`. LED polarity is still **(verify)** by eye. Firmware: `firmware/src/main.cpp`,
version in `firmware/platformio.ini` (`FORMCOACH_FW_VERSION`).

## Battery safety (paste verbatim into every hardware doc, CLAUDE.md rule 9)

- Protected cells only; verify polarity with a meter before soldering.
- Charge only through the XIAO's USB-C with the slide switch ON, while attended, on a non-flammable surface. Never charge a swollen, punctured or hot cell.
- Insulate the cell from the XIAO's underside pads (Kapton/foam tape); no screws or sharp edges touching the cell inside the case.
- If a cell is damaged, put it in a metal container away from flammables and dispose of it at a battery drop-off; do not throw it away.

Bring-up of a bare board uses USB power only; no battery is connected until step 4 of the
assembly order in `docs/03-hardware.md` §2.

## Where the board is: two working paths

| Path | When | Commands |
|---|---|---|
| **A. Windows PowerShell** | Robert's Windows host, teammates on Windows | `uvx platformio device list` → `uvx platformio run -d firmware -t upload --upload-port COM5` → `uvx platformio device monitor -d firmware -p COM5` |
| **B. WSL2 via usbipd-win** | Claude Code session in WSL wants `make fw-*` | admin PowerShell once: `winget install usbipd-win`; `usbipd list`; `usbipd bind --busid <id>` (once). Each plug-in: `usbipd attach --wsl --busid <id>`. Then in WSL: `ls /dev/ttyACM*`, `make fw-upload PORT=/dev/ttyACM0` after adding `PORT` (see Makefile note), `make fw-monitor`. Needs `sudo usermod -aG dialout $USER` + re-login once. `usbipd detach --busid <id>` when done. |

Linux: `/dev/ttyACM0` + `dialout`. macOS: `/dev/cu.usbmodem*`. WSL2 has **no Bluetooth**; any
BLE step runs on Windows Python or a phone. Path A works from a Windows-side clone (OneDrive
path or `wslpath -w "$(pwd)"`); building on a `\\wsl.localhost\...` UNC path is not
recommended. The Makefile's `fw-upload` has no port argument yet; pass `--upload-port` directly
or add `PORT` to the target in the same PR.

## Procedure and what to see

1. **Build first**, no board: `make fw-build` (or `uvx platformio run -d firmware`). First run
   downloads ~1 GB of toolchain; CI already proves it compiles.
2. **Plug in** with a data cable, no hub. Port appears (Device Manager "Ports", or
   `uvx platformio device list`; VID:PID `2886:0056`). No port → charge-only cable, or hold
   **BOOT**, tap **RESET**, release **BOOT** to force the ROM bootloader, then retry.
3. **Upload**. Success ends with `Hard resetting via RTS pin...`. Failure "Failed to connect" →
   step 2's BOOT/RESET, then press **RESET** once after the upload so the app runs.
4. **Monitor** at 115200 (`monitor_filters = time, default`). Expected, in order:
   ```
   # FormCoach fw 0.1.0  chip=ESP32-S3 rev=<n>  psram=8388608
   # sample=19 B batch=99 B rate=50 Hz
   # i2c scan
   #   no BMI160 at 0x68/0x69 (check SDA=D4, SCL=D5, 3V3, GND)      <- bare board
   # ble advertising as FormCoach-XXXX
   t_ms,ax,ay,az,gx,gy,gz,flags,seq
   ```
   plus `# button pressed at <ms>` when D1 is shorted to GND, and a ~1 Hz LED blink. The
   banner is printed once at boot; the monitor waits up to 3 s for USB, so open it, then press
   RESET. Nothing at all → wrong port or the board is still in the bootloader (press RESET).
   `psram=0` → wrong board in `platformio.ini`.
5. **BLE check**: nRF Connect (phone) shows `FormCoach-XXXX` advertising and its MAC. XXXX is
   the low 16 bits of the efuse MAC, so it differs per board and is the kit's BLE name.
6. **IMU**: power off, wire 3V3→VCC, GND→GND, D4→SDA, D5→SCL (`docs/03-hardware.md` §2), power
   on, monitor: `#   device at 0x69  <- BMI160?` (or 0x68). Still "no BMI160" → swap SDA/SCL
   first (most common), then meter 3V3 at the module, then try another module.
7. **Record**: a row in `docs/devices.md` (kit, owner, MAC, BLE name, IMU addr, FW version);
   a STATUS entry; a DECISIONS paragraph for every **(verify)** that was settled (LED
   polarity, I2C address, module pull-ups, which BMI160 library compiles).

## Rules

- Never instruct anyone to charge unattended, bypass the protection circuit, or solder a
  battery before the meter check. Battery steps come after the IMU is confirmed.
- Do not change the pin map (`docs/02-system-design.md` §2.3) or the BLE protocol without
  asking Robert (CLAUDE.md rule 7).
- A hardware doc or howto is under 120 lines and contains the battery block above verbatim.
- Do not spend more than an hour on a laptop's Bluetooth stack; fall back to USB serial
  (`docs/03-hardware.md` §3).
