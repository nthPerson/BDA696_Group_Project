# Device registry

One row per assembled wearable. `SessionRecorder` reads the kit number from the BLE name and
stores it in `meta.json`. Update firmware version on every flash.

| Kit | Owner | MAC | BLE name | IMU addr | FW version | Calibrated | Known issues |
|---|---|---|---|---|---|---|---|
| K1 | Robert (bench unit) | 28:84:85:B3:8F:05 | FormCoach-8428 | (IMU not wired yet) | 1.1.0 | — | bare board verified 2026-10-08: boot, TFLM, BLE, serial commands; antenna must be clipped on |
| K2 | | | | | | | |
| K3 | | | | | | | |
| K4 | | | | | | | |
| K5 | | | | | | | |
| K6 (spare) | | | | | | | |
