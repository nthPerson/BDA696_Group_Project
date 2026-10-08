# firmware/

PlatformIO project for the FormCoach wearable (Seeed XIAO ESP32-S3, Arduino framework).

| Command (repo root) | Direct | What |
|---|---|---|
| `make fw-build` | `pio run -d firmware` | compile (also runs in CI) |
| `make fw-upload` | `pio run -d firmware -t upload` | flash over USB-C |
| `make fw-monitor` | `pio device monitor -d firmware` | serial monitor, 115200 baud |

Without a global PlatformIO install, `uvx platformio run -d firmware` works from the repo root
(that is what the Makefile does). Flashing and serial-port notes per OS are in the root README;
wiring, pin map and the **mandatory LiPo safety rules** are in `docs/03-hardware.md`.

`include/protocol.h` is the packet contract shared with `src/formcoach/io/protocol.py`.
`model/` will hold `gate_model.tflite`, `gate_model_data.cc` and `preprocess.json` (Checkpoint 5).

## WSL2: handing the USB port to Linux

Windows holds the XIAO's USB port until usbipd-win passes it through. Once per port, in an
**admin** PowerShell: `usbipd bind --busid <id>` (find the id with `usbipd list`; the XIAO is
`303a:1001`). After that, from WSL: `make attach-xiao` (or `firmware/tools/attach-xiao.sh`)
attaches with auto-reattach so flashing resets do not lose the port; `firmware/tools/attach-xiao.sh
detach` returns it to Windows. Then `make device-check PORT=/dev/ttyACM0 KIT=K1`.
