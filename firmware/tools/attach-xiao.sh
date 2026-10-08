#!/usr/bin/env bash
# attach-xiao.sh — hand the XIAO ESP32-S3's USB port from Windows to WSL2, from inside WSL.
#
# Adapted from Robert's Car-Sounds satellite-recorder tool. usbipd-win >= 4.0 needs elevation
# only for the one-time `bind`; `attach` runs unelevated. On "Not shared" this script triggers
# a single Windows UAC prompt for the bind and then attaches.
#
# The XIAO is matched by VID:PID, never by name (the name embeds a COM number that changes):
#   303a:1001  Espressif USB JTAG/serial (ROM bootloader, or firmware built with USB-JTAG CDC)
#   2886:0056  Seeed XIAO ESP32S3 running the Arduino core's native USB CDC (what our firmware uses)
#
# Usage:  firmware/tools/attach-xiao.sh            -> prints "Ready: /dev/ttyACM0"
#         firmware/tools/attach-xiao.sh detach     -> give the port back to Windows
# This WSL does not inherit the Windows PATH, so Windows executables use /mnt/c/... paths.

set -euo pipefail

VIDPIDS="303a:1001|2886:0056"

USBIPD="$(command -v usbipd.exe || true)"
[ -n "$USBIPD" ] || USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
[ -x "$USBIPD" ] || {
  echo "usbipd.exe not found: install usbipd-win on Windows (winget install usbipd) and re-run." >&2
  exit 1
}
PWSH="$(command -v powershell.exe || true)"
[ -n "$PWSH" ] || PWSH="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# usbipd is a Windows exe: CRLF output — strip \r or every string compare silently fails.
line="$("$USBIPD" list | tr -d '\r' | grep -m1 -E "$VIDPIDS" || true)"
if [ -z "$line" ]; then
  echo "XIAO not visible to Windows (no $VIDPIDS device on the bus)." >&2
  echo "Use a DATA cable in the XIAO's USB-C; if the board is deep-asleep press the button;" >&2
  echo "if it never enumerates hold BOOT while plugging in (ROM bootloader)." >&2
  exit 1
fi
busid="$(awk '{print $1}' <<<"$line")"
state="$(grep -oE '(Not shared|Shared \(forced\)|Shared|Attached)$' <<<"$line" || true)"
echo "XIAO at busid $busid — state: ${state:-unknown}"

if [ "${1:-}" = "detach" ]; then
  pkill -f "attach --wsl --auto-attach --busid $busid" 2>/dev/null || true
  "$USBIPD" detach --busid "$busid" && echo "Detached $busid (back to Windows)."
  exit 0
fi

# --auto-attach keeps re-attaching after every re-enumeration (esptool resets the board when
# flashing, and RESET/BOOT presses do too); it stays running in the background until detach.
attach() {
  nohup "$USBIPD" attach --wsl --auto-attach --busid "$busid" > /tmp/usbipd-attach-"$busid".log 2>&1 &
  sleep 2
}
case "$state" in
  Attached) : ;;
  Shared|"Shared (forced)") attach ;;
  "Not shared")
    echo "One-time bind needed — approve the Windows UAC prompt..."
    "$PWSH" -NoProfile -Command \
      "Start-Process usbipd -Verb RunAs -Wait -ArgumentList 'bind','--busid','$busid'"
    attach
    ;;
  *) echo "Unrecognized state '${state:-}' — inspect with: \"$USBIPD\" list" >&2; exit 1 ;;
esac

for _ in $(seq 1 20); do
  port="$(ls /dev/ttyACM* 2>/dev/null | head -1 || true)"
  if [ -n "$port" ]; then
    echo "Ready: $port   (make fw-upload PORT=$port; make fw-monitor PORT=$port)"
    exit 0
  fi
  sleep 0.5
done
echo "Attached, but no /dev/ttyACM* appeared within 10 s. Press RESET on the XIAO, or hold BOOT" >&2
echo "while re-plugging; stale attachment: \"$USBIPD\" detach --busid $busid and re-run." >&2
exit 1
