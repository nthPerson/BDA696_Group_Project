# /// script
# requires-python = ">=3.11"
# dependencies = ["bleak>=0.22"]
# ///
"""ble-check.py — verify a FormCoach wearable over Bluetooth from a laptop with a BLE radio.

Runs anywhere `uv` runs (Windows, macOS, Linux with BlueZ); WSL2 has no Bluetooth, so on
Robert's machine run it from Windows:  uv run firmware\\tools\\ble-check.py  [--name FormCoach-8428]
(from WSL: /mnt/c/Users/<you>/.local/bin/uv.exe run "\\\\wsl.localhost\\...\\ble-check.py").

Checks: the device advertises the FormCoach service UUID (scan), connects, reads the status
characteristic (firmware version, session id, gate state, uptime, calibrated), subscribes to
IMU notifications for a few seconds (counts packets; 0 is expected on a bare board without an
IMU), writes PING / START_SESSION / STOP_SESSION to the control characteristic (watch the
serial monitor for `# pong` / `# session start`). Standalone on purpose: it only needs bleak.
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import sys
import time

from bleak import BleakClient, BleakScanner

SERVICE_UUID = "7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001"
IMU_CHAR_UUID = "7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001"
CONTROL_CHAR_UUID = "7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001"
STATUS_CHAR_UUID = "7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001"
CMD_LED_GOOD_REP, CMD_LED_FAULT, CMD_START, CMD_STOP, CMD_PING = 0x01, 0x02, 0x03, 0x04, 0x10
STATUS = struct.Struct("<BBBHBIBB")  # mirrors FcStatus / protocol.Status (12 bytes)
SAMPLE, HEADER = struct.Struct("<IhhhhhhBH"), struct.Struct("<HBB")


async def main(name_prefix: str, address: str | None, seconds: float) -> int:
    ok = True
    print(f"scanning 8 s for {address or name_prefix + '*'} …")
    found = None
    adv_uuids: list[str] = []
    for d, adv in (await BleakScanner.discover(timeout=8.0, return_adv=True)).values():
        if (address and d.address.lower() == address.lower()) or (
            not address and d.name and d.name.startswith(name_prefix)
        ):
            found, adv_uuids = d, [u.lower() for u in adv.service_uuids]
            print(f"  found {d.name}  address={d.address}  rssi={adv.rssi} dBm  uuids={adv_uuids}")
            break
    if found is None:
        print("FAIL: no FormCoach device found (board on? within a few metres? phone shows it?)")
        return 1
    has_uuid = SERVICE_UUID in adv_uuids
    print(f"  [{'PASS' if has_uuid else 'FAIL'}] service UUID advertised")
    ok &= has_uuid
    n_packets = 0
    n_samples = 0
    last_seq = None
    drops = 0

    def on_imu(_h, data: bytearray):
        nonlocal n_packets, n_samples, last_seq, drops
        n_packets += 1
        if len(data) < HEADER.size:
            return
        _sid, n, _r = HEADER.unpack_from(data, 0)
        for i in range(n):
            s = SAMPLE.unpack_from(data, HEADER.size + i * SAMPLE.size)
            seq = s[-1]
            if last_seq is not None:
                drops += (seq - last_seq - 1) % 65536 if 0 < (seq - last_seq - 1) % 65536 < 32768 else 0
            last_seq = seq
            n_samples += 1

    async with BleakClient(found.address, timeout=15.0) as client:
        print(f"  connected, MTU {client.mtu_size}")
        st = STATUS.unpack(bytes(await client.read_gatt_char(STATUS_CHAR_UUID)))
        print(f"  [PASS] status: fw {st[0]}.{st[1]}.{st[2]}  session={st[3]}  gate={st[4]}  "
              f"uptime={st[5]} s  calibrated={st[6]}  session_active={st[7]}")
        await client.write_gatt_char(CONTROL_CHAR_UUID, bytes([CMD_PING]), response=True)
        print("  wrote PING (serial monitor should show '# pong')")
        await client.start_notify(IMU_CHAR_UUID, on_imu)
        await client.write_gatt_char(CONTROL_CHAR_UUID, bytes([CMD_START]), response=True)
        t0 = time.monotonic()
        await asyncio.sleep(seconds)
        await client.write_gatt_char(CONTROL_CHAR_UUID, bytes([CMD_STOP]), response=True)
        await client.stop_notify(IMU_CHAR_UUID)
        dt = time.monotonic() - t0
        st2 = STATUS.unpack(bytes(await client.read_gatt_char(STATUS_CHAR_UUID)))
        print(f"  [{'PASS' if st2[3] == st[3] + 1 else 'FAIL'}] START/STOP commands: session id {st[3]} -> {st2[3]}")
        ok &= st2[3] == st[3] + 1
        rate = n_samples / dt if dt else 0.0
        print(f"  IMU notifications: {n_packets} packets, {n_samples} samples in {dt:.1f} s "
              f"({rate:.1f} Hz), {drops} dropped")
        if n_samples == 0:
            print("  [INFO] no samples: expected on a bare board (no IMU); with the IMU expect ~50 Hz")
        else:
            good = abs(rate - 50) < 3 and drops == 0
            print(f"  [{'PASS' if good else 'FAIL'}] 50 Hz with 0 drops over BLE")
            ok &= good
        await client.write_gatt_char(CONTROL_CHAR_UUID, bytes([CMD_LED_GOOD_REP]), response=True)
        print("  wrote LED_GOOD_REP (watch the LED flash)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="FormCoach", help="name prefix to match (default FormCoach)")
    ap.add_argument("--address", default=None, help="connect to this address instead of scanning by name")
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to subscribe to IMU notifications")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.name, a.address, a.seconds)))
