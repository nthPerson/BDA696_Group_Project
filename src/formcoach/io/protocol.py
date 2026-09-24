"""BLE / serial packet layout shared with the wearable firmware.

This file mirrors ``firmware/include/protocol.h``. Change both together; ``tests/test_protocol.py``
pins the byte sizes so a drift on either side fails CI. Layout is little-endian and packed.

Sample (19 bytes)::

    t_ms   uint32   device millis at sample
    ax..az int16x3  raw accel LSB  (+-8 g   -> 4096 LSB/g)
    gx..gz int16x3  raw gyro  LSB  (+-1000 dps -> 32.8 LSB/dps)
    flags  uint8    bit0 gate_state, bit1 session_active, bit2 button_pressed, bit3 calibrated
    seq    uint16   rolling sample counter (drop detection)

Batch = header (4 bytes: session_id uint16, n uint8, reserved uint8) + n samples.
One BLE notification carries BATCH_SAMPLES samples (100 ms at 50 Hz).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntEnum

# ---- Identity -------------------------------------------------------------------------------
DEVICE_NAME_PREFIX = "FormCoach-"
SERVICE_UUID = "7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001"
IMU_CHAR_UUID = "7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001"
CONTROL_CHAR_UUID = "7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001"
STATUS_CHAR_UUID = "7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001"
REQUESTED_MTU = 185

# ---- Sampling and gate constants (keep identical to protocol.h and model/preprocess.json) ----
SAMPLE_RATE_HZ = 50
WINDOW_SAMPLES = 100  # 2 s
WINDOW_STRIDE_SAMPLES = 25  # gate runs every 0.5 s
BATCH_SAMPLES = 5  # samples per BLE notification (100 ms)
GATE_ON_WINDOWS = 2  # consecutive "active" windows to open the gate (1.0 s)
GATE_OFF_WINDOWS = 12  # consecutive "idle" windows to close it (6 s)

# ---- Unit conversion ------------------------------------------------------------------------
ACCEL_LSB_PER_G = 4096.0  # BMI160 at +-8 g
GYRO_LSB_PER_DPS = 32.8  # BMI160 at +-1000 dps
G_TO_MS2 = 9.80665
DPS_TO_RADS = math.pi / 180.0


class Command(IntEnum):
    """1-byte commands written by the laptop to the control characteristic."""

    LED_GOOD_REP = 0x01
    LED_FAULT = 0x02
    START_SESSION = 0x03
    STOP_SESSION = 0x04
    CALIBRATE = 0x05
    PING = 0x10


class Flag(IntEnum):
    """Bit positions of ``Sample.flags``."""

    GATE_STATE = 1 << 0
    SESSION_ACTIVE = 1 << 1
    BUTTON_PRESSED = 1 << 2
    CALIBRATED = 1 << 3


SAMPLE_STRUCT = struct.Struct("<IhhhhhhBH")
BATCH_HEADER_STRUCT = struct.Struct("<HBB")
SAMPLE_SIZE = SAMPLE_STRUCT.size  # 19
BATCH_HEADER_SIZE = BATCH_HEADER_STRUCT.size  # 4
MAX_BATCH_SIZE = BATCH_HEADER_SIZE + BATCH_SAMPLES * SAMPLE_SIZE  # 99


@dataclass(frozen=True, slots=True)
class Sample:
    """One raw IMU sample exactly as the wearable sends it."""

    t_ms: int
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    flags: int = 0
    seq: int = 0

    @property
    def gate_state(self) -> bool:
        return bool(self.flags & Flag.GATE_STATE)

    @property
    def session_active(self) -> bool:
        return bool(self.flags & Flag.SESSION_ACTIVE)

    def to_si(self) -> tuple[float, float, float, float, float, float]:
        """Return ``(ax, ay, az, gx, gy, gz)`` in m/s^2 and rad/s."""
        a = G_TO_MS2 / ACCEL_LSB_PER_G
        g = DPS_TO_RADS / GYRO_LSB_PER_DPS
        return (self.ax * a, self.ay * a, self.az * a, self.gx * g, self.gy * g, self.gz * g)


def pack_sample(s: Sample) -> bytes:
    return SAMPLE_STRUCT.pack(s.t_ms, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq)


def unpack_sample(buf: bytes | memoryview, offset: int = 0) -> Sample:
    return Sample(*SAMPLE_STRUCT.unpack_from(buf, offset))


def pack_batch(session_id: int, samples: list[Sample]) -> bytes:
    if not 0 < len(samples) <= BATCH_SAMPLES:
        raise ValueError(f"batch must hold 1..{BATCH_SAMPLES} samples, got {len(samples)}")
    header = BATCH_HEADER_STRUCT.pack(session_id, len(samples), 0)
    return header + b"".join(pack_sample(s) for s in samples)


def unpack_batch(buf: bytes | memoryview) -> tuple[int, list[Sample]]:
    """Return ``(session_id, samples)``; raises ``ValueError`` on a truncated packet."""
    if len(buf) < BATCH_HEADER_SIZE:
        raise ValueError(f"packet too short for header: {len(buf)} bytes")
    session_id, n, _reserved = BATCH_HEADER_STRUCT.unpack_from(buf, 0)
    expected = BATCH_HEADER_SIZE + n * SAMPLE_SIZE
    if len(buf) < expected:
        raise ValueError(f"packet declares {n} samples ({expected} bytes) but has {len(buf)}")
    samples = [unpack_sample(buf, BATCH_HEADER_SIZE + i * SAMPLE_SIZE) for i in range(n)]
    return session_id, samples


SERIAL_CSV_HEADER = "t_ms,ax,ay,az,gx,gy,gz,flags,seq"


def parse_serial_line(line: str) -> Sample | None:
    """Parse one dev-mode CSV line printed by the firmware; ``None`` for non-data lines."""
    parts = line.strip().split(",")
    if len(parts) != 9 or parts[0] == "t_ms":
        return None
    try:
        return Sample(*(int(p) for p in parts))
    except ValueError:
        return None
