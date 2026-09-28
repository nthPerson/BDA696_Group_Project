"""Source-agnostic IMU sample and the ``IMUSource`` protocol (docs/02 §4.1).

Every source (replay, serial, BLE) yields :class:`SISample` in SI units: ``t`` seconds on the
device/session clock, accel m/s², gyro rad/s, the raw ``flags`` byte and ``seq`` counter, and
``t_arrival`` (laptop ``time.monotonic()`` when the sample was received; ``None`` for replay).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from formcoach.io import protocol


@dataclass(slots=True)
class SISample:
    t: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    flags: int = 0
    seq: int = 0
    t_arrival: float | None = None

    @property
    def a(self) -> float:
        """|a| in m/s²."""
        return math.sqrt(self.ax * self.ax + self.ay * self.ay + self.az * self.az)

    @property
    def gate_state(self) -> bool:
        return bool(self.flags & protocol.Flag.GATE_STATE)

    @property
    def session_active(self) -> bool:
        return bool(self.flags & protocol.Flag.SESSION_ACTIVE)


class IMUSource(Protocol):
    """What the pipeline needs from any IMU source."""

    def iter_samples(self) -> Iterator[SISample]: ...

    def close(self) -> None: ...


def from_raw(sample: protocol.Sample, t_arrival: float | None = None) -> SISample:
    """Convert a wire-format :class:`protocol.Sample` (LSB, ms) to SI."""
    ax, ay, az, gx, gy, gz = sample.to_si()
    return SISample(
        sample.t_ms / 1000.0, ax, ay, az, gx, gy, gz, sample.flags, sample.seq, t_arrival
    )
