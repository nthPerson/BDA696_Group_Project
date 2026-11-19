"""A software wearable: emits the same packets the firmware would, for tests and dry runs
(``formcoach record --source fake``). Idle noise, then a 1 Hz curl-like oscillation from
``active_from_s`` with the gate flag set — so `session check` has something to report."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np

from formcoach.io import protocol
from formcoach.io.ble import BatchDecoder
from formcoach.io.serial_source import Stats
from formcoach.io.source import SISample

GATE_ON_DELAY_S = 1.0  # the firmware's hysteresis: 2 windows of 0.5 s to open ...
GATE_OFF_DELAY_S = 6.0  # ... and 12 windows to close


def default_schedule(t: float) -> bool:
    """True while the fake person is "exercising" at time ``t`` (s): ``t % 30`` in [5, 25), so
    5 s of rest, 20 s of work and 5 s of rest, repeating."""
    return 5.0 <= t % 30.0 < 25.0


class FakeSource:
    def __init__(
        self, duration_s: float = 10.0, fs: int = 50, active_from_s: float = 3.0, seed: int = 0
    ):
        self.duration_s, self.fs, self.active_from_s, self.seed = (
            duration_s,
            fs,
            active_from_s,
            seed,
        )
        self.decoder = BatchDecoder()

    def raw_samples(self) -> Iterator[protocol.Sample]:
        rng = np.random.default_rng(self.seed)
        n = int(self.duration_s * self.fs)
        for i in range(n):
            t = i / self.fs
            active = t >= self.active_from_s
            amp = 0.35 if active else 0.01
            ax = amp * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.005)
            az = 1.0 + (0.2 * np.cos(2 * np.pi * 1.0 * t) if active else 0) + rng.normal(0, 0.005)
            gy = (60.0 * np.cos(2 * np.pi * 1.0 * t) if active else 0) + rng.normal(0, 0.3)
            flags = (protocol.Flag.GATE_STATE if active else 0) | protocol.Flag.SESSION_ACTIVE
            yield protocol.Sample(
                t_ms=round(t * 1000),
                ax=int(ax * protocol.ACCEL_LSB_PER_G),
                ay=int(rng.normal(0, 0.005) * protocol.ACCEL_LSB_PER_G),
                az=int(az * protocol.ACCEL_LSB_PER_G),
                gx=int(rng.normal(0, 0.3) * protocol.GYRO_LSB_PER_DPS),
                gy=int(gy * protocol.GYRO_LSB_PER_DPS),
                gz=0,
                flags=int(flags),
                seq=i & 0xFFFF,
            )

    def iter_samples(self) -> Iterator[SISample]:
        """Samples go through pack_batch/unpack_batch exactly like the BLE path."""
        batch: list[protocol.Sample] = []
        for raw in self.raw_samples():
            batch.append(raw)
            if len(batch) == protocol.BATCH_SAMPLES:
                yield from self.decoder.feed(protocol.pack_batch(1, batch))
                batch = []
        if batch:
            yield from self.decoder.feed(protocol.pack_batch(1, batch))

    @property
    def stats(self):
        return self.decoder.stats

    def close(self) -> None:
        pass


class FakeWatch:
    """A software wearable paced like the real one (an ``IMUSource``): 5-sample batches at ``fs``
    Hz through ``pack_batch``/``unpack_batch``, ``t_arrival`` stamped from ``clock``.

    While ``schedule(t)`` is true (``t`` = seconds on ``clock`` since :meth:`iter_samples` began)
    the wrist swings like a curl (0.4 Hz); otherwise it rests. The gate flag follows the schedule
    with the firmware's hysteresis: set after it has been true for 1 s, cleared after it has
    been false for 6 s. The iterator is endless unless ``duration_s`` is given or :meth:`close`
    is called. Commands passed to :meth:`send` are recorded in ``sent``.
    """

    device_name = "FormCoach-FAKE"

    def __init__(
        self,
        *,
        schedule: Callable[[float], bool] = default_schedule,
        fs: int = 50,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        duration_s: float | None = None,
        seed: int = 0,
    ):
        """``fs`` is samples per second; ``duration_s`` the run length in seconds (``None``:
        endless); ``seed`` seeds the sensor noise."""
        self.schedule = schedule
        self.fs = fs
        self.clock = clock
        self.sleep = sleep
        self.duration_s = duration_s
        self.seed = seed
        self.decoder = BatchDecoder()
        self.sent: list[protocol.Command] = []
        self._born = clock()
        self._closed = False
        self._gate = False

    @property
    def stats(self) -> Stats:
        return self.decoder.stats

    @property
    def status(self) -> protocol.Status:
        """Firmware 0.0.0, calibrated; ``uptime_s`` follows the clock, ``gate_state`` the flag."""
        return protocol.Status(0, 0, 0, 1, int(self._gate), int(self.clock() - self._born), 1, 1)

    def send(self, cmd: protocol.Command) -> None:
        self.sent.append(cmd)

    def close(self) -> None:
        """End :meth:`iter_samples` at its next batch."""
        self._closed = True

    def iter_samples(self) -> Iterator[SISample]:
        rng = np.random.default_rng(self.seed)
        t0 = self.clock()
        on_since: float | None = None  # schedule true since (s)
        off_since: float | None = None  # schedule false since (s)
        active_since = 0.0
        i = 0
        while not self._closed:
            if self.duration_s is not None and i / self.fs >= self.duration_s:
                return
            wait = (i + protocol.BATCH_SAMPLES) / self.fs - (self.clock() - t0)
            if wait > 0:
                self.sleep(wait)
            if self._closed:
                return
            now = self.clock() - t0
            if self.schedule(now):
                off_since = None
                if on_since is None:
                    on_since = active_since = now
                if not self._gate and now - on_since >= GATE_ON_DELAY_S:
                    self._gate = True
            else:
                on_since = None
                if off_since is None:
                    off_since = now
                if self._gate and now - off_since >= GATE_OFF_DELAY_S:
                    self._gate = False
            moving = on_since is not None
            batch = []
            for _ in range(protocol.BATCH_SAMPLES):
                batch.append(self._sample(i, moving, i / self.fs - active_since, rng))
                i += 1
            yield from self.decoder.feed(protocol.pack_batch(1, batch), self.clock())

    def _sample(self, i: int, moving: bool, since: float, rng) -> protocol.Sample:
        """Sample ``i``: a 0.4 Hz curl-like swing ``since`` seconds after the movement began, or
        rest, plus noise."""
        w = 2 * np.pi * 0.4 * since
        ax = (0.3 * np.sin(w) if moving else 0.0) + rng.normal(0, 0.005)
        az = 1.0 + (0.45 * np.cos(w) if moving else 0.0) + rng.normal(0, 0.005)
        gy = (60.0 * np.cos(w) if moving else 0.0) + rng.normal(0, 0.3)
        flags = protocol.Flag.SESSION_ACTIVE | (protocol.Flag.GATE_STATE if self._gate else 0)
        return protocol.Sample(
            t_ms=round(i / self.fs * 1000),
            ax=int(ax * protocol.ACCEL_LSB_PER_G),
            ay=int(rng.normal(0, 0.005) * protocol.ACCEL_LSB_PER_G),
            az=int(az * protocol.ACCEL_LSB_PER_G),
            gx=int(rng.normal(0, 0.3) * protocol.GYRO_LSB_PER_DPS),
            gy=int(gy * protocol.GYRO_LSB_PER_DPS),
            gz=0,
            flags=int(flags),
            seq=i & 0xFFFF,
        )
