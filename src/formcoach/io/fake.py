"""A software wearable: emits the same packets the firmware would, for tests and dry runs
(``formcoach record --source fake``). Idle noise, then a 1 Hz curl-like oscillation from
``active_from_s`` with the gate flag set — so `session check` has something to report."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from formcoach.io import protocol
from formcoach.io.ble import BatchDecoder
from formcoach.io.source import SISample


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
