"""BLE IMU source (docs/02 §4.1 ``BLESource``) via ``bleak`` (the ``device`` extra).

Notifications on the IMU characteristic carry 5-sample batches (``protocol.unpack_batch``);
:class:`BatchDecoder` turns them into SI samples with drop accounting and is unit-tested with
fake packets. :class:`BLESource` runs bleak's asyncio loop in a background thread, reconnects
with backoff, and exposes the same ``iter_samples()`` / ``send()`` as the serial source.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Iterator

from formcoach.io import protocol
from formcoach.io.serial_source import Stats
from formcoach.io.source import SISample, from_raw


class BatchDecoder:
    def __init__(self) -> None:
        self.stats = Stats()
        self.session_id: int | None = None

    def feed(self, data: bytes, t_arrival: float | None = None) -> list[SISample]:
        try:
            session_id, samples = protocol.unpack_batch(data)
        except ValueError:
            self.stats.bad_packets += 1
            return []
        self.session_id = session_id
        out = []
        for raw in samples:
            self.stats.account_seq(raw.seq)
            out.append(from_raw(raw, t_arrival))
        return out


def _bleak():
    try:
        import bleak
    except ImportError as exc:
        raise ImportError("BLE input needs bleak: `uv sync --extra device`") from exc
    return bleak


class BLESource:
    """Connect to the first ``FormCoach-XXXX`` device (or ``address``) and stream samples."""

    def __init__(
        self,
        name_prefix: str = protocol.DEVICE_NAME_PREFIX,
        address: str | None = None,
        scan_timeout: float = 10.0,
        queue_size: int = 5000,
    ):
        self.bleak = _bleak()
        self.name_prefix = name_prefix
        self.address = address
        self.scan_timeout = scan_timeout
        self.decoder = BatchDecoder()
        self.q: queue.Queue[SISample | None] = queue.Queue(maxsize=queue_size)
        self.cmd_q: queue.Queue[protocol.Command] = queue.Queue()
        self.status: protocol.Status | None = None
        self.device_name: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ble", daemon=True)
        self._thread.start()

    @property
    def stats(self) -> Stats:
        return self.decoder.stats

    def send(self, cmd: protocol.Command) -> None:
        self.cmd_q.put(cmd)

    def _run(self) -> None:
        import asyncio

        asyncio.run(self._main())

    async def _find(self):
        bleak = self.bleak
        if self.address:
            return self.address
        devices = await bleak.BleakScanner.discover(timeout=self.scan_timeout)
        for d in devices:
            if d.name and d.name.startswith(self.name_prefix):
                self.device_name = d.name
                return d.address
        raise RuntimeError(
            f"no {self.name_prefix}* device found in {self.scan_timeout:.0f} s; is the wearable on?"
        )

    async def _main(self) -> None:
        import asyncio

        bleak = self.bleak
        backoff = 1.0
        while not self._stop.is_set():
            try:
                address = await self._find()
                async with bleak.BleakClient(address) as client:
                    backoff = 1.0

                    def on_imu(_h, data: bytearray) -> None:
                        for s in self.decoder.feed(bytes(data), time.monotonic()):
                            try:
                                self.q.put_nowait(s)
                            except queue.Full:
                                self.stats.dropped += 1

                    def on_status(_h, data: bytearray) -> None:
                        with contextlib.suppress(ValueError):
                            self.status = protocol.unpack_status(bytes(data))

                    await client.start_notify(protocol.IMU_CHAR_UUID, on_imu)
                    await client.start_notify(protocol.STATUS_CHAR_UUID, on_status)
                    while client.is_connected and not self._stop.is_set():
                        try:
                            cmd = self.cmd_q.get_nowait()
                            await client.write_gatt_char(protocol.CONTROL_CHAR_UUID, bytes([cmd]))
                        except queue.Empty:
                            await asyncio.sleep(0.05)
            except Exception as exc:
                if self._stop.is_set():
                    break
                self.q.put(None)  # wake the consumer so it can print a status line
                print(f"# ble: {exc}; reconnecting in {backoff:.0f} s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def iter_samples(self) -> Iterator[SISample]:
        while not self._stop.is_set():
            try:
                s = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if s is not None:
                yield s

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
