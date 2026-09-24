"""USB-serial IMU source (docs/02 §4.1 ``SerialSource``): the firmware prints one CSV line per
50 Hz sample (``protocol.SERIAL_CSV_HEADER``) and accepts one-letter commands.

pyserial is the ``device`` extra; :class:`LineDecoder` is pure Python so the parsing and drop
accounting are unit-tested without a port. Ports: ``/dev/ttyACM0`` (Linux, add yourself to
``dialout``), ``/dev/cu.usbmodem*`` (macOS), ``COM5`` (Windows).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass

from formcoach.io import protocol
from formcoach.io.source import SISample, from_raw


@dataclass
class Stats:
    samples: int = 0
    dropped: int = 0  # missing seq numbers
    bad_lines: int = 0
    comment_lines: int = 0
    bad_packets: int = 0  # BLE only
    last_seq: int | None = None

    def account_seq(self, seq: int) -> None:
        """Count missing sequence numbers. A jump back to a small number that is not a 16-bit
        wrap (e.g. 40000 -> 0) is a device restart and is not counted as drops."""
        if self.last_seq is not None:
            restart = seq < self.last_seq and seq < 256 and self.last_seq < 65536 - 256
            gap = (seq - self.last_seq - 1) % 65536
            if not restart and 0 < gap < 32768:
                self.dropped += gap
        self.last_seq = seq
        self.samples += 1


class LineDecoder:
    """Feed serial lines; get :class:`SISample` or ``None`` (header, comment or junk)."""

    def __init__(self) -> None:
        self.stats = Stats()

    def feed(self, line: str, t_arrival: float | None = None) -> SISample | None:
        s = line.strip()
        if not s:
            return None
        if s.startswith("#") or s == protocol.SERIAL_CSV_HEADER:
            self.stats.comment_lines += 1
            return None
        raw = protocol.parse_serial_line(s)
        if raw is None:
            self.stats.bad_lines += 1
            return None
        self.stats.account_seq(raw.seq)
        return from_raw(raw, t_arrival)


def _pyserial():
    try:
        import serial
    except ImportError as exc:
        raise ImportError(
            "serial input needs pyserial: `uv sync --extra device` "
            "(then --port /dev/ttyACM0 or COM5)"
        ) from exc
    return serial


def list_ports() -> list[str]:
    """Serial ports visible to pyserial (empty list without the extra)."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    return [p.device for p in lp.comports()]


class SerialSource:
    """Iterate SI samples from the wearable's USB serial CSV stream."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        serial = _pyserial()
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.decoder = LineDecoder()

    @property
    def stats(self) -> Stats:
        return self.decoder.stats

    def send(self, cmd: protocol.Command) -> None:
        self.ser.write((protocol.SERIAL_COMMANDS[cmd] + "\n").encode("ascii"))

    def iter_samples(self) -> Iterator[SISample]:
        while True:
            line = self.ser.readline().decode("ascii", errors="replace")
            if not line:
                continue  # timeout: keep waiting (Ctrl-C to stop)
            s = self.decoder.feed(line, time.monotonic())
            if s is not None:
                yield s

    def close(self) -> None:
        # closing must never raise inside a finally block
        with contextlib.suppress(Exception):
            self.ser.close()
