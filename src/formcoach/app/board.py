"""``BoardLink``: a non-blocking adapter between an IMU source (serial or BLE) and the live demo
loop (docs/superpowers/specs/2026-11-02-live-demo-design.md §4.5).

A daemon thread pumps ``source.iter_samples()`` into a queue; the live loop calls :meth:`poll`
once per frame to drain whatever arrived since the last call, without blocking, and reads
:meth:`alive`, :meth:`gate_open` and :meth:`state` from the most recently drained sample. The
board counts as alive only if a sample arrived within ``silence_s`` seconds (default 2 s) of
``clock()``. Never imports ``bleak`` or ``cv2``: it only needs an object with ``iter_samples()``
and, optionally, ``send()``/``close()``/``device_name`` (serial or BLE sources both qualify).
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable

from formcoach.io import protocol
from formcoach.io.source import IMUSource, SISample

SILENCE_S = 2.0  # seconds without a sample before the board is treated as silent


class BoardLink:
    """Drain an :class:`IMUSource` on a background thread without blocking the live loop."""

    def __init__(
        self,
        source: IMUSource,
        *,
        silence_s: float = SILENCE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``silence_s``: seconds without a sample before :meth:`state` reports ``"silent"``.
        ``clock``: returns seconds on whatever clock the caller's timestamps use (default
        ``time.monotonic``); injectable for tests."""
        self.source = source
        self.silence_s = silence_s
        self.clock = clock
        self.heard = False
        self._last_sample: SISample | None = None
        self._last_arrival: float | None = None
        self._q: queue.Queue[tuple[SISample, float]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, name="board-link-pump", daemon=True)
        self._thread.start()

    @property
    def name(self) -> str:
        """The board's advertised name once scanned/connected, else ``"board"``."""
        return getattr(self.source, "device_name", None) or "board"

    def _pump(self) -> None:
        try:
            for sample in self.source.iter_samples():
                if self._stop.is_set():
                    return
                arrival = sample.t_arrival or self.clock()
                self._q.put((sample, arrival))
        except Exception:
            # the source died (board off, port unplugged, ...); the live loop reads this as
            # "silent" through state()/alive(), never as an exception.
            return

    def poll(self) -> list[SISample]:
        """Drain whatever samples have arrived since the last call. Never blocks."""
        out: list[SISample] = []
        while True:
            try:
                sample, arrival = self._q.get_nowait()
            except queue.Empty:
                break
            self.heard = True
            self._last_sample = sample
            self._last_arrival = arrival
            out.append(sample)
        return out

    def alive(self) -> bool:
        """``True`` if a sample arrived within ``silence_s`` seconds of ``clock()``."""
        if self._last_arrival is None:
            return False
        return (self.clock() - self._last_arrival) <= self.silence_s

    def gate_open(self) -> bool:
        """``True`` if the board is :meth:`alive` and its latest sample's gate flag is set."""
        return bool(self.alive() and self._last_sample is not None and self._last_sample.gate_state)

    def state(self) -> str:
        """``"waiting"`` (never heard from), ``"ok"`` (alive) or ``"silent"`` (heard before, now
        quiet for more than ``silence_s`` seconds)."""
        if not self.heard:
            return "waiting"
        return "ok" if self.alive() else "silent"

    def send(self, cmd: protocol.Command) -> None:
        """Forward ``cmd`` to the source's ``send`` if it has one. Never raises."""
        send = getattr(self.source, "send", None)
        if send is None:
            return
        with contextlib.suppress(Exception):
            send(cmd)

    def close(self) -> None:
        """Stop the pump thread and close the source. Never raises or hangs."""
        self._stop.set()
        with contextlib.suppress(Exception):
            self.source.close()
        self._thread.join(timeout=1.0)
