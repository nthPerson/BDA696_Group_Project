"""``BoardLink`` (docs/superpowers/specs/2026-11-02-live-demo-design.md §4.5): a non-blocking
adapter between an IMU source (serial or BLE) and the live demo loop. An injected clock and
plain generator-based fake sources stand in for real hardware."""

from __future__ import annotations

import time

from formcoach.app.board import SILENCE_S, BoardLink
from formcoach.io import protocol
from formcoach.io.source import SISample


def _sample(flags: int = 0, seq: int = 0) -> SISample:
    """A minimal sample; ``t_arrival=None`` so ``BoardLink`` stamps it with the injected clock."""
    return SISample(t=0.0, ax=0.0, ay=0.0, az=9.8, gx=0.0, gy=0.0, gz=0.0, flags=flags, seq=seq)


class _Source:
    """A fake ``IMUSource``: ``iter_samples`` is a plain generator over a fixed list."""

    def __init__(self, samples, device_name: str | None = None) -> None:
        self._samples = list(samples)
        self.device_name = device_name
        self.sent: list[protocol.Command] = []
        self.closed = False

    def iter_samples(self):
        yield from self._samples

    def send(self, cmd: protocol.Command) -> None:
        self.sent.append(cmd)

    def close(self) -> None:
        self.closed = True


class _RaisingSendSource(_Source):
    def send(self, cmd: protocol.Command) -> None:
        raise RuntimeError("send boom")


class _NoSendSource:
    """A source with no ``send`` at all (e.g. a read-only replay source)."""

    def __init__(self, samples) -> None:
        self._samples = list(samples)

    def iter_samples(self):
        yield from self._samples

    def close(self) -> None:
        pass


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.005) -> bool:
    """Poll ``predicate`` with a bounded deadline instead of a fixed sleep (the pump runs on its
    own daemon thread, so the test must wait for it without guessing how long that takes)."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return predicate()
        time.sleep(interval)


def _drain_until_heard(link: BoardLink, timeout: float = 1.0) -> None:
    assert _wait_until(lambda: link.poll() or link.heard, timeout=timeout), "pump never delivered"


def test_waiting_then_ok_then_silent():
    clock = [0.0]
    link = BoardLink(_Source([_sample()]), silence_s=2.0, clock=lambda: clock[0])
    try:
        assert link.state() == "waiting"
        assert link.heard is False

        _drain_until_heard(link)
        assert link.state() == "ok"
        assert link.heard is True

        clock[0] = 2.5
        assert link.state() == "silent"
    finally:
        link.close()


def test_gate_open_requires_flag_and_alive():
    clock = [0.0]
    gated = _Source([_sample(flags=protocol.Flag.GATE_STATE)])
    link = BoardLink(gated, silence_s=2.0, clock=lambda: clock[0])
    try:
        assert link.gate_open() is False  # nothing heard yet -> not alive

        _drain_until_heard(link)
        assert link.alive() is True
        assert link.gate_open() is True  # alive and the flag bit is set

        clock[0] = 2.5
        assert link.alive() is False
        assert link.gate_open() is False  # no longer alive, even though the flag was set
    finally:
        link.close()

    ungated = _Source([_sample(flags=0)])
    link2 = BoardLink(ungated, silence_s=2.0, clock=lambda: 0.0)
    try:
        _drain_until_heard(link2)
        assert link2.alive() is True
        assert link2.gate_open() is False  # alive, but the gate flag is clear
    finally:
        link2.close()


def test_send_forwards_and_swallows_errors():
    ok_src = _Source([])
    link = BoardLink(ok_src, clock=lambda: 0.0)
    link.send(protocol.Command.LED_GOOD_REP)
    assert ok_src.sent == [protocol.Command.LED_GOOD_REP]
    link.close()

    raising_src = _RaisingSendSource([])
    link2 = BoardLink(raising_src, clock=lambda: 0.0)
    link2.send(protocol.Command.LED_FAULT)  # must not raise
    link2.close()

    no_send_src = _NoSendSource([])
    link3 = BoardLink(no_send_src, clock=lambda: 0.0)
    link3.send(protocol.Command.PING)  # must not raise
    link3.close()


def test_name_falls_back_to_board():
    link = BoardLink(_NoSendSource([]), clock=lambda: 0.0)
    assert link.name == "board"
    link.close()

    named_src = _Source([], device_name=None)
    link2 = BoardLink(named_src, clock=lambda: 0.0)
    assert link2.name == "board"
    named_src.device_name = "FormCoach-8428"  # e.g. BLESource fills this in once scanned
    assert link2.name == "FormCoach-8428"
    link2.close()


def test_close_never_raises_or_hangs():
    link = BoardLink(_RaisingSendSource([_sample()]), clock=lambda: 0.0)

    class _Boom:
        def close(self) -> None:
            raise RuntimeError("close boom")

    link.source.close = _Boom().close
    t0 = time.monotonic()
    link.close()  # must not raise despite source.close() raising
    assert time.monotonic() - t0 < 2.0


def test_default_silence_s_matches_module_constant():
    assert SILENCE_S == 2.0
    link = BoardLink(_Source([]), clock=lambda: 0.0)
    assert link.silence_s == SILENCE_S
    link.close()
