"""``BoardLink``'s default clock (src/formcoach/app/board.py): tests/test_board.py always
injects ``clock=`` explicitly. This file covers the path where it is left at its default.

Important finding, reported in full in the task's response rather than worked around: the
``__init__`` signature is::

    def __init__(self, source, *, silence_s: float = SILENCE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:

``clock``'s default is a plain default-argument *expression*, evaluated exactly once when
``board.py`` is first imported (i.e. it captures the ``time.monotonic`` function object that
existed at that moment), not a lookup performed inside ``__init__`` at construction time. That
means ``monkeypatch.setattr(time, "monotonic", fake)`` or the equivalent
``monkeypatch.setattr(board.time, "monotonic", fake)`` (same module object either way) cannot
change what a later ``BoardLink(source)`` built *without* an explicit ``clock=`` will call: the
default value already is the original function, patch or no patch. This is proven below instead
of assumed. Per the task instructions, since the monkeypatch cannot reach the bound default, no
src change was made and the ">2s gate closes / re-opens via the patched default clock" scenario
asked for is reported as not coverable this way, rather than worked around with a real-time
sleep (disallowed: tests must not need real timers).
"""

from __future__ import annotations

import time

from formcoach.app import board
from formcoach.app.board import BoardLink
from formcoach.io import protocol
from formcoach.io.source import SISample


def _sample(flags: int = 0, seq: int = 0) -> SISample:
    return SISample(t=0.0, ax=0.0, ay=0.0, az=9.8, gx=0.0, gy=0.0, gz=0.0, flags=flags, seq=seq)


class _Source:
    """A fake ``IMUSource``: ``iter_samples`` is a plain generator over a fixed list."""

    def __init__(self, samples) -> None:
        self._samples = list(samples)

    def iter_samples(self):
        yield from self._samples

    def close(self) -> None:
        pass


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.005) -> bool:
    """Poll ``predicate`` with a bounded deadline instead of a fixed sleep: only used here to
    wait for the pump thread to deliver a sample (an async event), never to test the 2 s silence
    window itself."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return predicate()
        time.sleep(interval)


def test_default_clock_param_is_bound_at_import_time_not_by_a_later_monkeypatch(monkeypatch):
    """Proves the finding above: patching ``time.monotonic`` (via ``board.time.monotonic``, the
    name board.py actually references - the same module object as the global ``time`` module)
    does not change the ``clock`` a fresh, default-clock ``BoardLink`` will call.

    Production change that makes this fail: rewriting ``__init__`` to look the clock up lazily,
    e.g. ``clock: Callable[[], float] | None = None`` then ``self.clock = clock or
    time.monotonic`` evaluated inside the method body - that *would* pick up a patch applied
    before construction, flipping every assertion here.
    """
    real_monotonic = time.monotonic
    calls: list[str] = []

    def fake_monotonic() -> float:
        calls.append("fake")
        return 999.0

    monkeypatch.setattr(board.time, "monotonic", fake_monotonic)
    assert board.time.monotonic is fake_monotonic  # the patch did land on the module

    link = BoardLink(_Source([]))
    try:
        assert link.clock is real_monotonic
        assert link.clock is not fake_monotonic
        value = link.clock()
        assert value != 999.0
        assert calls == []  # the patched function was never reached through the default
    finally:
        link.close()


def test_default_clock_still_reports_waiting_before_anything_arrives():
    """A sanity check that the default (real) clock path works at all for the cheap, timing-free
    parts of the state machine: before any sample has arrived, ``state()``/``alive()``/
    ``gate_open()`` must agree regardless of which clock backs them.

    Production change that makes this fail: any of the three methods returning something other
    than "never heard yet" before ``poll()`` has drained a sample (e.g. ``alive()`` returning
    ``True`` when ``_last_arrival`` is still ``None``).
    """
    link = BoardLink(_Source([]))
    try:
        assert link.state() == "waiting"
        assert link.alive() is False
        assert link.gate_open() is False
    finally:
        link.close()


def test_default_clock_reports_alive_and_gate_open_for_a_real_sample():
    """With no injected clock (the production default), a sample that just arrived over the real
    wall clock is reported alive, and - with its gate flag set - ``gate_open()`` is ``True``: the
    same arithmetic tests/test_board.py exercises with a fake clock, exercised here against the
    real ``time.monotonic`` the default actually calls.

    Production change that makes this fail: inverting or breaking the ``alive()`` comparison
    (``(self.clock() - self._last_arrival) <= self.silence_s``), e.g. using ``<`` so a
    same-instant sample (common on a fast machine) is reported not alive.
    """
    link = BoardLink(_Source([_sample(flags=protocol.Flag.GATE_STATE)]))
    try:
        assert _wait_until(lambda: link.poll() or link.heard, timeout=1.0), "pump never delivered"
        assert link.alive() is True
        assert link.gate_open() is True
        assert link.state() == "ok"
    finally:
        link.close()
