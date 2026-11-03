"""``BLESource._find`` scans in 2 s rounds via ``find_device_by_filter`` (docs/superpowers/specs/
2026-11-02-live-demo-design.md §4.5). A fake ``bleak`` module stands in so this needs no real
Bluetooth; ``BLESource.__new__`` is used so no thread or real bleak import starts."""

from __future__ import annotations

import asyncio
import threading
import time
import types

import pytest

from formcoach.io.ble import BLESource


class _FakeDevice:
    def __init__(self, name: str) -> None:
        self.name = name
        self.address = "AA:BB:CC:DD:EE:FF"


def _source(find_device_by_filter) -> BLESource:
    """Build a ``BLESource`` without running ``__init__`` (no thread, no real bleak)."""
    src = BLESource.__new__(BLESource)
    src.bleak = types.SimpleNamespace(
        BleakScanner=types.SimpleNamespace(find_device_by_filter=find_device_by_filter)
    )
    src.name_prefix = "FormCoach-"
    src.address = None
    src.scan_round_s = 2.0
    src._stop = threading.Event()
    src.device_name = None
    return src


def test_find_scans_in_2s_rounds_until_found():
    calls: list[float] = []
    device = _FakeDevice("FormCoach-8428")

    async def fake_find(match, timeout):
        calls.append(timeout)
        # the real predicate must accept (device, advertisement_data) and match by name prefix
        assert match(_FakeDevice("FormCoach-8428"), None) is True
        assert match(_FakeDevice("OtherThing"), None) is False
        assert match(types.SimpleNamespace(name=None), None) is False
        if len(calls) < 3:
            return None
        return device

    src = _source(fake_find)
    t0 = time.monotonic()
    result = asyncio.run(src._find())
    elapsed = time.monotonic() - t0

    assert calls == [2.0, 2.0, 2.0]
    assert result is device
    assert src.device_name == "FormCoach-8428"
    assert elapsed < 1.0  # no sleeping / backoff between rounds


def test_find_returns_given_address_without_scanning():
    async def fake_find(match, timeout):
        raise AssertionError("must not scan when an address is given")

    src = _source(fake_find)
    src.address = "AA:BB:CC:DD:EE:FF"
    result = asyncio.run(src._find())
    assert result == "AA:BB:CC:DD:EE:FF"
    assert src.device_name is None  # unchanged: the given-address path never scans


def test_find_stops_when_asked():
    async def fake_find(match, timeout):
        raise AssertionError("must not scan once _stop is set")

    src = _source(fake_find)
    src._stop.set()
    with pytest.raises(RuntimeError):
        asyncio.run(src._find())
