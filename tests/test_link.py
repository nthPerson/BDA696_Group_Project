"""``LinkManager`` (Bench design §5.2): scan, connect to a chosen watch, health, disconnect and
kit labels. Injected factories, scanner and clock stand in for hardware; ``scan_devices`` and
``BLESource`` logging are tested with a fake ``bleak`` module, so nothing opens a radio or a
port."""

from __future__ import annotations

import json
import time
import types

import pytest

from formcoach.app.board import BoardLink
from formcoach.app.link import STATUS_KEYS, DeviceInfo, LinkManager
from formcoach.io import ble, protocol, serial_source
from formcoach.io.serial_source import LineDecoder, Stats
from formcoach.io.source import SISample


def _sample(seq: int = 0) -> SISample:
    """``t_arrival=None`` so the ``BoardLink`` stamps it with the injected clock."""
    return SISample(t=0.0, ax=0.0, ay=0.0, az=9.8, gx=0.0, gy=0.0, gz=0.0, seq=seq)


class _Source:
    """A fake BLE-like source: ``stats``, an optional ``status`` packet, ``send``/``close``."""

    def __init__(self, id, samples=(), status: protocol.Status | None = None) -> None:
        self.id = id
        self.device_name = None
        self.stats = Stats()
        self.status = status
        self.sent: list[protocol.Command] = []
        self.closed = False
        self._samples = list(samples)

    def iter_samples(self):
        yield from self._samples

    def send(self, cmd: protocol.Command) -> None:
        self.sent.append(cmd)

    def close(self) -> None:
        self.closed = True


class _SerialLike:
    """A fake serial source: a real ``LineDecoder`` and no ``status`` packet."""

    def __init__(self, port) -> None:
        self.port = port
        self.decoder = LineDecoder()
        self.closed = False

    @property
    def stats(self) -> Stats:
        return self.decoder.stats

    def iter_samples(self):
        return iter(())

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _manager(tmp_path, *, made=None, logs=None, scanner=None, clock=None, **source_kw):
    """A manager whose ``ble`` factory builds :class:`_Source` objects (recorded in ``made``)."""
    made = [] if made is None else made

    def ble_factory(id):
        src = _Source(id, **source_kw)
        made.append(src)
        return src

    kw = {} if logs is None else {"on_log": lambda level, text: logs.append((level, text))}
    return LinkManager(
        kits_file=tmp_path / "kits.json",
        clock=clock or _Clock(),
        factories={"ble": ble_factory, "serial": _SerialLike},
        scanner=scanner or (lambda timeout_s: []),
        **kw,
    )


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return predicate()
        time.sleep(0.005)
    return True


def test_scan_returns_devices_strongest_first_with_kit_labels(tmp_path, monkeypatch):
    (tmp_path / "kits.json").write_text(json.dumps({"BB": "K2"}))
    timeouts: list[float] = []

    def scanner(timeout_s):
        timeouts.append(timeout_s)
        return [("FormCoach-A", "AA", -80), ("FormCoach-C", "CC", None), ("FormCoach-B", "BB", -40)]

    logs: list[tuple[str, str]] = []
    m = _manager(tmp_path, scanner=scanner, logs=logs)
    found = m.scan("ble", timeout_s=3.0)

    assert timeouts == [3.0]
    assert found == [
        DeviceInfo(kind="ble", id="BB", name="FormCoach-B", rssi=-40, kit="K2"),
        DeviceInfo(kind="ble", id="AA", name="FormCoach-A", rssi=-80, kit=None),
        DeviceInfo(kind="ble", id="CC", name="FormCoach-C", rssi=None, kit=None),
    ]
    texts = [t for _lvl, t in logs]
    assert any("scanning" in t for t in texts)
    assert any("3 device" in t for t in texts)

    (fake,) = m.scan("fake")
    assert fake.kind == "fake" and fake.name == "FormCoach-FAKE" and fake.rssi is None

    monkeypatch.setattr(serial_source, "list_ports", lambda: ["/dev/ttyACM0", "COM5"])
    ports = m.scan("serial")
    assert [(d.kind, d.id, d.name) for d in ports] == [
        ("serial", "/dev/ttyACM0", "/dev/ttyACM0"),
        ("serial", "COM5", "COM5"),
    ]

    with pytest.raises(ValueError):
        m.scan("carrier-pigeon")


def test_scan_devices_filters_by_name_prefix_and_sorts_by_rssi(monkeypatch):
    def dev(name, address):
        return types.SimpleNamespace(name=name, address=address)

    def adv(rssi, local_name=None):
        return types.SimpleNamespace(rssi=rssi, local_name=local_name)

    calls: list[dict] = []

    async def discover(**kwargs):
        calls.append(kwargs)
        return {
            "11": (dev("FormCoach-1111", "11"), adv(-70)),
            "22": (dev(None, "22"), adv(-50, local_name="FormCoach-2222")),  # name only in adv
            "33": (dev("Headphones", "33"), adv(-30)),
            "44": (dev("FormCoach-4444", "44"), adv(None)),
        }

    fake_bleak = types.SimpleNamespace(BleakScanner=types.SimpleNamespace(discover=discover))
    monkeypatch.setattr(ble, "_bleak", lambda: fake_bleak)

    found = ble.scan_devices(timeout_s=1.5)

    assert calls == [{"timeout": 1.5, "return_adv": True}]
    assert found == [
        ("FormCoach-2222", "22", -50),
        ("FormCoach-1111", "11", -70),
        ("FormCoach-4444", "44", None),
    ]


def test_connect_wraps_the_source_and_status_reports_firmware_rate_and_drops(tmp_path):
    clock = _Clock()
    made: list[_Source] = []
    m = _manager(
        tmp_path,
        made=made,
        clock=clock,
        scanner=lambda t: [("FormCoach-8428", "AA", -60)],
        samples=[_sample(0), _sample(1)],
        status=protocol.Status(1, 2, 3, 7, 0, 42, 1, 1),
    )
    m.scan("ble")
    m.set_kit_label("AA", "K4")

    board = m.connect("ble", "AA")

    assert isinstance(board, BoardLink)
    assert m.board is board
    (src,) = made
    assert board.source is src and src.id == "AA"

    st = m.status()
    assert (st["kind"], st["id"], st["name"], st["kit"]) == ("ble", "AA", "FormCoach-8428", "K4")
    assert st["state"] == "waiting"
    assert st["firmware"] == "1.2.3"
    assert st["uptime_s"] == 42
    assert st["calibrated"] is True
    assert st["rate_hz"] is None  # one reading is not a rate yet

    # status() never polls the board: once the pump has queued both samples, status() leaves
    # them for the session's poll
    assert _wait_until(lambda: board._q.qsize() == 2)
    m.status()
    got = board.poll()
    assert [s.seq for s in got] == [0, 1]
    assert m.status()["state"] == "ok"

    src.stats.dropped, src.stats.bad_packets = 3, 2
    for t, n in [(1.0, 50), (2.0, 100), (3.0, 150), (3.5, 200)]:
        clock.t, src.stats.samples = t, n
        st = m.status()
    # over the last 2 s: from the last reading at or before 3.5 - 2 = 1.5 (t=1.0, 50 samples)
    assert st["rate_hz"] == pytest.approx(150 / 2.5)
    assert st["samples"] == 200 and st["dropped"] == 3 and st["bad_packets"] == 2
    assert st["p_active"] is None and st["infer_class"] is None and st["infer_ms"] is None
    m.disconnect()


def test_connecting_again_closes_the_previous_source(tmp_path):
    made: list[_Source] = []
    m = _manager(tmp_path, made=made)
    first = m.connect("ble", "AA")
    second = m.connect("ble", "BB")

    assert made[0].closed is True
    assert made[1].closed is False
    assert first is not second and m.board is second and second.source is made[1]
    assert m.status()["id"] == "BB"
    m.disconnect()


def test_disconnect_closes_and_status_state_is_none(tmp_path):
    made: list[_Source] = []
    m = _manager(tmp_path, made=made, status=protocol.Status(1, 0, 0, 1, 0, 5, 1, 1))
    m.connect("ble", "AA")
    m.disconnect()

    assert made[0].closed is True
    assert m.board is None
    st = m.status()
    assert st["state"] == "none"
    assert st["kind"] is None and st["id"] is None and st["firmware"] is None
    m.disconnect()  # a second disconnect is a no-op


def test_kit_label_is_persisted_and_survives_a_new_manager(tmp_path):
    kits = tmp_path / "sub" / "kits.json"  # the directory is created on first write
    scanner = lambda t: [("FormCoach-8428", "AA", -60)]  # noqa: E731
    m = LinkManager(kits_file=kits, scanner=scanner, factories={"ble": _Source})
    m.set_kit_label("AA", "K3")
    assert json.loads(kits.read_text()) == {"AA": "K3"}

    m2 = LinkManager(kits_file=kits, scanner=scanner, factories={"ble": _Source})
    assert m2.scan("ble")[0].kit == "K3"
    m2.connect("ble", "AA")
    assert m2.status()["kit"] == "K3"
    m2.set_kit_label("AA", "K5")
    assert m2.status()["kit"] == "K5"  # the connected kit's label follows the change
    m2.disconnect()

    m2.set_kit_label("AA", None)
    assert json.loads(kits.read_text()) == {}
    assert LinkManager(kits_file=kits, scanner=scanner).scan("ble")[0].kit is None


def test_identify_sends_led_good_rep(tmp_path):
    made: list[_Source] = []
    logs: list[tuple[str, str]] = []
    m = _manager(tmp_path, made=made, logs=logs)
    m.identify()  # no link: a warning, not an exception
    assert logs[-1][0] == "warn"

    m.connect("ble", "AA")
    m.identify()
    m.send(protocol.Command.CALIBRATE)
    assert made[0].sent == [protocol.Command.LED_GOOD_REP, protocol.Command.CALIBRATE]
    m.disconnect()


def test_status_has_every_key_even_with_no_link(tmp_path):
    expected = {
        "kind",
        "id",
        "name",
        "kit",
        "state",
        "firmware",
        "uptime_s",
        "calibrated",
        "rate_hz",
        "samples",
        "dropped",
        "bad_packets",
        "p_active",
        "infer_class",
        "infer_ms",
    }
    assert set(STATUS_KEYS) == expected

    m = _manager(tmp_path)
    st = m.status()
    assert set(st) == expected
    assert st["state"] == "none"
    assert all(v is None for k, v in st.items() if k != "state")

    class _Bare:  # no stats, no status, no decoder, no name
        def iter_samples(self):
            return iter(())

        def close(self):
            pass

    m2 = LinkManager(kits_file=tmp_path / "k.json", factories={"bare": lambda id: _Bare()})
    m2.connect("bare")
    st = m2.status()
    assert set(st) == expected
    assert st["kind"] == "bare" and st["state"] == "waiting"
    assert st["firmware"] is None and st["samples"] is None and st["rate_hz"] is None
    m2.disconnect()


def test_link_log_receives_connect_and_disconnect_lines(tmp_path, capsys):
    clock = _Clock()
    logs: list[tuple[str, str]] = []
    m = _manager(tmp_path, logs=logs, clock=clock, samples=[_sample()])
    board = m.connect("ble", "AA")
    assert any(lvl == "info" and "connect" in t and "AA" in t for lvl, t in logs)

    m.status()  # still waiting: no state-change line
    n = len(logs)
    assert _wait_until(lambda: board.poll() or board.heard)
    m.status()
    m.status()  # unchanged: logged once
    assert [t for _l, t in logs[n:]].count("link ok") == 1
    clock.t = 10.0
    m.status()
    assert logs[-1][0] == "warn" and "silent" in logs[-1][1]

    m.disconnect()
    assert logs[-1][0] == "info" and "disconnect" in logs[-1][1] and "AA" in logs[-1][1]

    # the default on_log prints nothing
    quiet = _manager(tmp_path)
    quiet.connect("ble", "AA")
    quiet.status()
    quiet.disconnect()
    assert capsys.readouterr().out == ""


def test_a_source_that_fails_to_open_leaves_no_link_and_logs_an_error(tmp_path):
    logs: list[tuple[str, str]] = []
    made: list[_Source] = []
    m = _manager(tmp_path, made=made, logs=logs)
    m.connect("ble", "AA")

    def broken(id):
        raise ImportError("serial input needs pyserial")

    m.factories["serial"] = broken
    with pytest.raises(ImportError):
        m.connect("serial", "COM5")

    assert made[0].closed is True  # the previous link was closed first
    assert m.board is None and m.status()["state"] == "none"
    assert logs[-1][0] == "error" and "pyserial" in logs[-1][1]


def test_serial_status_carries_p_active_from_infer_lines(tmp_path):
    m = _manager(tmp_path)
    board = m.connect("serial", "/dev/ttyACM0")
    src = board.source
    st = m.status()
    assert st["kind"] == "serial" and st["name"] == "/dev/ttyACM0"
    assert st["p_active"] is None and st["firmware"] is None

    src.decoder.feed("# infer us=13390 arena=6196 class=1 p_active=0.93")
    st = m.status()
    assert st["p_active"] == pytest.approx(0.93)
    assert st["infer_class"] == 1
    assert st["infer_ms"] == pytest.approx(13.39)
    assert st["firmware"] is None and st["uptime_s"] is None and st["calibrated"] is None
    m.disconnect()


def _fake_bleak(find_device_by_filter, client_error: Exception | None = None):
    class _Client:
        def __init__(self, target) -> None:
            self.target = target

        async def __aenter__(self):
            raise client_error or RuntimeError("no client in tests")

        async def __aexit__(self, *exc):
            return False

    return types.SimpleNamespace(
        BleakScanner=types.SimpleNamespace(find_device_by_filter=find_device_by_filter),
        BleakClient=_Client,
    )


def test_ble_source_log_callback_replaces_print(capsys, monkeypatch):
    async def find_nothing(match, timeout):
        return None

    monkeypatch.setattr(ble, "_bleak", lambda: _fake_bleak(find_nothing))
    lines: list[str] = []
    src = ble.BLESource(scan_round_s=0.001, log=lines.append)
    try:
        assert _wait_until(lambda: any("switched on?" in t for t in lines))
    finally:
        src.close()

    assert any("scanning for FormCoach-*" in t for t in lines)
    assert capsys.readouterr().out == ""


def test_default_ble_factory_routes_ble_lines_to_on_log(tmp_path, monkeypatch):
    async def must_not_scan(match, timeout):
        raise AssertionError("an address was given")

    monkeypatch.setattr(
        ble, "_bleak", lambda: _fake_bleak(must_not_scan, client_error=OSError("adapter off"))
    )
    logs: list[tuple[str, str]] = []
    m = LinkManager(on_log=lambda level, text: logs.append((level, text)), kits_file=tmp_path / "k")
    board = m.connect("ble", "AA:BB")
    try:
        assert isinstance(board.source, ble.BLESource) and board.source.address == "AA:BB"
        assert _wait_until(lambda: any("reconnecting" in t for _l, t in logs))
        assert ("warn", "# ble: adapter off; reconnecting in 1 s") in logs
    finally:
        m.disconnect()
