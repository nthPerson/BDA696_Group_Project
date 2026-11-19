"""``LinkManager``: the watch link a UI drives (Bench design §5.2, Bench plan Task 4).

It replaces "the first ``FormCoach-`` name wins": :meth:`LinkManager.scan` lists nearby devices
(BLE name, identifier and signal strength; USB serial ports; the fake watch),
:meth:`~LinkManager.connect` opens the chosen one and wraps it in a :class:`BoardLink` (hand
:attr:`~LinkManager.board` to ``LiveSession.set_board``), :meth:`~LinkManager.status` reports
its health and :meth:`~LinkManager.disconnect` closes it. Every kit advertises the same name, so
a kit label (K1…K6) can be attached to an identifier; labels are kept in a small JSON file on
this laptop (:data:`KITS_FILE`), and :meth:`~LinkManager.identify` flashes the connected kit's
LED.

Thread safety: :meth:`connect`/:meth:`disconnect` may run on a worker thread while a UI thread
calls :meth:`status` several times a second. ``status()`` only takes a short lock and reads
counters the sources already keep; it never touches the radio and never polls the board (the
live session owns ``BoardLink.poll``). No hardware is touched in tests: ``factories`` and
``scanner`` replace the real sources and the BLE scan.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from formcoach.app.board import BoardLink
from formcoach.io import protocol, serial_source
from formcoach.io.source import IMUSource

KITS_FILE = Path.home() / ".formcoach" / "kits.json"
RATE_WINDOW_S = 2.0  # seconds of sample counts behind status()["rate_hz"]
FAKE_ID = "fake"  # the fake watch's identifier in scan results

STATUS_KEYS = (
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
)

Factory = Callable[[str | None], IMUSource]
Scanner = Callable[[float], list[tuple[str, str, int | None]]]


@dataclass(frozen=True)
class DeviceInfo:
    """One scan result: ``kind`` ``"ble"``/``"serial"``/``"fake"``, ``id`` the identifier to pass
    to :meth:`LinkManager.connect` (BLE MAC or macOS UUID, serial port, ``"fake"``), ``name`` as
    advertised (the port for serial), ``rssi`` signal strength in dBm (BLE only, else ``None``)
    and ``kit`` the label attached to ``id`` on this laptop, if any."""

    kind: str
    id: str
    name: str
    rssi: int | None
    kit: str | None


def _no_log(level: str, text: str) -> None:
    """The default ``on_log``: discard."""


def _ble_level(text: str) -> str:
    """Level for a ``BLESource`` progress line: trouble (reconnecting, nothing found yet) is a
    warning, the rest is information."""
    return "warn" if ("reconnecting" in text or "found yet" in text) else "info"


class LinkManager:
    """Scan for, connect to and watch one wearable at a time.

    ``on_log(level, text)`` (``level`` ``"info"``/``"warn"``/``"error"``) receives scan start and
    result count, connects, link state changes, disconnects and every ``BLESource`` progress
    line; the default discards them. ``kits_file`` holds kit labels as JSON ``{id: kit}``.
    ``clock`` returns seconds (``time.monotonic`` by default) and is shared with the
    :class:`BoardLink`. ``factories`` maps a kind to a callable building its source from an
    identifier (merged over the defaults ``BLESource(address=id)``, ``SerialSource(id)``,
    ``FakeWatch()``); ``scanner(timeout_s)`` replaces :func:`formcoach.io.ble.scan_devices`.
    """

    def __init__(
        self,
        *,
        on_log: Callable[[str, str], None] = _no_log,
        kits_file: Path = KITS_FILE,
        clock: Callable[[], float] = time.monotonic,
        factories: dict[str, Factory] | None = None,
        scanner: Scanner | None = None,
    ) -> None:
        self.on_log = on_log
        self.kits_file = Path(kits_file)
        self.clock = clock
        self.factories: dict[str, Factory] = {
            "ble": self._open_ble,
            "serial": self._open_serial,
            "fake": self._open_fake,
            **(factories or {}),
        }
        self.scanner: Scanner = scanner or self._scan_ble
        self._lock = threading.Lock()  # guards the fields below; held only briefly
        self._switch = threading.Lock()  # serialises connect/disconnect (may be slow)
        self._labels = threading.Lock()  # serialises set_kit_label (memory change + file write)
        self._log_failed = False  # on_log raised once already (reported to stderr)
        self._kits: dict[str, str] = self._load_kits()
        self._names: dict[str, str] = {}  # identifier -> advertised name, from the last scans
        self._board: BoardLink | None = None
        self._kind: str | None = None
        self._id: str | None = None
        self._state: str | None = None  # last BoardLink.state() seen by status()
        self._error_seen: BoardLink | None = None  # the link whose last_error was logged
        self._counts: deque[tuple[float, int]] = deque()  # (clock, stats.samples) readings

    # ---- default sources (need the ``device`` extra for BLE and serial) -----------------------

    def _open_ble(self, id: str | None) -> IMUSource:
        from formcoach.io.ble import BLESource

        return BLESource(address=id, log=lambda text: self._log(_ble_level(text), text))

    def _open_serial(self, id: str | None) -> IMUSource:
        from formcoach.io.serial_source import SerialSource

        if not id:
            ports = serial_source.list_ports()
            raise FileNotFoundError(f"a serial port is required (visible: {ports or 'none'})")
        return SerialSource(id)

    def _open_fake(self, id: str | None) -> IMUSource:
        from formcoach.io.fake import FakeWatch

        return FakeWatch()

    def _scan_ble(self, timeout_s: float) -> list[tuple[str, str, int | None]]:
        from formcoach.io.ble import scan_devices

        return scan_devices(timeout_s)

    # ---- logging and kit labels ---------------------------------------------------------------

    def _log(self, level: str, text: str) -> None:
        """Hand one line to ``on_log``. A broken UI callback must not break the link: its first
        failure is printed to stderr once, later ones are ignored."""
        try:
            self.on_log(level, text)
        except Exception as exc:
            if not self._log_failed:
                self._log_failed = True
                print(
                    f"formcoach link: on_log failed ({type(exc).__name__}: {exc}); "
                    "further failures are ignored",
                    file=sys.stderr,
                )

    def _load_kits(self) -> dict[str, str]:
        try:
            data = json.loads(self.kits_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            self._log("warn", f"kit labels unreadable ({self.kits_file}): {exc}; starting empty")
            return {}
        if not isinstance(data, dict):
            self._log("warn", f"kit labels in {self.kits_file} are not a JSON object; ignored")
            return {}
        return {str(k): str(v) for k, v in data.items() if v}

    def set_kit_label(self, id: str, kit: str | None) -> None:
        """Attach kit label ``kit`` (e.g. ``"K3"``) to identifier ``id``, or remove it with
        ``None`` or ``""``, and save all labels to ``kits_file`` (its directory is created).

        Calls are serialised, so the file always holds what memory holds. If the file cannot be
        written the label is put back as it was and the ``OSError`` is raised.
        """
        with self._labels:
            with self._lock:
                previous = self._kits.get(id)
                if kit:
                    self._kits[id] = kit
                else:
                    self._kits.pop(id, None)
                kits = dict(self._kits)
            try:
                self.kits_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.kits_file.with_name(self.kits_file.name + ".tmp")
                tmp.write_text(json.dumps(kits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(tmp, self.kits_file)
            except OSError:
                with self._lock:
                    if previous is None:
                        self._kits.pop(id, None)
                    else:
                        self._kits[id] = previous
                raise
        self._log("info", f"kit label {id} -> {kit or '(none)'}")

    # ---- scan / connect / disconnect ----------------------------------------------------------

    def scan(self, kind: str = "ble", timeout_s: float = 4.0) -> list[DeviceInfo]:
        """List devices of ``kind``: ``"ble"`` scans the radio for ``timeout_s`` seconds
        (blocking; strongest first), ``"serial"`` lists the visible ports, ``"fake"`` returns the
        one fake watch. Raises ``ValueError`` for another kind; a scan error is logged and
        re-raised."""
        if kind == "ble":
            self._log("info", f"scanning for watches ({timeout_s:g} s)")
            try:
                found = self.scanner(timeout_s)
            except Exception as exc:
                self._log("error", f"scan failed: {exc}")
                raise
            found = sorted(found, key=lambda d: (d[2] is None, -(d[2] or 0)))
            with self._lock:
                self._names.update({ident: name for name, ident, _rssi in found})
                out = [DeviceInfo("ble", i, n, r, self._kits.get(i)) for n, i, r in found]
        elif kind == "serial":
            ports = serial_source.list_ports()
            with self._lock:
                out = [DeviceInfo("serial", p, p, None, self._kits.get(p)) for p in ports]
        elif kind == "fake":
            from formcoach.io.fake import FakeWatch

            with self._lock:
                kit = self._kits.get(FAKE_ID)
            out = [DeviceInfo("fake", FAKE_ID, FakeWatch.device_name, None, kit)]
        else:
            raise ValueError(f"unknown link kind {kind!r}; choose ble, serial or fake")
        self._log("info", f"found {len(out)} device(s) ({kind})")
        return out

    def _describe(self, kind: str | None, id: str | None) -> str:
        """``ble AA:BB (FormCoach-8428, kit K2)``-style text for log lines. Call with ``_lock``
        held."""
        name, kit = self._names.get(id or ""), self._kits.get(id or "")
        extras = [x for x in (name, kit and f"kit {kit}") if x]
        text = f"{kind} {id or '(first found)'}"
        return f"{text} ({', '.join(extras)})" if extras else text

    def connect(self, kind: str, id: str | None = None) -> BoardLink:
        """Close any current link, then open ``kind`` (a key of ``factories``) at identifier
        ``id`` (BLE address or macOS UUID, ``None`` for the first watch found; serial port;
        ignored by the fake watch) and return its :class:`BoardLink`. If the source cannot be
        opened (missing extra, port busy), the error is logged and re-raised and there is no
        link."""
        if kind not in self.factories:
            raise ValueError(f"unknown link kind {kind!r}; choose {', '.join(self.factories)}")
        with self._switch:
            self._close_current()
            with self._lock:
                desc = self._describe(kind, id)
            self._log("info", f"connecting to {desc}")
            try:
                source = self.factories[kind](id)
            except Exception as exc:
                self._log("error", f"could not open {desc}: {exc}")
                raise
            board = BoardLink(source, clock=self.clock)
            with self._lock:
                self._board, self._kind, self._id = board, kind, id
                self._state = board.state()
                self._counts.clear()
            return board

    def disconnect(self) -> None:
        """Close the current link, if any (a no-op otherwise)."""
        with self._switch:
            self._close_current()

    def _close_current(self) -> None:
        """Detach the current link under the state lock, then close it outside it (closing joins
        threads and may take a second)."""
        with self._lock:
            board, kind, id = self._board, self._kind, self._id
            if board is None:
                return
            desc = self._describe(kind, id)
            self._board = self._kind = self._id = self._state = None
            self._counts.clear()
        board.close()
        self._log("info", f"disconnected from {desc}")

    @property
    def board(self) -> BoardLink | None:
        """The current link (hand it to ``LiveSession.set_board``), or ``None``."""
        with self._lock:
            return self._board

    # ---- commands -----------------------------------------------------------------------------

    def send(self, cmd: protocol.Command) -> None:
        """Send ``cmd`` to the connected watch; with no link, log a warning instead."""
        board = self.board
        if board is None:
            self._log("warn", f"no watch connected; {cmd.name} not sent")
            return
        board.send(cmd)

    def identify(self) -> None:
        """Flash the connected kit's LED (``LED_GOOD_REP``) so it can be told apart."""
        if self.board is not None:
            self._log("info", "identify: flashing the connected watch's LED")
        self.send(protocol.Command.LED_GOOD_REP)

    # ---- health -------------------------------------------------------------------------------

    def status(self) -> dict:
        """Link health; every key of :data:`STATUS_KEYS` is present, unknown values are ``None``.

        ``kind``/``id``/``name``/``kit`` identify the link; ``state`` is ``"none"`` with no link,
        else ``BoardLink.state()`` (``"waiting"``, ``"ok"``, ``"silent"``). From the watch's
        status packet: ``firmware`` ``"major.minor.patch"``, ``uptime_s`` (s), ``calibrated``.
        From the decoder: ``samples``, ``dropped`` (missing sequence numbers), ``bad_packets``
        and ``rate_hz`` (samples per second over the last :data:`RATE_WINDOW_S` seconds of
        ``clock``). From the latest serial ``# infer`` line: ``p_active`` (0..1),
        ``infer_class`` and ``infer_ms`` (model time, ms). Cheap; never blocks on the radio and
        never polls the board. A change of ``state`` is logged, and so is, once per link, the
        exception that stopped the source (``BoardLink.last_error``), at level ``"error"``.
        """
        out: dict = dict.fromkeys(STATUS_KEYS)
        changed: str | None = None
        died: BaseException | None = None
        with self._lock:
            board = self._board
            if board is None:
                out["state"] = "none"
                return out
            error = getattr(board, "last_error", None)
            if error is not None and self._error_seen is not board:
                self._error_seen = board  # each link's failure is logged once
                died = error
            source = board.source
            name = getattr(source, "device_name", None) or self._names.get(self._id or "")
            out.update(
                kind=self._kind,
                id=self._id,
                name=name or self._id,
                kit=self._kits.get(self._id or ""),
                state=board.state(),
            )
            if out["state"] != self._state:
                self._state = changed = out["state"]
            stats = getattr(source, "stats", None)
            if stats is not None:
                out.update(
                    samples=stats.samples, dropped=stats.dropped, bad_packets=stats.bad_packets
                )
                out["rate_hz"] = self._rate(stats.samples)
        st = getattr(source, "status", None)
        if isinstance(st, protocol.Status):
            out.update(firmware=st.firmware, uptime_s=st.uptime_s, calibrated=bool(st.calibrated))
        infer = getattr(getattr(source, "decoder", None), "last_infer", None)
        if infer is not None:
            out.update(p_active=infer.p_active, infer_class=infer.cls, infer_ms=infer.us / 1000)
        if died is not None:
            self._log("error", f"the watch link stopped: {type(died).__name__}: {died}")
        if changed is not None:
            self._log("warn" if changed == "silent" else "info", f"link {changed}")
        return out

    def _rate(self, samples: int) -> float | None:
        """Samples per second from the last reading at or before ``RATE_WINDOW_S`` seconds ago
        (or the oldest reading, early on) to now; ``None`` until two readings exist. Call with
        ``_lock`` held."""
        now = self.clock()
        counts = self._counts
        counts.append((now, samples))
        while len(counts) > 1 and counts[1][0] <= now - RATE_WINDOW_S:
            counts.popleft()
        t0, n0 = counts[0]
        if now <= t0:
            return None
        return (samples - n0) / (now - t0)
