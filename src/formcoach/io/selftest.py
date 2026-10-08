"""``formcoach device check``: the assembly-day hardware self-test (docs/assembly-day.md §3).

Reads the firmware's serial output for a few seconds and turns it into PASS/FAIL checks a
teammate can read: boot banner and firmware version, IMU found and at which address, TFLite
Micro gate loaded (or energy fallback), BLE name, 50 Hz sample rate, no dropped samples,
gravity ≈ 1 g while the board rests flat, gyro bias near zero, inference time. With
``--interactive`` it also walks through pressing the button (session + button flags) and
shaking the board (gate flag). The raw log and a JSON summary are saved under
``reports/logs/<kit>_<timestamp>.log`` for `formcoach eval device` and the device registry.
:func:`analyze` is pure Python (unit-tested on canned lines); :func:`run_live` needs pyserial.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from formcoach.io import protocol
from formcoach.io.serial_source import LineDecoder

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "reports" / "logs"
RATE_TOL_HZ = 2.0
GRAVITY_TOL_G = 0.1
GYRO_BIAS_MAX_LSB = 150  # ≈ 4.6 dps
INFER_MAX_MS = 20.0  # docs/02 §6.3 target
RE_FW = re.compile(r"#\s*FormCoach fw\s+(\S+)")
RE_IMU = re.compile(r"#\s*bmi160\s+(ok|NOT FOUND)\s+at\s+(0x[0-9A-Fa-f]{2})")
RE_GATE_TFLM = re.compile(r"#\s*gate:\s*tflm ok arena=(\d+)")
RE_GATE_MODE = re.compile(r"#\s*gate:\s*(int8 CNN \(TFLite Micro\)|motion-energy rule)")
RE_BLE = re.compile(r"#\s*ble advertising as\s+(FormCoach-[0-9A-Fa-f]{4})")
RE_FLASH = re.compile(r"#\s*flash app=(\d+)")
RE_INFER = re.compile(r"#\s*infer us=(\d+)")
RE_SESSION = re.compile(r"#\s*session (start|stop) id=(\d+)")
RE_CALIB = re.compile(r"#\s*calibrated g=")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    value: float | None = None


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    firmware: str | None = None
    ble_name: str | None = None
    imu_address: str | None = None
    flash_bytes: int | None = None
    arena_bytes: int | None = None
    n_samples: int = 0
    duration_s: float = 0.0
    kit: str | None = None
    port: str | None = None
    when: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)


def analyze(
    lines: Iterable[str],
    phases: dict[str, tuple[float, float]] | None = None,
    live_pong: bool | None = None,
) -> Report:
    """Turn captured serial lines into a :class:`Report`.

    ``phases`` maps ``"button"``/``"shake"`` to ``(t_start_s, t_end_s)`` windows on the sample
    clock (``t_ms / 1000`` of the first sample = 0) during which the button was held / the board
    was shaken; ``live_pong`` records whether ``# pong`` answered a ping (live runs only).
    """
    rep = Report()
    dec = LineDecoder()
    samples = []
    infer_us: list[int] = []
    gate_mode = None
    sessions = 0
    calibrated = False
    for raw in lines:
        ln = raw.rstrip("\r\n")
        if m := RE_FW.search(ln):
            rep.firmware = m.group(1)
        elif m := RE_IMU.search(ln):
            rep.imu_address = m.group(2) if m.group(1) == "ok" else None
            rep.checks.append(
                Check("imu", m.group(1) == "ok", f"bmi160 {m.group(1)} at {m.group(2)}")
            )
        elif m := RE_GATE_TFLM.search(ln):
            rep.arena_bytes = int(m.group(1))
        elif m := RE_GATE_MODE.search(ln):
            gate_mode = m.group(1)
        elif m := RE_BLE.search(ln):
            rep.ble_name = m.group(1)
        elif m := RE_FLASH.search(ln):
            rep.flash_bytes = int(m.group(1))
        elif m := RE_INFER.search(ln):
            infer_us.append(int(m.group(1)))
        elif RE_SESSION.search(ln):
            sessions += 1
        elif RE_CALIB.search(ln):
            calibrated = True
        s = dec.feed(ln)
        if s is not None:
            samples.append(s)
    rep.checks.insert(
        0,
        Check("banner", rep.firmware is not None, f"firmware {rep.firmware or 'banner not seen'}"),
    )
    if not any(c.name == "imu" for c in rep.checks):
        rep.checks.append(
            Check("imu", False, "no '# bmi160' line (banner not captured? press RESET)")
        )
    tflm = rep.arena_bytes is not None and gate_mode == "int8 CNN (TFLite Micro)"
    rep.checks.append(
        Check(
            "gate",
            tflm,
            f"tflm arena={rep.arena_bytes} B"
            if tflm
            else f"gate mode: {gate_mode or 'unknown'} (energy fallback or no line)",
            rep.arena_bytes,
        )
    )
    rep.checks.append(
        Check("ble", rep.ble_name is not None, rep.ble_name or "no '# ble advertising' line")
    )
    if live_pong is not None:
        rep.checks.append(
            Check(
                "ping",
                live_pong,
                "'# pong' received" if live_pong else "no '# pong' after sending P",
            )
        )
    rep.n_samples = len(samples)
    if samples:
        t = np.array([s.t for s in samples])
        t0 = t[0]
        rep.duration_s = float(t[-1] - t0)
        rate = len(samples) / rep.duration_s if rep.duration_s > 0 else 0.0
        rep.checks.append(
            Check(
                "rate",
                abs(rate - protocol.SAMPLE_RATE_HZ) <= RATE_TOL_HZ,
                f"{rate:.1f} Hz over {rep.duration_s:.1f} s ({len(samples)} samples)",
                rate,
            )
        )
        rep.checks.append(
            Check(
                "drops",
                dec.stats.dropped == 0,
                f"{dec.stats.dropped} dropped, {dec.stats.bad_lines} bad lines",
                dec.stats.dropped,
            )
        )
        acc = np.array([[s.ax, s.ay, s.az] for s in samples]) / protocol.G_TO_MS2
        gyr = (
            np.array([[s.gx, s.gy, s.gz] for s in samples])
            * protocol.GYRO_LSB_PER_DPS
            / protocol.DPS_TO_RADS
        )
        still = np.ones(len(samples), bool)
        if phases:
            for a, b in phases.values():
                still &= ~((t - t0 >= a) & (t - t0 <= b))
        if still.sum() < 10:
            still = np.ones(len(samples), bool)
        g = float(np.linalg.norm(acc[still], axis=1).mean())
        rep.checks.append(
            Check(
                "gravity", abs(g - 1.0) <= GRAVITY_TOL_G, f"|a| = {g:.3f} g at rest (expect 1.0)", g
            )
        )
        bias = float(np.abs(gyr[still].mean(axis=0)).max())
        rep.checks.append(
            Check(
                "gyro_bias",
                bias <= GYRO_BIAS_MAX_LSB,
                f"max |gyro mean| = {bias:.0f} LSB ({bias / protocol.GYRO_LSB_PER_DPS:.1f} dps)",
                bias,
            )
        )
        flags = np.array([s.flags for s in samples])
        if phases and "button" in phases:
            a, b = phases["button"]
            win = (t - t0 >= a) & (t - t0 <= b)
            pressed = bool((flags[win] & protocol.Flag.BUTTON_PRESSED).any())
            rep.checks.append(
                Check(
                    "button",
                    pressed,
                    "button flag seen while held"
                    if pressed
                    else "button flag never set (D1 wiring / pull-up)",
                )
            )
            sess = bool((flags[win] & protocol.Flag.SESSION_ACTIVE).any()) or sessions > 0
            rep.checks.append(
                Check("session", sess, "session toggled" if sess else "no session start")
            )
        if phases and "shake" in phases:
            a, b = phases["shake"]
            win = (t - t0 >= a) & (t - t0 <= b)
            gate = bool((flags[win] & protocol.Flag.GATE_STATE).any())
            rep.checks.append(
                Check(
                    "gate_flag",
                    gate,
                    "gate opened while shaking"
                    if gate
                    else "gate never opened (shake harder / longer than 1 s, check IMU)",
                )
            )
    else:
        rep.checks.append(
            Check("rate", False, "no CSV samples (IMU missing, or the monitor is not at 115200)")
        )
    if infer_us:
        med = float(np.median(infer_us)) / 1000.0
        rep.checks.append(
            Check(
                "inference",
                med <= INFER_MAX_MS,
                f"median {med:.2f} ms, max {max(infer_us) / 1000:.2f} ms "
                f"({len(infer_us)} inferences)",
                med,
            )
        )
    if calibrated:
        rep.checks.append(Check("calibration", True, "calibration stored"))
    return rep


def format_report(rep: Report) -> str:
    lines = [f"FormCoach device check  kit={rep.kit or '-'}  port={rep.port or '-'}  {rep.when}"]
    for c in rep.checks:
        lines.append(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name:<12} {c.detail}")
    lines.append(
        f"  firmware {rep.firmware}  ble {rep.ble_name}  imu {rep.imu_address}  "
        f"flash {rep.flash_bytes}  arena {rep.arena_bytes}"
    )
    lines.append(
        "RESULT: "
        + ("PASS" if rep.passed else "FAIL — see the FAIL lines above and docs/assembly-day.md §5")
    )
    return "\n".join(lines)


def save_log(
    lines: Iterable[str], rep: Report, out_dir: Path = DEFAULT_LOG_DIR, kit: str | None = None
) -> Path:
    """Write ``<kit>_<timestamp>.log`` (raw lines) and ``.json`` (the report) under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{kit or rep.kit or 'kit'}_{stamp}"
    base.with_suffix(".log").write_text(
        "\n".join(ln.rstrip("\r\n") for ln in lines) + "\n", encoding="utf-8"
    )
    doc = asdict(rep)
    doc["passed"] = rep.passed
    base.with_suffix(".json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return base.with_suffix(".log")


def run_live(
    port: str,
    *,
    duration_s: float = 8.0,
    kit: str | None = None,
    interactive: bool = False,
    out_dir: Path = DEFAULT_LOG_DIR,
    say: Callable[[str], None] = print,
    baud: int = 115200,
) -> tuple[Report, Path]:
    """Capture the boot banner (toggling DTR/RTS resets the board) and ``duration_s`` of samples,
    optionally with button/shake phases, then analyse and save the log."""
    from formcoach.io.serial_source import _pyserial

    serial = _pyserial()
    lines: list[str] = []
    phases: dict[str, tuple[float, float]] = {}
    t_first: float | None = None
    with serial.Serial(port, baud, timeout=0.2) as ser:
        ser.setDTR(False)
        ser.setRTS(True)  # pulse the auto-reset circuit so the banner is captured
        time.sleep(0.1)
        ser.setRTS(False)
        ser.setDTR(False)
        t_start = time.monotonic()
        ser.write(b"P\n")
        pong = False

        def pump(until: float):
            nonlocal t_first, pong
            while time.monotonic() < until:
                raw = ser.readline().decode("ascii", errors="replace")
                if not raw:
                    continue
                lines.append(raw)
                if "# pong" in raw:
                    pong = True
                if t_first is None and protocol.parse_serial_line(raw) is not None:
                    t_first = protocol.parse_serial_line(raw).t_ms / 1000.0

        def sample_clock() -> float:
            last = next(
                (
                    protocol.parse_serial_line(ln)
                    for ln in reversed(lines)
                    if protocol.parse_serial_line(ln)
                ),
                None,
            )
            return (last.t_ms / 1000.0 - (t_first or 0.0)) if last else 0.0

        say(f"capturing {duration_s:.0f} s — keep the board flat and still on the desk")
        pump(t_start + duration_s)
        if interactive:
            say("now press and HOLD the button for 3 s (short press toggles the session)…")
            a = sample_clock()
            pump(time.monotonic() + 4.0)
            phases["button"] = (a, sample_clock())
            say("release; now SHAKE the board for 4 s…")
            a = sample_clock()
            pump(time.monotonic() + 5.0)
            phases["shake"] = (a, sample_clock())
            say("done; keep still for 2 s")
            pump(time.monotonic() + 2.0)
    rep = analyze(lines, phases or None, live_pong=pong)
    rep.kit, rep.port = kit, port
    path = save_log(lines, rep, out_dir, kit)
    return rep, path
