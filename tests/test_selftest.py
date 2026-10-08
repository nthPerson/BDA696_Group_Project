"""`formcoach device check`: the assembly-day self-test, driven here by a fake serial stream."""

from __future__ import annotations

import numpy as np

from formcoach.io import protocol, selftest


def _lines(n_s: float = 6.0, *, imu=True, flags_seq=None, az_g=1.0, gyro_bias=5, drop_every=0):
    """Boot banner + CSV lines like firmware v1.1 prints."""
    out = [
        "# FormCoach fw 1.1.0  chip=ESP32-S3 rev=0",
        "# sample=19 B batch=99 B status=12 B rate=50 Hz",
        "# flash app=642745",
        f"# bmi160 {'ok at 0x69' if imu else 'NOT FOUND at 0x00'}",
        "# calibration none (long-press to calibrate)",
        "# gate: tflm ok arena=28000 in=600 B classes=6",
        "# gate: int8 CNN (TFLite Micro)",
        "# ble advertising as FormCoach-1A2B",
        protocol.SERIAL_CSV_HEADER,
    ]
    if not imu:
        return out
    rng = np.random.default_rng(0)
    seq = 0
    for i in range(int(n_s * 50)):
        t_ms = 20 * i
        flags = flags_seq(i) if flags_seq else 0
        ax, ay = int(rng.normal(0, 20)), int(rng.normal(0, 20))
        az = int(az_g * 4096 + rng.normal(0, 20))
        if drop_every and i % drop_every == 0:
            seq += 1  # skip one seq number
        out.append(f"{t_ms},{ax},{ay},{az},{gyro_bias},{-gyro_bias},0,{flags},{seq}")
        seq += 1
        if i % 25 == 0:
            out.append(f"# infer us={1800 + i} arena=28000 class=0 p_active=0.05")
    return out


def test_healthy_board_passes_every_static_check():
    rep = selftest.analyze(_lines())
    checks = {c.name: c for c in rep.checks}
    assert checks["banner"].ok and "1.1.0" in checks["banner"].detail
    assert checks["imu"].ok and "0x69" in checks["imu"].detail
    assert checks["gate"].ok and "tflm" in checks["gate"].detail
    assert checks["ble"].ok and "FormCoach-1A2B" in checks["ble"].detail
    assert checks["rate"].ok and abs(checks["rate"].value - 50.0) < 1.0
    assert checks["drops"].ok and checks["drops"].value == 0
    assert checks["gravity"].ok and abs(checks["gravity"].value - 1.0) < 0.05
    assert checks["gyro_bias"].ok
    assert checks["inference"].ok and 1.5 < checks["inference"].value < 3.0
    assert rep.passed and rep.firmware == "1.1.0" and rep.ble_name == "FormCoach-1A2B"
    assert rep.imu_address == "0x69"


def test_missing_imu_and_drops_are_reported():
    rep = selftest.analyze(_lines(imu=False))
    checks = {c.name: c for c in rep.checks}
    assert not checks["imu"].ok and not rep.passed
    assert not checks["rate"].ok  # no samples at all
    rep2 = selftest.analyze(_lines(drop_every=50))
    assert not {c.name: c for c in rep2.checks}["drops"].ok


def test_wrong_gravity_and_energy_fallback_flagged():
    lines = _lines(az_g=0.5)
    lines = [
        ln.replace("# gate: int8 CNN (TFLite Micro)", "# gate: motion-energy rule") for ln in lines
    ]
    rep = selftest.analyze(lines)
    checks = {c.name: c for c in rep.checks}
    assert not checks["gravity"].ok
    assert not checks["gate"].ok and "energy" in checks["gate"].detail


def test_interactive_phases_detect_button_session_and_gate():
    def flags(i):  # 0-2 s idle, 2-4 s session+button, 4-6 s gate open
        if i < 100:
            return 0
        if i < 200:
            return int(protocol.Flag.SESSION_ACTIVE | protocol.Flag.BUTTON_PRESSED)
        return int(protocol.Flag.GATE_STATE | protocol.Flag.SESSION_ACTIVE)

    rep = selftest.analyze(
        _lines(flags_seq=flags), phases={"button": (2.0, 4.0), "shake": (4.0, 6.0)}
    )
    checks = {c.name: c for c in rep.checks}
    assert checks["button"].ok and checks["session"].ok and checks["gate_flag"].ok


def test_report_text_and_json(tmp_path):
    rep = selftest.analyze(_lines())
    text = selftest.format_report(rep)
    assert "PASS" in text and "rate" in text and "gravity" in text
    p = selftest.save_log(_lines(), rep, tmp_path, kit="K1")
    assert p.exists() and p.name.startswith("K1_") and (p.with_suffix(".json")).exists()
