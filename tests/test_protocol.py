"""The Python packet mirror must match firmware/include/protocol.h byte for byte."""

import re
from pathlib import Path

import pytest

from formcoach.io import protocol as p

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "include" / "protocol.h"


def test_sizes_are_pinned():
    assert p.SAMPLE_SIZE == 19
    assert p.BATCH_HEADER_SIZE == 4
    assert p.MAX_BATCH_SIZE == 99
    assert p.MAX_BATCH_SIZE <= p.REQUESTED_MTU - 3  # ATT header


def test_sample_round_trip():
    s = p.Sample(
        t_ms=123456, ax=-4096, ay=12, az=4095, gx=-32768, gy=0, gz=32767, flags=0b1011, seq=65535
    )
    assert p.unpack_sample(p.pack_sample(s)) == s
    assert s.gate_state is True
    assert s.session_active is True


def test_batch_round_trip_and_truncation():
    samples = [
        p.Sample(t_ms=20 * i, ax=i, ay=0, az=4096, gx=0, gy=0, gz=0, seq=i) for i in range(5)
    ]
    buf = p.pack_batch(session_id=7, samples=samples)
    assert len(buf) == p.MAX_BATCH_SIZE
    sid, out = p.unpack_batch(buf)
    assert sid == 7
    assert out == samples
    with pytest.raises(ValueError):
        p.unpack_batch(buf[:-1])
    with pytest.raises(ValueError):
        p.pack_batch(1, [])


def test_to_si_units():
    s = p.Sample(t_ms=0, ax=4096, ay=0, az=0, gx=328, gy=0, gz=0)
    ax, _, _, gx, _, _ = s.to_si()
    assert ax == pytest.approx(9.80665)
    assert gx == pytest.approx(10 * 3.141592653589793 / 180, rel=1e-3)


def test_serial_csv_parsing():
    assert p.parse_serial_line(p.SERIAL_CSV_HEADER) is None
    assert p.parse_serial_line("# boot") is None
    s = p.parse_serial_line("100,1,2,3,4,5,6,1,42\n")
    assert s == p.Sample(100, 1, 2, 3, 4, 5, 6, 1, 42)


def _header_constants() -> dict[str, str]:
    text = HEADER.read_text()
    return dict(re.findall(r"^#define\s+FC_(\w+)\s+(\S+)", text, flags=re.MULTILINE))


def test_constants_match_c_header():
    c = _header_constants()
    assert c["SAMPLE_RATE_HZ"] == str(p.SAMPLE_RATE_HZ)
    assert c["WINDOW_SAMPLES"] == str(p.WINDOW_SAMPLES)
    assert c["BATCH_SAMPLES"] == str(p.BATCH_SAMPLES)
    assert c["GATE_ON_WINDOWS"] == str(p.GATE_ON_WINDOWS)
    assert c["GATE_OFF_WINDOWS"] == str(p.GATE_OFF_WINDOWS)
    assert c["REQUESTED_MTU"] == str(p.REQUESTED_MTU)
    assert c["SAMPLE_SIZE"] == str(p.SAMPLE_SIZE)
    assert c["BATCH_HEADER_SIZE"] == str(p.BATCH_HEADER_SIZE)
    text = HEADER.read_text()
    for uuid in (p.SERVICE_UUID, p.IMU_CHAR_UUID, p.CONTROL_CHAR_UUID, p.STATUS_CHAR_UUID):
        assert uuid in text
    for cmd in p.Command:
        assert f"0x{cmd.value:02X}" in text, cmd
