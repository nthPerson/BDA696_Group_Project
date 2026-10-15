"""The browser copy of the protocol (web/ble-viewer/protocol.js) must match protocol.py.

Third mirror after firmware/include/protocol.h and src/formcoach/io/protocol.py; this test
reads protocol.js as text, so it needs no browser and no Node.
"""

import re
from pathlib import Path

from formcoach.io import protocol as p

JS = Path(__file__).resolve().parents[1] / "web" / "ble-viewer" / "protocol.js"
SAMPLE_ROW = re.compile(
    r"\{ t_ms: (\d+), ax: (-?\d+), ay: (-?\d+), az: (-?\d+), "
    r"gx: (-?\d+), gy: (-?\d+), gz: (-?\d+), flags: (\d+), seq: (\d+) \}"
)


def _js() -> str:
    return JS.read_text(encoding="utf-8")


def _const(name: str) -> str:
    m = re.search(rf"^export const {name} = (.+);$", _js(), flags=re.MULTILINE)
    assert m, f"{name} not found in protocol.js"
    return m.group(1)


def _object(name: str) -> dict[str, int]:
    m = re.search(rf"export const {name} = Object\.freeze\(\{{(.*?)\}}\);", _js(), flags=re.S)
    assert m, f"{name} not found in protocol.js"
    return {k: int(v, 0) for k, v in re.findall(r"(\w+): (0x[0-9a-fA-F]+|\d+)", m.group(1))}


def test_strings_match():
    for name in (
        "DEVICE_NAME_PREFIX",
        "SERVICE_UUID",
        "IMU_CHAR_UUID",
        "CONTROL_CHAR_UUID",
        "STATUS_CHAR_UUID",
        "SERIAL_CSV_HEADER",
    ):
        assert _const(name) == f'"{getattr(p, name)}"', name


def test_numbers_match():
    for name in (
        "SAMPLE_RATE_HZ",
        "BATCH_SAMPLES",
        "SAMPLE_SIZE",
        "BATCH_HEADER_SIZE",
        "MAX_BATCH_SIZE",
        "STATUS_SIZE",
    ):
        assert int(_const(name)) == getattr(p, name), name
    assert float(_const("ACCEL_LSB_PER_G")) == p.ACCEL_LSB_PER_G
    assert float(_const("GYRO_LSB_PER_DPS")) == p.GYRO_LSB_PER_DPS


def test_commands_and_flags_match():
    assert _object("Command") == {c.name: c.value for c in p.Command}
    assert _object("Flag") == {f.name: f.value for f in p.Flag}


def test_golden_batch_is_what_protocol_py_packs():
    samples = [p.Sample(*map(int, row)) for row in SAMPLE_ROW.findall(_js())]
    assert len(samples) == p.BATCH_SAMPLES
    session_id = int(_const("GOLDEN_SESSION_ID"))
    assert _const("GOLDEN_BATCH_HEX") == f'"{p.pack_batch(session_id, samples).hex()}"'


def test_golden_status_is_what_protocol_py_packs():
    v = _object("GOLDEN_STATUS")
    st = p.Status(
        v["fwMajor"],
        v["fwMinor"],
        v["fwPatch"],
        v["sessionId"],
        v["gateState"],
        v["uptimeS"],
        v["calibrated"],
        v["sessionActive"],
    )
    assert _const("GOLDEN_STATUS_HEX") == f'"{p.pack_status(st).hex()}"'
