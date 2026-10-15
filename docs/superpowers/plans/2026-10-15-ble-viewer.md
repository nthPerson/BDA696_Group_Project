# BLE Live Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static Web Bluetooth page that connects to a FormCoach wearable and shows live
accel/gyro traces, status and flags, sends control commands and records a serial-format CSV,
served locally by `formcoach viewer` and published to GitHub Pages.

**Architecture:** Plain ES modules under `web/ble-viewer/` with no build step. Each module
does one thing: `protocol.js` (wire format), `buffer.js` (ring + stats), `recorder.js` (CSV),
`sim.js` (synthetic wearable), `ble.js` (Web Bluetooth link), `charts.js` (uPlot), and
`app.js` (DOM wiring). The simulator and the BLE link hand the same raw `DataView`s to the
same decoder. Python only serves files (`formcoach.app.viewer`) and pins the JS protocol
copy against `protocol.py` in CI.

**Tech Stack:** HTML/CSS/ES2020 modules, Web Bluetooth, uPlot 1.6.32 (vendored, MIT), Node 20
`node:test` (local only), Python 3.11 `http.server`, typer, pytest, GitHub Actions Pages.

**Spec:** `docs/superpowers/specs/2026-10-15-ble-viewer-design.md`

**Deviation from the spec (intentional):** the spec lists one `app.js`. This plan splits it into
`ble.js`, `buffer.js`, `recorder.js`, `sim.js`, `charts.js` and a thin `app.js`, so the ring
buffer, recorder and simulator can be unit-tested in Node without a DOM. The behaviour is
unchanged.

## Global Constraints

- Protocol constants are copied verbatim from `src/formcoach/io/protocol.py`: service
  `7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001`, IMU `…0002…`, control `…0003…`, status `…0004…`;
  sample 19 B `<IhhhhhhBH`, header 4 B `<HBB`, batch ≤ 99 B, status 12 B `<BBBHBIBB`;
  accel 4096 LSB/g, gyro 32.8 LSB/dps; 50 Hz, 5 samples per notification.
- CSV header is exactly `t_ms,ax,ay,az,gx,gy,gz,flags,seq` (raw LSB), i.e. `FC_SERIAL_CSV_HEADER`.
- No npm install and no build step. `web/ble-viewer/package.json` exists only to mark `.js` as ES
  modules for `node --test`.
- The Node test is local only. It is not added to CI (Robert, 2026-10-15).
- No change to firmware, `protocol.h` or `protocol.py`.
- Never `cd` in Bash (session hook). Use absolute paths or `git -C`.
- Ruff line length 100. Pre-commit runs trailing-whitespace and end-of-file fixers, so exclude
  `web/ble-viewer/vendor/`.
- Commits are conventional, each ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **`seq` wraps 65535 → 0, and the first sample after a reconnect.** A wrap must not count as
   a drop, and neither must the jump after a reconnect. Pinned in Task 2 (`buffer.test.mjs`).
2. **The device reboots mid-stream (`t_ms` goes backwards).** The plot must clear rather than
   draw two clocks, and the jump is not counted as drops. Pinned in Task 2.
3. **Windows serves `.js` as `text/plain`.** Python's `mimetypes` reads the Windows registry,
   and browsers refuse module scripts with the wrong MIME type, which gives a blank page on
   teammates' Windows machines. Pinned in Task 4 (`test_viewer_server.py` checks the
   Content-Type).
4. **Overlapping GATT operations.** A second button click while a write is in flight gives
   Chrome's "GATT operation already in progress". Writes are serialized through a promise
   queue (Task 5, `ble.js`) and exercised by the fake-BLE Playwright smoke (Task 7).
5. **A reconnect doubles the listeners.** Re-adding a fresh closure on every reconnect would
   duplicate every packet. One handler function per link, reused (Task 5), is checked by the
   fake-BLE smoke in Task 7: about 10 pkt/s after a reconnect, not 20.

---

### Task 1: Protocol mirror (`protocol.js`) with CI pin test and Node decoder test

**Files:**
- Create: `web/ble-viewer/protocol.js`
- Create: `web/ble-viewer/package.json`
- Create: `web/ble-viewer/test/protocol.test.mjs`
- Test: `tests/test_web_protocol.py`

**Interfaces:**
- Produces: the constants `DEVICE_NAME_PREFIX, SERVICE_UUID, IMU_CHAR_UUID, CONTROL_CHAR_UUID,
  STATUS_CHAR_UUID, SAMPLE_RATE_HZ, BATCH_SAMPLES, SAMPLE_SIZE, BATCH_HEADER_SIZE,
  MAX_BATCH_SIZE, STATUS_SIZE, ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS, SERIAL_CSV_HEADER`; the
  frozen objects `Command {LED_GOOD_REP, LED_FAULT, START_SESSION, STOP_SESSION, CALIBRATE,
  PING}` and `Flag {GATE_STATE, SESSION_ACTIVE, BUTTON_PRESSED, CALIBRATED}`; and these functions:
  `decodeBatch(view: DataView) → {sessionId, samples: Sample[], error: string|null}` (never throws)
  `encodeBatch(sessionId, samples: Sample[]) → DataView` (throws RangeError outside 1..5)
  `decodeStatus(view) → {fwMajor, fwMinor, fwPatch, sessionId, gateState, uptimeS, calibrated,
  sessionActive, firmware: "M.m.p", error: null} | {error: string}`
  `encodeStatus(st) → DataView`, `hexToView(hex) → DataView`, `viewToHex(view) → string`.
  `Sample = {t_ms, ax, ay, az, gx, gy, gz, flags, seq}` (raw LSB integers).
  Golden vectors: `GOLDEN_SESSION_ID, GOLDEN_SAMPLES, GOLDEN_BATCH_HEX, GOLDEN_STATUS,
  GOLDEN_STATUS_HEX`.

- [ ] **Step 1: Write the failing pytest pin test** `tests/test_web_protocol.py`

```python
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
```

- [ ] **Step 2: Run it and check it fails**

Run: `uv run pytest tests/test_web_protocol.py -q`
Expected: 5 FAIL with `FileNotFoundError: …web/ble-viewer/protocol.js`

- [ ] **Step 3: Write `web/ble-viewer/protocol.js`**

```js
// FormCoach BLE protocol: the browser's copy of firmware/include/protocol.h and
// src/formcoach/io/protocol.py. tests/test_web_protocol.py checks every constant and both
// golden vectors below against protocol.py, so change all three files together.
// Wire format: little-endian, packed. Values stay raw LSB here; buffer.js converts units.

// ---- Identity ---------------------------------------------------------------------------------
export const DEVICE_NAME_PREFIX = "FormCoach-";
export const SERVICE_UUID = "7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001";
export const IMU_CHAR_UUID = "7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001";
export const CONTROL_CHAR_UUID = "7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001";
export const STATUS_CHAR_UUID = "7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001";

// ---- Sampling, sizes, scales ------------------------------------------------------------------
export const SAMPLE_RATE_HZ = 50;
export const BATCH_SAMPLES = 5;
export const SAMPLE_SIZE = 19;
export const BATCH_HEADER_SIZE = 4;
export const MAX_BATCH_SIZE = 99;
export const STATUS_SIZE = 12;
export const ACCEL_LSB_PER_G = 4096.0;
export const GYRO_LSB_PER_DPS = 32.8;
export const SERIAL_CSV_HEADER = "t_ms,ax,ay,az,gx,gy,gz,flags,seq";

// ---- Commands (control characteristic, 1 byte) and sample flag bits ---------------------------
export const Command = Object.freeze({
  LED_GOOD_REP: 0x01,
  LED_FAULT: 0x02,
  START_SESSION: 0x03,
  STOP_SESSION: 0x04,
  CALIBRATE: 0x05,
  PING: 0x10,
});

export const Flag = Object.freeze({
  GATE_STATE: 0x01,
  SESSION_ACTIVE: 0x02,
  BUTTON_PRESSED: 0x04,
  CALIBRATED: 0x08,
});

// ---- Golden vectors: negative values, every flag bit, a seq wrap, t_ms at the uint32 max. ------
// GOLDEN_BATCH_HEX == protocol.pack_batch(GOLDEN_SESSION_ID, GOLDEN_SAMPLES).hex() in Python.
export const GOLDEN_SESSION_ID = 7;
export const GOLDEN_SAMPLES = Object.freeze([
  { t_ms: 1000, ax: 0, ay: 0, az: 4096, gx: 0, gy: 0, gz: 0, flags: 0, seq: 65534 },
  { t_ms: 1020, ax: -4096, ay: 2048, az: -1, gx: -32768, gy: 328, gz: 32767, flags: 1, seq: 65535 },
  { t_ms: 1040, ax: 32767, ay: -32768, az: 100, gx: -328, gy: 0, gz: 1, flags: 2, seq: 0 },
  { t_ms: 1060, ax: 1, ay: -1, az: 4095, gx: 10, gy: -10, gz: 0, flags: 4, seq: 1 },
  { t_ms: 4294967295, ax: -2, ay: 3, az: -4, gx: 5, gy: -6, gz: 7, flags: 9, seq: 2 },
]);
export const GOLDEN_BATCH_HEX = "07000500e803000000000000001000000000000000fefffc03000000f00008ffff00804801ff7f01ffff10040000ff7f00806400b8fe00000100020000240400000100ffffff0f0a00f6ff0000040100fffffffffeff0300fcff0500faff0700090200";
export const GOLDEN_STATUS = Object.freeze({
  fwMajor: 1,
  fwMinor: 1,
  fwPatch: 0,
  sessionId: 3,
  gateState: 1,
  uptimeS: 312,
  calibrated: 1,
  sessionActive: 1,
});
export const GOLDEN_STATUS_HEX = "010100030001380100000101";

/**
 * Decode one IMU notification (FcBatch). Never throws: a short packet comes back with `error`.
 * @param {DataView} view characteristic value (Web Bluetooth may hand a non-zero byteOffset)
 * @returns {{sessionId: number, samples: object[], error: string|null}} samples are
 *   {t_ms, ax, ay, az, gx, gy, gz, flags, seq} in raw LSB (accel 4096/g, gyro 32.8/dps)
 */
export function decodeBatch(view) {
  if (view.byteLength < BATCH_HEADER_SIZE) {
    return { sessionId: 0, samples: [], error: `packet too short for header: ${view.byteLength} bytes` };
  }
  const sessionId = view.getUint16(0, true);
  const n = view.getUint8(2);
  const expected = BATCH_HEADER_SIZE + n * SAMPLE_SIZE;
  if (view.byteLength < expected) {
    return {
      sessionId,
      samples: [],
      error: `packet declares ${n} samples (${expected} bytes) but has ${view.byteLength}`,
    };
  }
  const samples = [];
  for (let i = 0; i < n; i++) samples.push(decodeSample(view, BATCH_HEADER_SIZE + i * SAMPLE_SIZE));
  return { sessionId, samples, error: null };
}

function decodeSample(v, o) {
  return {
    t_ms: v.getUint32(o, true),
    ax: v.getInt16(o + 4, true),
    ay: v.getInt16(o + 6, true),
    az: v.getInt16(o + 8, true),
    gx: v.getInt16(o + 10, true),
    gy: v.getInt16(o + 12, true),
    gz: v.getInt16(o + 14, true),
    flags: v.getUint8(o + 16),
    seq: v.getUint16(o + 17, true),
  };
}

/** Pack 1..BATCH_SAMPLES raw samples into an FcBatch exactly as the firmware does. */
export function encodeBatch(sessionId, samples) {
  if (samples.length < 1 || samples.length > BATCH_SAMPLES) {
    throw new RangeError(`batch must hold 1..${BATCH_SAMPLES} samples, got ${samples.length}`);
  }
  const v = new DataView(new ArrayBuffer(BATCH_HEADER_SIZE + samples.length * SAMPLE_SIZE));
  v.setUint16(0, sessionId, true);
  v.setUint8(2, samples.length);
  v.setUint8(3, 0);
  samples.forEach((s, i) => {
    const o = BATCH_HEADER_SIZE + i * SAMPLE_SIZE;
    v.setUint32(o, s.t_ms, true);
    v.setInt16(o + 4, s.ax, true);
    v.setInt16(o + 6, s.ay, true);
    v.setInt16(o + 8, s.az, true);
    v.setInt16(o + 10, s.gx, true);
    v.setInt16(o + 12, s.gy, true);
    v.setInt16(o + 14, s.gz, true);
    v.setUint8(o + 16, s.flags);
    v.setUint16(o + 17, s.seq, true);
  });
  return v;
}

/** Decode the 12-byte FcStatus (status characteristic). Never throws. */
export function decodeStatus(view) {
  if (view.byteLength < STATUS_SIZE) return { error: `status packet too short: ${view.byteLength} bytes` };
  const st = {
    fwMajor: view.getUint8(0),
    fwMinor: view.getUint8(1),
    fwPatch: view.getUint8(2),
    sessionId: view.getUint16(3, true),
    gateState: view.getUint8(5),
    uptimeS: view.getUint32(6, true),
    calibrated: view.getUint8(10),
    sessionActive: view.getUint8(11),
  };
  return { ...st, firmware: `${st.fwMajor}.${st.fwMinor}.${st.fwPatch}`, error: null };
}

/** Pack an FcStatus (used by the simulator). */
export function encodeStatus(st) {
  const v = new DataView(new ArrayBuffer(STATUS_SIZE));
  v.setUint8(0, st.fwMajor);
  v.setUint8(1, st.fwMinor);
  v.setUint8(2, st.fwPatch);
  v.setUint16(3, st.sessionId, true);
  v.setUint8(5, st.gateState);
  v.setUint32(6, st.uptimeS, true);
  v.setUint8(10, st.calibrated);
  v.setUint8(11, st.sessionActive);
  return v;
}

export function hexToView(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return new DataView(bytes.buffer);
}

export function viewToHex(view) {
  let out = "";
  for (let i = 0; i < view.byteLength; i++) out += view.getUint8(i).toString(16).padStart(2, "0");
  return out;
}
```

- [ ] **Step 4: Run the pin test and check it passes**

Run: `uv run pytest tests/test_web_protocol.py -q`
Expected: `5 passed`

- [ ] **Step 5: Write `web/ble-viewer/package.json` and the Node test**

`web/ble-viewer/package.json`:
```json
{
  "private": true,
  "type": "module",
  "description": "Nothing to install. This file only tells Node the .js files are ES modules (node --test)."
}
```

`web/ble-viewer/test/protocol.test.mjs`:
```js
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  decodeBatch,
  decodeStatus,
  encodeBatch,
  encodeStatus,
  GOLDEN_BATCH_HEX,
  GOLDEN_SAMPLES,
  GOLDEN_SESSION_ID,
  GOLDEN_STATUS,
  GOLDEN_STATUS_HEX,
  hexToView,
  MAX_BATCH_SIZE,
  viewToHex,
} from "../protocol.js";

test("decodes the golden batch packed by protocol.py", () => {
  const out = decodeBatch(hexToView(GOLDEN_BATCH_HEX));
  assert.equal(out.error, null);
  assert.equal(out.sessionId, GOLDEN_SESSION_ID);
  assert.deepEqual(out.samples, GOLDEN_SAMPLES);
});

test("encodeBatch reproduces the golden bytes", () => {
  const v = encodeBatch(GOLDEN_SESSION_ID, GOLDEN_SAMPLES);
  assert.equal(v.byteLength, MAX_BATCH_SIZE);
  assert.equal(viewToHex(v), GOLDEN_BATCH_HEX);
  assert.throws(() => encodeBatch(1, []), RangeError);
});

test("a truncated packet returns an error and never throws", () => {
  const short = decodeBatch(hexToView(GOLDEN_BATCH_HEX.slice(0, -2)));
  assert.match(short.error, /declares 5 samples \(99 bytes\) but has 98/);
  assert.deepEqual(short.samples, []);
  assert.match(decodeBatch(new DataView(new ArrayBuffer(2))).error, /too short for header/);
});

test("decodes a DataView that starts in the middle of its buffer", () => {
  const golden = new Uint8Array(hexToView(GOLDEN_BATCH_HEX).buffer);
  const padded = new Uint8Array(3 + golden.length);
  padded.set(golden, 3);
  const out = decodeBatch(new DataView(padded.buffer, 3, golden.length));
  assert.deepEqual(out.samples, GOLDEN_SAMPLES);
});

test("status round trip matches protocol.py", () => {
  assert.deepEqual(decodeStatus(hexToView(GOLDEN_STATUS_HEX)), {
    ...GOLDEN_STATUS,
    firmware: "1.1.0",
    error: null,
  });
  assert.equal(viewToHex(encodeStatus(GOLDEN_STATUS)), GOLDEN_STATUS_HEX);
  assert.match(decodeStatus(new DataView(new ArrayBuffer(11))).error, /too short/);
});
```

- [ ] **Step 6: Run the Node test and check it passes**

Run: `node --test /home/robert/academics/BDA696_Group_Project/web/ble-viewer/test/protocol.test.mjs`
Expected: `# pass 5`, `# fail 0`

- [ ] **Step 7: Commit**

```bash
git -C <repo> add web/ble-viewer/protocol.js web/ble-viewer/package.json web/ble-viewer/test/protocol.test.mjs tests/test_web_protocol.py
git -C <repo> commit -m "feat: web BLE viewer protocol mirror pinned against protocol.py"
```

---

### Task 2: Ring buffer, rate meter and CSV recorder

**Files:**
- Create: `web/ble-viewer/buffer.js`, `web/ble-viewer/recorder.js`
- Test: `web/ble-viewer/test/buffer.test.mjs`, `web/ble-viewer/test/recorder.test.mjs`

**Interfaces:**
- Consumes: `ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS, SERIAL_CSV_HEADER` from `protocol.js`.
- Produces:
  `class SampleRing(capacity=500)` with fields `length, received, drops, badPackets, resets,
  version`, the getter `latest` (Sample|null), and the methods `push(sample)`, `resetLink()`,
  `clear()`, `dropRate() → 0..1` and `series() → {t, ax, ay, az, amag, gx, gy, gz}`
  (Float64Arrays, oldest to newest; `t` in s relative to the newest sample, ≤ 0; accel in g,
  gyro in dps).
  `class RateMeter(windowMs=2000)` with `mark(nowMs, count=1)` and `rate(nowMs) → events/s`.
  `class Recorder` with `active`, `count`, `start()`, `add(samples)`, `stop() → csv string`
  and `csv()`; `csvFilename(device: string, date: Date) → "<device>_YYYYMMDD-HHMMSS.csv"`.

- [ ] **Step 1: Write the failing tests**

`web/ble-viewer/test/buffer.test.mjs`:
```js
import assert from "node:assert/strict";
import { test } from "node:test";
import { RateMeter, SampleRing } from "../buffer.js";

const mk = (seq, t_ms) => ({ t_ms, ax: 4096, ay: 0, az: -4096, gx: 328, gy: 0, gz: 0, flags: 0, seq });

test("counts seq gaps, and the 65535 -> 0 wrap is not a drop", () => {
  const r = new SampleRing(10);
  [65534, 65535, 0, 3].forEach((seq, i) => r.push(mk(seq, 20 * i)));
  assert.equal(r.received, 4);
  assert.equal(r.drops, 2); // 1 and 2 are missing
  assert.equal(r.dropRate(), 2 / 6);
});

test("the first sample after resetLink (a reconnect) is not counted as drops", () => {
  const r = new SampleRing(10);
  r.push(mk(10, 200));
  r.resetLink();
  r.push(mk(500, 10200));
  assert.equal(r.drops, 0);
  assert.equal(r.length, 2); // the plot is kept across a reconnect
});

test("t_ms going backwards (device reboot) clears the plot without counting drops", () => {
  const r = new SampleRing(10);
  r.push(mk(100, 5000));
  r.push(mk(101, 5020));
  r.push(mk(0, 40));
  assert.equal(r.length, 1);
  assert.equal(r.drops, 0);
  assert.equal(r.resets, 1);
});

test("keeps the newest `capacity` samples, oldest first, time relative to the newest", () => {
  const r = new SampleRing(3);
  for (let i = 0; i < 5; i++) r.push(mk(i, 20 * i));
  assert.equal(r.length, 3);
  assert.equal(r.latest.seq, 4);
  const t = Array.from(r.series().t);
  assert.equal(t.length, 3);
  [-0.04, -0.02, 0].forEach((want, i) => assert.ok(Math.abs(t[i] - want) < 1e-12, `t[${i}]=${t[i]}`));
});

test("series converts raw LSB to g and dps and computes |a|", () => {
  const r = new SampleRing(4);
  r.push(mk(0, 0));
  const s = r.series();
  assert.equal(s.ax[0], 1);
  assert.equal(s.az[0], -1);
  assert.ok(Math.abs(s.amag[0] - Math.SQRT2) < 1e-12);
  assert.ok(Math.abs(s.gx[0] - 10) < 1e-12);
});

test("version changes on every push and clear (the render loop redraws on change)", () => {
  const r = new SampleRing(4);
  const v0 = r.version;
  r.push(mk(0, 0));
  assert.notEqual(r.version, v0);
  const v1 = r.version;
  r.clear();
  assert.notEqual(r.version, v1);
  assert.equal(r.length, 0);
});

test("RateMeter reports events per second over its window", () => {
  const m = new RateMeter(2000);
  for (let i = 0; i < 30; i++) m.mark(100 * i); // 10 per second for 3 s
  assert.equal(m.rate(2900), 10);
  assert.equal(m.rate(10000), 0); // stale events age out
});
```

`web/ble-viewer/test/recorder.test.mjs`:
```js
import assert from "node:assert/strict";
import { test } from "node:test";
import { GOLDEN_SAMPLES, SERIAL_CSV_HEADER } from "../protocol.js";
import { csvFilename, Recorder } from "../recorder.js";

test("records raw rows under the serial CSV header, only while active", () => {
  const r = new Recorder();
  r.add(GOLDEN_SAMPLES); // not started: ignored
  r.start();
  r.add(GOLDEN_SAMPLES.slice(0, 2));
  assert.equal(r.count, 2);
  const lines = r.stop().trimEnd().split("\n");
  assert.deepEqual(lines, [
    SERIAL_CSV_HEADER,
    "1000,0,0,4096,0,0,0,0,65534",
    "1020,-4096,2048,-1,-32768,328,32767,1,65535",
  ]);
  r.add(GOLDEN_SAMPLES); // stopped: ignored
  assert.equal(r.count, 2);
  assert.equal(r.active, false);
});

test("start() begins a fresh recording", () => {
  const r = new Recorder();
  r.start();
  r.add(GOLDEN_SAMPLES);
  r.stop();
  r.start();
  assert.equal(r.count, 0);
});

test("csvFilename is <device>_YYYYMMDD-HHMMSS.csv with unsafe characters replaced", () => {
  const d = new Date(2026, 9, 15, 9, 5, 7);
  assert.equal(csvFilename("FormCoach-8428", d), "FormCoach-8428_20261015-090507.csv");
  assert.equal(csvFilename("a b/c", d), "a_b_c_20261015-090507.csv");
  assert.equal(csvFilename(undefined, d), "FormCoach_20261015-090507.csv");
});
```

- [ ] **Step 2: Run them and check they fail**

Run: `node --test <repo>/web/ble-viewer/test/buffer.test.mjs <repo>/web/ble-viewer/test/recorder.test.mjs`
Expected: FAIL with `Cannot find module …/buffer.js` / `…/recorder.js`

- [ ] **Step 3: Write `web/ble-viewer/buffer.js`**

```js
// Last N decoded samples plus link statistics. Pure: no DOM, unit-tested with node --test.
import { ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS } from "./protocol.js";

const SEQ_MOD = 65536;

/** Fixed-capacity ring of raw samples (oldest overwritten) with drop accounting. */
export class SampleRing {
  constructor(capacity = 500) {
    this.capacity = capacity;
    this._buf = new Array(capacity);
    this._start = 0;
    this.length = 0;
    this.received = 0; // samples accepted since the page loaded
    this.drops = 0; // samples missing according to seq gaps
    this.badPackets = 0; // incremented by the caller for undecodable notifications
    this.resets = 0; // device reboots seen (t_ms went backwards)
    this.version = 0; // changes on every push/clear; the render loop redraws when it changes
    this._lastSeq = null;
    this._lastT = null;
  }

  /** Forget the seq/t baseline (call before every (re)connect) but keep the plot. */
  resetLink() {
    this._lastSeq = null;
    this._lastT = null;
  }

  clear() {
    this._start = 0;
    this.length = 0;
    this.resetLink();
    this.version++;
  }

  push(s) {
    if (this._lastT !== null && s.t_ms < this._lastT) {
      this.clear(); // the device rebooted: the old samples belong to another clock
      this.resets++;
    }
    if (this._lastSeq !== null) {
      const gap = (s.seq - this._lastSeq - 1 + SEQ_MOD) % SEQ_MOD;
      if (gap < SEQ_MOD / 2) this.drops += gap; // a larger "gap" is a duplicate or reorder
    }
    this._lastSeq = s.seq;
    this._lastT = s.t_ms;
    const idx = (this._start + this.length) % this.capacity;
    if (this.length < this.capacity) this.length++;
    else this._start = (this._start + 1) % this.capacity;
    this._buf[idx] = s;
    this.received++;
    this.version++;
  }

  get latest() {
    return this.length ? this._buf[(this._start + this.length - 1) % this.capacity] : null;
  }

  /** Fraction of expected samples that never arrived (0..1). */
  dropRate() {
    const expected = this.received + this.drops;
    return expected ? this.drops / expected : 0;
  }

  /** uPlot-ready columns, oldest to newest: t in s relative to the newest sample (<= 0),
   * accel in g, |a| in g, gyro in dps. */
  series() {
    const n = this.length;
    const cols = ["t", "ax", "ay", "az", "amag", "gx", "gy", "gz"];
    const out = Object.fromEntries(cols.map((c) => [c, new Float64Array(n)]));
    const t0 = n ? this.latest.t_ms : 0;
    for (let i = 0; i < n; i++) {
      const s = this._buf[(this._start + i) % this.capacity];
      const ax = s.ax / ACCEL_LSB_PER_G;
      const ay = s.ay / ACCEL_LSB_PER_G;
      const az = s.az / ACCEL_LSB_PER_G;
      out.t[i] = (s.t_ms - t0) / 1000;
      out.ax[i] = ax;
      out.ay[i] = ay;
      out.az[i] = az;
      out.amag[i] = Math.hypot(ax, ay, az);
      out.gx[i] = s.gx / GYRO_LSB_PER_DPS;
      out.gy[i] = s.gy / GYRO_LSB_PER_DPS;
      out.gz[i] = s.gz / GYRO_LSB_PER_DPS;
    }
    return out;
  }
}

/** Events per second over a sliding window (reads low for the first `windowMs`). */
export class RateMeter {
  constructor(windowMs = 2000) {
    this.windowMs = windowMs;
    this._events = []; // [timeMs, count]
  }

  mark(nowMs, count = 1) {
    this._events.push([nowMs, count]);
    this._trim(nowMs);
  }

  rate(nowMs) {
    this._trim(nowMs);
    let total = 0;
    for (const [, c] of this._events) total += c;
    return total / (this.windowMs / 1000);
  }

  _trim(nowMs) {
    while (this._events.length && this._events[0][0] <= nowMs - this.windowMs) this._events.shift();
  }
}
```

- [ ] **Step 4: Write `web/ble-viewer/recorder.js`**

```js
// Records raw samples and produces the same CSV the firmware prints over USB serial
// (FC_SERIAL_CSV_HEADER), so `protocol.parse_serial_line` and existing tooling read it unchanged.
import { SERIAL_CSV_HEADER } from "./protocol.js";

export class Recorder {
  constructor() {
    this._rows = [];
    this.active = false;
  }

  get count() {
    return this._rows.length;
  }

  start() {
    this._rows = [];
    this.active = true;
  }

  add(samples) {
    if (!this.active) return;
    for (const s of samples) {
      this._rows.push(`${s.t_ms},${s.ax},${s.ay},${s.az},${s.gx},${s.gy},${s.gz},${s.flags},${s.seq}`);
    }
  }

  stop() {
    this.active = false;
    return this.csv();
  }

  csv() {
    return `${[SERIAL_CSV_HEADER, ...this._rows].join("\n")}\n`;
  }
}

/** `<device>_YYYYMMDD-HHMMSS.csv` in local time, with characters outside [A-Za-z0-9_-] replaced. */
export function csvFilename(device, date = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  const stamp =
    `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-` +
    `${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
  const safe = (device || "FormCoach").replace(/[^A-Za-z0-9_-]+/g, "_");
  return `${safe}_${stamp}.csv`;
}
```

- [ ] **Step 5: Run the tests and check they pass**

Run: `node --test <repo>/web/ble-viewer/test/*.test.mjs`
Expected: `# fail 0` (5 + 7 + 3 = 15 pass)

- [ ] **Step 6: Commit**

```bash
git -C <repo> add web/ble-viewer/buffer.js web/ble-viewer/recorder.js web/ble-viewer/test/buffer.test.mjs web/ble-viewer/test/recorder.test.mjs
git -C <repo> commit -m "feat: web BLE viewer ring buffer, rate meter and serial-format CSV recorder"
```

---

### Task 3: Synthetic wearable (`sim.js`)

**Files:**
- Create: `web/ble-viewer/sim.js`
- Test: `web/ble-viewer/test/sim.test.mjs`

**Interfaces:**
- Consumes: `encodeBatch, encodeStatus, Command, Flag, BATCH_SAMPLES, SAMPLE_RATE_HZ,
  ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS` from `protocol.js`.
- Produces: `class Simulator({onPacket(view), onStatus(view), onDisconnect()}, seed=1)` with the
  same surface as `BleLink` (Task 5): `name` ("Simulator"), `hasControl` (true), `connected`
  (getter), `async connect()` (restarts device time at 0), `async reconnect()` (resumes),
  `async disconnect()` and `async send(cmd)`. It also has the pure helpers
  `sampleAt(tMs) → Sample`, `nextBatch() → DataView` (advances 100 ms) and
  `statusView() → DataView`, plus the fields `sessionId` and `sessionActive`.

- [ ] **Step 1: Write the failing test** `web/ble-viewer/test/sim.test.mjs`

```js
import assert from "node:assert/strict";
import { test } from "node:test";
import { SampleRing } from "../buffer.js";
import { Command, decodeBatch, decodeStatus, Flag } from "../protocol.js";
import { Simulator } from "../sim.js";

const quiet = { onPacket() {}, onStatus() {}, onDisconnect() {} };
const g = (s) => Math.hypot(s.ax, s.ay, s.az) / 4096;

test("simulated batches decode cleanly: 5 samples per 100 ms, continuous seq", () => {
  const sim = new Simulator(quiet);
  const ring = new SampleRing(500);
  for (let i = 0; i < 100; i++) {
    const { samples, error } = decodeBatch(sim.nextBatch());
    assert.equal(error, null);
    assert.equal(samples.length, 5);
    for (const s of samples) ring.push(s);
  }
  assert.equal(ring.received, 500);
  assert.equal(ring.drops, 0);
  assert.equal(ring.latest.t_ms, 9980);
});

test("about 1 g while still; the gate flag and the gyro follow the curl", () => {
  const sim = new Simulator(quiet);
  const still = sim.sampleAt(8000); // 6 s moving, then 4 s still
  assert.ok(Math.abs(g(still) - 1) < 0.03, `|a| = ${g(still)}`);
  assert.equal(still.flags & Flag.GATE_STATE, 0);
  const moving = sim.sampleAt(1500); // peak angular rate of the 0.5 Hz curl
  assert.ok(Math.abs(g(moving) - 1) < 0.03);
  assert.ok(moving.flags & Flag.GATE_STATE);
  assert.ok(Math.abs(moving.gx / 32.8) > 150, `gx = ${moving.gx / 32.8} dps`);
  assert.ok(still.flags & Flag.CALIBRATED);
});

test("START/STOP commands toggle the session flag and bump the session id", async () => {
  const statuses = [];
  const sim = new Simulator({ ...quiet, onStatus: (v) => statuses.push(decodeStatus(v)) });
  await sim.send(Command.START_SESSION);
  assert.ok(sim.sampleAt(0).flags & Flag.SESSION_ACTIVE);
  assert.equal(statuses.at(-1).sessionId, 1);
  assert.equal(statuses.at(-1).sessionActive, 1);
  await sim.send(Command.STOP_SESSION);
  assert.equal(sim.sampleAt(0).flags & Flag.SESSION_ACTIVE, 0);
  await sim.send(Command.PING); // accepted, no effect
  assert.equal(sim.sessionId, 1);
});
```

- [ ] **Step 2: Run it and check it fails**

Run: `node --test <repo>/web/ble-viewer/test/sim.test.mjs`
Expected: FAIL with `Cannot find module …/sim.js`

- [ ] **Step 3: Write `web/ble-viewer/sim.js`**

```js
// A synthetic wearable for demos without hardware, browsers without Web Bluetooth, and tests.
// It packs real FcBatch bytes with encodeBatch(), so every simulated packet goes through the
// same decodeBatch() path as a real notification.
import {
  ACCEL_LSB_PER_G,
  BATCH_SAMPLES,
  Command,
  encodeBatch,
  encodeStatus,
  Flag,
  GYRO_LSB_PER_DPS,
  SAMPLE_RATE_HZ,
} from "./protocol.js";

const DT_MS = 1000 / SAMPLE_RATE_HZ; // 20 ms
const BATCH_MS = DT_MS * BATCH_SAMPLES; // 100 ms, one notification
const MOVE_S = 6; // three 0.5 Hz "curls"...
const STILL_S = 4; // ...then rest
const CURL_HZ = 0.5;
const CURL_RAD = 2.0; // ~115 degrees of forearm rotation

/** Small seeded PRNG (mulberry32) so tests are reproducible. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const int16 = (v) => Math.max(-32768, Math.min(32767, Math.round(v)));

export class Simulator {
  constructor(callbacks, seed = 1) {
    this.cb = callbacks;
    this.name = "Simulator";
    this.hasControl = true;
    this.sessionId = 0;
    this.sessionActive = false;
    this._tMs = 0;
    this._seq = 0;
    this._noise = mulberry32(seed);
    this._timer = null;
  }

  get connected() {
    return this._timer !== null;
  }

  /** The raw sample at device time tMs: gravity rotating about x with the curl, plus noise. */
  sampleAt(tMs) {
    const cycle = (tMs / 1000) % (MOVE_S + STILL_S);
    const moving = cycle < MOVE_S;
    const w = 2 * Math.PI * CURL_HZ;
    // theta = A(1 - cos wt)/2 starts and ends each curl at rest (theta and rate both 0)
    const theta = moving ? (CURL_RAD * (1 - Math.cos(w * cycle))) / 2 : 0;
    const rate = moving ? (CURL_RAD * w * Math.sin(w * cycle)) / 2 : 0; // rad/s
    const n = () => 2 * this._noise() - 1;
    const flags =
      (moving ? Flag.GATE_STATE : 0) | (this.sessionActive ? Flag.SESSION_ACTIVE : 0) | Flag.CALIBRATED;
    return {
      t_ms: tMs >>> 0,
      ax: int16((0.01 * n()) * ACCEL_LSB_PER_G),
      ay: int16((Math.sin(theta) + 0.01 * n()) * ACCEL_LSB_PER_G),
      az: int16((Math.cos(theta) + 0.01 * n()) * ACCEL_LSB_PER_G),
      gx: int16(((rate * 180) / Math.PI + 0.5 * n()) * GYRO_LSB_PER_DPS),
      gy: int16(0.5 * n() * GYRO_LSB_PER_DPS),
      gz: int16(0.5 * n() * GYRO_LSB_PER_DPS),
      flags,
      seq: this._seq,
    };
  }

  /** The next 100 ms of samples as one encoded notification. */
  nextBatch() {
    const samples = [];
    for (let i = 0; i < BATCH_SAMPLES; i++) {
      samples.push(this.sampleAt(this._tMs));
      this._tMs += DT_MS;
      this._seq = (this._seq + 1) & 0xffff;
    }
    return encodeBatch(this.sessionId, samples);
  }

  statusView() {
    return encodeStatus({
      fwMajor: 0,
      fwMinor: 0,
      fwPatch: 0,
      sessionId: this.sessionId,
      gateState: this.sampleAt(this._tMs).flags & Flag.GATE_STATE ? 1 : 0,
      uptimeS: Math.floor(this._tMs / 1000),
      calibrated: 1,
      sessionActive: this.sessionActive ? 1 : 0,
    });
  }

  async connect() {
    this._tMs = 0;
    this._seq = 0;
    return this.reconnect();
  }

  async reconnect() {
    if (this._timer !== null) return;
    this._timer = setInterval(() => {
      this.cb.onPacket(this.nextBatch());
      if (this._tMs % 1000 === 0) this.cb.onStatus(this.statusView());
    }, BATCH_MS);
    this.cb.onStatus(this.statusView());
  }

  async disconnect() {
    if (this._timer === null) return;
    clearInterval(this._timer);
    this._timer = null;
    this.cb.onDisconnect();
  }

  async send(cmd) {
    if (cmd === Command.START_SESSION && !this.sessionActive) {
      this.sessionActive = true;
      this.sessionId = (this.sessionId + 1) & 0xffff;
    } else if (cmd === Command.STOP_SESSION) {
      this.sessionActive = false;
    }
    this.cb.onStatus(this.statusView());
  }
}
```

- [ ] **Step 4: Run all Node tests and check they pass**

Run: `node --test <repo>/web/ble-viewer/test/*.test.mjs`
Expected: `# fail 0` (18 pass)

- [ ] **Step 5: Commit**

```bash
git -C <repo> add web/ble-viewer/sim.js web/ble-viewer/test/sim.test.mjs
git -C <repo> commit -m "feat: synthetic wearable for the web BLE viewer (real FcBatch bytes)"
```

---

### Task 4: Local server: `formcoach viewer`, `make ble-viewer`, Node tests in `make test`

**Files:**
- Create: `src/formcoach/app/viewer.py`
- Modify: `src/formcoach/cli.py` (new `viewer` command after the `demo` command)
- Modify: `Makefile` (`.PHONY`, the `test` target, a new `ble-viewer` target)
- Test: `tests/test_viewer_server.py`

**Interfaces:**
- Produces: `formcoach.app.viewer.WEB_ROOT: Path`, `VIEWER_PATH = "/ble-viewer/"`,
  `make_server(port: int = 8000, root: Path = WEB_ROOT) -> ThreadingHTTPServer`
  (binds 127.0.0.1; port 0 picks a free one), `viewer_url(server) -> str`.
  CLI: `formcoach viewer [--port 8000] [--open/--no-open]`. Make: `make ble-viewer [VIEWER_PORT=8000]`.

- [ ] **Step 1: Write the failing test** `tests/test_viewer_server.py`

```python
"""`formcoach viewer` serves web/ on localhost with JavaScript MIME types browsers accept."""

import threading
from urllib.request import urlopen

from typer.testing import CliRunner

from formcoach.app.viewer import WEB_ROOT, make_server, viewer_url
from formcoach.cli import app


def test_server_serves_the_page_and_modules_as_javascript():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = viewer_url(server)
        assert url.startswith("http://localhost:") and url.endswith("/ble-viewer/")
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(f"{base}/ble-viewer/") as r:
            assert r.status == 200
            assert "FormCoach" in r.read().decode("utf-8")
        # Windows can map .js to text/plain in the registry; module scripts then refuse to load.
        with urlopen(f"{base}/ble-viewer/protocol.js") as r:
            assert r.headers["Content-Type"].startswith("text/javascript")
            assert r.headers["Cache-Control"] == "no-store"
    finally:
        server.shutdown()
        server.server_close()


def test_web_root_is_the_repo_web_dir():
    assert (WEB_ROOT / "ble-viewer" / "protocol.js").is_file()


def test_viewer_command_is_registered():
    result = CliRunner().invoke(app, ["viewer", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
```

Note: `/ble-viewer/` serves `index.html`, which Task 5 writes. To keep this task green on its
own, Step 3b creates a placeholder `index.html` that contains "FormCoach". Task 5 overwrites it.

- [ ] **Step 2: Run it and check it fails**

Run: `uv run pytest tests/test_viewer_server.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'formcoach.app.viewer'`

- [ ] **Step 3: Write `src/formcoach/app/viewer.py`**

```python
"""Serve the browser BLE live viewer (``web/``) on http://localhost.

Web Bluetooth only runs in a secure context: https (the GitHub Pages copy) or localhost (this
server). Standard library only, so it works from a plain ``make setup`` install.
"""

from __future__ import annotations

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[3] / "web"
VIEWER_PATH = "/ble-viewer/"


class _Handler(SimpleHTTPRequestHandler):
    # Explicit types: on Windows, mimetypes reads the registry, which can map .js to
    # text/plain, and browsers refuse to run module scripts served that way.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always serve the files on disk
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # one line per request would bury the URL the user needs


def make_server(port: int = 8000, root: Path = WEB_ROOT) -> ThreadingHTTPServer:
    """Serve ``root`` on 127.0.0.1:``port`` (0 picks a free port). Call ``serve_forever()``."""
    handler = functools.partial(_Handler, directory=str(root))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def viewer_url(server: ThreadingHTTPServer) -> str:
    """The viewer's URL. ``localhost`` (a secure context) is what WSL2 forwards to Windows."""
    return f"http://localhost:{server.server_address[1]}{VIEWER_PATH}"
```

(If ruff flags `A002`, that rule isn't selected in this repo, so drop the `noqa` instead.)

- [ ] **Step 3b: Write a placeholder `web/ble-viewer/index.html`** (Task 5 replaces it)

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FormCoach Live IMU</title></head>
<body><p>FormCoach live IMU viewer (placeholder)</p></body></html>
```

- [ ] **Step 4: Add the CLI command** to `src/formcoach/cli.py`, directly after the `demo`
  command and its helper `_open_source`, before `@app.command("record")`:

```python
@app.command("viewer")
def viewer(
    port: Annotated[int, typer.Option(help="Local port to serve on.")] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the page in the default browser.")
    ] = True,
) -> None:
    """Serve the browser BLE live viewer on http://localhost (Chrome or Edge).

    Web Bluetooth needs https or localhost; the hosted copy is on GitHub Pages. See
    docs/howto/ble-viewer.md.
    """
    import webbrowser

    from formcoach.app.viewer import make_server, viewer_url

    try:
        server = make_server(port)
    except OSError as err:
        console.print(f"[red]Cannot listen on port {port}: {err}[/] (try --port {port + 1})")
        raise typer.Exit(1) from err
    url = viewer_url(server)
    console.print(f"BLE viewer: [cyan]{url}[/]  (Chrome or Edge; Ctrl+C to stop)")
    console.print("  WSL: open that URL in Chrome/Edge on Windows (WSL2 forwards localhost).")
    console.print("  Hosted copy: https://nthperson.github.io/BDA696_Group_Project/ble-viewer/")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("viewer stopped")
    finally:
        server.server_close()
```

- [ ] **Step 5: Update the `Makefile`**

Add `ble-viewer` to `.PHONY`, add `VIEWER_PORT ?= 8000` beside `MODEL ?= cnn`, and replace the
`test` target and add `ble-viewer` after `demo`:

```make
test: ## run the pytest suite (+ the web viewer's JS tests when node is installed)
	$(RUN) pytest
	@if command -v node >/dev/null 2>&1; then node --test web/ble-viewer/test/*.test.mjs; \
	else echo "node not found: skipped the optional web/ble-viewer JS tests"; fi

ble-viewer: ## serve the browser BLE live viewer on http://localhost:8000/ble-viewer/ (VIEWER_PORT=8000)
	$(RUN) formcoach viewer --port $(VIEWER_PORT)
```

(`PORT` already means a serial port in this Makefile, hence `VIEWER_PORT`.)

- [ ] **Step 6: Run the tests and check they pass**

Run: `uv run pytest tests/test_viewer_server.py tests/test_cli.py tests/test_repo_contract.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git -C <repo> add src/formcoach/app/viewer.py src/formcoach/cli.py Makefile tests/test_viewer_server.py web/ble-viewer/index.html
git -C <repo> commit -m "feat: formcoach viewer serves the web BLE viewer on localhost; make ble-viewer"
```

---

### Task 5: The page: vendored uPlot, `charts.js`, `ble.js`, `index.html`, `app.js`

**Files:**
- Create: `web/ble-viewer/vendor/uPlot.iife.min.js`, `web/ble-viewer/vendor/uPlot.min.css`,
  `web/ble-viewer/vendor/LICENSE`, `web/ble-viewer/vendor/README.md`
- Create: `web/ble-viewer/charts.js`, `web/ble-viewer/ble.js`, `web/ble-viewer/app.js`
- Modify (overwrite placeholder): `web/ble-viewer/index.html`
- Modify: `.pre-commit-config.yaml` (top-level `exclude: ^web/ble-viewer/vendor/`)

**Interfaces:**
- Consumes: everything in Tasks 1–3. `uPlot` is a global from the IIFE build.
- Produces: `bluetoothAvailable() → bool`;
  `class BleLink({onPacket, onStatus, onDisconnect})` with `name`, `device`, `hasControl`,
  `connected`, `async connect()` (opens Chrome's chooser; needs a user click),
  `async reconnect()`, `async disconnect()` and `async send(cmd)` (serialized);
  `class Charts(accelEl, gyroEl)` with `update(series)`.
  DOM ids used by the Task 7 smoke: `connect, simulate, disconnect, device, pkt-rate, hz,
  drops, bad, unsupported, message, fw, uptime, calibrated, session, pill-gate, pill-button,
  pill-session, chart-accel, chart-gyro, ping, led-good, led-fault, session-toggle,
  calibrate, record, rec-count`.

- [ ] **Step 1: Vendor uPlot 1.6.32** (downloaded into its own empty scratch directory, extracted
  there, and only the three files copied out)

```bash
S=<scratchpad>/uplot && mkdir -p "$S" && npm pack uplot@1.6.32 --pack-destination "$S" --silent
tar -xzf "$S/uplot-1.6.32.tgz" -C "$S"
V=<repo>/web/ble-viewer/vendor && mkdir -p "$V"
cp "$S/package/dist/uPlot.iife.min.js" "$S/package/dist/uPlot.min.css" "$S/package/LICENSE" "$V/"
sha256sum "$V"/uPlot.iife.min.js "$V"/uPlot.min.css
```

Write `web/ble-viewer/vendor/README.md` with: the source (`npm pack uplot@1.6.32`, MIT,
https://github.com/leeoniya/uPlot), the two sha256 values printed above, and the bump procedure
(the same three commands with the new version, update this file, open the viewer and Simulate).

Add to the top of `.pre-commit-config.yaml` (after the comment line):
```yaml
exclude: ^web/ble-viewer/vendor/  # vendored minified files stay byte-identical to upstream
```

- [ ] **Step 2: Write `web/ble-viewer/ble.js`**

```js
// One Web Bluetooth connection to a FormCoach wearable. Callbacks receive raw DataViews, the
// same shape the Simulator produces. Chrome/Edge only, and only in a secure context.
import { CONTROL_CHAR_UUID, IMU_CHAR_UUID, SERVICE_UUID, STATUS_CHAR_UUID } from "./protocol.js";

export function bluetoothAvailable() {
  return typeof navigator !== "undefined" && "bluetooth" in navigator && !!navigator.bluetooth;
}

export class BleLink {
  constructor(callbacks) {
    this.cb = callbacks;
    this.device = null;
    this.hasControl = false;
    this._control = null;
    this._writes = Promise.resolve();
    // One function object per link: addEventListener ignores a repeated add of the same
    // function, so a reconnect (which may hand back the same characteristic objects) never
    // doubles the packet stream.
    this._onImu = (e) => this.cb.onPacket(e.target.value);
    this._onStatus = (e) => this.cb.onStatus(e.target.value);
    this._onGone = () => {
      this._control = null;
      this.hasControl = false;
      this.cb.onDisconnect();
    };
  }

  get name() {
    return this.device?.name || "FormCoach";
  }

  get connected() {
    return !!this.device?.gatt?.connected;
  }

  /** Open the browser's device chooser (must run inside a click handler), then connect. */
  async connect() {
    const device = await navigator.bluetooth.requestDevice({ filters: [{ services: [SERVICE_UUID] }] });
    if (this.device !== device) {
      device.addEventListener("gattserverdisconnected", this._onGone);
      this.device = device;
    }
    await this._open();
  }

  /** Reconnect to the device picked earlier, without the chooser. */
  async reconnect() {
    if (!this.device) return this.connect();
    await this._open();
  }

  async disconnect() {
    if (this.device?.gatt?.connected) this.device.gatt.disconnect(); // fires gattserverdisconnected
  }

  /** Write one command byte; writes are queued so two clicks never overlap a GATT operation. */
  send(cmd) {
    const op = this._writes.then(() => {
      if (!this._control) throw new Error("not connected, or this firmware has no control characteristic");
      return this._control.writeValueWithResponse(Uint8Array.of(cmd));
    });
    this._writes = op.catch(() => {});
    return op;
  }

  async _open() {
    const server = await this.device.gatt.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    const imu = await service.getCharacteristic(IMU_CHAR_UUID);
    imu.addEventListener("characteristicvaluechanged", this._onImu);
    await imu.startNotifications();
    this._control = await service.getCharacteristic(CONTROL_CHAR_UUID).catch(() => null);
    this.hasControl = !!this._control;
    const status = await service.getCharacteristic(STATUS_CHAR_UUID).catch(() => null);
    if (status) {
      status.addEventListener("characteristicvaluechanged", this._onStatus);
      await status.startNotifications().catch(() => {});
      this.cb.onStatus(await status.readValue());
    }
  }
}
```

- [ ] **Step 3: Write `web/ble-viewer/charts.js`**

```js
// Two scrolling uPlot strip charts (accel in g, gyro in dps) over the last 10 s.
/* global uPlot */

const HEIGHT = 220;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const fmt = (_u, v) => (v == null ? "–" : v.toFixed(2));

/** y range grows with the data but never shrinks below +-minAbs, so a still kit stays readable. */
const atLeast = (minAbs) => (_u, min, max) => [Math.min(-minAbs, min ?? 0), Math.max(minAbs, max ?? 0)];

function makeChart(el, title, unit, series, minAbs) {
  const axis = { stroke: cssVar("--muted"), grid: { stroke: cssVar("--grid"), width: 1 }, ticks: { show: false } };
  const opts = {
    title,
    width: el.clientWidth || 600,
    height: HEIGHT,
    scales: { x: { time: false, range: [-10, 0] }, y: { range: atLeast(minAbs) } },
    axes: [{ ...axis, label: "seconds ago" }, { ...axis, label: unit, size: 60 }],
    series: [
      { label: "t (s)", value: fmt },
      ...series.map(([label, color]) => ({ label, stroke: cssVar(color), width: 1.5, value: fmt })),
    ],
    cursor: { drag: { x: false, y: false } },
  };
  return new uPlot(opts, [[], ...series.map(() => [])], el);
}

export class Charts {
  constructor(accelEl, gyroEl) {
    this._els = [accelEl, gyroEl];
    this.accel = makeChart(
      accelEl,
      "Acceleration",
      "g",
      [["ax", "--c-x"], ["ay", "--c-y"], ["az", "--c-z"], ["|a|", "--c-mag"]],
      1.5,
    );
    this.gyro = makeChart(gyroEl, "Angular rate", "dps", [["gx", "--c-x"], ["gy", "--c-y"], ["gz", "--c-z"]], 200);
    new ResizeObserver(() => this.resize()).observe(accelEl);
  }

  /** @param {ReturnType<import("./buffer.js").SampleRing["series"]>} s */
  update(s) {
    this.accel.setData([s.t, s.ax, s.ay, s.az, s.amag]);
    this.gyro.setData([s.t, s.gx, s.gy, s.gz]);
  }

  resize() {
    this.accel.setSize({ width: this._els[0].clientWidth, height: HEIGHT });
    this.gyro.setSize({ width: this._els[1].clientWidth, height: HEIGHT });
  }
}
```

- [ ] **Step 4: Write `web/ble-viewer/index.html`** (overwrites the Task 4 placeholder)

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FormCoach Live IMU</title>
<link rel="stylesheet" href="vendor/uPlot.min.css">
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f6f7; --panel: #ffffff; --fg: #1b1b1f; --muted: #6a6a74; --border: #dcdce1;
    --grid: #ececf0; --accent: #2457d6; --ok: #15803d; --warn: #a16207; --err: #c62828;
    --c-x: #d6336c; --c-y: #2b8a3e; --c-z: #1c64d1; --c-mag: #1b1b1f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111114; --panel: #1a1a1f; --fg: #ececf1; --muted: #9b9ba8; --border: #30303a;
      --grid: #27272e; --accent: #7aa7ff; --ok: #4ade80; --warn: #facc15; --err: #f87171;
      --c-x: #f783ac; --c-y: #69db7c; --c-z: #74c0fc; --c-mag: #ececf1;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header, main, footer { max-width: 1100px; margin: 0 auto; padding: 12px 16px; }
  h1 { font-size: 18px; margin: 0 0 8px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
  .status { margin-top: 8px; color: var(--muted); }
  .status b, .stat b { color: var(--fg); font-variant-numeric: tabular-nums; }
  .stat { color: var(--muted); }
  .device { font-weight: 600; }
  button { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
           background: var(--panel); color: var(--fg); cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .pill { padding: 2px 10px; border-radius: 999px; border: 1px solid var(--border);
          font-size: 12px; font-weight: 600; letter-spacing: .04em; color: var(--muted); }
  .pill.on { background: var(--ok); border-color: var(--ok); color: #fff; }
  .banner, .message { margin-top: 8px; padding: 8px 12px; border-radius: 6px; border: 1px solid; }
  .banner, .message.error { border-color: var(--err); color: var(--err); }
  .message.warn { border-color: var(--warn); color: var(--warn); }
  .message.info { border-color: var(--border); color: var(--muted); }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
           padding: 8px; margin-bottom: 12px; }
  .spacer { flex: 1; }
  .uplot, .u-title, .u-legend { color: var(--fg); font-family: inherit; }
  .u-title { font-size: 14px; font-weight: 600; }
  footer { color: var(--muted); font-size: 12px; }
  footer a { color: var(--accent); }
</style>
<script src="vendor/uPlot.iife.min.js"></script>
<script type="module" src="app.js"></script>
</head>
<body>
<header>
  <h1>FormCoach live IMU</h1>
  <div class="row">
    <button id="connect" class="primary">Connect</button>
    <button id="simulate">Simulate</button>
    <button id="disconnect" disabled>Disconnect</button>
    <span id="device" class="device">not connected</span>
    <span class="stat"><b id="pkt-rate">0.0</b> pkt/s</span>
    <span class="stat"><b id="hz">0.0</b> Hz</span>
    <span class="stat">drops <b id="drops">0</b></span>
    <span class="stat">bad packets <b id="bad">0</b></span>
  </div>
  <div id="unsupported" class="banner" hidden></div>
  <div id="message" class="message" hidden></div>
  <div class="row status">
    <span>fw <b id="fw">–</b></span>
    <span>up <b id="uptime">–</b></span>
    <span>calibrated <b id="calibrated">–</b></span>
    <span>session <b id="session">–</b></span>
    <span id="pill-gate" class="pill" title="Gate open: the wearable thinks an exercise is in progress">GATE</span>
    <span id="pill-button" class="pill" title="Button held">BUTTON</span>
    <span id="pill-session" class="pill" title="Session active">SESSION</span>
  </div>
</header>
<main>
  <section class="panel"><div id="chart-accel"></div></section>
  <section class="panel"><div id="chart-gyro"></div></section>
  <section class="panel row">
    <button id="ping" data-needs-control disabled>Ping</button>
    <button id="led-good" data-needs-control disabled>LED: good rep</button>
    <button id="led-fault" data-needs-control disabled>LED: fault</button>
    <button id="session-toggle" data-needs-control disabled>Start session</button>
    <button id="calibrate" data-needs-control disabled>Calibrate…</button>
    <span class="spacer"></span>
    <span id="rec-count" class="stat"></span>
    <button id="record" disabled>● Record</button>
  </section>
</main>
<footer>
  Chrome or Edge on Windows, macOS or Android. Only one app can be connected to a kit at a time.
  Help: <a href="https://github.com/nthPerson/BDA696_Group_Project/blob/main/docs/howto/ble-viewer.md">docs/howto/ble-viewer.md</a>
</footer>
</body>
</html>
```

- [ ] **Step 5: Write `web/ble-viewer/app.js`**

```js
// FormCoach BLE live viewer: wires the page to a source (BleLink or Simulator), keeps the last
// 10 s in a SampleRing, and redraws on animation frames, decoupled from BLE notifications.
// Design: docs/superpowers/specs/2026-10-15-ble-viewer-design.md
import { BleLink, bluetoothAvailable } from "./ble.js";
import { RateMeter, SampleRing } from "./buffer.js";
import { Charts } from "./charts.js";
import { Command, decodeBatch, decodeStatus, Flag, SAMPLE_RATE_HZ } from "./protocol.js";
import { csvFilename, Recorder } from "./recorder.js";
import { Simulator } from "./sim.js";

const WINDOW_S = 10;
const CONNECT_HINT =
  "Only one app can be connected to a kit at a time: close ble-check.py, formcoach demo/record " +
  "and other viewer tabs. Is the antenna clipped on, and is the kit's LED blinking (advertising)?";
const MTU_HINT = "a full batch needs a negotiated MTU of at least 102 bytes";

const $ = (id) => document.getElementById(id);
const ring = new SampleRing(WINDOW_S * SAMPLE_RATE_HZ);
const packetRate = new RateMeter();
const sampleRate = new RateMeter();
const recorder = new Recorder();
const charts = new Charts($("chart-accel"), $("chart-gyro"));

let link = null; // the active source: BleLink or Simulator
let status = null; // last decoded FcStatus
let drawnVersion = -1;

const callbacks = {
  onPacket(view) {
    const now = performance.now();
    packetRate.mark(now);
    const { samples, error } = decodeBatch(view);
    if (error) {
      ring.badPackets++;
      showMessage(`Bad packet: ${error} (${MTU_HINT}).`, "warn");
      return;
    }
    for (const s of samples) ring.push(s);
    sampleRate.mark(now, samples.length);
    recorder.add(samples);
  },
  onStatus(view) {
    const st = decodeStatus(view);
    if (!st.error) status = st;
  },
  onDisconnect() {
    const kept = recorder.active ? " The recording is kept: press Stop to download it." : "";
    showMessage(`Disconnected at ${new Date().toLocaleTimeString()}.${kept}`, "warn");
    updateButtons();
  },
};

function showMessage(text, kind = "info") {
  const el = $("message");
  el.textContent = text;
  el.className = `message ${kind}`;
  el.hidden = false;
}

function clearMessage() {
  $("message").hidden = true;
}

function updateButtons() {
  const connected = !!link && link.connected;
  const ble = link instanceof BleLink;
  const sim = link instanceof Simulator;
  $("connect").textContent = ble && link.device && !connected ? "Reconnect" : "Connect";
  $("connect").disabled = !bluetoothAvailable() || !window.isSecureContext || (ble && connected);
  $("simulate").textContent = sim && connected ? "Stop simulation" : "Simulate";
  $("simulate").disabled = ble && connected;
  $("disconnect").disabled = !(ble && connected);
  for (const b of document.querySelectorAll("[data-needs-control]")) {
    b.disabled = !(connected && link.hasControl);
    b.title = connected && !link.hasControl ? "This firmware has no control characteristic" : "";
  }
  $("record").disabled = !connected && !recorder.active;
  $("record").textContent = recorder.active ? "■ Stop & download CSV" : "● Record";
  $("device").textContent = link ? `${link.name}${connected ? "" : " (disconnected)"}` : "not connected";
}

async function connectBle() {
  clearMessage();
  if (link instanceof Simulator) await link.disconnect();
  if (!(link instanceof BleLink)) {
    link = new BleLink(callbacks);
    status = null;
    ring.clear();
  }
  ring.resetLink();
  try {
    await (link.device ? link.reconnect() : link.connect());
    clearMessage();
  } catch (err) {
    const cancelled = err.name === "NotFoundError" && /cancel/i.test(err.message);
    if (!cancelled) showMessage(`Could not connect: ${err.message}. ${CONNECT_HINT}`, "error");
  }
  updateButtons();
}

async function toggleSimulation() {
  clearMessage();
  if (link instanceof Simulator && link.connected) {
    await link.disconnect();
  } else {
    link = new Simulator(callbacks);
    status = null;
    ring.clear();
    await link.connect();
  }
  updateButtons();
}

async function send(cmd, label) {
  try {
    await link.send(cmd);
    showMessage(`Sent ${label}.`, "info");
  } catch (err) {
    showMessage(`${label} failed: ${err.message}`, "error");
  }
}

function download(text, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const latestFlags = () => ring.latest?.flags ?? 0;

$("connect").addEventListener("click", connectBle);
$("simulate").addEventListener("click", toggleSimulation);
$("disconnect").addEventListener("click", () => link?.disconnect());
$("ping").addEventListener("click", () => send(Command.PING, "PING (the serial monitor prints '# pong')"));
$("led-good").addEventListener("click", () => send(Command.LED_GOOD_REP, "LED good rep"));
$("led-fault").addEventListener("click", () => send(Command.LED_FAULT, "LED fault"));
$("session-toggle").addEventListener("click", () =>
  latestFlags() & Flag.SESSION_ACTIVE
    ? send(Command.STOP_SESSION, "STOP_SESSION")
    : send(Command.START_SESSION, "START_SESSION"),
);
$("calibrate").addEventListener("click", () => {
  if (window.confirm("Lay the kit flat and keep it still for 3 s after you press OK.")) {
    send(Command.CALIBRATE, "CALIBRATE (keep it flat and still for 3 s)");
  }
});
$("record").addEventListener("click", () => {
  if (recorder.active) download(recorder.stop(), csvFilename(link?.name));
  else recorder.start();
  updateButtons();
});

function fmtUptime(s) {
  const m = Math.floor(s / 60);
  return m ? `${m} min ${s % 60} s` : `${s} s`;
}

function frame() {
  const now = performance.now();
  if (ring.version !== drawnVersion) {
    charts.update(ring.series());
    drawnVersion = ring.version;
  }
  $("pkt-rate").textContent = packetRate.rate(now).toFixed(1);
  $("hz").textContent = sampleRate.rate(now).toFixed(1);
  $("drops").textContent = ring.drops ? `${ring.drops} (${(100 * ring.dropRate()).toFixed(1)} %)` : "0";
  $("bad").textContent = String(ring.badPackets);
  const f = latestFlags();
  $("pill-gate").classList.toggle("on", !!(f & Flag.GATE_STATE));
  $("pill-button").classList.toggle("on", !!(f & Flag.BUTTON_PRESSED));
  $("pill-session").classList.toggle("on", !!(f & Flag.SESSION_ACTIVE));
  $("session-toggle").textContent = f & Flag.SESSION_ACTIVE ? "Stop session" : "Start session";
  $("fw").textContent = status ? (link instanceof Simulator ? "simulated" : status.firmware) : "–";
  $("uptime").textContent = status ? fmtUptime(status.uptimeS) : "–";
  $("calibrated").textContent = status ? (status.calibrated ? "yes" : "no") : "–";
  $("session").textContent = status ? `#${status.sessionId}${status.sessionActive ? " (active)" : ""}` : "–";
  $("rec-count").textContent = recorder.active ? `recording: ${recorder.count} samples` : "";
  requestAnimationFrame(frame);
}

function init() {
  const banner = $("unsupported");
  if (!window.isSecureContext) {
    banner.textContent =
      "This page must be opened over https (GitHub Pages) or from http://localhost (make ble-viewer). Simulate still works.";
    banner.hidden = false;
  } else if (!bluetoothAvailable()) {
    banner.textContent =
      "This browser has no Web Bluetooth. Use Chrome or Edge on Windows, macOS or Android. Simulate still works.";
    banner.hidden = false;
  }
  updateButtons();
  requestAnimationFrame(frame);
}

init();
```

- [ ] **Step 6: Smoke-check in a headless browser (Simulate)**

Start the server in the background: `uv run formcoach viewer --no-open --port 8765`.
Open `http://localhost:8765/ble-viewer/` with Playwright and check that the console has no
errors, then click `#simulate`, wait 5 s, and read the stats.
Expected: `#pkt-rate` ≈ 10.0, `#hz` ≈ 50.0, `#drops` = 0, `#bad` = 0, both charts drawn, the GATE
pill toggling with the 6 s/4 s cycle, `#fw` = "simulated". Take a screenshot.

- [ ] **Step 7: Commit**

```bash
git -C <repo> add web/ble-viewer .pre-commit-config.yaml
git -C <repo> commit -m "feat: web BLE live viewer page (Web Bluetooth, uPlot charts, controls, CSV, simulator)"
```

---

### Task 6: GitHub Pages deployment

**Files:**
- Create: `.github/workflows/pages.yml`
- Create: `web/index.html` (the site root redirects to the viewer)

- [ ] **Step 1: Write `web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FormCoach Tools</title>
<meta http-equiv="refresh" content="0; url=ble-viewer/">
</head>
<body><p><a href="ble-viewer/">FormCoach BLE live viewer</a></p></body>
</html>
```

- [ ] **Step 2: Write `.github/workflows/pages.yml`**

```yaml
# Publishes web/ (the BLE live viewer) to GitHub Pages:
# https://nthperson.github.io/BDA696_Group_Project/ble-viewer/
# One-time setup by a repo admin: Settings -> Pages -> Source: GitHub Actions.
name: Pages (BLE viewer)

on:
  push:
    branches: [main]
    paths: ["web/**", ".github/workflows/pages.yml"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}ble-viewer/
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 3: Validate the YAML**

Run: `uv run pre-commit run check-yaml --files .github/workflows/pages.yml`
Expected: `Passed`

- [ ] **Step 4: Commit**

```bash
git -C <repo> add .github/workflows/pages.yml web/index.html
git -C <repo> commit -m "chore: publish web/ (BLE viewer) to GitHub Pages on merge to main"
```

---

### Task 7: Fake-BLE browser smoke, docs and handover

**Files:**
- Create: `docs/howto/ble-viewer.md`
- Modify: `README.md` (Everyday commands table, plus a "See your kit live" line)
- Modify: `docs/DECISIONS.md` (append ADR-0026)
- Modify: `docs/STATUS.md` (new top entry)

- [ ] **Step 1: Fake-BLE smoke (Playwright, not committed).** Before the page loads, install a
  fake `navigator.bluetooth` with `page.addInitScript`. Its `requestDevice()` resolves a device
  named `FormCoach-TEST` whose IMU characteristic fires a `characteristicvaluechanged` every
  100 ms with a 99-byte batch built by an inline DataView encoder (seq continuous), whose
  status `readValue()` returns `hexToView("010100030001380100000101")` bytes, and whose control
  `writeValueWithResponse` records the bytes and rejects with
  `NetworkError("GATT operation already in progress.")` if a previous write is still pending
  (each write resolves after 50 ms). `gatt.disconnect()` dispatches `gattserverdisconnected`.
  Check that:
  1. Connect gives `#device` = `FormCoach-TEST`, `#fw` = `1.1.0`, pkt/s ≈ 10, Hz ≈ 50, drops 0.
  2. Clicking Ping, LED good and LED fault quickly in a row records the bytes `[0x10, 0x01, 0x02]`
     with no error message (the write queue works).
  3. Record, wait 1 s, then Disconnect gives the message "Disconnected at … The recording is
     kept". Stop gives a CSV download whose first line is `t_ms,ax,ay,az,gx,gy,gz,flags,seq`.
  4. Reconnect (without the chooser) gives pkt/s ≈ 10 again, **not 20** (no duplicate listeners),
     and drops 0 (`resetLink`).
  5. The console shows no errors at any point.

- [ ] **Step 2: Windows reachability check (WSL → Windows localhost)**

Run (with the server from Task 5 running):
`/mnt/c/Windows/System32/curl.exe -sI http://localhost:8765/ble-viewer/protocol.js`
Expected: `HTTP/1.0 200 OK` and `Content-Type: text/javascript`. If curl.exe is missing or WSL
uses mirrored networking, record the result in the howto as a (verify) note rather than
failing.

- [ ] **Step 3: Write `docs/howto/ble-viewer.md`** for Bryce and the team, in the style of
  `docs/howto/replay-demo.md`, with these sections:
  1. *What it is*: one paragraph and the Pages URL
     `https://nthperson.github.io/BDA696_Group_Project/ble-viewer/`.
  2. *Open it*: the hosted URL (no clone needed), or `make ble-viewer` /
     `uv run formcoach viewer` → `http://localhost:8000/ble-viewer/`. WSL: open the printed
     URL in Chrome/Edge on Windows.
  3. *Browser support* table: Windows Chrome/Edge yes; macOS Chrome/Edge yes, with the first-time
     macOS Bluetooth permission (System Settings → Privacy & Security → Bluetooth → allow
     the browser); Android Chrome yes; Linux Chrome needs
     `chrome://flags/#enable-experimental-web-platform-features` (verify); Safari, Firefox and
     iOS have no Web Bluetooth, so Simulate only.
  4. *Connect your kit*: power on, antenna clipped, LED blinking, Connect, pick
     `FormCoach-XXXX` in the chooser (only kits advertising the FormCoach service appear).
     One client at a time.
  5. *What you should see* (the bring-up checklist): flat and still → |a| ≈ 1.00 g and az ≈ ±1 g;
     rotate → gyro traces; shake → GATE lights; about 10 pkt/s, about 50 Hz, drops 0; the
     status strip shows fw and calibrated.
  6. *Controls*: what each button sends (byte values) and what to look for on the kit or the
     serial monitor (`# pong`, `# session start id=N`, LED flash, calibration needs it flat
     for 3 s).
  7. *Recording*: the CSV columns and units (raw LSB: accel ÷ 4096 = g, gyro ÷ 32.8 = dps). It
     has the same format as the serial CSV. It is not a team recording: `formcoach record`
     remains the tool for data collection (`data/team/`, MANIFEST). **Do not commit CSVs of a
     teammate.**
  8. *Troubleshooting* table: the spec §6 rows, plus "|a| ≈ 4 g at rest and the gyro flat at 0
     → firmware from before ADR-0025 (BMI160 range bug); reflash from main once `fw/imu-bringup`
     merges", plus "blank page from `make ble-viewer` → hard-refresh; the server sends no-store".
  9. *For maintainers*: the file map, keeping `protocol.js` in sync (the pytest pin, golden
     vectors from `protocol.pack_batch`), `make test` running the Node tests when Node is
     installed, bumping uPlot (`vendor/README.md`), and Pages deploying on merge to main.
  Include no LiPo text unless the doc gives hardware handling instructions. Section 4 does
  ("power on"), so add the battery note from `docs/03-hardware.md` §4 verbatim (CLAUDE.md
  rule 9): copy it exactly from that file.

- [ ] **Step 4: README.** Add to the Everyday commands table, after `make demo`:
  `| make ble-viewer | uv run formcoach viewer | browser live view of a kit over Bluetooth (Chrome/Edge); hosted at https://nthperson.github.io/BDA696_Group_Project/ble-viewer/ (docs/howto/ble-viewer.md) |`

- [ ] **Step 5: ADR-0026** (append to `docs/DECISIONS.md`, matching the existing heading format)

```markdown
## ADR-0026 · Browser BLE viewer: Web Bluetooth + vendored uPlot, no build step, GitHub Pages · 2026-10-15 · accepted
Context: the team needs to see a kit's live IMU stream for the prototype demo and for per-kit
bring-up, on macOS and Windows laptops (WSL2 has no Bluetooth), without installing anything.
Options: (a) a Python GUI on bleak + matplotlib/Streamlit, which needs `--extra device`, runs on
the Windows side for WSL users, and is slow at 50 Hz × 7 traces; (b) a web page with Chart.js
or Plotly from a CDN, which stutters with streaming traces and needs internet at the demo; (c)
plain ES modules + Web Bluetooth + uPlot vendored in the repo. Decision: (c), under
`web/ble-viewer/`, with no npm build. It is served by `formcoach viewer` on localhost (Web
Bluetooth needs a secure context) and deployed to GitHub Pages by `.github/workflows/pages.yml`.
`protocol.js` is a third mirror of the protocol, pinned against `protocol.py` by
`tests/test_web_protocol.py` (constants and golden packets). The Node unit tests run locally
via `make test` and are not in CI. Consequences: Chrome/Edge only (no Safari, Firefox or iOS),
and the page has a simulator for everyone else. uPlot is bumped by hand
(`web/ble-viewer/vendor/README.md`). ADR-0025 is on the unmerged `fw/imu-bringup` branch,
hence the number gap on main.
```

- [ ] **Step 6: STATUS entry** (new top entry in `docs/STATUS.md`, using the existing
  heading/section format): what changed (files, URL, `make ble-viewer`), Next (Robert enables
  Pages; Robert tests K1 from Windows Chrome per the spec §7.4; then finish `fw/imu-bringup`),
  open questions (Linux Chrome flag (verify)), blockers (none), and who should read what
  (Bryce: `docs/howto/ble-viewer.md`).

- [ ] **Step 7: Full gates**

Run: `make -C <repo> lint test`
Expected: ruff clean; pytest all pass; `node --test` `# fail 0`.

- [ ] **Step 8: Commit**

```bash
git -C <repo> add docs/howto/ble-viewer.md README.md docs/DECISIONS.md docs/STATUS.md
git -C <repo> commit -m "docs: BLE viewer howto, README, ADR-0026, STATUS"
```

---

### Task 8: Review and PR

- [ ] **Step 1:** Run the `formcoach-reviewer` subagent (Opus) on `git diff main...feat/ble-viewer`.
  Fix confirmed findings, then re-run `make lint test`.
- [ ] **Step 2:** `git -C <repo> push -u origin feat/ble-viewer`, then
  `gh pr create --base main` with a description a teammate can follow: what it is, how to try
  it (Simulate needs no hardware), the screenshot, the test evidence, Robert's two manual steps
  (enable Pages; the K1 hardware check) and the `🤖 Generated with [Claude Code]` footer.
