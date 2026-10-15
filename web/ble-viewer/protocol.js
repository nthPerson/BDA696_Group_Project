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
