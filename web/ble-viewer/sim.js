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
      ax: int16(0.01 * n() * ACCEL_LSB_PER_G),
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
