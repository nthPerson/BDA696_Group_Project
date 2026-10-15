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

  /** Forget the seq baseline (call before every reconnect) but keep the plot. The time baseline
   * is kept on purpose: a kit that rebooted while disconnected comes back with a smaller t_ms,
   * and push() must still see that to drop the old clock. */
  resetLink() {
    this._lastSeq = null;
  }

  /** Empty the plot for a new source (a new clock). The counters are kept; see resetStats(). */
  clear() {
    this._start = 0;
    this.length = 0;
    this._lastSeq = null;
    this._lastT = null;
    this.version++;
  }

  /** Zero received/drops/badPackets/resets, for a newly chosen kit or the simulator. */
  resetStats() {
    this.received = 0;
    this.drops = 0;
    this.badPackets = 0;
    this.resets = 0;
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
