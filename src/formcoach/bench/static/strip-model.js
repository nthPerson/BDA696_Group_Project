// The strip charts' data, without the DOM: what strips.js draws. Times are seconds on the
// laptop's session clock (frame t_mono, IMU arrival time); event times (frame clock) are moved
// onto it with the latest frame's offset t_mono - t. The server judges every rep; this module
// only places the judgements on the time axis.
import { Series } from "./series.js";

const IMU_CAPACITY = 16000; // 300 s at 50 Hz
const FRAME_CAPACITY = 10000; // 300 s at ~30 fps
const MAX_REPS = 500;
const DEFAULT_RATE_HZ = 50;
const EPOCH_EVENTS = new Set(["counters_reset", "exercise"]); // the counters start again at rep 0

const finite = (x) => typeof x === "number" && Number.isFinite(x);
const counterOf = (arm) => (arm === "l" || arm === "r" ? arm : "all");

/**
 * The rule parameter whose metric the metric lane shows: `key` when it is a rule parameter of
 * `params`, else the first rule parameter; `null` when there is none. `limit` is the value in
 * force (`values[key]`, else the default); `exercise` the rule's exercise.
 * @returns {{key:string, exercise:string, code:string, metric:string, unit:string, limit:number}|null}
 */
export function metricChoice(params, values, key) {
  const rules = (params ?? []).filter((p) => p.group === "rules" && p.key.startsWith("rule."));
  const p = rules.find((x) => x.key === key) ?? rules[0];
  if (!p) return null;
  const [, exercise, code, ...metric] = p.key.split(".");
  return { key: p.key, exercise, code, metric: metric.join("."), unit: p.unit, limit: values?.[p.key] ?? p.default };
}

/** The charts' data model on one shared Timeline. */
export class StripData {
  /** @param {import("./series.js").Timeline} timeline */
  constructor(timeline, { imuCapacity = IMU_CAPACITY, frameCapacity = FRAME_CAPACITY, maxReps = MAX_REPS } = {}) {
    this.timeline = timeline;
    this.imu = new Series(["ax", "ay", "az", "gx", "gy", "gz"], imuCapacity, timeline);
    // band_hi/band_lo: the counter's enter/exit thresholds (deg), higher first, per arm
    this.frames = new Series(
      ["device", "energy", "p_active", "angle_l", "angle_r", "band_hi_l", "band_lo_l", "band_hi_r", "band_lo_r"],
      frameCapacity, timeline,
    );
    this.reps = []; // {arm, rep_id, exercise, epoch, t0, t1 (session clock, s), mark: good|warn|drop, faults, metrics}
    this.maxReps = maxReps;
    this.ops = {}; // "CODE metric" -> op ("gt", "lt", ...) learnt from the faults seen
    this.metric = null; // see setMetric
    this._epoch = 0;
    this._offset = 0; // t_mono - t of the latest frame with both times
    this._rateHz = DEFAULT_RATE_HZ;
    this._pActive = null;
  }

  /** Oldest time any series still holds (s), or the newest when empty. */
  get oldestT() {
    const t = Math.min(this.imu.oldestT, this.frames.oldestT);
    return Number.isFinite(t) ? t : this.timeline.newest;
  }

  /**
   * An `imu` message (parallel arrays; `t` the arrival time, s). The watch sends samples in
   * batches that arrive together, so a run of equal arrival times is spread back from that time
   * at the link's sample rate (never before the previous sample). Samples without a time are
   * skipped.
   */
  pushImu(msg) {
    const n = msg.t.length;
    const dt = 1 / this._rateHz;
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && msg.t[j + 1] === msg.t[i]) j++;
      const tArr = msg.t[i];
      if (finite(tArr)) {
        for (let k = i; k <= j; k++) {
          const t = Math.max(this.imu.latestT, tArr - (j - k) * dt);
          this.imu.push(t, [msg.ax[k], msg.ay[k], msg.az[k], msg.gx[k], msg.gy[k], msg.gz[k]]);
        }
      }
      i = j + 1;
    }
  }

  /** A frame header: gate bits, each counter's angle and band (squat's `all` counter shows as left). */
  pushFrame(h) {
    if (!finite(h.t_mono) || h.t_mono < this.frames.latestT) return;
    if (finite(h.t)) this._offset = h.t_mono - h.t;
    const c = h.counters ?? {};
    const arm = (k) => c[k] ?? (k === "l" ? c.all : null);
    const band = (s) => (s && finite(s.enter) && finite(s.exit)
      ? [Math.max(s.enter, s.exit), Math.min(s.enter, s.exit)] : [null, null]);
    const [l, r] = [arm("l"), arm("r")];
    this.frames.push(h.t_mono, [
      h.gates?.device ? 1 : 0,
      h.gates?.energy ? 1 : 0,
      this._pActive,
      l?.angle ?? null,
      r?.angle ?? null,
      ...band(l),
      ...band(r),
    ]);
  }

  /**
   * A server `event`: `rep` and `rep_dropped` become marks; `rejudged` rewrites the faults of
   * this epoch's reps; `counters_reset`, `exercise` and a capture that starts recording begin a
   * new epoch (rep ids start at 0 again, so older reps are no longer re-judged).
   */
  pushEvent(ev) {
    const p = ev.payload ?? {};
    if (ev.kind === "rep" || ev.kind === "rep_dropped") {
      const faults = p.faults ?? [];
      for (const f of faults) if (f.code && f.op) this.ops[`${f.code} ${f.metric}`] = f.op;
      const t1 = (finite(p.t_end) ? p.t_end : ev.t) + this._offset;
      const t0 = (finite(p.t_start) ? p.t_start : ev.t) + this._offset;
      this.reps.push({
        arm: counterOf(p.arm), rep_id: p.rep_id ?? null, exercise: p.exercise ?? null, epoch: this._epoch, t0, t1,
        mark: ev.kind === "rep_dropped" ? "drop" : faults.length ? "warn" : "good",
        faults, metrics: p.metrics ?? {},
      });
      if (this.reps.length > this.maxReps) this.reps.shift();
    } else if (ev.kind === "rejudged") {
      const now = new Map((p.reps ?? []).map((r) => [`${counterOf(r.arm)} ${r.rep_id}`, r.faults ?? []]));
      for (const r of this.reps) {
        if (r.epoch !== this._epoch || r.mark === "drop") continue;
        const faults = now.get(`${r.arm} ${r.rep_id}`);
        if (!faults) continue;
        for (const f of faults) if (f.code && f.op) this.ops[`${f.code} ${f.metric}`] = f.op;
        r.faults = faults;
        r.mark = faults.length ? "warn" : "good";
      }
    } else if (EPOCH_EVENTS.has(ev.kind) || (ev.kind === "capture" && p.state === "recording")) {
      this._epoch++;
    }
  }

  /** The `link` message's `status`: its sample rate spreads IMU batches; p_active (0-1) joins the gates lane. */
  pushLink(status) {
    const hz = status?.rate_hz;
    this._rateHz = finite(hz) && hz >= 10 && hz <= 400 ? hz : DEFAULT_RATE_HZ;
    this._pActive = finite(status?.p_active) ? status.p_active : null;
  }

  /** Which rule metric the metric lane shows (see {@link metricChoice}), or null for none. */
  setMetric(choice) {
    this.metric = choice;
  }

  /**
   * Everything the lanes draw for the timeline's window; x values are seconds relative to the
   * window's right edge (<= 0). `gates` and `angle` are uPlot data arrays; marks and metric bars
   * are rep spans [x0, x1].
   */
  view() {
    const { spanS } = this.timeline;
    const tEnd = this.timeline.tEnd;
    const rel = (d) => [d[0].map((t) => t - tEnd), ...d.slice(1)];
    const imu = rel(this.imu.window(tEnd, spanS));
    const fr = rel(this.frames.window(tEnd, spanS));
    const visible = this.reps.filter((r) => r.t1 >= tEnd - spanS && r.t0 <= tEnd);
    const marks = visible.map((r) => ({ arm: r.arm, mark: r.mark, x0: r.t0 - tEnd, x1: r.t1 - tEnd }));
    return {
      tEnd,
      spanS,
      behindS: Math.max(0, this.timeline.newest - tEnd),
      accel: imu.slice(0, 4),
      gyro: [imu[0], imu[4], imu[5], imu[6]],
      gates: fr.slice(0, 4),
      angle: [fr[0], ...fr.slice(4)],
      marks,
      metric: this._metricView(visible, tEnd),
    };
  }

  _metricView(visible, tEnd) {
    const m = this.metric;
    if (!m) return null;
    const bars = [];
    for (const r of visible) {
      const value = r.metrics?.[m.metric];
      if (r.mark === "drop" || !finite(value)) continue;
      if (m.exercise && r.exercise && r.exercise !== m.exercise) continue; // another exercise's rep
      bars.push({ arm: r.arm, x0: r.t0 - tEnd, x1: r.t1 - tEnd, value, flagged: r.faults.some((f) => f.code === m.code) });
    }
    const all = [0, m.limit, ...bars.map((b) => b.value)].filter(finite);
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = 0.15 * (hi - lo);
    if (hi > 0) hi += pad;
    if (lo < 0) lo -= pad;
    return { ...m, op: this.ops[`${m.code} ${m.metric}`] ?? null, bars, range: [lo, hi] };
  }
}
