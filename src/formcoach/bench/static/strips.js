// Stacked strip charts on one shared Timeline (uPlot, vendored; loaded as a page global).
// The push* methods only store data; the page calls render() from requestAnimationFrame.
// The enter/exit band, rep marks and metric bars are drawn with uPlot hooks on the lane's canvas.
/* global uPlot */
import { Series } from "./series.js";

const LANE_HEIGHT = 120;
const IMU_CAPACITY = 16000; // 300 s at 50 Hz
const FRAME_CAPACITY = 10000; // 300 s at ~30 fps
const MAX_REPS = 500;

export const ALL_LANES = ["accel", "gyro", "gates", "angle", "metrics"];

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const atLeast = (minAbs) => (_u, min, max) => [Math.min(-minAbs, min ?? 0), Math.max(minAbs, max ?? 0)];
const fmt = (_u, v) => (v == null ? "–" : v.toFixed(2));
// The gates lane stacks three traces in one plot: device bit high band, energy bit middle, p_active 0-1.
const DEVICE_BASE = 2.1;
const ENERGY_BASE = 1.1;
const lift = (base) => (v) => (v == null ? null : base + 0.8 * v);

/**
 * @param {HTMLElement} container one div per lane is appended to it
 * @param {import("./series.js").Timeline} timeline
 * @param {string[]} lanes subset of ALL_LANES to show (always in the order of ALL_LANES)
 */
export class StripCharts {
  constructor(container, timeline, lanes = ALL_LANES) {
    this.timeline = timeline;
    this.imu = new Series(["ax", "ay", "az", "gx", "gy", "gz"], IMU_CAPACITY, timeline);
    this.frames = new Series(["device", "energy", "p_active", "angle_l", "angle_r"], FRAME_CAPACITY, timeline);
    this.reps = []; // {t (t_mono clock, s), mark: "good"|"warn"|"drop", ratio, sev}
    this.band = { l: null, r: null }; // latest [enter, exit] (deg) per counter
    this._clockOffset = 0; // t_mono - t of the latest frame: puts event times on the chart clock
    this._pActive = NaN;
    this._tEnd = 0;
    this.lanes = ALL_LANES.filter((n) => lanes.includes(n)).map((name) => this._makeLane(container, name));
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => this.resize()).observe(container);
  }

  /** An `imu` message (parallel arrays; `t` the arrival time, s). Samples without a time are skipped. */
  pushImu(msg) {
    for (let i = 0; i < msg.t.length; i++) {
      if (msg.t[i] == null) continue;
      this.imu.push(msg.t[i], [msg.ax[i], msg.ay[i], msg.az[i], msg.gx[i], msg.gy[i], msg.gz[i]]);
    }
  }

  /** A frame header: gate bits, primary angle per counter (a squat's single counter `all` shows as left). */
  pushFrame(h) {
    const c = h.counters ?? {};
    this.frames.push(h.t_mono, [
      h.gates?.device ? 1 : 0,
      h.gates?.energy ? 1 : 0,
      this._pActive,
      (c.l ?? c.all)?.angle ?? null,
      c.r?.angle ?? null,
    ]);
    for (const k of ["l", "r"]) {
      const s = c[k] ?? (k === "l" ? c.all : null);
      this.band[k] = s && s.enter != null && s.exit != null ? [s.enter, s.exit] : null;
    }
    this._clockOffset = h.t_mono - h.t;
  }

  /** A server `event` message: `rep` and `rep_dropped` become marks; flagged reps also get a metric bar. */
  pushEvent(ev) {
    const t = ev.t + this._clockOffset;
    if (ev.kind === "rep_dropped") this.reps.push({ t, mark: "drop", ratio: null, sev: null });
    else if (ev.kind === "rep") {
      const faults = ev.payload?.faults ?? [];
      const top = faults.find((f) => f.severity === "warn") ?? faults[0];
      const ratio = top && top.threshold ? top.value / top.threshold : null;
      this.reps.push({ t, mark: faults.length ? "warn" : "good", ratio, sev: top?.severity ?? null });
    } else return;
    if (this.reps.length > MAX_REPS) this.reps.shift();
  }

  /** The `link` message's `status`: remembers p_active (0-1) for the gates lane. */
  pushLink(status) {
    this._pActive = status?.p_active ?? NaN;
  }

  /** Redraw every lane for the timeline's current window. Call from requestAnimationFrame. */
  render() {
    const { spanS } = this.timeline;
    const tEnd = this.timeline.tEnd;
    const rel = (d) => [d[0].map((t) => t - tEnd), ...d.slice(1)]; // x = seconds ago (<= 0)
    const imu = rel(this.imu.window(tEnd, spanS));
    const fr = rel(this.frames.window(tEnd, spanS));
    const data = {
      accel: imu.slice(0, 4),
      gyro: [imu[0], imu[4], imu[5], imu[6]],
      gates: [fr[0], fr[1].map(lift(DEVICE_BASE)), fr[2].map(lift(ENERGY_BASE)), fr[3]],
      angle: [fr[0], fr[4], fr[5]],
      metrics: [[-spanS, 0]],
    };
    this._tEnd = tEnd;
    for (const lane of this.lanes) lane.u.setData(data[lane.name]);
  }

  resize() {
    for (const lane of this.lanes) lane.u.setSize({ width: lane.el.clientWidth || 600, height: LANE_HEIGHT });
  }

  _makeLane(container, name) {
    const el = document.createElement("div");
    el.className = "strip-lane";
    container.appendChild(el);
    const axis = { stroke: cssVar("--muted"), grid: { stroke: cssVar("--grid"), width: 1 }, ticks: { show: false } };
    const line = (label, color, extra = {}) => ({ label, stroke: cssVar(color), width: 1.5, value: fmt, ...extra });
    const stepped = uPlot.paths.stepped({ align: 1 });
    const spec = {
      accel: { unit: "m/s²", y: atLeast(12), series: [line("ax", "--cx"), line("ay", "--cy"), line("az", "--cz")] },
      gyro: { unit: "rad/s", y: atLeast(3), series: [line("gx", "--cx"), line("gy", "--cy"), line("gz", "--cz")] },
      gates: {
        unit: "gates",
        y: [-0.1, 3.1],
        series: [
          line("device gate", "--accent", { paths: stepped }),
          line("energy gate", "--warn", { paths: stepped }),
          line("p_active", "--ok"),
        ],
      },
      angle: { unit: "deg", y: [0, 180], series: [line("left angle", "--cx"), line("right angle", "--cz")] },
      metrics: { unit: "× limit", y: [0, 2], series: [] },
    }[name];
    const hooks = {
      angle: { drawClear: [(u) => this._drawBand(u)], draw: [(u) => this._drawMarks(u)] },
      metrics: { draw: [(u) => this._drawMetricBars(u)] },
    }[name] ?? {};
    const u = new uPlot(
      {
        width: el.clientWidth || 600,
        height: LANE_HEIGHT,
        legend: { show: spec.series.length > 0 },
        scales: { x: { time: false, range: () => [-this.timeline.spanS, 0] }, y: { range: spec.y } },
        axes: [{ ...axis, label: "seconds ago" }, { ...axis, label: spec.unit, size: 60 }],
        series: [{ label: "seconds ago", value: fmt }, ...spec.series],
        cursor: { drag: { x: false, y: false } },
        hooks,
      },
      [[], ...spec.series.map(() => [])],
      el,
    );
    return { name, el, u };
  }

  // Shaded enter/exit band of each counter, drawn under the angle lines.
  _drawBand(u) {
    const { ctx, bbox } = u;
    ctx.save();
    ctx.fillStyle = "rgba(122,167,255,0.12)";
    for (const b of Object.values(this.band)) {
      if (!b) continue;
      const y1 = u.valToPos(Math.max(...b), "y", true);
      const y2 = u.valToPos(Math.min(...b), "y", true);
      ctx.fillRect(bbox.left, y1, bbox.width, y2 - y1);
    }
    ctx.restore();
  }

  // Rep marks: a faint vertical line and a dot (green good, amber flagged, grey dropped).
  _drawMarks(u) {
    const { ctx, bbox } = u;
    const colour = { good: cssVar("--ok"), warn: cssVar("--warn"), drop: cssVar("--muted") };
    ctx.save();
    for (const r of this.reps) {
      const x = u.valToPos(r.t - this._tEnd, "x", true);
      if (x < bbox.left || x > bbox.left + bbox.width) continue;
      ctx.fillStyle = colour[r.mark];
      ctx.strokeStyle = colour[r.mark];
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, bbox.top);
      ctx.lineTo(x, bbox.top + bbox.height);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(x, bbox.top + 7, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  // One bar per flagged rep: value / limit of its most severe fault, against a dashed limit line at 1.
  _drawMetricBars(u) {
    const { ctx, bbox } = u;
    const yLimit = u.valToPos(1, "y", true);
    ctx.save();
    ctx.strokeStyle = cssVar("--err");
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(bbox.left, yLimit);
    ctx.lineTo(bbox.left + bbox.width, yLimit);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const r of this.reps) {
      if (r.ratio == null) continue;
      const x = u.valToPos(r.t - this._tEnd, "x", true);
      if (x < bbox.left || x > bbox.left + bbox.width) continue;
      const top = u.valToPos(Math.min(2, Math.max(0, r.ratio)), "y", true);
      ctx.fillStyle = r.sev === "warn" ? cssVar("--warn") : cssVar("--accent");
      ctx.fillRect(x - 5, top, 10, bbox.top + bbox.height - top);
    }
    ctx.restore();
  }
}
