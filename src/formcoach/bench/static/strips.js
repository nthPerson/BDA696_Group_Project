// Stacked strip charts on one shared time axis (uPlot, vendored; loaded as a page global before
// this module runs). The data lives in a StripData (strip-model.js, no DOM); this module only
// draws it. Each lane is a grid row: an HTML label on the left, a uPlot chart on the right
// without its own x axis; one axis strip under the lanes shows the time for all of them.
// The page calls render() from requestAnimationFrame.
/* global uPlot */

export const ALL_LANES = ["accel", "gyro", "gates", "angle", "metrics"];

const Y_AXIS_PX = 46; // every lane's y axis is this wide, so the plots line up with the time axis
const AXIS_STRIP_PX = 22;
const ROW = { accel: "1.1fr", gyro: "1fr", gates: "0.8fr", angle: "1.4fr", metrics: "0.9fr" };
// The gates lane stacks its traces in one plot: watch bit (top), energy gate bit, model p (bottom).
const BASE = { device: 2.1, energy: 1.1, p: 0.1 };
const lift = (base) => (v) => (v == null ? null : base + 0.8 * v);
const MARK_GLYPH = { good: "✓", warn: "!", drop: "×" };
const OP_WORD = { gt: "above", ge: "above", lt: "below", le: "below" };

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const atLeast = (minAbs) => (_u, min, max) => [Math.min(-minAbs, min ?? 0), Math.max(minAbs, max ?? 0)];

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

/** Seconds relative to now as axis text: "now", "-5 s". */
function agoText(v) {
  const r = Math.round(v * 10) / 10;
  return Math.abs(r) < 0.05 ? "now" : `${r} s`;
}

/**
 * @param {HTMLElement} container the lanes and the time axis are appended to it (a CSS grid)
 * @param {import("./strip-model.js").StripData} data
 * @param {string[]} lanes subset of ALL_LANES to show (always in the order of ALL_LANES)
 */
export class StripCharts {
  constructor(container, data, lanes = ALL_LANES) {
    this.data = data;
    this.container = container;
    this.colours = {
      muted: cssVar("--muted"), grid: cssVar("--grid"), ok: cssVar("--ok"), warn: cssVar("--warn"),
      err: cssVar("--err"), accent: cssVar("--accent"), cx: cssVar("--cx"), cy: cssVar("--cy"),
      cz: cssVar("--cz"), off: cssVar("--off"),
    };
    this.view = null; // the StripData view drawn last
    this._hasModel = false;
    this.lanes = ALL_LANES.filter((n) => lanes.includes(n)).map((name) => this._makeLane(name));
    this.axis = this._makeAxis();
    container.style.gridTemplateRows = [...this.lanes.map((l) => ROW[l.name]), `${AXIS_STRIP_PX}px`].join(" ");
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  /** A lane's label element (title `b` and a key line `span`), for the page to adjust. */
  labelOf(name) {
    return this.lanes.find((l) => l.name === name)?.label ?? null;
  }

  /** Redraw every lane for the timeline's current window. */
  render() {
    const v = this.data.view();
    this.view = v;
    const hasModel = v.gates[3].some((x) => x != null);
    if (hasModel !== this._hasModel) {
      this._hasModel = hasModel;
      const key = this.labelOf("gates")?.querySelector(".model");
      if (key) key.hidden = !hasModel;
    }
    const xs = v.gates[0];
    const series = {
      accel: v.accel,
      gyro: v.gyro,
      gates: [xs, v.gates[1].map(lift(BASE.device)), v.gates[2].map(lift(BASE.energy)), v.gates[3].map(lift(BASE.p))],
      angle: v.angle,
      metrics: [[-v.spanS, 0], [null, null]],
    };
    for (const lane of this.lanes) lane.u.setData(series[lane.name]);
    // the axis strip's x is seconds relative to now (the lanes' is relative to the window's
    // right edge): its range moves when scrolled back, so uPlot recomputes the tick labels
    this.axis.u.setData([[-v.spanS - v.behindS, -v.behindS], [null, null]]);
  }

  resize() {
    for (const lane of [...this.lanes, this.axis]) {
      const w = Math.max(50, Math.floor(lane.cell.clientWidth));
      const h = Math.max(lane === this.axis ? AXIS_STRIP_PX : 20, Math.floor(lane.cell.clientHeight));
      if (w !== lane.w || h !== lane.h) {
        lane.w = w;
        lane.h = h;
        lane.u.setSize({ width: w, height: h });
      }
    }
  }

  _yAxis(extra = {}) {
    const c = this.colours;
    return { size: Y_AXIS_PX, stroke: c.muted, grid: { stroke: c.grid, width: 1 }, ticks: { show: false }, font: "10px system-ui, sans-serif", gap: 3, ...extra };
  }

  _plot(cell, series, y, extra = {}) {
    const span = () => [-this.data.timeline.spanS, 0];
    return new uPlot(
      {
        width: 300,
        height: 60,
        legend: { show: false },
        padding: [6, 8, 4, 0],
        scales: { x: { time: false, range: span }, y: { range: y } },
        axes: [{ show: false }, this._yAxis(extra.yAxis)],
        series: [{}, ...series],
        cursor: { sync: { key: "bench-strips" }, drag: { x: false, y: false }, points: { show: false } },
        ...(extra.opts ?? {}),
      },
      [[], ...series.map(() => [])],
      cell,
    );
  }

  _makeLane(name) {
    const c = this.colours;
    const row = el("div", "strip-lane");
    const label = el("div", "strip-label");
    const cell = el("div", "strip-plot");
    row.append(label, cell);
    this.container.appendChild(row);
    const line = (stroke, extra = {}) => ({ stroke, width: 1.5, ...extra });
    const stepped = uPlot.paths.stepped({ align: 1 });
    const bandLine = { stroke: "rgba(155,155,168,0.55)", width: 1, paths: stepped };
    let u;
    if (name === "accel" || name === "gyro") {
      const accel = name === "accel";
      label.innerHTML = `<b>${accel ? "Accel" : "Gyro"}</b><span><i class="k-cx">x</i> <i class="k-cy">y</i> <i class="k-cz">z</i> · ${accel ? "m/s²" : "rad/s"}</span>`;
      u = this._plot(cell, [line(c.cx), line(c.cy), line(c.cz)], atLeast(accel ? 12 : 3));
    } else if (name === "gates") {
      label.innerHTML = '<b>Gate</b><span><i class="k-ok">watch bit</i> · <i class="k-accent">laptop energy</i><i class="model" hidden> · <i class="k-cx">model p</i></i></span><span class="sub">high = open</span>';
      u = this._plot(cell, [line(c.ok, { paths: stepped, width: 2 }), line(c.accent, { paths: stepped, width: 2 }), line(c.cx)], [0, 3.05], {
        yAxis: {
          splits: () => [BASE.p + 0.4, BASE.energy + 0.4, BASE.device + 0.4],
          values: () => [this._hasModel ? "model" : "", "energy", "watch"],
          grid: { show: false },
        },
      });
    } else if (name === "angle") {
      label.innerHTML = '<b>Angle</b><span><i class="k-cx">L</i> <i class="k-cz">R</i> · enter/exit band · reps</span>';
      const fill = "rgba(122,167,255,0.08)";
      u = this._plot(cell, [line(c.cx, { width: 1.8 }), line(c.cz, { width: 1.8 }), bandLine, bandLine, bandLine, bandLine], [0, 200], {
        opts: { bands: [{ series: [3, 4], fill }, { series: [5, 6], fill }], hooks: { draw: [(p) => this._drawMarks(p)] } },
      });
    } else {
      label.innerHTML = "<b>Per-rep metric</b><span>vs limit</span>";
      u = this._plot(cell, [{ stroke: "transparent" }], () => this.view?.metric?.range ?? [0, 1], {
        opts: { hooks: { draw: [(p) => this._drawMetric(p)] } },
      });
    }
    return { name, row, label, cell, u, w: 0, h: 0 };
  }

  _makeAxis() {
    const row = el("div", "strip-lane strip-axis");
    const label = el("div", "strip-label", "<span>seconds</span>");
    const cell = el("div", "strip-plot");
    row.append(label, cell);
    this.container.appendChild(row);
    const c = this.colours;
    const u = new uPlot(
      {
        width: 300,
        height: AXIS_STRIP_PX,
        legend: { show: false },
        padding: [0, 8, 0, 0],
        cursor: { show: false },
        scales: { x: { time: false, range: (_u, min, max) => [min, max] }, y: { range: [0, 1] } },
        axes: [
          {
            size: AXIS_STRIP_PX, stroke: c.muted, grid: { show: false }, ticks: { show: true, stroke: c.grid, size: 4 },
            font: "10px system-ui, sans-serif", gap: 2, values: (_u, splits) => splits.map(agoText),
          },
          { size: Y_AXIS_PX, values: () => [], grid: { show: false }, ticks: { show: false }, stroke: "transparent" },
        ],
        series: [{}, { stroke: "transparent" }],
      },
      [[], []],
      cell,
    );
    return { name: "axis", row, label, cell, u, w: 0, h: 0 };
  }

  // Rep marks along the top of the angle lane: one row per arm (left above right), a bar over
  // the rep's span with a glyph: ✓ good (green), ! flagged (amber), × dropped (grey).
  _drawMarks(u) {
    const v = this.view;
    if (!v) return;
    const { ctx, bbox } = u;
    const px = devicePixelRatio || 1;
    const c = this.colours;
    const fill = { good: c.ok, warn: c.warn, drop: c.off };
    const rowH = 10 * px;
    ctx.save();
    ctx.font = `bold ${8 * px}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const m of v.marks) {
      const x0 = Math.max(bbox.left, u.valToPos(m.x0, "x", true));
      const x1 = Math.min(bbox.left + bbox.width, u.valToPos(m.x1, "x", true));
      if (x1 - x0 < 1) continue;
      const y = bbox.top + (m.arm === "r" ? rowH + 1 * px : 0);
      ctx.fillStyle = fill[m.mark];
      ctx.fillRect(x0 + px, y, x1 - x0 - 2 * px, rowH - px);
      if (x1 - x0 > 12 * px) {
        ctx.fillStyle = m.mark === "drop" ? "#ececf1" : "#111";
        ctx.fillText(`${m.arm === "all" ? "" : m.arm.toUpperCase()}${MARK_GLYPH[m.mark]}`, (x0 + x1) / 2, y + rowH / 2);
      }
    }
    ctx.restore();
  }

  // The chosen rule metric per rep against its limit: a bar from zero to the rep's value
  // (left arm in the left half of the rep's span, right arm in the right half), amber with "!"
  // when the server flagged the rep for that rule, green with "✓" when not; the limit as a
  // dashed line, and the side of the limit that gets flagged shaded and named, once a fault of
  // this rule has shown which side that is.
  _drawMetric(u) {
    const m = this.view?.metric;
    if (!m) return;
    const { ctx, bbox } = u;
    const px = devicePixelRatio || 1;
    const c = this.colours;
    const yLimit = u.valToPos(m.limit, "y", true);
    const y0 = u.valToPos(Math.min(Math.max(0, m.range[0]), m.range[1]), "y", true);
    ctx.save();
    const side = OP_WORD[m.op];
    if (side) {
      ctx.fillStyle = "rgba(248,113,113,0.07)";
      if (side === "above") ctx.fillRect(bbox.left, bbox.top, bbox.width, yLimit - bbox.top);
      else ctx.fillRect(bbox.left, yLimit, bbox.width, bbox.top + bbox.height - yLimit);
    }
    ctx.font = `bold ${9 * px}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    for (const b of m.bars) {
      let x0 = u.valToPos(b.x0, "x", true);
      let x1 = u.valToPos(b.x1, "x", true);
      const w = x1 - x0;
      if (b.arm === "l") x1 = x0 + w / 2;
      if (b.arm === "r") x0 += w / 2;
      x0 = Math.max(bbox.left, x0 + w * 0.06);
      x1 = Math.min(bbox.left + bbox.width, x1 - w * 0.06);
      if (x1 - x0 < 1) continue;
      const yv = u.valToPos(b.value, "y", true);
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = b.flagged ? c.warn : c.ok;
      ctx.fillRect(x0, Math.min(yv, y0), x1 - x0, Math.max(1, Math.abs(y0 - yv)));
      ctx.globalAlpha = 1;
      if (x1 - x0 >= 8 * px) {
        const up = yv <= y0;
        const inside = Math.abs(y0 - yv) >= 13 * px;
        ctx.fillStyle = inside ? "#111" : b.flagged ? c.warn : c.ok;
        ctx.textBaseline = up === inside ? "top" : "bottom";
        const ty = inside ? yv + (up ? 2 : -2) * px : yv + (up ? -2 : 2) * px;
        ctx.fillText(b.flagged ? "!" : "✓", (x0 + x1) / 2, ty);
      }
    }
    ctx.strokeStyle = c.err;
    ctx.lineWidth = px;
    ctx.setLineDash([6 * px, 4 * px]);
    ctx.beginPath();
    ctx.moveTo(bbox.left, yLimit);
    ctx.lineTo(bbox.left + bbox.width, yLimit);
    ctx.stroke();
    ctx.restore();
  }
}

/** The metric lane's key line: "<metric> per rep · flagged above limit 35°". */
export function metricKeyText(metric, limitText) {
  if (!metric) return "no rule metric for this exercise";
  const side = OP_WORD[metric.op];
  return side ? `flagged ${side} the limit (${limitText}, dashed)` : `limit ${limitText} (dashed)`;
}
