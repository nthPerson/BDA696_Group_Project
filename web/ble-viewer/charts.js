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
