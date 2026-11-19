import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Timeline } from "../series.js";
import { metricChoice, StripData } from "../strip-model.js";

const golden = JSON.parse(readFileSync(new URL("../../../../../tests/golden/bench_messages.json", import.meta.url), "utf8"));

// a frame header at frame time t; t_mono = t + 1000 (the session clock runs 1000 s ahead)
const frame = (t, o = {}) => ({
  type: "frame", t, t_mono: t + 1000, gates: { device: true, energy: false },
  counters: { l: { angle: 90, enter: 70, exit: 140 }, r: { angle: 100, enter: 72, exit: 138 } }, ...o,
});
const rep = (arm, id, t0, t1, faults = [], metrics = { duration_s: t1 - t0 }) => ({
  type: "event", kind: "rep", t: t1, payload: { arm, rep_id: id, t_start: t0, t_end: t1, faults, metrics },
});
const fast = { code: "CURL_TOO_FAST", metric: "duration_s", op: "lt", value: 0.8, threshold: 1.0, severity: "warn" };

function make() {
  const tl = new Timeline(20, 300);
  return { tl, d: new StripData(tl) };
}

test("frames feed the gate bits, angles and the enter/exit band (high, low) of each arm", () => {
  const { d } = make();
  d.pushFrame(frame(1));
  d.pushFrame(frame(2, { gates: { device: false, energy: true }, counters: { all: { angle: 120, enter: 100, exit: 160 } } }));
  const v = d.view();
  assert.equal(v.tEnd, 1002);
  assert.deepEqual(v.gates, [[-1, 0], [1, 0], [0, 1], [null, null]]);
  // squat's single counter "all" is drawn as the left arm
  assert.deepEqual(v.angle, [[-1, 0], [90, 120], [100, null], [140, 160], [70, 100], [138, null], [72, null]]);
});

test("frames with a non-finite clock are skipped and do not move the clock offset", () => {
  const { d } = make();
  d.pushFrame(frame(1));
  d.pushFrame(frame(2, { t_mono: null }));
  d.pushFrame(frame(3, { t: Number.NaN }));
  d.pushFrame(frame(0.5)); // older than the newest row: out of order, skipped
  assert.equal(d.frames.length, 2); // t = 1 and t = 3 (the NaN frame time keeps its row, not its offset)
  d.pushEvent(rep("l", 0, 0.5, 1.5));
  assert.deepEqual(d.reps.map((r) => [r.t0, r.t1]), [[1000.5, 1001.5]]);
});

test("imu batches stamped with one arrival time are spread back at the sample rate", () => {
  const { d } = make();
  d.pushLink({ rate_hz: 50, p_active: null });
  const msg = { ...golden.imu, t: [10, 10, 10], ax: [1, 2, 3], ay: [0, 0, 0], az: [9, 9, 9], gx: [0, 0, 0], gy: [0, 0, 0], gz: [0, 0, 0] };
  d.pushImu(msg);
  d.pushImu({ ...msg, t: [10.02, 10.02, null], ax: [4, 5, 6] }); // the null-time sample is skipped
  const [t, ax] = d.imu.window(100, 1000);
  assert.deepEqual(t.map((x) => Math.round(x * 1000) / 1000), [9.96, 9.98, 10, 10, 10.02]);
  assert.deepEqual(ax, [1, 2, 3, 4, 5]); // never earlier than the previous batch's last sample
});

test("rep and rep_dropped events become marks on the chart clock; flagged reps are warn", () => {
  const { d } = make();
  d.pushFrame(frame(5));
  d.pushEvent(rep("l", 0, 1, 2.5));
  d.pushEvent(rep("r", 0, 1.2, 2.6, [fast]));
  d.pushEvent({ type: "event", kind: "rep_dropped", t: 4, payload: { arm: "l", t_start: 3, t_end: 4, reason: "too_short" } });
  d.pushEvent({ type: "event", kind: "gate_open", t: 4, payload: {} });
  const v = d.view();
  const r3 = (x) => Math.round(x * 1000) / 1000;
  assert.deepEqual(v.marks.map((m) => [m.arm, m.mark, r3(m.x0), r3(m.x1)]), [["l", "good", -4, -2.5], ["r", "warn", -3.8, -2.4], ["l", "drop", -2, -1]]);
});

test("rejudged rewrites the faults of this epoch's reps; a counter reset starts a new epoch", () => {
  const { d } = make();
  d.pushFrame(frame(5));
  d.pushEvent(rep("l", 0, 1, 2));
  d.pushEvent({ type: "event", kind: "counters_reset", t: 3, payload: {} });
  d.pushEvent(rep("l", 0, 3, 4));
  d.pushEvent({ type: "event", kind: "rejudged", t: 4, payload: { totals: {}, reps: [{ arm: "l", rep_id: 0, faults: [fast] }] } });
  assert.deepEqual(d.reps.map((r) => r.mark), ["good", "warn"]); // the older rep 0 is from before the reset
  d.pushEvent({ type: "event", kind: "capture", t: 4, payload: { state: "recording" } }); // recording resets the counters too
  d.pushEvent({ type: "event", kind: "rejudged", t: 4, payload: { totals: {}, reps: [{ arm: "l", rep_id: 0, faults: [] }] } });
  assert.deepEqual(d.reps.map((r) => r.mark), ["good", "warn"]);
});

test("metric bars: the metric's value per rep against the limit, flagged by the server's faults", () => {
  const { d } = make();
  d.pushFrame(frame(10));
  d.pushEvent(rep("l", 0, 1, 2.5, [], { duration_s: 1.5 }));
  d.pushEvent(rep("r", 0, 1, 1.8, [fast], { duration_s: 0.8 }));
  d.pushEvent(rep("l", 1, 3, 4, [], {})); // no value for the metric: no bar
  d.setMetric({ key: "rule.curl.CURL_TOO_FAST.duration_s", code: "CURL_TOO_FAST", metric: "duration_s", unit: "s", limit: 1.0 });
  const m = d.view().metric;
  assert.equal(m.limit, 1.0);
  assert.equal(m.op, "lt"); // learnt from the fault: flagged below the limit
  assert.deepEqual(m.bars.map((b) => [b.arm, b.value, b.flagged]), [["l", 1.5, false], ["r", 0.8, true]]);
  assert.equal(m.range[0], 0);
  assert.ok(m.range[1] > 1.5);
});

test("metric bars leave out reps of another exercise", () => {
  const { d } = make();
  d.pushFrame(frame(10));
  const curl = rep("l", 0, 1, 2, [], { elbow_max: 150 });
  curl.payload.exercise = "curl";
  const press = rep("l", 0, 3, 4, [], { elbow_max: 160 });
  press.payload.exercise = "press";
  d.pushEvent(curl);
  d.pushEvent(press);
  d.setMetric({ key: "rule.press.PRESS_NO_LOCKOUT.elbow_max", exercise: "press", code: "PRESS_NO_LOCKOUT", metric: "elbow_max", unit: "deg", limit: 155 });
  assert.deepEqual(d.view().metric.bars.map((b) => b.value), [160]);
  assert.equal(d.view().marks.length, 2); // the marks stay: they are history
});

test("metric range takes in a negative limit and negative values; op is null until a fault is seen", () => {
  const { d } = make();
  d.pushFrame(frame(10));
  d.pushEvent(rep(null, 0, 1, 2, [], { knee_track_at_bottom: -0.05 }));
  d.setMetric({ key: "rule.squat.SQUAT_KNEE_VALGUS.knee_track_at_bottom", code: "SQUAT_KNEE_VALGUS", metric: "knee_track_at_bottom", unit: "", limit: -0.1 });
  const m = d.view().metric;
  assert.equal(m.op, null);
  assert.ok(m.range[0] < -0.1 && m.range[1] >= 0);
  assert.deepEqual(m.bars.map((b) => [b.arm, b.value]), [["all", -0.05]]);
  d.setMetric(null);
  assert.equal(d.view().metric, null);
});

test("the view follows the timeline: paused windows and seconds behind", () => {
  const { tl, d } = make();
  for (let t = 0; t <= 60; t += 1) d.pushFrame(frame(t));
  tl.pause();
  tl.scrollBy(-30);
  const v = d.view();
  assert.equal(v.tEnd, 1030);
  assert.equal(v.behindS, 30);
  assert.equal(v.gates[0].at(-1), 0);
  assert.equal(v.gates[0][0], -20);
  assert.equal(d.oldestT, 1000);
});

test("metricChoice: the rule parameters of the exercise, with the limit now in force", () => {
  const params = [
    { key: "rule.curl.CURL_SWING.upper_arm_trunk_drift", label: "CURL_SWING: upper_arm_trunk_drift", group: "rules", unit: "deg", default: 35 },
    { key: "gate.energy_threshold", label: "Energy threshold", group: "gate", unit: "(m/s²)²", default: 3 },
  ];
  assert.deepEqual(metricChoice(params, { "rule.curl.CURL_SWING.upper_arm_trunk_drift": 30 }, null), {
    key: "rule.curl.CURL_SWING.upper_arm_trunk_drift", exercise: "curl", code: "CURL_SWING", metric: "upper_arm_trunk_drift", unit: "deg", limit: 30,
  });
  assert.equal(metricChoice(params, {}, "rule.curl.CURL_SWING.upper_arm_trunk_drift").limit, 35);
  assert.equal(metricChoice(params, {}, "rule.press.X.y").key, "rule.curl.CURL_SWING.upper_arm_trunk_drift"); // unknown key: first rule
  assert.equal(metricChoice([params[1]], {}, null), null);
});
