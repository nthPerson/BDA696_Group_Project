import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { reduce as reduceFromPage } from "../lab.js";
import {
  LOG_LINES, captureView, counterKeys, defaultRulesPath, devicesFor, exerciseOptions, idTail,
  initialState, isRulesConflict, paramText, reduce, rejudgeLine, uptimeText,
} from "../lab-state.js";

const golden = JSON.parse(readFileSync(new URL("../../../../../tests/golden/bench_messages.json", import.meta.url), "utf8"));
const ev = (kind, payload = {}, t = 1) => ({ type: "event", kind, t, payload });
const run = (msgs, s = initialState()) => msgs.reduce(reduce, s);
const hello = () => structuredClone(golden.hello);

test("lab.js exports the same pure reduce", () => {
  assert.equal(reduceFromPage, reduce);
});

test("hello gives the server state, totals, link and capture; the input state is not changed", () => {
  const s0 = initialState();
  const s = reduce(s0, hello());
  assert.equal(s0.server, null);
  assert.equal(s.server.exercise, "curl");
  assert.deepEqual(s.totals, golden.hello.state.totals);
  assert.equal(s.link.kit, "K2");
  assert.equal(s.capture.state, "idle");
  assert.deepEqual(s.capture.last, golden.hello.state.capture.last);
});

test("rejudged updates the totals and remembers the previous ones", () => {
  const s = run([hello(), ev("rejudged", {
    totals: { l: { reps: 2, good: 2, flagged: 0, faults_by_code: {} }, r: { reps: 1, good: 0, flagged: 1, faults_by_code: {} } },
    reps: [{ arm: "l", rep_id: 0, faults: [] }],
  })]);
  assert.equal(s.totals.l.good, 2);
  assert.deepEqual(rejudgeLine(s), { good: 2, flagged: 1, was: { good: 2, flagged: 1 } });
  // golden totals: l 1 good 1 flagged, r 1 good -> was 2 good 1 flagged; now 2 good + 1 flagged
  // without a receive time every re-judging starts a gesture: "was" is the totals just before it
  const s2 = reduce(s, ev("rejudged", { totals: { l: { reps: 2, good: 0, flagged: 2 }, r: { reps: 1, good: 0, flagged: 1 } }, reps: [] }, 4));
  assert.deepEqual(rejudgeLine(s2), { good: 0, flagged: 3, was: { good: 2, flagged: 1 } });
  assert.equal(rejudgeLine(reduce(s2, ev("counters_reset"))).was, null);
});

test("re-judgings are grouped into slider gestures by the page clock, not the frame time", () => {
  const tot = (good, flagged) => ({ l: { reps: good + flagged, good, flagged }, r: { reps: 0, good: 0, flagged: 0 } });
  const rj = (totals, rx) => ({ ...ev("rejudged", { totals, reps: [] }, 7.5), rx }); // the source ended: every t is 7.5
  let s = reduce(reduce(initialState(), hello()), { type: "state", state: { ...golden.state.state, totals: tot(5, 0) } });
  s = reduce(s, rj(tot(4, 1), 1000)); // first move
  s = reduce(s, rj(tot(3, 2), 1500)); // same drag, 0.5 s later
  assert.deepEqual(rejudgeLine(s), { good: 3, flagged: 2, was: { good: 5, flagged: 0 } });
  s = reduce(s, rj(tot(1, 4), 9000)); // a new move 7.5 s later, same frame time
  assert.deepEqual(rejudgeLine(s), { good: 1, flagged: 4, was: { good: 3, flagged: 2 } });
});

test("rejudged rewrites the faults of the reps held for the dots and the last cue", () => {
  const fault = { code: "CURL_TOO_FAST", metric: "duration_s", op: "lt", value: 0.8, threshold: 1, severity: "warn", message: "Slow down" };
  let s = run([hello(), ev("rep", { arm: "l", rep_id: 0, faults: [] }), ev("rep", { arm: "r", rep_id: 0, faults: [] })]);
  assert.equal(s.lastRep.arm, "r");
  s = reduce(s, ev("rejudged", { totals: {}, reps: [{ arm: "r", rep_id: 0, faults: [fault] }] }));
  assert.deepEqual(s.reps.r.map((p) => p.faults.length), [1]);
  assert.deepEqual(s.reps.l.map((p) => p.faults.length), [0]);
  assert.equal(s.lastRep.faults[0].code, "CURL_TOO_FAST");
});

test("capture events move the capture panel: countdown, recording, review, kept", () => {
  let s = run([hello(), { type: "frame", ...golden.frame, t: 10 }]);
  s = reduce(s, ev("capture", { state: "countdown", delay_s: 5 }, 10));
  assert.deepEqual(captureView(s), { state: "countdown", countdownS: 5, elapsedS: null });
  s = reduce(s, { type: "frame", ...golden.frame, t: 12.5 });
  assert.equal(captureView(s).countdownS, 3);
  s = reduce(s, ev("capture", { state: "recording", t_mono: 1015, labels: { subject: "S9" } }, 15));
  s = reduce(s, { type: "frame", ...golden.frame, t: 27 });
  assert.deepEqual(captureView(s), { state: "recording", countdownS: null, elapsedS: 12 });
  // the state message that follows keeps the start time
  s = reduce(s, { type: "state", state: { ...golden.state.state, capture: { state: "recording", labels: { subject: "S9" }, summary: null, last: null } } });
  assert.equal(captureView(s).elapsedS, 12);
  const summary = { duration_s: 12, frames: 360 };
  s = reduce(s, ev("capture", { state: "review", summary }, 27));
  assert.equal(s.capture.state, "review");
  assert.deepEqual(s.capture.summary, summary);
  s = reduce(s, ev("capture", { state: "idle", dir: "data/team/S9/x", check: { problems: [] }, warnings: [] }, 28));
  assert.equal(s.capture.state, "idle");
  assert.equal(s.capture.summary, null);
  assert.deepEqual(s.capture.last, { dir: "data/team/S9/x", check: { problems: [] }, warnings: [] });
  assert.equal(reduce(s, ev("capture", { state: "idle", discarded: true })).capture.discarded, true);
});

test("a page that connects mid-recording shows the state without an elapsed time", () => {
  const st = { ...golden.state.state, capture: { state: "recording", labels: null, summary: null, last: null } };
  const s = reduce(initialState(), { type: "hello", v: 1, state: st });
  assert.deepEqual(captureView(s), { state: "recording", countdownS: null, elapsedS: null });
});

test("scan results replace that kind's list; the connected device is listed first", () => {
  let s = run([hello(), { type: "scan_started", kind: "ble" }]);
  assert.equal(s.scanning, "ble");
  s = reduce(s, golden.scan);
  assert.equal(s.scanning, null);
  assert.equal(s.scans.ble.length, 2);
  s = reduce(s, { type: "scan", kind: "ble", devices: [{ kind: "ble", id: "11:22:33:44:55:66", name: "FormCoach-8428", rssi: -70, kit: null }] });
  assert.deepEqual(s.scans.ble.map((d) => d.id), ["11:22:33:44:55:66"]);
  const list = devicesFor(s, "ble");
  assert.deepEqual(list.map((d) => [d.id, d.connected]), [["AA:BB:CC:DD:EE:FF", true], ["11:22:33:44:55:66", false]]);
  assert.deepEqual(devicesFor(s, "serial"), []);
});

test("link messages replace the status; log keeps the newest lines", () => {
  let s = run([hello(), golden.link]);
  assert.equal(s.link.firmware, "0.4.1");
  for (let i = 0; i < LOG_LINES + 5; i++) s = reduce(s, { type: "log", level: "info", text: `line ${i}`, t: i, rx: 0 });
  assert.equal(s.log.length, LOG_LINES);
  assert.equal(s.log.at(-1).text, `line ${LOG_LINES + 4}`);
  assert.equal(s.log[0].text, "line 5");
});

test("errors are kept until dismissed; session error events join them; the socket state is tracked", () => {
  let s = run([hello(), golden.error, ev("error", { where: "on_event", error: "KeyError: x" })]);
  assert.deepEqual(s.errors.map((e) => e.text), ["unknown parameter 'x'", "on_event: KeyError: x"]);
  s = reduce(s, { type: "dismiss_error", id: s.errors[0].id });
  assert.deepEqual(s.errors.map((e) => e.text), ["on_event: KeyError: x"]);
  assert.equal(reduce(s, { type: "socket_close" }).connected, false);
  assert.equal(reduce(s, { type: "socket_open" }).connected, true);
});

test("frames update the live totals and the frame slice; rules_saved is remembered", () => {
  const s = run([hello(), golden.frame, ev("rules_saved", { path: "runs/rules/curl.yaml", overwritten: true })]);
  assert.equal(s.frame.gate, true);
  assert.equal(s.frame.fps, 29.97);
  assert.deepEqual(s.totals, golden.frame.totals);
  assert.deepEqual(s.rulesSaved, { path: "runs/rules/curl.yaml", overwritten: true });
});

test("rep events are kept per counter for the dots and cleared by a reset or a new exercise", () => {
  let s = run([hello(), ev("rep", { arm: "l", rep_id: 0, faults: [] }), ev("rep", { arm: null, rep_id: 0, faults: [] })]);
  assert.equal(s.reps.l.length, 1);
  assert.equal(s.reps.all.length, 1);
  s = reduce(s, ev("exercise", { exercise: "squat" }));
  assert.deepEqual(s.reps, { l: [], r: [], all: [] });
  assert.equal(s.lastRep, null);
});

test("helpers: counters, ids, uptime, parameter text, rules path, exercise options", () => {
  assert.deepEqual(counterKeys("squat"), ["all"]);
  assert.deepEqual(counterKeys("curl"), ["l", "r"]);
  assert.equal(idTail("AA:BB:CC:B3:8E:94"), "…B3:8E:94");
  assert.equal(idTail("COM5"), "COM5");
  assert.equal(uptimeText(45), "45 s");
  assert.equal(uptimeText(720), "12 min");
  assert.equal(uptimeText(3900), "1 h 05 min");
  assert.equal(uptimeText(null), "–");
  assert.equal(paramText({ unit: "deg", step: 1 }, 35), "35°");
  assert.equal(paramText({ unit: "s", step: 0.1 }, 1), "1.0 s");
  assert.equal(paramText({ unit: "", step: 0.01 }, 0.35), "0.35");
  assert.equal(paramText({ unit: "windows", step: 1 }, 12), "12 win");
  assert.equal(defaultRulesPath("curl", new Date(2026, 10, 18, 9, 5)), "runs/rules/curl-20261118-0905.yaml");
  assert.deepEqual(exerciseOptions(golden.hello.state).map((o) => o.text), ["curl", "press (untuned)", "raise (untuned)", "squat (untuned)", "idle (no counting)"]);
  assert.equal(isRulesConflict({ type: "error", cmd: "save_rules", text: "'a.yaml' already exists; send overwrite: true to replace it" }), true);
  assert.equal(isRulesConflict({ type: "error", cmd: "save_rules", text: "not allowed" }), false);
  assert.equal(isRulesConflict(golden.error), false);
});

test("paramLabel: rule parameters lose the exercise prefix; others keep the server's label", async () => {
  const { paramLabel } = await import("../lab-state.js");
  assert.equal(paramLabel({ key: "rule.curl.CURL_SWING.upper_arm_trunk_drift", label: "CURL_SWING: upper_arm_trunk_drift" }), "Swing · upper_arm_trunk_drift");
  assert.equal(paramLabel({ key: "rule.curl.CURL_PARTIAL_ROM.elbow_min", label: "x" }), "Partial rom · elbow_min");
  assert.equal(paramLabel({ key: "rule.squat.KNEE_VALGUS.knee_track_at_bottom", label: "x" }), "Knee valgus · knee_track_at_bottom");
  assert.equal(paramLabel({ key: "gate.energy_threshold", label: "Energy threshold" }), "Energy threshold");
});
