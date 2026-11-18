import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { reduce as reduceFromPage } from "../coach.js";
import { CUE_TTL_S, MAX_DOTS, counterKeys, initialState, reduce, view } from "../coach-state.js";
import { clockText } from "../format.js";

const golden = JSON.parse(readFileSync(new URL("../../../../../tests/golden/bench_messages.json", import.meta.url), "utf8"));
const ev = (kind, payload = {}, t = 1) => ({ type: "event", kind, t, payload });
const rep = (arm, rep_id, faults = []) => ev("rep", { arm, rep_id, t_start: 0, duration_s: 2, faults, metrics: {} });
const warn = { code: "CURL_SWING", severity: "warn", message: "Keep your upper arm still", metric: "x", value: 1, threshold: 0, op: "gt" };
const tip = { code: "CURL_TIP", severity: "info", message: "Slow down a little", metric: "x", value: 1, threshold: 0, op: "gt" };
const run = (msgs, s = initialState()) => msgs.reduce(reduce, s);
const hello = () => structuredClone(golden.hello);
const tick = (now) => ({ type: "tick", now });
const withTotals = (totals) => ({ type: "frame", t: 1, gate: true, totals, exercise: "curl", link_state: "ok", capture_state: "idle" });
const v = (s) => view(s, clockText);

test("coach.js exports the same pure reduce", () => {
  assert.equal(reduceFromPage, reduce);
});

test("hello gives exercise, totals, link and capture; the input state is not changed", () => {
  const s0 = initialState();
  const s = reduce(s0, hello());
  assert.equal(s0.exercise, null);
  assert.equal(s.exercise, "curl");
  assert.equal(s.totals.l.reps, 2);
  assert.equal(s.linkState, "ok");
  assert.equal(s.connected, true);
  const m = v(s);
  assert.deepEqual(m.counts.map((c) => [c.label, c.count]), [["L", 2], ["R", 1]]);
});

test("a rep event sets that arm's cue, count (from totals) and dots", () => {
  let s = run([hello(), tick(10), rep("l", 6, [warn]), withTotals({ l: { reps: 7, good: 6, flagged: 1 }, r: { reps: 7, good: 7, flagged: 0 } })]);
  let m = v(s);
  assert.equal(m.band.kind, "warn");
  assert.equal(m.band.text, "Keep your upper arm still");
  assert.equal(m.band.who, "LEFT ARM · REP 7");
  assert.equal(m.band.icon, "!");
  assert.equal(m.counts[0].count, 7);
  assert.equal(m.counts[0].dots.at(-1), "");
  assert.equal(m.counts[0].dots.filter((d) => d === "w").length, 1);
  assert.equal(m.counts[0].dots.length, MAX_DOTS);
  assert.equal(s.pendingTone, "fault");
});

test("a good rep shows a green Good rep cue and queues the good tone", () => {
  const s = run([hello(), rep("r", 0)]);
  const m = v(s);
  assert.deepEqual([m.band.kind, m.band.text, m.band.icon, m.band.who], ["good", "Good rep", "✓", "RIGHT ARM · REP 1"]);
  assert.equal(s.pendingTone, "good");
});

test("a second arm's rep on the same frame keeps both cues; the more severe shows", () => {
  let s = run([hello(), tick(1), rep("l", 3, [warn]), rep("r", 3)]);
  assert.equal(Object.keys(s.cues).length, 2);
  assert.equal(v(s).band.kind, "warn"); // the fault beats Good rep although the good one is newer
  assert.equal(s.pendingTone, "fault"); // and its tone wins
  s = run([rep("l", 4, [tip]), rep("r", 4, [warn])], s);
  assert.equal(v(s).band.who, "RIGHT ARM · REP 5");
  s = run([rep("l", 5, [warn])], s); // equal severity: the newer
  assert.equal(v(s).band.who, "LEFT ARM · REP 6");
  s = run([rep("r", 5, [tip])], s); // warn beats info
  assert.equal(v(s).band.kind, "warn");
});

test("an arm's cue is replaced by its next rep", () => {
  const s = run([hello(), rep("l", 0, [warn]), rep("l", 1)]);
  assert.equal(v(s).band.text, "Good rep");
  assert.equal(v(s).band.who, "LEFT ARM · REP 2");
});

test("the cue expires after 8 s of tick; the other arm's later cue lives on", () => {
  let s = run([hello(), tick(100), rep("l", 0, [warn])]);
  s = run([tick(104), rep("r", 0)], s);
  s = run([tick(100 + CUE_TTL_S - 0.1)], s);
  assert.equal(v(s).band.kind, "warn");
  s = run([tick(100 + CUE_TTL_S)], s);
  assert.equal(v(s).band.kind, "good"); // r is still live (4 s old)
  s = run([tick(104 + CUE_TTL_S)], s);
  assert.equal(v(s).band.kind, "neutral");
  assert.equal(v(s).band.text, "Ready");
});

test("the tone is queued until tone_played; a good rep after a fault in one frame stays a fault", () => {
  let s = run([hello(), rep("l", 0, [warn]), rep("r", 0)]);
  assert.equal(s.pendingTone, "fault");
  s = reduce(s, { type: "tone_played" });
  assert.equal(s.pendingTone, null);
  s = reduce(s, rep("l", 1));
  assert.equal(s.pendingTone, "good");
});

test("squat shows one counter", () => {
  const s = run([{ ...hello(), state: { ...hello().state, exercise: "squat", totals: { all: { reps: 4, good: 3, flagged: 1 } } } }, rep(null, 3)]);
  assert.deepEqual(counterKeys(s), ["all"]);
  const m = v(s);
  assert.equal(m.counts.length, 1);
  assert.equal(m.counts[0].count, 4);
  assert.equal(m.exercise, "SQUAT");
  assert.equal(m.band.who, "REP 4");
  assert.equal(m.band.arm, null);
});

test("arm exercises show L and R even before any totals", () => {
  const s = reduce(initialState(), { type: "frame", t: 0, gate: false, exercise: "press" });
  assert.deepEqual(v(s).counts.map((c) => [c.label, c.count]), [["L", 0], ["R", 0]]);
});

test("dots are capped at 12 (the newest) and reset on counters_reset", () => {
  let s = run([hello()]);
  for (let i = 0; i < 20; i++) s = reduce(s, rep("l", i, i === 19 ? [warn] : []));
  s = reduce(s, withTotals({ l: { reps: 20, good: 19, flagged: 1 }, r: { reps: 0, good: 0, flagged: 0 } }));
  const dots = v(s).counts[0].dots;
  assert.equal(dots.length, 12);
  assert.equal(dots.filter((d) => d !== "").length, 12);
  assert.equal(dots.at(-1), "w");
  assert.equal(s.reps.l.length, MAX_DOTS);
  s = reduce(s, ev("counters_reset"));
  assert.deepEqual(s.reps, { l: [], r: [], all: [] });
  assert.deepEqual(s.cues, {});
  s = reduce(s, withTotals({ l: { reps: 0, good: 0, flagged: 0 }, r: { reps: 0, good: 0, flagged: 0 } }));
  assert.equal(v(s).counts[0].dots.filter((d) => d !== "").length, 0);
  assert.equal(v(s).band.text, "Ready");
});

test("an exercise change and the start of a recording clear dots and cues", () => {
  let s = run([hello(), rep("l", 0), ev("exercise", { exercise: "press" })]);
  assert.deepEqual(s.cues, {});
  s = run([rep("l", 0), ev("capture", { state: "countdown", delay_s: 3 }, 5)], s);
  assert.equal(s.reps.l.length, 1); // a countdown does not clear
  s = reduce(s, ev("capture", { state: "recording", labels: {} }, 8));
  assert.deepEqual(s.reps.l, []);
  assert.deepEqual(s.cues, {});
});

test("countdown state text counts down from the capture deadline", () => {
  let s = run([hello(), ev("capture", { state: "countdown", delay_s: 3 }, 10), { type: "frame", t: 10.2, gate: false }]);
  assert.equal(v(s).strip.rec[0], "GET READY 3");
  assert.equal(v(s).band.text, "Get ready 3");
  s = reduce(s, { type: "frame", t: 11.5, gate: false });
  assert.equal(v(s).strip.rec[0], "GET READY 2");
  s = reduce(s, { type: "frame", t: 12.9, gate: false });
  assert.equal(v(s).band.text, "Get ready 1");
  s = reduce(s, ev("capture", { state: "idle" }, 13));
  assert.equal(v(s).strip.rec[0], "READY");
  assert.equal(v(s).band.text, "Ready");
});

test("a countdown seen only through the state has no number", () => {
  const h = hello();
  h.state.capture.state = "countdown";
  const s = reduce(initialState(), h);
  assert.equal(v(s).strip.rec[0], "GET READY");
});

test("recording time is counted from when the page saw it start", () => {
  let s = run([hello(), tick(50), ev("capture", { state: "recording", labels: {} }, 7), tick(92.4)]);
  assert.equal(v(s).strip.rec[0], "● REC 0:42");
  assert.equal(v(s).strip.rec[1], "r");
  // the frame header repeating the state does not restart the clock
  s = run([{ type: "frame", t: 9, gate: true, capture_state: "recording" }, tick(100)], s);
  assert.equal(v(s).strip.rec[0], "● REC 0:50");
  s = run([ev("capture", { state: "review" }, 70)], s);
  assert.equal(v(s).strip.rec[0], "READY");
});

test("a page that opens mid-recording shows REC without a time", () => {
  const h = hello();
  h.state.capture.state = "recording";
  const s = run([h, tick(500), { type: "frame", t: 3, gate: true, capture_state: "recording" }, tick(520)]);
  assert.equal(v(s).strip.rec[0], "● REC");
});

test("a recording seen first through a state message starts the clock then", () => {
  const h2 = hello();
  h2.state.capture.state = "recording";
  const s = run([hello(), tick(30), { type: "state", state: h2.state }, tick(35)]);
  assert.equal(v(s).strip.rec[0], "● REC 0:05");
});

test("gate and watch words", () => {
  let s = run([hello(), { type: "frame", t: 1, gate: true }]);
  assert.deepEqual(v(s).strip.gate, ["GATE OPEN", "g"]);
  assert.deepEqual(v(s).strip.watch, ["WATCH LINKED", "b"]);
  s = run([{ type: "frame", t: 2, gate: false, link_state: "silent" }], s);
  assert.deepEqual(v(s).strip.gate, ["GATE CLOSED", "n"]);
  assert.deepEqual(v(s).strip.watch, ["WATCH SILENT", "a"]);
  for (const st of ["none", "waiting"]) {
    s = reduce(s, { type: "link", status: { state: st } });
    assert.deepEqual(v(s).strip.watch, ["NO WATCH", "n"]);
  }
  s = reduce(s, { type: "link", status: { state: "ok" } });
  assert.equal(v(s).strip.watch[0], "WATCH LINKED");
});

test("a closed socket is flagged and keeps the counts", () => {
  let s = run([hello(), withTotals({ l: { reps: 9, good: 9, flagged: 0 }, r: { reps: 8, good: 8, flagged: 0 } })]);
  s = reduce(s, { type: "socket_close" });
  assert.equal(v(s).disconnected, true);
  assert.equal(v(s).counts[0].count, 9);
  s = reduce(s, { type: "socket_open" });
  assert.equal(v(s).disconnected, false);
});

test("two-digit counts are plain numbers", () => {
  const s = run([hello(), withTotals({ l: { reps: 12, good: 12, flagged: 0 }, r: { reps: 11, good: 11, flagged: 0 } })]);
  assert.deepEqual(v(s).counts.map((c) => c.count), [12, 11]);
});

test("a cue hidden behind the other arm's fault is not fresh when promoted after the fault expires", () => {
  let s = run([hello(), tick(100), rep("l", 0, [warn]), tick(101), rep("r", 0)]);
  assert.equal(v(s).band.who, "LEFT ARM · REP 1");
  assert.equal(v(s).band.fresh, true);
  s = reduce(s, tick(100 + CUE_TTL_S));
  assert.equal(v(s).band.who, "RIGHT ARM · REP 1"); // promoted: still displayed
  assert.equal(v(s).band.fresh, false); // but never confirmed
});

test("a rep whose cue is on top at once is fresh, on every view until replaced", () => {
  let s = run([hello(), tick(1), rep("l", 0)]);
  assert.equal(v(s).band.fresh, true);
  s = reduce(s, tick(2));
  assert.equal(v(s).band.fresh, true); // the DOM code dedupes the send by key
  assert.equal(v(s).band.key, v(reduce(s, tick(3))).band.key);
  s = reduce(s, rep("l", 1, [warn]));
  assert.equal(v(s).band.fresh, true);
  assert.equal(v(s).band.rep_id, 1);
});

test("two arms finishing on one frame with equal severity: only the newer is fresh", () => {
  let s = run([hello(), tick(1), rep("l", 0), rep("r", 0)]);
  assert.equal(v(s).band.who, "RIGHT ARM · REP 1");
  assert.equal(v(s).band.fresh, true);
  assert.equal(s.cues.l.fresh, false);
  s = reduce(s, rep("r", 1, [warn])); // later frames: the newer fault replaces r; l stays unconfirmed
  s = run([tick(2), rep("r", 2)], s);
  s = reduce(s, tick(1 + CUE_TTL_S - 0.5));
  assert.equal(s.cues.l.fresh, false);
});
