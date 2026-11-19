import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { clockText, cueText, marginText, repDots } from "../format.js";

const golden = JSON.parse(readFileSync(new URL("../../../../../tests/golden/bench_messages.json", import.meta.url), "utf8"));

const fault = (o = {}) => ({ code: "C", severity: "warn", metric: "upper_arm_trunk_drift", value: 41, threshold: 35, op: "gt", message: "Elbow drifting", ...o });
const rep = (o = {}) => ({ kind: "rep", t: 1, payload: { arm: "l", rep_id: 6, faults: [], ...o } });

test("cueText: clean rep, who from the arm, rep number is rep_id + 1", () => {
  assert.deepEqual(cueText(rep()), { who: "LEFT ARM · REP 7", text: "Good rep", kind: "good" });
  assert.equal(cueText(rep({ arm: "r" })).who, "RIGHT ARM · REP 7");
  assert.equal(cueText(rep({ arm: null })).who, "REP 7");
  assert.equal(cueText(rep({ arm: "all" })).who, "REP 7");
  assert.equal(cueText(golden.event).text, "Good rep"); // the golden rep has no faults
});

test("cueText names the most severe fault: warn beats info, first wins a tie", () => {
  const info = fault({ severity: "info", message: "Slow down a little" });
  const w1 = fault({ message: "first warn" });
  const w2 = fault({ message: "second warn" });
  assert.deepEqual(cueText(rep({ faults: [info, w1, w2] })), { who: "LEFT ARM · REP 7", text: "first warn", kind: "warn" });
  assert.equal(cueText(rep({ faults: [info, fault({ severity: "info", message: "later" })] })).text, "Slow down a little");
  assert.equal(cueText(rep({ faults: [info] })).kind, "info");
  assert.equal(cueText(rep().payload).text, "Good rep"); // a bare payload works too
});

test("marginText: op, degrees only for degree metrics, seconds for duration_s", () => {
  assert.equal(marginText(fault()), "upper_arm_trunk_drift 41° > limit 35° (6° over)");
  assert.equal(marginText(fault({ op: "ge" })), "upper_arm_trunk_drift 41° > limit 35° (6° over)");
  assert.equal(
    marginText(fault({ metric: "duration_s", value: 0.8, threshold: 1, op: "lt" })),
    "duration_s 0.8 s < limit 1 s (0.2 s under)",
  );
  assert.equal(marginText(fault({ metric: "something_else", value: 3.04, threshold: 2, op: "le" })), "something_else 3 < limit 2 (1 under)");
});

test("repDots: g for clean, w for flagged, in order", () => {
  const ev = [rep(), rep({ faults: [fault()] }), rep()];
  assert.deepEqual(repDots({ reps: 3, good: 2, flagged: 1 }, ev), ["g", "w", "g"]);
  assert.deepEqual(repDots(undefined, []), []);
  assert.deepEqual(repDots(undefined, ev), ["g", "w", "g"]);
});

test("repDots: counted reps the page never saw come first as neutral dots, seen ones in order", () => {
  const ev = [rep({ faults: [fault()] }), rep()];
  assert.deepEqual(repDots({ reps: 5, good: 2, flagged: 3 }, ev), ["u", "u", "u", "w", "g"]);
  assert.deepEqual(repDots({ reps: 0, good: 0, flagged: 0 }, ev), ["w", "g"]); // never fewer than seen
});

test("clockText is m:ss", () => {
  assert.equal(clockText(42), "0:42");
  assert.equal(clockText(0), "0:00");
  assert.equal(clockText(59.9), "0:59");
  assert.equal(clockText(605), "10:05");
  assert.equal(clockText(-3), "0:00");
  assert.equal(clockText(NaN), "0:00");
});
