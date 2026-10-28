import assert from "node:assert/strict";
import { test } from "node:test";
import { GOLDEN_SAMPLES, SERIAL_CSV_HEADER } from "../protocol.js";
import { csvFilename, Recorder } from "../recorder.js";

test("records raw rows under the serial CSV header, only while active", () => {
  const r = new Recorder();
  r.add(GOLDEN_SAMPLES); // not started: ignored
  r.start();
  r.add(GOLDEN_SAMPLES.slice(0, 2));
  assert.equal(r.count, 2);
  const lines = r.stop().trimEnd().split("\n");
  assert.deepEqual(lines, [
    SERIAL_CSV_HEADER,
    "1000,0,0,4096,0,0,0,0,65534",
    "1020,-4096,2048,-1,-32768,328,32767,1,65535",
  ]);
  r.add(GOLDEN_SAMPLES); // stopped: ignored
  assert.equal(r.count, 2);
  assert.equal(r.active, false);
});

test("start() begins a fresh recording", () => {
  const r = new Recorder();
  r.start();
  r.add(GOLDEN_SAMPLES);
  r.stop();
  r.start();
  assert.equal(r.count, 0);
});

test("csvFilename is <device>_YYYYMMDD-HHMMSS.csv with unsafe characters replaced", () => {
  const d = new Date(2026, 9, 15, 9, 5, 7);
  assert.equal(csvFilename("FormCoach-8428", d), "FormCoach-8428_20261015-090507.csv");
  assert.equal(csvFilename("a b/c", d), "a_b_c_20261015-090507.csv");
  assert.equal(csvFilename(undefined, d), "FormCoach_20261015-090507.csv");
});
