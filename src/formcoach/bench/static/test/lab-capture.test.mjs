import assert from "node:assert/strict";
import { test } from "node:test";
import { shortDir, summaryRows } from "../lab-capture.js";

test("summaryRows: duration, frames, IMU, reps per arm, faults, gate-open share and cue delay", () => {
  const rows = Object.fromEntries(summaryRows({
    duration_s: 12.34, frames: 370, pose_frames: 360, imu_samples: 617, rate_hz: 50.04, dropped: 2,
    reps_by_arm: { l: 7, r: 6 }, faults: { l: { CURL_SWING: 2 }, r: {} }, gate_open_pct: 84.2, latency_ms_median: 181.6,
  }));
  assert.deepEqual(rows, {
    Duration: "12.3 s", Frames: "370 (pose found in 360)", IMU: "50.0 Hz · 2 drops · 617 samples", Reps: "L 7 · R 6",
    Faults: "L CURL_SWING ×2", "Gate open": "84% of frames", "Cue delay": "median 182 ms",
  });
  const idle = Object.fromEntries(summaryRows({ duration_s: 0, frames: 0, reps_by_arm: {}, faults: {}, latency_ms_median: null }));
  assert.equal(idle.Reps, "none");
  assert.equal(idle.Faults, "none");
  assert.equal(idle["Cue delay"], "– (no coach page cue yet)");
  assert.equal(Object.fromEntries(summaryRows({ reps_by_arm: { all: 9 } })).Reps, "all 9");
});

test("shortDir keeps the folder from data/team on, with / separators", () => {
  assert.equal(shortDir("/home/x/repo/data/team/S9/20261118-150310"), "data/team/S9/20261118-150310");
  assert.equal(shortDir("C:\\repo\\data\\team\\S1\\20261118-150310"), "data/team/S1/20261118-150310");
  assert.equal(shortDir("elsewhere/S1/x"), "elsewhere/S1/x");
});
