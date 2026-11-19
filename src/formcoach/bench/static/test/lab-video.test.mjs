import assert from "node:assert/strict";
import { test } from "node:test";
import { HISTORY_FRAMES, LandmarkHistory, VideoPanel, anglesOf } from "../lab-video.js";

test("the landmark history holds 300 s at 30 fps", () => {
  assert.equal(HISTORY_FRAMES, 9000);
  assert.equal(new LandmarkHistory().capacity, 9000);
});

test("at() finds the newest entry at or before t, within the gap allowed", () => {
  const h = new LandmarkHistory(4);
  [1, 2, 3, 4, 5].forEach((t) => h.push(t, [[t, 0, 1]], { l: t, r: null }));
  assert.equal(h.length, 4); // the oldest (t = 1) was dropped
  assert.equal(h.at(3.2).t, 3);
  assert.equal(h.at(3).t, 3);
  assert.deepEqual(h.at(5.4).lm, [[5, 0, 1]]);
  assert.equal(h.at(5.6), null); // more than 0.5 s after the newest
  assert.equal(h.at(1.5), null); // before the oldest kept
  assert.equal(h.at(5.6, 1).t, 5);
});

test("non-finite and out-of-order times are not kept; a frame without a person keeps null", () => {
  const h = new LandmarkHistory(10);
  h.push(Number.NaN, []);
  h.push(2, null);
  h.push(1, [[0, 0, 1]]);
  assert.equal(h.length, 1);
  assert.equal(h.at(2).lm, null);
});

test("the panel's skeleton follows every frame header even when no picture comes (A4)", () => {
  const canvas = () => ({ hidden: false, width: 0, height: 0, clientWidth: 0, clientHeight: 0, getContext: () => null });
  const root = { querySelector: (sel) => (sel === "canvas.vid" ? vid : skel), style: {} };
  const vid = canvas();
  const skel = canvas();
  const panel = new VideoPanel(root, new LandmarkHistory(10));
  const header = (t, x) => ({ t_mono: t, lm: [[x, 0.5, 1]], counters: { l: { angle: 40 }, r: { angle: null } } });
  panel.onFrame(header(1, 0.1), null, { paused: false, videoOn: true }); // OpenCV missing: no JPEG
  assert.deepEqual(panel.header.lm, [[0.1, 0.5, 1]]);
  panel.onFrame(header(2, 0.2), null, { paused: false, videoOn: true });
  assert.deepEqual(panel.header.lm, [[0.2, 0.5, 1]]);
  assert.equal(panel.bitmap, null);
  panel.onFrame(header(3, 0.3), null, { paused: true, videoOn: true }); // paused: held
  assert.deepEqual(panel.header.lm, [[0.2, 0.5, 1]]);
});

test("the panel's arm angles come from the arm counters only (squat lights no arm)", () => {
  assert.deepEqual(anglesOf({ counters: { all: { angle: 90 } } }), { l: null, r: null });
  assert.deepEqual(anglesOf({ counters: { l: { angle: 50 }, r: { angle: 60 } } }), { l: 50, r: 60 });
});
