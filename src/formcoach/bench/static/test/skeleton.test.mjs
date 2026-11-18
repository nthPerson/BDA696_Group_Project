import assert from "node:assert/strict";
import { test } from "node:test";
import { EDGES, drawSkeleton, layout } from "../skeleton.js";

const lm = (over = {}) => {
  const a = Array.from({ length: 33 }, () => [0.5, 0.5, 1]);
  for (const [i, v] of Object.entries(over)) a[i] = v;
  return a;
};

test("EDGES are the 12 body edges of the overlay, as MediaPipe-33 indices", () => {
  assert.equal(EDGES.length, 12);
  assert.deepEqual(EDGES[0], [11, 13]); // left shoulder - left elbow
  assert.deepEqual(EDGES[11], [26, 28]); // right knee - right ankle
});

test("layout scales to pixels and mirror flips x", () => {
  const l = lm({ 11: [0.25, 0.5, 0.9] });
  const plain = layout(l, 200, 100, { mirror: false });
  const mir = layout(l, 200, 100, { mirror: true });
  const p = (r) => r.points.find((q) => q.i === 11);
  assert.deepEqual(p(plain), { i: 11, x: 50, y: 50, v: 0.9 });
  assert.equal(p(mir).x, 150);
  assert.equal(p(mir).y, 50);
});

test("layout drops low-visibility and missing points and the edges that touch them", () => {
  const r = layout(lm({ 13: [0.1, 0.1, 0.49], 16: null }), 100, 100, { mirror: false });
  const ids = r.points.map((p) => p.i);
  assert.ok(!ids.includes(13) && !ids.includes(16));
  assert.equal(ids.length, 31);
  assert.ok(!r.edges.some((e) => e.includes(13) || e.includes(16)));
  assert.equal(r.edges.length, 12 - 3); // 11-13, 13-15, 14-16 are gone
  assert.deepEqual(layout(null, 10, 10, { mirror: true }), { points: [], edges: [] });
});

test("drawSkeleton draws edges and joints on a context, highlighting tracked arms", () => {
  const calls = [];
  const ctx = new Proxy({}, { get: (_t, k) => (typeof k === "string" ? (...a) => calls.push([k, a]) : undefined), set: (_t, k, v) => (calls.push(["set:" + k, v]), true) });
  drawSkeleton(ctx, lm(), 100, 100, { mirror: true, angles: { l: 90, r: null } });
  assert.equal(calls.filter((c) => c[0] === "stroke").length, 12);
  assert.equal(calls.filter((c) => c[0] === "arc").length, 33 - 11);
  assert.equal(calls.filter((c) => c[0] === "set:strokeStyle" && c[1] === "#7aa7ff").length, 2);
});
