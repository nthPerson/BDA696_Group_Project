import assert from "node:assert/strict";
import { test } from "node:test";
import { Series, Timeline } from "../series.js";

test("Series drops the oldest rows beyond capacity", () => {
  const s = new Series(["a", "b"], 3);
  for (let i = 0; i < 5; i++) s.push(i, [i * 10, -i]);
  assert.equal(s.length, 3);
  assert.equal(s.latestT, 4);
  assert.deepEqual(s.window(4, 100), [[2, 3, 4], [20, 30, 40], [-2, -3, -4]]);
});

test("window returns only rows inside the span, NaN as null", () => {
  const s = new Series(["a"], 10);
  [0, 1, 2, 3, 4].forEach((t) => s.push(t, [t === 2 ? null : t]));
  assert.deepEqual(s.window(3, 2), [[1, 2, 3], [1, null, 3]]);
  assert.deepEqual(s.window(10, 1), [[], []]);
  assert.equal(new Series(["a"], 2).latestT, -Infinity);
});

test("Timeline follows the newest time of any registered series", () => {
  const tl = new Timeline(20, 300);
  const a = new Series(["v"], 5, tl);
  const b = new Series(["v"], 5, tl);
  a.push(10, [0]);
  b.push(12, [0]);
  a.push(11, [0]);
  assert.equal(tl.tEnd, 12);
  assert.equal(tl.paused, false);
});

test("pause freezes tEnd, scrollBy moves it and clamps to the kept range", () => {
  const tl = new Timeline(20, 100);
  const s = new Series(["v"], 5, tl);
  s.push(500, [0]);
  tl.scrollBy(-10);
  assert.equal(tl.tEnd, 500); // running: scrollBy does nothing
  tl.pause();
  s.push(510, [0]);
  assert.equal(tl.tEnd, 500);
  assert.equal(tl.paused, true);
  tl.scrollBy(-30);
  assert.equal(tl.tEnd, 470);
  tl.scrollBy(-1000);
  assert.equal(tl.tEnd, 410); // newest 510 - keepS 100
  tl.scrollBy(+1000);
  assert.equal(tl.tEnd, 510);
  tl.resume();
  s.push(520, [0]);
  assert.equal(tl.tEnd, 520);
});

test("a frozen window is pushed forward when its history ages out", () => {
  const tl = new Timeline(20, 100);
  const s = new Series(["v"], 5, tl);
  s.push(0, [0]);
  tl.pause();
  s.push(250, [0]);
  assert.equal(tl.tEnd, 150);
});

test("window finds its rows by binary search in a wrapped ring, ties included", () => {
  const s = new Series(["v"], 8);
  // 13 rows: the ring has wrapped, oldest kept row is t = 5; two rows share t = 9
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11].forEach((t, k) => s.push(t, [k]));
  assert.deepEqual(s.window(11, 100), [[5, 6, 7, 8, 9, 9, 10, 11], [5, 6, 7, 8, 9, 10, 11, 12]]);
  assert.deepEqual(s.window(9, 1), [[8, 9, 9], [8, 9, 10]]);
  assert.deepEqual(s.window(9.5, 0.4), [[], []]);
  assert.deepEqual(s.window(4, 10), [[], []]); // all before the oldest kept row
  assert.deepEqual(s.window(50, 10), [[], []]); // all after the newest
  assert.deepEqual(new Series(["v"], 4).window(1, 1), [[], []]);
});

test("window touches only the rows it returns (no copy of the whole ring)", () => {
  const s = new Series(["v"], 20000);
  for (let i = 0; i < 20000; i++) s.push(i / 50, [i]);
  let reads = 0;
  const t = s._t;
  s._t = new Proxy(t, { get: (o, k) => { if (typeof k === "string" && /^\d+$/.test(k)) reads++; const v = o[k]; return typeof v === "function" ? v.bind(o) : v; } });
  const [ts] = s.window(399.98, 1);
  assert.equal(ts.length, 51);
  assert.ok(reads < 200, `read ${reads} times`);
});

test("oldestT is the first kept row's time", () => {
  const s = new Series(["v"], 3);
  assert.equal(s.oldestT, Infinity);
  [1, 2, 3, 4].forEach((t) => s.push(t, [0]));
  assert.equal(s.oldestT, 2);
});
