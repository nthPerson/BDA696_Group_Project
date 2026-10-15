import assert from "node:assert/strict";
import { test } from "node:test";
import { RateMeter, SampleRing } from "../buffer.js";

const mk = (seq, t_ms) => ({ t_ms, ax: 4096, ay: 0, az: -4096, gx: 328, gy: 0, gz: 0, flags: 0, seq });

test("counts seq gaps, and the 65535 -> 0 wrap is not a drop", () => {
  const r = new SampleRing(10);
  [65534, 65535, 0, 3].forEach((seq, i) => r.push(mk(seq, 20 * i)));
  assert.equal(r.received, 4);
  assert.equal(r.drops, 2); // 1 and 2 are missing
  assert.equal(r.dropRate(), 2 / 6);
});

test("the first sample after resetLink (a reconnect) is not counted as drops", () => {
  const r = new SampleRing(10);
  r.push(mk(10, 200));
  r.resetLink();
  r.push(mk(500, 10200));
  assert.equal(r.drops, 0);
  assert.equal(r.length, 2); // the plot is kept across a reconnect
});

test("t_ms going backwards (device reboot) clears the plot without counting drops", () => {
  const r = new SampleRing(10);
  r.push(mk(100, 5000));
  r.push(mk(101, 5020));
  r.push(mk(0, 40));
  assert.equal(r.length, 1);
  assert.equal(r.drops, 0);
  assert.equal(r.resets, 1);
});

test("keeps the newest `capacity` samples, oldest first, time relative to the newest", () => {
  const r = new SampleRing(3);
  for (let i = 0; i < 5; i++) r.push(mk(i, 20 * i));
  assert.equal(r.length, 3);
  assert.equal(r.latest.seq, 4);
  const t = Array.from(r.series().t);
  assert.equal(t.length, 3);
  [-0.04, -0.02, 0].forEach((want, i) => assert.ok(Math.abs(t[i] - want) < 1e-12, `t[${i}]=${t[i]}`));
});

test("series converts raw LSB to g and dps and computes |a|", () => {
  const r = new SampleRing(4);
  r.push(mk(0, 0));
  const s = r.series();
  assert.equal(s.ax[0], 1);
  assert.equal(s.az[0], -1);
  assert.ok(Math.abs(s.amag[0] - Math.SQRT2) < 1e-12);
  assert.ok(Math.abs(s.gx[0] - 10) < 1e-12);
});

test("version changes on every push and clear (the render loop redraws on change)", () => {
  const r = new SampleRing(4);
  const v0 = r.version;
  r.push(mk(0, 0));
  assert.notEqual(r.version, v0);
  const v1 = r.version;
  r.clear();
  assert.notEqual(r.version, v1);
  assert.equal(r.length, 0);
});

test("RateMeter reports events per second over its window", () => {
  const m = new RateMeter(2000);
  for (let i = 0; i < 30; i++) m.mark(100 * i); // 10 per second for 3 s
  assert.equal(m.rate(2900), 10);
  assert.equal(m.rate(10000), 0); // stale events age out
});

test("a reboot during a disconnect is still detected on the first sample after Reconnect", () => {
  const r = new SampleRing(500);
  for (let i = 0; i < 100; i++) r.push(mk(i, 300000 + 20 * i)); // uptime ~300 s
  r.resetLink(); // Reconnect after the kit rebooted
  r.push(mk(0, 800));
  assert.equal(r.resets, 1);
  assert.equal(r.length, 1); // the old clock is gone, so x stays ascending
  assert.equal(r.drops, 0);
});

test("clear() starts a new clock: a new source may start at any t_ms", () => {
  const r = new SampleRing(10);
  r.push(mk(5, 90000));
  r.clear();
  r.push(mk(0, 0));
  assert.equal(r.resets, 0);
  assert.equal(r.length, 1);
});

test("resetStats() zeroes the counters for a new source", () => {
  const r = new SampleRing(10);
  r.push(mk(0, 0));
  r.push(mk(5, 20));
  r.badPackets = 2;
  r.resetStats();
  assert.deepEqual([r.received, r.drops, r.badPackets, r.resets], [0, 0, 0, 0]);
});
