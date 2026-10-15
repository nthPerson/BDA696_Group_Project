import assert from "node:assert/strict";
import { test } from "node:test";
import { SampleRing } from "../buffer.js";
import { Command, decodeBatch, decodeStatus, Flag } from "../protocol.js";
import { Simulator } from "../sim.js";

const quiet = { onPacket() {}, onStatus() {}, onDisconnect() {} };
const g = (s) => Math.hypot(s.ax, s.ay, s.az) / 4096;

test("simulated batches decode cleanly: 5 samples per 100 ms, continuous seq", () => {
  const sim = new Simulator(quiet);
  const ring = new SampleRing(500);
  for (let i = 0; i < 100; i++) {
    const { samples, error } = decodeBatch(sim.nextBatch());
    assert.equal(error, null);
    assert.equal(samples.length, 5);
    for (const s of samples) ring.push(s);
  }
  assert.equal(ring.received, 500);
  assert.equal(ring.drops, 0);
  assert.equal(ring.latest.t_ms, 9980);
});

test("about 1 g while still; the gate flag and the gyro follow the curl", () => {
  const sim = new Simulator(quiet);
  const still = sim.sampleAt(8000); // 6 s moving, then 4 s still
  assert.ok(Math.abs(g(still) - 1) < 0.03, `|a| = ${g(still)}`);
  assert.equal(still.flags & Flag.GATE_STATE, 0);
  const moving = sim.sampleAt(1500); // peak angular rate of the 0.5 Hz curl
  assert.ok(Math.abs(g(moving) - 1) < 0.03);
  assert.ok(moving.flags & Flag.GATE_STATE);
  assert.ok(Math.abs(moving.gx / 32.8) > 150, `gx = ${moving.gx / 32.8} dps`);
  assert.ok(still.flags & Flag.CALIBRATED);
});

test("START/STOP commands toggle the session flag and bump the session id", async () => {
  const statuses = [];
  const sim = new Simulator({ ...quiet, onStatus: (v) => statuses.push(decodeStatus(v)) });
  await sim.send(Command.START_SESSION);
  assert.ok(sim.sampleAt(0).flags & Flag.SESSION_ACTIVE);
  assert.equal(statuses.at(-1).sessionId, 1);
  assert.equal(statuses.at(-1).sessionActive, 1);
  await sim.send(Command.STOP_SESSION);
  assert.equal(sim.sampleAt(0).flags & Flag.SESSION_ACTIVE, 0);
  await sim.send(Command.PING); // accepted, no effect
  assert.equal(sim.sessionId, 1);
});
