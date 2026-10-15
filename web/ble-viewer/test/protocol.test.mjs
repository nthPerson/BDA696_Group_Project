import assert from "node:assert/strict";
import { test } from "node:test";
import {
  decodeBatch,
  decodeStatus,
  encodeBatch,
  encodeStatus,
  GOLDEN_BATCH_HEX,
  GOLDEN_SAMPLES,
  GOLDEN_SESSION_ID,
  GOLDEN_STATUS,
  GOLDEN_STATUS_HEX,
  hexToView,
  MAX_BATCH_SIZE,
  viewToHex,
} from "../protocol.js";

test("decodes the golden batch packed by protocol.py", () => {
  const out = decodeBatch(hexToView(GOLDEN_BATCH_HEX));
  assert.equal(out.error, null);
  assert.equal(out.sessionId, GOLDEN_SESSION_ID);
  assert.deepEqual(out.samples, GOLDEN_SAMPLES);
});

test("encodeBatch reproduces the golden bytes", () => {
  const v = encodeBatch(GOLDEN_SESSION_ID, GOLDEN_SAMPLES);
  assert.equal(v.byteLength, MAX_BATCH_SIZE);
  assert.equal(viewToHex(v), GOLDEN_BATCH_HEX);
  assert.throws(() => encodeBatch(1, []), RangeError);
});

test("a truncated packet returns an error and never throws", () => {
  const short = decodeBatch(hexToView(GOLDEN_BATCH_HEX.slice(0, -2)));
  assert.match(short.error, /declares 5 samples \(99 bytes\) but has 98/);
  assert.deepEqual(short.samples, []);
  assert.match(decodeBatch(new DataView(new ArrayBuffer(2))).error, /too short for header/);
});

test("decodes a DataView that starts in the middle of its buffer", () => {
  const golden = new Uint8Array(hexToView(GOLDEN_BATCH_HEX).buffer);
  const padded = new Uint8Array(3 + golden.length);
  padded.set(golden, 3);
  const out = decodeBatch(new DataView(padded.buffer, 3, golden.length));
  assert.deepEqual(out.samples, GOLDEN_SAMPLES);
});

test("status round trip matches protocol.py", () => {
  assert.deepEqual(decodeStatus(hexToView(GOLDEN_STATUS_HEX)), {
    ...GOLDEN_STATUS,
    firmware: "1.1.0",
    error: null,
  });
  assert.equal(viewToHex(encodeStatus(GOLDEN_STATUS)), GOLDEN_STATUS_HEX);
  assert.match(decodeStatus(new DataView(new ArrayBuffer(11))).error, /too short/);
});
