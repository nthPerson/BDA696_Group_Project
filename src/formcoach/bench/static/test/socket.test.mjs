import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { BenchSocket, decodeFrame } from "../socket.js";

const golden = JSON.parse(readFileSync(new URL("../../../../../tests/golden/bench_messages.json", import.meta.url), "utf8"));

const toBuffer = (hex) => {
  const u8 = Uint8Array.from(Buffer.from(hex, "hex"));
  return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
};

class FakeWS {
  static all = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    FakeWS.all.push(this);
  }
  send(s) { this.sent.push(s); }
  close() { this.readyState = 3; this.onclose?.(); }
  open() { this.readyState = 1; this.onopen?.(); }
  drop() { this.readyState = 3; this.onclose?.(); }
  recv(data) { this.onmessage?.({ data }); }
}

function make(handlers = {}) {
  FakeWS.all = [];
  const timers = [];
  const sock = new BenchSocket("ws://x/ws", {
    WebSocket: FakeWS,
    setTimer: (fn, ms) => timers.push({ fn, ms }) - 1,
    clearTimer: () => {},
    ...handlers,
  });
  return { sock, timers, ws: () => FakeWS.all.at(-1) };
}

test("decodeFrame gives the golden header and JPEG length", () => {
  const { header, jpeg } = decodeFrame(toBuffer(golden.frame_hex));
  // the golden `frame` message is the same header without an image; the hex frame has one
  assert.deepEqual(header, { ...golden.frame, has_image: true });
  assert.equal(jpeg.length, golden.frame_jpeg_len);
});

test("decodeFrame: no JPEG bytes gives null; a short buffer throws", () => {
  const head = new TextEncoder().encode('{"type":"frame"}');
  const buf = new Uint8Array(4 + head.length);
  new DataView(buf.buffer).setUint32(0, head.length);
  buf.set(head, 4);
  assert.deepEqual(decodeFrame(buf.buffer), { header: { type: "frame" }, jpeg: null });
  assert.throws(() => decodeFrame(new ArrayBuffer(2)));
  assert.throws(() => decodeFrame(buf.buffer.slice(0, 10)));
});

test("messages are routed to their handlers; hello and state both reach onState", () => {
  const got = [];
  const names = ["Hello", "State", "Imu", "Event", "Link", "Scan", "Log", "Error"];
  const h = Object.fromEntries(names.map((n) => [`on${n}`, (...a) => got.push([n, ...a])]));
  h.onFrame = (hd, jpeg) => got.push(["Frame", hd.i, jpeg.length]);
  const { ws } = make(h);
  ws().open();
  for (const k of ["hello", "state", "imu", "event", "link", "scan", "log", "error"]) ws().recv(JSON.stringify(golden[k]));
  ws().recv(toBuffer(golden.frame_hex));
  assert.deepEqual(got.map((g) => g[0]), ["Hello", "State", "State", "Imu", "Event", "Link", "Scan", "Log", "Error", "Frame"]);
  assert.deepEqual(got[1][1], golden.hello.state);
  assert.deepEqual(got[2][1], golden.state.state);
  assert.deepEqual(got[5][1], golden.link.status);
  assert.deepEqual(got[9], ["Frame", 7, golden.frame_jpeg_len]);
});

test("a malformed message goes to onError and does not throw", () => {
  const errs = [];
  const { ws } = make({ onError: (m) => errs.push(m) });
  ws().recv("{not json");
  assert.equal(errs.length, 1);
  assert.match(errs[0].text, /bad message/);
});

test("send writes {cmd, ...fields} when open and nothing otherwise", () => {
  const { sock, ws } = make();
  assert.equal(sock.send("scan", { kind: "ble" }), false);
  ws().open();
  assert.equal(sock.send("set_exercise", { exercise: "curl" }), true);
  assert.equal(sock.send("disconnect"), true);
  assert.deepEqual(ws().sent.map((s) => JSON.parse(s)), [{ cmd: "set_exercise", exercise: "curl" }, { cmd: "disconnect" }]);
});

test("reconnects with backoff 0.5 s doubling to 5 s; an open resets it", () => {
  const { timers, ws } = make();
  const delays = [];
  for (let i = 0; i < 6; i++) {
    ws().drop();
    delays.push(timers.at(-1).ms);
    timers.at(-1).fn();
  }
  assert.deepEqual(delays, [500, 1000, 2000, 4000, 5000, 5000]);
  assert.equal(FakeWS.all.length, 7);
  ws().open();
  ws().drop();
  assert.equal(timers.at(-1).ms, 500);
});

test("close() stops reconnecting", () => {
  const { sock, timers } = make();
  sock.close();
  assert.equal(timers.length, 0);
});
