// FormCoach BLE live viewer: wires the page to a source (BleLink or Simulator), keeps the last
// 10 s in a SampleRing, and redraws on animation frames, decoupled from BLE notifications.
// Design: docs/superpowers/specs/2026-10-15-ble-viewer-design.md
import { BleLink, bluetoothAvailable } from "./ble.js";
import { RateMeter, SampleRing } from "./buffer.js";
import { Charts } from "./charts.js";
import { ACCEL_LSB_PER_G, Command, decodeBatch, decodeStatus, Flag, SAMPLE_RATE_HZ } from "./protocol.js";
import { csvFilename, Recorder } from "./recorder.js";
import { Simulator } from "./sim.js";

const WINDOW_S = 10;
const CONNECT_HINT =
  "Only one app can be connected to a kit at a time: close ble-check.py, formcoach demo/record " +
  "and other viewer tabs. Is the antenna clipped on, and is the kit's LED blinking (advertising)?";
const MTU_HINT = "a full batch needs a negotiated MTU of at least 102 bytes";

const $ = (id) => document.getElementById(id);
const ring = new SampleRing(WINDOW_S * SAMPLE_RATE_HZ);
const packetRate = new RateMeter();
const sampleRate = new RateMeter();
const recorder = new Recorder();
const charts = new Charts($("chart-accel"), $("chart-gyro"));

let link = null; // the active source: BleLink or Simulator
let status = null; // last decoded FcStatus
let busy = false; // a chooser/connect is in progress: source buttons are locked
let recordingFrom = null; // name of the source the open recording belongs to
let drawnVersion = -1;

function handlePacket(view) {
  const now = performance.now();
  packetRate.mark(now);
  const { samples, error } = decodeBatch(view);
  if (error) {
    ring.badPackets++;
    showMessage(`Bad packet: ${error} (${MTU_HINT}).`, "warn");
    return;
  }
  for (const s of samples) ring.push(s);
  sampleRate.mark(now, samples.length);
  recorder.add(samples);
}

/** Callbacks for one source. Events from any source other than the active one are ignored,
 * so a link that is still finishing its connect after the user moved on cannot mix in. */
function newSource(Kind) {
  const cb = {
    onPacket: (view) => cb.owner === link && handlePacket(view),
    onStatus: (view) => {
      if (cb.owner !== link) return;
      const st = decodeStatus(view);
      if (!st.error) status = st;
    },
    onDisconnect: () => {
      if (cb.owner !== link) return;
      const kept = recorder.active ? " The recording is kept: press Stop to download it." : "";
      showMessage(`Disconnected at ${new Date().toLocaleTimeString()}.${kept}`, "warn");
      updateButtons();
    },
  };
  cb.owner = new Kind(cb);
  return cb.owner;
}

/** Make `next` the active source: save an open recording under the old source's name, start
 * a fresh plot and fresh counters, and release the old source. */
async function switchSource(next) {
  const prev = link;
  if (recorder.active) {
    const name = recordingFrom;
    download(recorder.stop(), csvFilename(name));
    showMessage(`Saved the recording from ${name} before switching.`, "info");
  }
  link = next;
  status = null;
  ring.clear();
  ring.resetStats();
  if (prev?.connected) await prev.disconnect();
}

function showMessage(text, kind = "info") {
  const el = $("message");
  el.textContent = text;
  el.className = `message ${kind}`;
  el.hidden = false;
}

function clearMessage() {
  $("message").hidden = true;
}

function updateButtons() {
  const connected = !!link && link.connected;
  const ble = link instanceof BleLink;
  const sim = link instanceof Simulator;
  $("connect").disabled = busy || !bluetoothAvailable() || !window.isSecureContext || (ble && connected);
  $("reconnect").hidden = !(ble && link.device && !connected);
  $("reconnect").disabled = busy;
  $("simulate").textContent = sim && connected ? "Stop simulation" : "Simulate";
  $("simulate").disabled = busy || (ble && connected);
  $("disconnect").disabled = busy || !(ble && connected);
  for (const b of document.querySelectorAll("[data-needs-control]")) {
    b.disabled = busy || !(connected && link.hasControl);
    b.title = connected && !link.hasControl ? "This firmware has no control characteristic" : "";
  }
  $("record").disabled = !connected && !recorder.active;
  $("record").textContent = recorder.active ? "■ Stop & download CSV" : "● Record";
  const named = sim || (ble && link.device);
  $("device").textContent = named ? `${link.name}${connected ? "" : " (disconnected)"}` : "not connected";
}

function connectError(err) {
  const cancelled = err.name === "NotFoundError" && /cancel/i.test(err.message);
  if (!cancelled) showMessage(`Could not connect: ${err.message}. ${CONNECT_HINT}`, "error");
}

/** Lock the source buttons while `fn` runs (chooser + GATT connect take seconds). */
async function whileBusy(fn) {
  busy = true;
  updateButtons();
  try {
    await fn();
  } finally {
    busy = false;
    updateButtons();
  }
}

/** Connect…: always opens the chooser, so any kit can be picked. Cancelling keeps the old source. */
const connectBle = () =>
  whileBusy(async () => {
    clearMessage();
    const fresh = newSource(BleLink);
    try {
      await fresh.choose();
    } catch (err) {
      connectError(err);
      return;
    }
    await switchSource(fresh);
    try {
      await fresh.reconnect();
      clearMessage();
    } catch (err) {
      connectError(err);
    }
  });

/** Reconnect: the kit picked last, without the chooser; the plot and counters are kept. */
const reconnectBle = () =>
  whileBusy(async () => {
    clearMessage();
    ring.resetLink();
    try {
      await link.reconnect();
      clearMessage();
    } catch (err) {
      connectError(err);
    }
  });

const toggleSimulation = () =>
  whileBusy(async () => {
    clearMessage();
    if (link instanceof Simulator && link.connected) {
      await link.disconnect();
      return;
    }
    const sim = newSource(Simulator);
    await switchSource(sim);
    await sim.connect();
  });

async function send(cmd, label) {
  try {
    await link.send(cmd);
    showMessage(`Sent ${label}.`, "info");
  } catch (err) {
    showMessage(`${label} failed: ${err.message}`, "error");
  }
}

function download(text, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const latestFlags = () => ring.latest?.flags ?? 0;

$("connect").addEventListener("click", connectBle);
$("reconnect").addEventListener("click", reconnectBle);
$("simulate").addEventListener("click", toggleSimulation);
$("disconnect").addEventListener("click", () => link?.disconnect());
$("ping").addEventListener("click", () => send(Command.PING, "PING (the serial monitor prints '# pong')"));
$("led-good").addEventListener("click", () => send(Command.LED_GOOD_REP, "LED good rep"));
$("led-fault").addEventListener("click", () => send(Command.LED_FAULT, "LED fault"));
$("session-toggle").addEventListener("click", () =>
  latestFlags() & Flag.SESSION_ACTIVE
    ? send(Command.STOP_SESSION, "STOP_SESSION")
    : send(Command.START_SESSION, "START_SESSION"),
);
$("calibrate").addEventListener("click", () => {
  if (window.confirm("Lay the kit flat and keep it still for 3 s after you press OK.")) {
    send(Command.CALIBRATE, "CALIBRATE (keep it flat and still for 3 s)");
  }
});
$("record").addEventListener("click", () => {
  if (recorder.active) {
    download(recorder.stop(), csvFilename(recordingFrom));
  } else {
    recordingFrom = link?.name;
    recorder.start();
  }
  updateButtons();
});

function fmtUptime(s) {
  const m = Math.floor(s / 60);
  return m ? `${m} min ${s % 60} s` : `${s} s`;
}

function frame() {
  const now = performance.now();
  if (ring.version !== drawnVersion) {
    charts.update(ring.series());
    drawnVersion = ring.version;
  }
  $("pkt-rate").textContent = packetRate.rate(now).toFixed(1);
  $("hz").textContent = sampleRate.rate(now).toFixed(1);
  $("drops").textContent = ring.drops ? `${ring.drops} (${(100 * ring.dropRate()).toFixed(1)} %)` : "0";
  $("bad").textContent = String(ring.badPackets);
  const last = ring.latest;
  $("amag").textContent = last ? (Math.hypot(last.ax, last.ay, last.az) / ACCEL_LSB_PER_G).toFixed(2) : "–";
  const f = latestFlags();
  $("pill-gate").classList.toggle("on", !!(f & Flag.GATE_STATE));
  $("pill-button").classList.toggle("on", !!(f & Flag.BUTTON_PRESSED));
  $("pill-session").classList.toggle("on", !!(f & Flag.SESSION_ACTIVE));
  $("session-toggle").textContent = f & Flag.SESSION_ACTIVE ? "Stop session" : "Start session";
  $("fw").textContent = status ? (link instanceof Simulator ? "simulated" : status.firmware) : "–";
  $("uptime").textContent = status ? fmtUptime(status.uptimeS) : "–";
  $("calibrated").textContent = status ? (status.calibrated ? "yes" : "no") : "–";
  $("session").textContent = status ? `#${status.sessionId}${status.sessionActive ? " (active)" : ""}` : "–";
  $("rec-count").textContent = recorder.active ? `recording: ${recorder.count} samples` : "";
  requestAnimationFrame(frame);
}

function init() {
  const banner = $("unsupported");
  if (!window.isSecureContext) {
    banner.textContent =
      "This page must be opened over https (GitHub Pages) or from http://localhost (make ble-viewer). Simulate still works.";
    banner.hidden = false;
  } else if (!bluetoothAvailable()) {
    banner.textContent =
      "This browser has no Web Bluetooth. Use Chrome or Edge on Windows, macOS or Android. Simulate still works.";
    banner.hidden = false;
  }
  updateButtons();
  requestAnimationFrame(frame);
}

init();
