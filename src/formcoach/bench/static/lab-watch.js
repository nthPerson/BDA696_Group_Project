// The lab console's left rail: the watch link (kind, scan list, kit labels, connect), the
// device's health and the link log.
import { clockOf, h, setText } from "./dom.js";
import { devicesFor, idTail, uptimeText } from "./lab-state.js";

const KINDS = [["ble", "BLE"], ["serial", "USB serial"], ["fake", "Fake"]];
const KITS = ["K1", "K2", "K3", "K4", "K5", "K6"];
const LINK_WORD = { ok: ["ok", "● linked"], waiting: ["warn", "◌ waiting"], silent: ["warn", "⚠ silent"] };
const LOG_ICON = { warn: "⚠ ", error: "✖ " };

const num = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const rssiText = (rssi) => (Number.isFinite(rssi) ? `${String(rssi).replace("-", "−")} dBm` : "no signal reading");

/**
 * Fill the three left-rail panels. `send(cmd, fields)` sends a command; `dispatch(action)`
 * updates the page state. Returns `render(state)`.
 */
export function mountWatch({ watch, health, log }, send, dispatch) {
  let kind = null; // the segment chosen; follows the connected link until the user picks one
  const seg = h("div", { class: "seg", role: "group", "aria-label": "How the watch is connected" },
    KINDS.map(([k, text]) => h("button", { type: "button", "data-kind": k, onclick: () => { kind = k; last.devs = null; render(last.s); } }, text)));
  const devs = h("div", { class: "devs" });
  const scanBtn = h("button", { class: "btn", type: "button", onclick: () => { send("scan", { kind }); dispatch({ type: "scan_started", kind }); } }, "Scan");
  const discBtn = h("button", { class: "btn", type: "button", onclick: () => send("disconnect") }, "Disconnect");
  watch.append(h("p", { class: "h" }, "Watch"), seg, devs, h("div", { class: "btns" }, scanBtn, discBtn));

  const kv = h("div", { class: "kv" });
  const cell = {};
  for (const [key, label] of [["kit", "Kit"], ["fw", "Firmware"], ["rate", "Sample rate"], ["drop", "Dropped"], ["cal", "Calibrated"], ["up", "Uptime"], ["inf", "Inference"]]) {
    cell[key] = h("span");
    kv.append(h("span", {}, label), cell[key]);
  }
  const none = h("p", { class: "muted" }, "No watch connected.");
  const devBtn = (action, text, title) => h("button", { class: "btn", type: "button", title, onclick: () => send("device", { action }) }, text);
  const devBtns = [devBtn("identify", "Identify", "Flash this kit's LED"), devBtn("led_good", "LED test", "Show the good-rep LED"), devBtn("calibrate", "Calibrate", "Hold the watch still while it calibrates")];
  health.append(h("p", { class: "h" }, "Device health"), none, kv, h("div", { class: "btns" }, devBtns));

  const lines = h("div", { class: "log", role: "log", "aria-live": "polite" });
  log.append(h("p", { class: "h" }, "Link log"), lines);

  const last = { s: null, devs: null, log: null };

  function deviceRow(d, s) {
    const state = s.link?.state;
    const [cls, word] = LINK_WORD[state] ?? ["", state ?? ""];
    const kit = h("select", {
      class: "kit", "aria-label": `Kit label for ${d.name ?? d.id} ${idTail(d.id)}`, title: "Kit label (remembered on this laptop)",
      onchange: (e) => send("label_kit", { id: d.id, kit: e.target.value || null }),
    }, h("option", { value: "" }, "kit –"), KITS.map((k) => h("option", { value: k, selected: d.kit === k }, k)));
    const action = d.connected
      ? h("span", { class: `pill ${cls}` }, word)
      : h("button", { class: "btn", type: "button", onclick: () => send("connect", { kind: d.kind ?? kind, id: d.id }) }, "Connect");
    return h("div", { class: `dev${d.connected ? " on" : ""}` },
      h("b", {}, d.name ?? d.id), action,
      h("small", {}, `${idTail(d.id)} · ${rssiText(d.rssi)}`), kit);
  }

  function render(s) {
    last.s = s;
    if (!s?.server) return;
    if (kind === null) kind = s.link?.kind ?? "ble";
    for (const b of seg.children) {
      const on = b.dataset.kind === kind;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
    const list = devicesFor(s, kind);
    const key = JSON.stringify([kind, list, s.link?.state, s.scanning, kind in s.scans]);
    if (key !== last.devs) {
      last.devs = key;
      const empty = s.scanning === kind ? "Scanning…" : kind in s.scans ? "Nothing found." : "Press Scan to list watches.";
      devs.replaceChildren(...(list.length ? list.map((d) => deviceRow(d, s)) : [h("p", { class: "muted" }, empty)]));
    }
    const scanning = s.scanning === kind;
    setText(scanBtn, scanning ? "Scanning…" : "Scan");
    scanBtn.disabled = scanning || !s.connected;
    const linked = s.link && s.link.state !== "none";
    discBtn.disabled = !linked || !s.connected;

    const L = s.link ?? {};
    none.hidden = !!linked;
    kv.hidden = !linked;
    devBtns.forEach((b) => { b.disabled = !linked || !s.connected; });
    if (linked) {
      setText(cell.kit, L.kit ?? "no label");
      setText(cell.fw, L.firmware ?? "–");
      setText(cell.rate, Number.isFinite(L.rate_hz) ? `${num(L.rate_hz)} Hz` : "–");
      setText(cell.drop, L.bad_packets ? `${L.dropped ?? 0} (bad packets ${L.bad_packets})` : L.dropped ?? "–");
      setText(cell.cal, L.calibrated == null ? "–" : L.calibrated ? "yes" : "no");
      setText(cell.up, uptimeText(L.uptime_s));
      setText(cell.inf, `${Number.isFinite(L.infer_ms) ? `${num(L.infer_ms)} ms` : "–"} (USB only)`);
    }

    if (s.log !== last.log) {
      last.log = s.log;
      const atBottom = lines.scrollHeight - lines.scrollTop - lines.clientHeight < 24;
      if (!s.log.length) lines.replaceChildren(h("div", { class: "muted" }, "Nothing yet: scans, links and silences since this page opened show here."));
      else lines.replaceChildren(...s.log.map((l) => h("div", { class: `l-${l.level}` },
        h("span", { class: "t" }, l.rx ? clockOf(l.rx) : ""), " ", `${LOG_ICON[l.level] ?? ""}${l.text}`)));
      if (atBottom) lines.scrollTop = lines.scrollHeight;
    }
  }
  return render;
}
