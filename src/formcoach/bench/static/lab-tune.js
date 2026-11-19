// The lab console's tuning panel: gate source, one slider per tuning parameter (grouped), the
// re-judged totals, reset and "Save as rules file…". Slider changes never edit a rules file:
// the server keeps them as overrides.
import { h, setText } from "./dom.js";
import { defaultRulesPath, paramLabel, paramText, rejudgeLine } from "./lab-state.js";

const SOURCES = [["device", "Watch", "The watch's own gate"], ["energy", "Energy", "The laptop's energy gate on the live IMU stream"], ["always_on", "Open", "Always open"], ["manual", "Hold", "Open or closed by hand"]];
const SEND_EVERY_MS = 150; // while a slider is dragged, at most this often
const QUIET_MS = 800; // a slider moved this recently is not reset from the server's state

const titleCase = (s) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();

/**
 * Fill the tuning panel. `send(cmd, fields)`; `onPick(key)` is told which rule slider the user
 * moved (the metric lane follows it). Returns `{render(state), saveConflict(errorMsg)}`.
 */
export function mountTune(root, send, onPick) {
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Gate source" },
    SOURCES.map(([src, text, title]) => h("button", { type: "button", "data-src": src, title, onclick: () => send("set_gate_source", { source: src }) }, text)));
  const holdBtn = h("button", { type: "button", class: "btn hold", onclick: () => send("set_manual_gate", { open: !last.s?.server?.manual_gate }) }, "");
  const hold = h("div", { class: "hold-row", hidden: true }, h("span", { class: "muted" }, "Held"), holdBtn);
  const groups = h("div", { class: "groups" });
  const judged = h("div", { class: "judged" });
  const resetBtn = h("button", { class: "btn", type: "button", onclick: () => send("reset_params") }, "Reset");
  const saveBtn = h("button", { class: "btn", type: "button", onclick: () => save() }, "Save as rules file…");
  const saved = h("div", { class: "saved", hidden: true });
  const rules = h("div", { class: "muted small" });
  const untuned = h("p", { class: "untuned", hidden: true });
  // a scrolling list (gate source and the sliders) above a footer that is always in view and
  // never covers a slider: the re-judged totals, Reset and Save
  const more = h("div", { class: "groups" });
  const list = h("div", { class: "tune-list" }, h("p", { class: "h" }, "Gate source"), seg, hold, groups, more, rules);
  const footer = h("div", { class: "tune-foot" }, judged, h("div", { class: "btns" }, resetBtn, saveBtn), saved);
  root.append(list, footer);
  root.after(untuned); // one line under the tuning panel

  const last = { s: null, paramsKey: null, exercise: null, savePath: null };
  const sliders = new Map(); // key -> {param, input, out, chg, moved, timer, sentAt}

  function save() {
    const ex = last.s?.server?.exercise ?? "rules";
    const path = prompt("Save the rules with these values as (relative to the folder the bench was started in, .yaml):", defaultRulesPath(ex));
    if (!path) return;
    last.savePath = path.trim();
    send("save_rules", { path: last.savePath });
  }

  /** The server said the file exists: ask, then send again with overwrite. */
  function saveConflict(msg) {
    const path = last.savePath;
    if (path && confirm(`${path} already exists. Replace it?`)) send("save_rules", { path, overwrite: true });
    else if (!path) console.warn(msg.text);
  }

  function slider(p, values) {
    const id = `param-${p.key.replace(/[^a-z0-9]+/gi, "-")}`;
    const input = h("input", { type: "range", id, min: p.min, max: p.max, step: p.step });
    const out = h("output", { for: id });
    const chg = h("span", { class: "chg", hidden: true });
    const s = { param: p, input, out, chg, moved: 0, timer: null, sentAt: 0 };
    const push = () => {
      clearTimeout(s.timer);
      s.timer = null;
      s.sentAt = performance.now();
      send("set_param", { key: p.key, value: Number(input.value) });
    };
    input.addEventListener("input", () => {
      s.moved = performance.now();
      setText(out, paramText(p, Number(input.value)));
      if (p.group === "rules") onPick(p.key);
      const wait = SEND_EVERY_MS - (performance.now() - s.sentAt);
      if (wait <= 0) push();
      else if (!s.timer) s.timer = setTimeout(push, wait);
    });
    input.addEventListener("change", () => { s.moved = performance.now(); push(); });
    input.addEventListener("focus", () => { if (p.group === "rules") onPick(p.key); });
    sliders.set(p.key, s);
    update(s, values);
    return h("div", { class: "sl" }, h("label", { for: id, title: p.label }, paramLabel(p), " ", chg), input, out);
  }

  function update(s, values) {
    const p = s.param;
    const v = values?.[p.key] ?? p.default;
    if (performance.now() - s.moved > QUIET_MS) {
      if (Number(s.input.value) !== v) s.input.value = v;
      setText(s.out, paramText(p, v));
    }
    const changed = values?.[p.key] != null && values[p.key] !== p.default;
    s.chg.hidden = !changed;
    setText(s.chg, `● default ${paramText(p, p.default)}`);
  }

  function build(server) {
    sliders.clear();
    const ex = server.exercise;
    const byGroup = (g) => server.params.filter((p) => p.group === g);
    const title = { rules: `${titleCase(ex)} rules`, counter: "Rep counter", gate: "Energy gate" };
    const note = { rules: " (re-judges this session)", gate: " (used when the source is Energy)" };
    const group = (g) => {
      const ps = byGroup(g);
      if (!ps.length) return [];
      const open = g !== "gate" || server.gate_source === "energy";
      return [h("details", { class: "grp", open },
        h("summary", { class: "h" }, title[g], note[g] ? h("span", { class: "note" }, note[g]) : null),
        ps.map((p) => slider(p, server.values)))];
    };
    groups.replaceChildren(...group("rules"));
    more.replaceChildren(...group("counter"), ...group("gate"));
  }

  function render(s) {
    last.s = s;
    const server = s?.server;
    if (!server) return;
    for (const b of seg.children) {
      const on = b.dataset.src === server.gate_source;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
    hold.hidden = server.gate_source !== "manual";
    setText(holdBtn, server.manual_gate ? "● Open (click to close)" : "○ Closed (click to open)");
    holdBtn.classList.toggle("primary", !!server.manual_gate);
    const key = JSON.stringify([server.exercise, server.params.map((p) => p.key)]);
    if (key !== last.paramsKey) {
      last.paramsKey = key;
      build(server);
    } else for (const sl of sliders.values()) update(sl, server.values);

    const j = rejudgeLine(s);
    const was = j.was ? ` (was ${j.was.good} · ${j.was.flagged})` : "";
    setText(judged, `With these values: ${j.good} good · ${j.flagged} flagged${was}`);
    judged.hidden = !server.params.some((p) => p.group === "rules");
    resetBtn.disabled = Object.keys(server.values ?? {}).length === 0;
    saved.hidden = !s.rulesSaved;
    if (s.rulesSaved) setText(saved, `✓ Saved ${s.rulesSaved.path}${s.rulesSaved.overwritten ? " (replaced)" : ""}`);
    setText(rules, `Rules file: ${server.rules ?? "none"}`);
    const tuned = server.tuned_exercises ?? [];
    untuned.hidden = tuned.includes(server.exercise) || server.exercise === "idle";
    setText(untuned, `Only ${tuned.join(", ") || "no exercise"} has thresholds tuned on a webcam; these are design defaults.`);
  }

  return { render, saveConflict };
}
