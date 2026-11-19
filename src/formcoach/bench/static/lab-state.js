// The lab console's state: a pure reducer over server messages and page actions, and the
// small text helpers the panels share. No DOM. The server judges reps and counts; this only
// keeps what it said.
//
// Messages: the server's (hello, state, frame header, imu is not stored, event, link, scan,
// log, error) and page actions: socket_open, socket_close, scan_started {kind},
// dismiss_error {id}. A `log` message may carry `rx`, the page's receive time (ms since epoch);
// a `rejudged` event may carry `rx` (ms, any monotonic page clock) to group a slider gesture.
import { clockText } from "./format.js";

export const LOG_LINES = 200;
const MAX_ERRORS = 5;
const MAX_REPS = 60; // per counter, for the rep dots
const EPOCH_EVENTS = new Set(["counters_reset", "exercise"]);
// Re-judgings received closer than this (ms, page clock `rx`) are one slider gesture: "was"
// keeps the totals from before the gesture, not from the previous step of the drag. The page
// clock, not the event's frame time: once the source has ended every event has the same t.
const GESTURE_MS = 2000;

const counterOf = (arm) => (arm === "l" || arm === "r" ? arm : "all");
const emptyReps = () => ({ l: [], r: [], all: [] });
const finite = (x) => typeof x === "number" && Number.isFinite(x);

/** The state before the first message. */
export function initialState() {
  return {
    connected: false,
    server: null, // the server's state (hello / state messages)
    link: null, // LinkManager.status(), from link messages and the state
    scans: {}, // kind -> [{kind, id, name, rssi, kit}]
    scanning: null, // kind of the scan running
    log: [], // {level, text, t, rx}
    errors: [], // {id, text, cmd}
    nextErrorId: 1,
    totals: {}, // live totals per counter
    was: null, // {good, flagged}: the totals before the latest re-judging gesture
    rejudgedRx: null, // page receive time (ms) of the latest re-judging
    reps: emptyReps(), // rep payloads of the current counters, per counter
    lastRep: null, // the newest rep payload
    frame: null, // the newest frame header (without landmarks)
    capture: { state: "idle", labels: null, summary: null, last: null, t0: null, deadline: null, discarded: false },
    rulesSaved: null, // {path, overwritten}
  };
}

function captureFromServer(prev, c) {
  if (!c) return prev;
  const same = c.state === prev.state;
  return {
    state: c.state,
    labels: c.labels,
    summary: c.summary,
    last: c.last,
    t0: same ? prev.t0 : null,
    deadline: same ? prev.deadline : null,
    discarded: same ? prev.discarded : false,
  };
}

function captureEvent(prev, ev) {
  const p = ev.payload ?? {};
  const next = { ...prev, state: p.state ?? prev.state, discarded: false };
  if (p.state === "countdown") next.deadline = ev.t + (p.delay_s ?? 0);
  if (p.state === "recording") {
    next.t0 = ev.t;
    next.deadline = null;
    if (p.labels) next.labels = p.labels;
  }
  if (p.state === "review") next.summary = p.summary ?? null;
  if (p.state === "idle") {
    next.summary = null;
    next.t0 = next.deadline = null;
    if (p.dir) next.last = { dir: p.dir, check: p.check, warnings: p.warnings };
    if (p.discarded) next.discarded = true;
  }
  return next;
}

function withError(s, text, cmd) {
  const errors = [...s.errors, { id: s.nextErrorId, text, cmd: cmd ?? null }].slice(-MAX_ERRORS);
  return { ...s, errors, nextErrorId: s.nextErrorId + 1 };
}

function onEvent(s, ev) {
  const p = ev.payload ?? {};
  switch (ev.kind) {
    case "rep": {
      const k = counterOf(p.arm);
      return { ...s, reps: { ...s.reps, [k]: [...s.reps[k], p].slice(-MAX_REPS) }, lastRep: p };
    }
    case "rejudged": {
      const now = new Map((p.reps ?? []).map((r) => [`${counterOf(r.arm)} ${r.rep_id}`, r.faults ?? []]));
      const fix = (rp) => {
        const f = now.get(`${counterOf(rp.arm)} ${rp.rep_id}`);
        return f ? { ...rp, faults: f } : rp;
      };
      const reps = Object.fromEntries(Object.entries(s.reps).map(([k, list]) => [k, list.map(fix)]));
      const rx = finite(ev.rx) ? ev.rx : null;
      const sameGesture = s.was && rx != null && s.rejudgedRx != null && rx - s.rejudgedRx < GESTURE_MS;
      return {
        ...s, was: sameGesture ? s.was : sumTotals(s.totals), rejudgedRx: rx, totals: p.totals ?? s.totals, reps,
        lastRep: s.lastRep && fix(s.lastRep),
      };
    }
    case "capture": {
      const capture = captureEvent(s.capture, ev);
      return p.state === "recording" ? { ...s, capture, reps: emptyReps(), lastRep: null, was: null } : { ...s, capture };
    }
    case "params_changed":
      return s.server ? { ...s, server: { ...s.server, values: p.values ?? s.server.values } } : s;
    case "rules_saved":
      return { ...s, rulesSaved: { path: p.path, overwritten: !!p.overwritten } };
    case "error":
      return withError(s, `${p.where ?? "session"}: ${p.error ?? "error"}`, null);
    default:
      return EPOCH_EVENTS.has(ev.kind) ? { ...s, reps: emptyReps(), lastRep: null, was: null } : s;
  }
}

/** The next state after `msg` (a server message or a page action). Never changes `state`. */
export function reduce(state, msg) {
  const s = state;
  switch (msg.type) {
    case "hello":
      return {
        ...s, connected: true, server: msg.state, totals: msg.state.totals ?? {}, link: msg.state.link,
        capture: captureFromServer(s.capture, msg.state.capture), reps: emptyReps(), lastRep: null, was: null,
      };
    case "state":
      return { ...s, server: msg.state, totals: msg.state.totals ?? s.totals, link: msg.state.link ?? s.link, capture: captureFromServer(s.capture, msg.state.capture) };
    case "frame": {
      const { lm: _lm, ...frame } = msg;
      return { ...s, frame, totals: msg.totals ?? s.totals };
    }
    case "link":
      return { ...s, link: msg.status };
    case "scan":
      return { ...s, scans: { ...s.scans, [msg.kind]: msg.devices ?? [] }, scanning: s.scanning === msg.kind ? null : s.scanning };
    case "log":
      return { ...s, log: [...s.log, { level: msg.level, text: msg.text, t: msg.t, rx: msg.rx ?? null }].slice(-LOG_LINES) };
    case "error":
      return { ...withError(s, msg.text, msg.cmd), scanning: msg.cmd === "scan" ? null : s.scanning };
    case "event":
      return onEvent(s, msg);
    case "socket_open":
      return { ...s, connected: true };
    case "socket_close":
      return { ...s, connected: false, scanning: null };
    case "scan_started":
      return { ...s, scanning: msg.kind };
    case "dismiss_error":
      return { ...s, errors: s.errors.filter((e) => e.id !== msg.id) };
    default:
      return s;
  }
}

// ---- views and text -------------------------------------------------------------------------

/** Good and flagged reps over every counter. */
export function sumTotals(totals) {
  let good = 0;
  let flagged = 0;
  for (const c of Object.values(totals ?? {})) {
    good += c?.good ?? 0;
    flagged += c?.flagged ?? 0;
  }
  return { good, flagged };
}

/** "with these values: G good · F flagged (was …)": the live sums and those before the latest re-judging. */
export function rejudgeLine(s) {
  return { ...sumTotals(s.totals), was: s.was };
}

/** The capture panel: its state, seconds left of a countdown and seconds recorded (null when unknown). */
export function captureView(s) {
  const c = s.capture;
  const t = s.frame?.t;
  const countdownS = c.state === "countdown" && finite(c.deadline) && finite(t) ? Math.max(0, Math.ceil(c.deadline - t - 1e-9)) : null;
  const elapsedS = c.state === "recording" && finite(c.t0) && finite(t) ? Math.max(0, t - c.t0) : null;
  return { state: c.state, countdownS, elapsedS };
}

const SOURCE_WORD = { device: "watch", energy: "energy", always_on: "always open", manual: "held" };

/**
 * The top bar's gate pill `{cls, text}`: live ("ok" while open) from the newest frame; neutral
 * "Gate · source ended" once the frame source has ended; neutral "Gate · <source>" while the
 * page is disconnected or before a frame.
 */
export function gatePill(s) {
  const f = s.frame;
  const src = SOURCE_WORD[f?.gate_source ?? s.server?.gate_source] ?? "";
  if (s.connected && s.server?.source === "ended") return { cls: "", text: "Gate · source ended" };
  if (!f || !s.connected) return { cls: "", text: `Gate · ${src}` };
  return { cls: f.gate ? "ok" : "", text: `${f.gate ? "● Gate open" : "○ Gate closed"} · ${src}` };
}

/** The top bar's capture pill `{hidden, cls, text}`; neutral (no colour) while the page is disconnected. */
export function recPill(s) {
  const v = captureView(s);
  const live = v.state === "recording" ? "rec" : "warn";
  return {
    hidden: v.state === "idle",
    cls: s.connected ? live : "",
    text: v.state === "countdown" ? `◔ Get ready${v.countdownS == null ? "" : ` ${v.countdownS}`}`
      : v.state === "recording" ? `● REC ${v.elapsedS == null ? "" : clockText(v.elapsedS)}`.trim() : "■ Take in review",
  };
}

/**
 * The last-cue line when there is no rep to show: "" for idle, "–" when reps were counted that
 * this page never saw (it opened or reconnected mid-set), "No rep yet" otherwise; null when a
 * rep was seen (the page draws it).
 */
export function lastCueText(s) {
  if (s.lastRep) return null;
  if (s.server?.exercise === "idle") return "";
  const counted = Object.values(s.totals ?? {}).some((c) => (c?.reps ?? 0) > 0);
  return counted ? "–" : "No rep yet";
}

/** The counter keys of an exercise: one ("all") for squat, else left and right. */
export function counterKeys(exercise) {
  return exercise === "squat" ? ["all"] : ["l", "r"];
}

/** The scanned devices of `kind`, the connected one first (added when the scan did not list it), each with `connected`. */
export function devicesFor(s, kind) {
  const link = s.link;
  const linked = link && link.state !== "none" && link.kind === kind ? link.id : null;
  const list = (s.scans[kind] ?? []).map((d) => ({ ...d, connected: d.id === linked }));
  const mine = list.find((d) => d.connected);
  const rest = list.filter((d) => !d.connected);
  if (mine) return [mine, ...rest];
  if (linked) return [{ kind, id: link.id, name: link.name, rssi: null, kit: link.kit, connected: true }, ...rest];
  return rest;
}

/** The end of a device identifier ("…B3:8E:94"); short ones whole. */
export function idTail(id) {
  const s = String(id ?? "");
  return s.length > 8 ? `…${s.slice(-8)}` : s;
}

/** Seconds as "45 s", "12 min" or "1 h 05 min"; "–" when unknown. */
export function uptimeText(sec) {
  if (!finite(sec)) return "–";
  if (sec < 60) return `${Math.round(sec)} s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min`;
  return `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
}

const DECIMALS = (step) => (step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step) - 1e-9)));
const UNIT = { deg: "°", s: " s", windows: " win", "(m/s²)²": "" };

/** A slider value with its unit, to the slider's step: "35°", "1.0 s", "0.35", "12 win". */
export function paramText(param, value) {
  if (!finite(value)) return "–";
  const unit = param.unit in UNIT ? UNIT[param.unit] : param.unit ? ` ${param.unit}` : "";
  return `${value.toFixed(DECIMALS(param.step ?? 1))}${unit}`;
}

/** The suggested file for "Save as rules file…": runs/rules/<exercise>-<YYYYMMDD-HHMM>.yaml. */
export function defaultRulesPath(exercise, date = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  const stamp = `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}`;
  return `runs/rules/${exercise}-${stamp}.yaml`;
}

/**
 * A slider's label: a rule parameter "rule.curl.CURL_SWING.upper_arm_trunk_drift" becomes
 * "Swing · upper_arm_trunk_drift" (the exercise prefix dropped from the rule code); others keep
 * the server's label.
 */
export function paramLabel(p) {
  const [kind, exercise, code, ...metric] = p.key.split(".");
  if (kind !== "rule" || !code || !metric.length) return p.label;
  const prefix = `${exercise.toUpperCase()}_`;
  const name = (code.startsWith(prefix) ? code.slice(prefix.length) : code).replaceAll("_", " ").toLowerCase();
  return `${name.charAt(0).toUpperCase()}${name.slice(1)} · ${metric.join(".")}`;
}

/** Whether an error says the rules file exists (the page then asks before sending overwrite). */
export function isRulesConflict(msg) {
  return msg?.cmd === "save_rules" && /already exists/.test(msg.text ?? "");
}

/** The exercise selector's options: the untuned ones say so, idle says it does not count. */
export function exerciseOptions(server) {
  const tuned = new Set(server?.tuned_exercises ?? []);
  return (server?.exercises ?? []).map((value) => ({
    value,
    tuned: tuned.has(value),
    text: value === "idle" ? "idle (no counting)" : tuned.has(value) ? value : `${value} (untuned)`,
  }));
}
