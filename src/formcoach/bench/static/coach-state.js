// The coach page's state: a pure reducer over server messages and page actions, and the
// view-model the DOM is drawn from. No DOM. The server judges reps and counts; this only keeps
// what it said and decides what is on screen.
//
// Messages: the server's (hello, state, frame header, event) and page actions:
// socket_open, socket_close, tick {now} (page clock, s), tone_played.
import { cueText, repDots } from "./format.js";

export const CUE_TTL_S = 8; // an arm's cue stays until its next rep or this long
export const MAX_DOTS = 12; // rep dots per counter
const EPOCH_EVENTS = new Set(["counters_reset", "exercise"]);
const RANK = { warn: 3, info: 2, good: 1 };
const ICON = { good: "✓", warn: "!", info: "i" };
const EXERCISE_NAME = { curl: "CURL", press: "PRESS", raise: "LATERAL RAISE", squat: "SQUAT", idle: "IDLE" };

const emptyReps = () => ({ l: [], r: [], all: [] });
const counterOf = (arm) => (arm === "l" || arm === "r" ? arm : "all");
const finite = (x) => typeof x === "number" && Number.isFinite(x);

/** The state before the first message. */
export function initialState() {
  return {
    connected: false,
    exercise: null,
    totals: {}, // per counter {reps, good, flagged, ...}
    gate: false,
    linkState: "none", // ok, silent, none or waiting
    videoOn: true,
    capture: { state: null, deadline: null, recStart: null }, // deadline: frame clock (s); recStart: page clock (s)
    frameT: null, // newest frame time (s, frame clock)
    now: 0, // page clock (s), set by tick
    reps: emptyReps(), // the last MAX_DOTS rep payloads per counter
    cues: {}, // per counter: {payload, cue, at, seq}
    seq: 0,
    pendingTone: null, // "good" | "fault" | null: what to play on the next animation frame
  };
}

/** Clears reps and cues (a new set: counters reset, exercise changed, recording started). */
function cleared(s) {
  return { ...s, reps: emptyReps(), cues: {}, pendingTone: null };
}

function withCapture(s, state, deadline) {
  if (state == null) return s;
  const prev = s.capture;
  if (state === prev.state) return deadline === undefined ? s : { ...s, capture: { ...prev, deadline } };
  const next = { state, deadline: state === "countdown" ? (deadline ?? null) : null, recStart: null };
  // A page that opens mid-recording (nothing known before) has no start time.
  if (state === "recording" && prev.state !== null) next.recStart = s.now;
  const moved = { ...s, capture: next };
  return state === "recording" && prev.state !== null ? cleared(moved) : moved;
}

function applyServerState(s, st) {
  let out = {
    ...s,
    exercise: st.exercise ?? s.exercise,
    totals: st.totals ?? s.totals,
    linkState: st.link?.state ?? s.linkState,
    videoOn: st.video?.on ?? s.videoOn,
  };
  out = withCapture(out, st.capture?.state);
  return out;
}

function onRep(s, p) {
  const k = counterOf(p.arm);
  const seq = s.seq + 1;
  const tone = p.faults?.length ? "fault" : "good";
  return {
    ...s,
    seq,
    reps: { ...s.reps, [k]: [...s.reps[k], p].slice(-MAX_DOTS) },
    cues: { ...s.cues, [k]: { payload: p, cue: cueText(p), at: s.now, seq } },
    pendingTone: s.pendingTone === "fault" ? "fault" : tone, // a fault tone wins within a frame
  };
}

function onEvent(s, ev) {
  const p = ev.payload ?? {};
  switch (ev.kind) {
    case "rep":
      return onRep(s, p);
    case "capture": {
      return withCapture(s, p.state, p.state === "countdown" ? ev.t + (p.delay_s ?? 0) : undefined);
    }
    default:
      return EPOCH_EVENTS.has(ev.kind) ? cleared(s) : s;
  }
}

/** The next state after `msg` (a server message or a page action). Never changes `state`. */
export function reduce(state, msg) {
  const s = state;
  switch (msg.type) {
    case "hello":
      return { ...cleared(applyServerState(s, msg.state)), connected: true };
    case "state":
      return applyServerState(s, msg.state);
    case "frame": {
      const out = {
        ...s,
        frameT: finite(msg.t) ? msg.t : s.frameT,
        gate: !!msg.gate,
        totals: msg.totals ?? s.totals,
        exercise: msg.exercise ?? s.exercise,
        linkState: msg.link_state ?? s.linkState,
      };
      return withCapture(out, msg.capture_state);
    }
    case "link":
      return { ...s, linkState: msg.status?.state ?? s.linkState };
    case "event":
      return onEvent(s, msg);
    case "tick": {
      const cues = {};
      for (const [k, c] of Object.entries(s.cues)) if (msg.now - c.at < CUE_TTL_S) cues[k] = c;
      const same = Object.keys(cues).length === Object.keys(s.cues).length;
      return { ...s, now: msg.now, cues: same ? s.cues : cues };
    }
    case "tone_played":
      return s.pendingTone ? { ...s, pendingTone: null } : s;
    case "socket_open":
      return { ...s, connected: true };
    case "socket_close":
      return { ...s, connected: false };
    default:
      return s;
  }
}

// ---- view-model -----------------------------------------------------------------------------

/** The counter keys on screen: one ("all") for squat, else left and right. */
export function counterKeys(s) {
  return s.exercise === "squat" || "all" in (s.totals ?? {}) ? ["all"] : ["l", "r"];
}

/** Seconds left of a capture countdown (whole, at least 1), or null when unknown. */
export function countdownSeconds(s) {
  const c = s.capture;
  if (c.state !== "countdown" || !finite(c.deadline) || !finite(s.frameT)) return null;
  return Math.max(1, Math.ceil(c.deadline - s.frameT - 1e-9));
}

/** Seconds recorded so far by the page's clock, or null when the page did not see it start. */
export function recordedSeconds(s) {
  return s.capture.state === "recording" && finite(s.capture.recStart) ? Math.max(0, s.now - s.capture.recStart) : null;
}

function bandOf(s) {
  let top = null;
  for (const c of Object.values(s.cues)) {
    const rank = RANK[c.cue.kind];
    if (!top || rank > RANK[top.cue.kind] || (rank === RANK[top.cue.kind] && c.seq > top.seq)) top = c;
  }
  if (top) {
    const p = top.payload;
    return {
      kind: top.cue.kind, icon: ICON[top.cue.kind], who: top.cue.who, text: top.cue.text,
      key: `${p.arm ?? "all"}:${p.rep_id}:${top.seq}`, arm: p.arm ?? null, rep_id: p.rep_id,
    };
  }
  const n = countdownSeconds(s);
  const counting = s.capture.state === "countdown";
  return {
    kind: "neutral", icon: "", who: "", text: counting && n != null ? `Get ready ${n}` : counting ? "Get ready" : "Ready",
    key: null, arm: null, rep_id: null,
  };
}

/** Everything the page draws, from the state: exercise, counts with dots, the cue band, the state strip. */
export function view(s, clockText) {
  const counts = counterKeys(s).map((k) => {
    const dots = repDots(s.totals?.[k], s.reps[k]).slice(-MAX_DOTS);
    return {
      key: k,
      label: k === "l" ? "L" : k === "r" ? "R" : "",
      count: s.totals?.[k]?.reps ?? 0,
      dots: [...dots, ...Array(MAX_DOTS - dots.length).fill("")],
    };
  });
  const link = { ok: ["WATCH LINKED", "b"], silent: ["WATCH SILENT", "a"] }[s.linkState] ?? ["NO WATCH", "n"];
  const cs = s.capture.state;
  const n = countdownSeconds(s);
  const rec = recordedSeconds(s);
  const recording =
    cs === "recording" ? [rec == null ? "● REC" : `● REC ${clockText(rec)}`, "r"]
    : cs === "countdown" ? [n == null ? "GET READY" : `GET READY ${n}`, "a"]
    : ["READY", "n"];
  return {
    exercise: EXERCISE_NAME[s.exercise] ?? String(s.exercise ?? "").toUpperCase(),
    counts,
    band: bandOf(s),
    strip: {
      gate: s.gate ? ["GATE OPEN", "g"] : ["GATE CLOSED", "n"],
      watch: link,
      rec: recording,
    },
    disconnected: !s.connected,
  };
}
