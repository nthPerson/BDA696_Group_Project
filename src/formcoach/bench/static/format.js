// Text for people, from the server's event payloads. Pure: no DOM. Rep numbers shown to
// people are rep_id + 1; "left"/"right" are the person's own sides.

// Metric names that are angles in degrees. Mirrors DEGREE_METRICS in
// src/formcoach/app/tuning.py; keep the two in step.
export const DEGREE_METRICS = new Set([
  "elbow_min", "elbow_max", "elbow_min_l", "elbow_min_r", "elbow_max_l", "elbow_max_r",
  "elbow_asymmetry", "elbow_top_asymmetry", "upper_arm_trunk_range", "upper_arm_trunk_drift",
  "trunk_incl_max", "abd_peak", "elbow_at_abd_peak", "knee_min", "trunk_incl_at_bottom",
]);

const ARM_NAMES = { l: "LEFT ARM", r: "RIGHT ARM" };
const OPS = { gt: ">", ge: ">", lt: "<", le: "<" };

const num = (x) => (x == null || Number.isNaN(x) ? "–" : String(Math.round(x * 10) / 10));
const unitOf = (metric) => (DEGREE_METRICS.has(metric) ? "°" : metric === "duration_s" ? " s" : "");

/** A rep event (the message or its payload) -> `{who, text, kind}`; kind is good, warn or info. */
export function cueText(repEvent) {
  const p = repEvent.payload ?? repEvent;
  const n = `REP ${(p.rep_id ?? 0) + 1}`;
  const arm = ARM_NAMES[p.arm];
  const who = arm ? `${arm} · ${n}` : n;
  const faults = p.faults ?? [];
  if (!faults.length) return { who, text: "Good rep", kind: "good" };
  let top = faults[0];
  for (const f of faults) if (f.severity === "warn" && top.severity !== "warn") top = f;
  return { who, text: top.message ?? top.code, kind: top.severity === "warn" ? "warn" : "info" };
}

/** A fault -> "upper_arm_trunk_drift 41° > limit 35° (6° over)" (" s" for duration_s, "under" for < limits). */
export function marginText(fault) {
  const u = unitOf(fault.metric);
  const op = OPS[fault.op] ?? "?";
  const below = op === "<";
  const margin = Math.abs(fault.value - fault.threshold);
  return `${fault.metric} ${num(fault.value)}${u} ${op} limit ${num(fault.threshold)}${u} (${num(margin)}${u} ${below ? "under" : "over"})`;
}

/**
 * One dot per rep, oldest first: "g" clean, "w" flagged, "u" counted but not seen by this page
 * (it opened or reconnected mid-set, or no longer holds the event): its verdict and its place
 * among the others are unknown, so it is drawn neutral, without a tick or "!". `events` are the
 * rep events (or payloads) of one counter; the unseen reps come first, then the seen ones in
 * their real order.
 * @param {{reps:number}|undefined} totalsForCounter
 */
export function repDots(totalsForCounter, events) {
  const dots = (events ?? []).map((e) => ((e.payload ?? e).faults?.length ? "w" : "g"));
  const missing = Math.max(0, (totalsForCounter?.reps ?? dots.length) - dots.length);
  return [...Array(missing).fill("u"), ...dots];
}

/** Seconds -> "m:ss" ("0:42", "12:05"); negative or non-finite values give "0:00". */
export function clockText(seconds) {
  const s = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
