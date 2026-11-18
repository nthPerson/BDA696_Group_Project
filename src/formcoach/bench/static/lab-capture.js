// The lab console's capture panel. It follows state.capture.state: idle (labels and Record),
// countdown (Get ready n, Stop), recording (elapsed time, Stop), review (summary, true reps,
// Keep / Discard); after Keep it shows the kept folder and the session check.
import { clockText } from "./format.js";
import { h, setText } from "./dom.js";
import { captureView, counterKeys } from "./lab-state.js";

const SUBJECTS = ["S1", "S2", "S3", "S4", "S5"];
const ARM_TEXT = { both: "both", l: "left", r: "right" };
const COUNTER_TEXT = { l: "L", r: "R", all: "all" };

/** A kept take's folder from "data/team" on (the server sends the full path), "/"-separated. */
export function shortDir(dir) {
  const s = String(dir ?? "").replaceAll("\\", "/");
  const i = s.lastIndexOf("data/team/");
  return i >= 0 ? s.slice(i) : s;
}

const pct = (x) => (Number.isFinite(x) ? `${Math.round(x)}%` : "–");

/** The review summary as [label, text] rows. */
export function summaryRows(summary) {
  const s = summary ?? {};
  const reps = Object.entries(s.reps_by_arm ?? {}).map(([k, n]) => `${COUNTER_TEXT[k] ?? k} ${n}`).join(" · ") || "none";
  const faults = Object.entries(s.faults ?? {})
    .flatMap(([k, codes]) => Object.entries(codes ?? {}).map(([code, n]) => `${COUNTER_TEXT[k] ?? k} ${code} ×${n}`));
  return [
    ["Duration", Number.isFinite(s.duration_s) ? `${s.duration_s.toFixed(1)} s` : "–"],
    ["Frames", `${s.frames ?? 0} (pose found in ${s.pose_frames ?? 0})`],
    ["IMU", `${Number.isFinite(s.rate_hz) ? s.rate_hz.toFixed(1) : "–"} Hz · ${s.dropped ?? 0} drops · ${s.imu_samples ?? 0} samples`],
    ["Reps", reps],
    ["Faults", faults.length ? faults.join(", ") : "none"],
    ["Gate open", `${pct(s.gate_open_pct)} of frames`],
    ["Cue delay", Number.isFinite(s.latency_ms_median) ? `median ${Math.round(s.latency_ms_median)} ms` : "– (no coach page cue yet)"],
  ];
}

/** Fill the capture panel; `send(cmd, fields)`. Returns `render(state)`. */
export function mountCapture(root, send) {
  const field = (id, label, input) => [h("label", { for: id }, label), input];
  const subject = h("input", { id: "cap-subject", list: "cap-subjects", value: "S1", autocomplete: "off", size: 4 });
  const exercise = h("output", { id: "cap-exercise", title: "Takes follow the session's exercise (top bar)" });
  const form = h("select", { id: "cap-form" });
  const arm = h("select", { id: "cap-arm" });
  const notes = h("input", { id: "cap-notes", placeholder: "camera 2.5 m, …" });
  const delay = h("select", { id: "cap-delay" }, [0, 5, 10].map((d) => h("option", { value: d, selected: d === 5 }, `${d} s`)));
  const autoStop = h("input", { type: "checkbox", id: "cap-auto" });
  const recBtn = h("button", { class: "btn recbtn", type: "button", onclick: () => record() }, "● Record set");
  const idle = h("div", { class: "cap-idle" },
    h("div", { class: "f" },
      field("cap-subject", "Subject", subject), field("cap-exercise", "Exercise", exercise),
      field("cap-form", "Form", form), field("cap-arm", "Arm", arm), field("cap-notes", "Notes", notes),
      field("cap-delay", "Start after", delay)),
    h("datalist", { id: "cap-subjects" }, SUBJECTS.map((s) => h("option", { value: s }))),
    h("label", { class: "check", for: "cap-auto" }, autoStop, " Stop 5 s after the gate closes"),
    h("div", { class: "btns" }, recBtn, h("span", { class: "muted small" }, "IMU + pose + labels, no video")));

  const big = h("div", { class: "cap-big", role: "status" });
  const stopBtn = h("button", { class: "btn", type: "button", onclick: () => send("capture_stop") }, "■ Stop");
  const running = h("div", { class: "cap-run", hidden: true }, big, h("div", { class: "muted small cap-labels" }), h("div", { class: "btns" }, stopBtn));

  const kv = h("div", { class: "kv" });
  const trueReps = h("input", { id: "cap-true", type: "number", min: 0, step: 1, inputmode: "numeric" });
  const trueHint = h("span", { class: "muted small" });
  const keepBtn = h("button", { class: "btn primary", type: "button", onclick: () => keep() }, "Keep take");
  const discardBtn = h("button", { class: "btn", type: "button", onclick: () => send("capture_discard") }, "Discard");
  const review = h("div", { class: "cap-review", hidden: true },
    h("p", { class: "h" }, "Take to review"), kv,
    h("div", { class: "f" }, field("cap-true", "True reps", trueReps)), trueHint,
    h("div", { class: "btns" }, keepBtn, discardBtn));

  const lastBox = h("div", { class: "cap-last", hidden: true, role: "status" });
  root.append(h("p", { class: "h" }, "Capture"), idle, running, review, lastBox);

  const last = { options: null, summary: null, state: null, lastKey: null };

  function record() {
    const labels = { subject: subject.value.trim(), exercise: exercise.value, form: form.value, arm: arm.value, notes: notes.value };
    send("capture_start", { labels, delay_s: Number(delay.value), auto_stop: autoStop.checked });
  }

  function keep() {
    const raw = trueReps.value.trim();
    const n = raw === "" ? null : Number(raw);
    if (n !== null && !(Number.isInteger(n) && n >= 0)) {
      trueReps.setCustomValidity("A whole number of reps, or empty");
      trueReps.reportValidity();
      return;
    }
    trueReps.setCustomValidity("");
    send("capture_keep", { true_reps: n });
  }

  function options(server) {
    const key = JSON.stringify([server.forms, server.arms]);
    if (key === last.options) return;
    last.options = key;
    form.replaceChildren(...server.forms.map((f) => h("option", { value: f }, f)));
    arm.replaceChildren(...server.arms.map((a) => h("option", { value: a }, ARM_TEXT[a] ?? a)));
  }

  function render(s) {
    const server = s?.server;
    if (!server) return;
    options(server);
    setText(exercise, server.exercise);
    exercise.value = server.exercise;
    const v = captureView(s);
    const c = s.capture;
    idle.hidden = v.state !== "idle";
    running.hidden = v.state !== "countdown" && v.state !== "recording";
    review.hidden = v.state !== "review";
    recBtn.disabled = !s.connected;
    if (v.state === "countdown") setText(big, v.countdownS == null ? "Get ready" : `Get ready ${v.countdownS}`);
    if (v.state === "recording") setText(big, `● REC ${v.elapsedS == null ? "" : clockText(v.elapsedS)}`.trim());
    big.classList.toggle("rec", v.state === "recording");
    setText(running.querySelector(".cap-labels"), c.labels ? `${c.labels.subject} · ${c.labels.exercise} · ${c.labels.form} · ${ARM_TEXT[c.labels.arm] ?? c.labels.arm}` : "");
    setText(stopBtn, v.state === "countdown" ? "■ Cancel" : "■ Stop");
    if (v.state === "review" && c.summary !== last.summary) {
      last.summary = c.summary;
      kv.replaceChildren(...summaryRows(c.summary).flatMap(([k, t]) => [h("span", {}, k), h("span", {}, t)]));
    }
    if (v.state !== last.state) {
      if (v.state === "review") trueReps.value = "";
      last.state = v.state;
    }
    const counters = counterKeys(server.exercise);
    setText(trueHint, counters.length === 2 ? "Per arm: with both arms, the reps done by each arm." : "");

    const lastKey = JSON.stringify([c.last, c.discarded, v.state]);
    if (lastKey !== last.lastKey) {
      last.lastKey = lastKey;
      lastBox.hidden = v.state !== "idle" || (!c.last && !c.discarded);
      lastBox.className = "cap-last";
      if (c.discarded) {
        lastBox.replaceChildren("Take discarded: nothing was written.");
      } else if (c.last) {
        const problems = c.last.check?.problems ?? [];
        const warnings = c.last.warnings ?? [];
        lastBox.classList.add(problems.length || warnings.length ? "bad" : "good");
        lastBox.replaceChildren(...[
          h("div", { title: c.last.dir }, "Kept ", h("code", {}, shortDir(c.last.dir))),
          problems.length
            ? h("div", {}, "! Session check found problems:", h("ul", {}, problems.map((p) => h("li", {}, p))))
            : h("div", {}, "✓ Session check: no problems"),
          warnings.length ? h("div", {}, "! Warnings:", h("ul", {}, warnings.map((w) => h("li", {}, w)))) : null,
        ].filter(Boolean));
      }
    }
  }
  return render;
}
