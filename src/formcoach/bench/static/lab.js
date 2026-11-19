// The lab console (/lab): one socket in, the pure reducer (lab-state.js) keeps the page state,
// and each panel module draws its part. The strip charts and the video panel are fed straight
// from the socket and drawn from requestAnimationFrame.
import { cueText, marginText, repDots } from "./format.js";
import { h, setText } from "./dom.js";
import { mountCapture } from "./lab-capture.js";
import {
  counterKeys, exerciseOptions, gatePill, initialState, isRulesConflict, lastCueText, paramLabel, paramText, recPill, reduce,
} from "./lab-state.js";
import { mountTune } from "./lab-tune.js";
import { LandmarkHistory, VideoPanel } from "./lab-video.js";
import { mountWatch } from "./lab-watch.js";
import { Timeline } from "./series.js";
import { BenchSocket } from "./socket.js";
import { metricChoice, StripData } from "./strip-model.js";
import { metricKeyText, StripCharts } from "./strips.js";

export { reduce } from "./lab-state.js";

const SPAN_S = 20; // seconds the charts show
const KEEP_S = 300; // seconds of history kept for scrolling back
const CHART_MS = 33; // the charts redraw at most about 30 times a second while running
const MAX_DOTS = 30; // at most this many rep dots per counter (the newest that fit)
const ANGLE_TITLE = { curl: "Elbow angle", press: "Elbow angle", raise: "Shoulder abduction", squat: "Knee angle" };
const COUNTER_TEXT = { l: "L", r: "R", all: "Reps" };
const DOT_GLYPH = { g: "✓", w: "!" }; // "u" (counted before this page saw it) has none
const DOT_TITLE = { g: "good", w: "flagged", u: "counted before this page saw it" };
const $ = (id) => document.getElementById(id);

function boot() {
  let state = initialState();
  let dirty = true; // every panel redraws
  let frameDirty = false; // a frame came: the panels that show frame values redraw
  let chartsDirty = true;
  let pickedMetric = null;
  const dispatch = (msg) => {
    state = reduce(state, msg);
    dirty = true;
  };

  const timeline = new Timeline(SPAN_S, KEEP_S);
  const data = new StripData(timeline);
  const charts = new StripCharts($("charts-lanes"), data);
  const history = new LandmarkHistory();
  const video = new VideoPanel($("video"), history);

  let sock = null;
  const send = (cmd, fields = {}) => {
    if (!sock?.send(cmd, fields)) dispatch({ type: "error", text: `not sent (${cmd}): the bench is not connected`, cmd });
  };

  const renderWatch = mountWatch({ watch: $("watch"), health: $("health"), log: $("linklog") }, send, dispatch);
  const tune = mountTune($("tune"), send, (key) => { pickedMetric = key; dirty = true; });
  const renderCapture = mountCapture($("capture"), send);
  const metricSelect = h("select", { class: "metric-pick", "aria-label": "Rule metric shown per rep", onchange: (e) => { pickedMetric = e.target.value; dirty = true; } });
  const metricLabel = charts.labelOf("metrics");
  metricLabel.replaceChildren(h("b", {}, "Per-rep metric"), metricSelect, h("span", { class: "metric-key" }));

  // ---- top bar, error bar, counts ------------------------------------------------------------
  $("exercise").addEventListener("change", (e) => send("set_exercise", { exercise: e.target.value }));
  $("reset-counters").addEventListener("click", () => send("reset_counters"));
  // a window of its own (not a tab, which would hide this page or be hidden by it): a hidden
  // coach page stops drawing, and its cue delays would be false
  $("open-coach").addEventListener("click", () => window.open("/coach", "formcoach-coach", "popup,width=1280,height=720"));
  $("video-toggle").addEventListener("click", () => send("set_video", { on: !(state.server?.video?.on ?? true) }));

  function renderTop(s) {
    const server = s.server;
    const sel = $("exercise");
    const opts = exerciseOptions(server);
    const key = JSON.stringify(opts);
    if (sel.dataset.key !== key) {
      sel.dataset.key = key;
      sel.replaceChildren(...opts.map((o) => h("option", { value: o.value }, o.text)));
    }
    if (server && sel.value !== server.exercise) sel.value = server.exercise; // a refused change snaps back
    sel.disabled = !server;

    const L = s.link;
    const link = $("pill-link");
    const tail = L?.id ? ` ${L.id.length > 8 ? `…${L.id.slice(-5)}` : L.id}` : "";
    const rate = Number.isFinite(L?.rate_hz) ? ` · ${Math.round(L.rate_hz)} Hz` : "";
    const [cls, text] = !s.connected ? ["", "○ Watch: no news (bench disconnected)"]
      : !L || L.state === "none" ? ["", "○ No watch"]
      : L.state === "ok" ? ["ok", `● Watch${L.kit ? ` ${L.kit}` : ""}${tail}${rate}`]
        : L.state === "silent" ? ["warn", `⚠ Watch silent${tail}`] : ["warn", `◌ Watch waiting${tail}`];
    link.className = `pill ${cls}`;
    setText(link, text);

    const f = s.frame;
    const gate = $("pill-gate");
    const g = gatePill(s);
    gate.className = `pill ${g.cls}`;
    setText(gate, g.text);

    const r = recPill(s);
    const rec = $("pill-rec");
    rec.hidden = r.hidden;
    rec.className = `pill ${r.cls}`;
    setText(rec, r.text);

    $("pill-socket").hidden = s.connected;
    const fps = Number.isFinite(f?.fps) ? ` · ${Math.round(f.fps)} fps` : "";
    setText($("source-text"), server ? `source: ${server.source}${fps}` : "");

    const bar = $("errbar");
    const err = s.errors.at(-1);
    bar.hidden = !err;
    if (err) {
      setText($("err-text"), `${err.text}${err.cmd ? ` (${err.cmd})` : ""}${s.errors.length > 1 ? `  · ${s.errors.length - 1} more` : ""}`);
      bar.dataset.id = err.id;
    }
  }
  $("err-dismiss").addEventListener("click", () => dispatch({ type: "dismiss_error", id: Number($("errbar").dataset.id) }));

  function renderCounts(s) {
    const server = s.server;
    if (!server) return;
    const keys = counterKeys(server.exercise);
    const arms = server.exercise === "squat" ? "" : " · both arms";
    setText($("mini-title"), server.exercise === "idle" ? "Idle · no counting" : `${server.exercise}${arms}`);
    const counts = $("counts");
    const dotsBox = $("dots");
    if (counts.dataset.keys !== keys.join()) {
      counts.dataset.keys = keys.join();
      counts.replaceChildren(...keys.map((k) => h("div", { class: "count" }, h("i", {}, COUNTER_TEXT[k]), h("b", { "data-k": k }, "0"))));
      dotsBox.replaceChildren(...keys.map((k) => h("div", { class: "dotrow" }, h("i", {}, k === "all" ? "" : COUNTER_TEXT[k]), h("div", { class: "dots", "data-k": k }))));
    }
    for (const k of keys) {
      setText(counts.querySelector(`b[data-k="${k}"]`), s.totals?.[k]?.reps ?? 0);
      const row = dotsBox.querySelector(`.dots[data-k="${k}"]`);
      // as many of the newest dots as fit the row (a dot and its gap are about 1.5 em)
      const fit = Math.max(3, Math.min(MAX_DOTS, Math.floor((row.clientWidth + 9) / 27)));
      const dots = repDots(s.totals?.[k], s.reps[k]).slice(-fit);
      const dkey = dots.join("");
      if (row.dataset.key !== dkey) {
        row.dataset.key = dkey;
        row.replaceChildren(...dots.map((d) => h("div", { class: `dot ${d}`, title: DOT_TITLE[d] }, DOT_GLYPH[d] ? h("span", {}, DOT_GLYPH[d]) : null)));
      }
    }
    const cue = $("cue");
    const margin = $("margin");
    const rep = s.lastRep;
    const none = lastCueText(s);
    if (none != null) {
      cue.className = "cue";
      setText(cue, none);
      setText(margin, "");
    } else {
      const c = cueText(rep);
      const n = (rep.rep_id ?? 0) + 1;
      const who = rep.arm === "l" || rep.arm === "r" ? `${rep.arm.toUpperCase()} rep ${n}` : `Rep ${n}`;
      cue.className = `cue ${c.kind}`;
      setText(cue, `${c.kind === "good" ? "✓" : c.kind === "warn" ? "!" : "i"} ${who}: ${c.text}`);
      const faults = rep.faults ?? [];
      const top = faults.find((x) => x.severity === "warn") ?? faults[0];
      setText(margin, top ? marginText(top) : "");
    }
    const on = server.video?.on ?? true;
    const vt = $("video-toggle");
    setText(vt, on ? "Video: on" : "Video: off (skeleton only)");
    vt.setAttribute("aria-pressed", String(on));
  }

  function updateMetric(s) {
    const server = s.server;
    if (!server) return;
    const rules = server.params.filter((p) => p.group === "rules");
    const key = JSON.stringify(rules.map((p) => p.key));
    if (metricSelect.dataset.key !== key) {
      metricSelect.dataset.key = key;
      metricSelect.replaceChildren(...rules.map((p) => h("option", { value: p.key, title: p.label }, paramLabel(p))));
    }
    const choice = metricChoice(server.params, server.values, pickedMetric);
    metricSelect.hidden = !choice;
    if (choice && metricSelect.value !== choice.key) metricSelect.value = choice.key;
    const prev = data.metric;
    if (JSON.stringify(prev) !== JSON.stringify(choice)) {
      data.setMetric(choice);
      chartsDirty = true;
    }
    const angle = charts.labelOf("angle")?.querySelector("b");
    if (angle) setText(angle, ANGLE_TITLE[server.exercise] ?? "Angle");
  }

  function renderMetricKey() {
    const m = charts.view?.metric ?? null;
    const p = m && state.server?.params.find((x) => x.key === m.key);
    setText(metricLabel.querySelector(".metric-key"), metricKeyText(m, p ? paramText(p, m.limit) : ""));
  }

  // ---- pause and scroll back -----------------------------------------------------------------
  const pauseBtn = $("pause");
  const scroll = $("scroll");
  const badge = $("behind");
  function setPaused(p) {
    if (p) timeline.pause();
    else timeline.resume();
    video.setPaused(p);
    chartsDirty = true;
    renderPause();
  }
  pauseBtn.addEventListener("click", () => setPaused(!timeline.paused));
  scroll.addEventListener("input", () => {
    if (!timeline.paused) setPaused(true);
    timeline.scrollBy(timeline.newest + Number(scroll.value) - timeline.tEnd);
    chartsDirty = true;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === " " && e.target === document.body) {
      e.preventDefault();
      setPaused(!timeline.paused);
    }
  });
  function renderPause() {
    const p = timeline.paused;
    setText(pauseBtn, p ? "▶ Resume" : "Pause");
    pauseBtn.setAttribute("aria-pressed", String(p));
    const back = Math.max(0, Math.min(KEEP_S, timeline.newest - data.oldestT));
    scroll.min = String(-Math.ceil(back));
    if (document.activeElement !== scroll || !p) scroll.value = String(-(timeline.newest - timeline.tEnd));
    const behind = Math.round(timeline.newest - timeline.tEnd);
    badge.hidden = !p;
    setText(badge, `paused, ${behind} s behind${behind >= 1 ? " · skeleton from kept landmarks (no video kept)" : ""}`);
  }

  // ---- socket --------------------------------------------------------------------------------
  sock = new BenchSocket(null, {
    onHello: (msg) => { dispatch(msg); chartsDirty = true; },
    onState: (st) => dispatch({ type: "state", state: st }),
    onFrame: (header, jpeg) => {
      state = reduce(state, header);
      frameDirty = true;
      data.pushFrame(header);
      video.onFrame(header, jpeg, { paused: timeline.paused, videoOn: state.server?.video?.on ?? true });
    },
    onImu: (msg) => data.pushImu(msg),
    onEvent: (msg) => { data.pushEvent(msg); dispatch({ ...msg, rx: performance.now() }); chartsDirty = true; },
    onLink: (status) => { data.pushLink(status); dispatch({ type: "link", status }); },
    onScan: (msg) => dispatch(msg),
    onLog: (msg) => dispatch({ ...msg, rx: Date.now() }),
    onError: (msg) => (isRulesConflict(msg) ? tune.saveConflict(msg) : dispatch(msg)),
    onOpen: () => dispatch({ type: "socket_open" }),
    onClose: () => dispatch({ type: "socket_close" }),
  });

  // ---- drawing -------------------------------------------------------------------------------
  let lastPauseRender = 0;
  let lastCharts = 0;
  function frame(now) {
    if (dirty) {
      dirty = false;
      frameDirty = true;
      updateMetric(state);
      renderWatch(state);
      tune.render(state);
    }
    if (frameDirty) {
      frameDirty = false;
      renderTop(state);
      renderCounts(state);
      renderCapture(state);
    }
    if (chartsDirty || (!timeline.paused && now - lastCharts >= CHART_MS)) {
      chartsDirty = false;
      lastCharts = now;
      charts.render();
      renderMetricKey();
    }
    if (now - lastPauseRender > 250) {
      lastPauseRender = now;
      renderPause();
    }
    video.render({ paused: timeline.paused, tEnd: timeline.tEnd });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

if (typeof document !== "undefined" && document.body?.classList.contains("lab")) boot();
