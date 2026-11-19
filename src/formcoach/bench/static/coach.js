// The coach page (/coach): the demo mode, read from 2 to 3 m. One socket in; the pure reducer
// (coach-state.js) keeps the state and `view()` says what is on screen; this file draws it, plays
// the tones and tells the server when a cue was drawn (cue_shown).
import { clockText } from "./format.js";
import { h, setText } from "./dom.js";
import { armAngles, drawSkeleton } from "./skeleton.js";
import { BenchSocket } from "./socket.js";
import { initialState, reduce, view } from "./coach-state.js";
import * as tones from "./tones.js";

export { reduce } from "./coach-state.js";

const CURSOR_HIDE_MS = 3000;
const BAND_MAX_VH = 13; // the cue text is as large as fits in two lines, between these sizes
const BAND_MIN_VH = 9;
const NO_PICTURE_ASPECT = 4 / 3; // the skeleton's frame when the server sends no image
const DOT_GLYPH = { g: "✓", w: "!" }; // "u" (counted, not seen) and "" (empty slot) have none
const $ = (id) => document.getElementById(id);

/** The mirrored picture and skeleton on two canvases that fill their box; the picture is letterboxed. */
class Video {
  constructor(root) {
    this.root = root;
    this.vid = root.querySelector("canvas.vid");
    this.skel = root.querySelector("canvas.skel");
    this.bitmap = null;
    this.lm = null;
    this.angles = null;
    this.aspect = NO_PICTURE_ASPECT;
    this._seq = 0;
    this._dirty = true;
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => { this._dirty = true; }).observe(root);
  }

  /** A frame from the socket. With video off (or no picture yet) the skeleton alone is drawn. */
  onFrame(header, jpeg, videoOn) {
    this.lm = header.lm ?? null;
    this.angles = armAngles(header.counters);
    this._dirty = true;
    if (!videoOn) {
      this._set(null);
      return;
    }
    if (!jpeg) return; // video on, frame without a picture: keep the last one
    const seq = ++this._seq;
    createImageBitmap(new Blob([jpeg], { type: "image/jpeg" })).then(
      (bmp) => (seq === this._seq ? this._set(bmp) : bmp.close()),
      () => {}, // an undecodable picture is skipped
    );
  }

  _set(bmp) {
    if (this.bitmap && this.bitmap !== bmp) this.bitmap.close();
    this.bitmap = bmp;
    this.aspect = bmp ? bmp.width / bmp.height : NO_PICTURE_ASPECT;
    this._dirty = true;
  }

  render() {
    if (!this._dirty) return;
    this._dirty = false;
    const px = devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.root.clientWidth * px));
    const hh = Math.max(1, Math.round(this.root.clientHeight * px));
    for (const c of [this.vid, this.skel]) {
      if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; }
    }
    // the picture's rectangle inside the box (contain)
    const rw = Math.min(w, hh * this.aspect);
    const rh = rw / this.aspect;
    const x = (w - rw) / 2;
    const y = (hh - rh) / 2;
    const vctx = this.vid.getContext("2d");
    vctx.clearRect(0, 0, w, hh);
    this.vid.hidden = !this.bitmap;
    if (this.bitmap) {
      vctx.setTransform(-1, 0, 0, 1, x + rw + x, 0); // mirrored, like a mirror
      vctx.drawImage(this.bitmap, x, y, rw, rh);
      vctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    const sctx = this.skel.getContext("2d");
    sctx.clearRect(0, 0, w, hh);
    if (this.lm) {
      sctx.save();
      sctx.translate(x, y);
      drawSkeleton(sctx, this.lm, rw, rh, { mirror: true, angles: this.angles });
      sctx.restore();
    }
  }
}

/**
 * Largest text size (vh) from BAND_MAX_VH down to BAND_MIN_VH at which the text fits its box (at
 * most two lines); text that still does not fit at the smallest size is clamped with an ellipsis.
 */
function fitCueText(el) {
  el.classList.remove("clamp");
  for (let vh = BAND_MAX_VH; vh >= BAND_MIN_VH; vh -= 0.5) {
    el.style.fontSize = `${vh}vh`;
    const overshoot = 0.2 * parseFloat(getComputedStyle(el).fontSize); // glyphs reach past the line box
    if (el.scrollHeight <= el.clientHeight + overshoot) return;
  }
  el.classList.add("clamp");
}

function boot() {
  let state = initialState();
  const dispatch = (msg) => { state = reduce(state, msg); };
  const video = new Video($("video"));
  const countsEl = $("counts");
  let countsKey = "";
  const sentCues = new Set();
  let shownKey = null;
  let sock = null;

  // ---- sound chip, full screen, cursor ---------------------------------------------------------
  const chip = $("sound-chip");
  chip.addEventListener("click", () => { if (tones.enable()) chip.hidden = true; });
  addEventListener("keydown", (e) => {
    if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey && !e.altKey) {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.().catch(() => {});
    }
  });
  let cursorTimer = null;
  const showCursor = () => {
    document.body.classList.remove("idle-cursor");
    clearTimeout(cursorTimer);
    cursorTimer = setTimeout(() => document.body.classList.add("idle-cursor"), CURSOR_HIDE_MS);
  };
  addEventListener("mousemove", showCursor);
  showCursor();
  // a hidden page gets no animation frames: a cue whose rep arrives then is never confirmed
  const onVisibility = () => dispatch({ type: "visibility", visible: document.visibilityState === "visible" });
  document.addEventListener("visibilitychange", onVisibility);
  onVisibility();

  // ---- drawing ---------------------------------------------------------------------------------
  function renderCounts(m) {
    const key = JSON.stringify(m.counts);
    if (key === countsKey) return;
    countsKey = key;
    countsEl.className = `counts${m.counts.length === 1 ? " one" : ""}`;
    countsEl.replaceChildren(...m.counts.map((c) => h("div", { class: "col" },
      h("i", { class: "lab" }, c.label),
      h("b", { class: "num" }, String(c.count)),
      h("div", { class: "dots", "aria-label": `${c.label || "all"} reps` },
        c.dots.map((d) => h("div", { class: `dot ${d}` }, DOT_GLYPH[d] ? h("span", {}, DOT_GLYPH[d]) : null))),
    )));
  }

  function renderBand(band) {
    const cue = $("cue");
    cue.className = `cue ${band.kind}`;
    setText($("cue-ico"), band.icon);
    setText($("cue-who"), band.who);
    const txt = $("cue-txt");
    if (txt.textContent !== band.text) {
      txt.textContent = band.text;
      fitCueText(txt);
    }
    // Only a cue drawn when its rep completed is confirmed (see coach-state.js). `sentCues` keeps
    // the send to once per rep: a cue that is hidden and shown again keeps the same key.
    if (band.key && band.fresh && band.key !== shownKey && !sentCues.has(band.key)) {
      sentCues.add(band.key);
      const { arm, rep_id } = band;
      requestAnimationFrame(() => sock?.send("cue_shown", { arm, rep_id })); // after the DOM update
    }
    shownKey = band.key;
  }

  function frame(ms) {
    dispatch({ type: "tick", now: ms / 1000 });
    if (state.pendingTone) { // at most one tone per animation frame; the reducer already let a fault win
      (state.pendingTone === "fault" ? tones.fault : tones.good)();
      dispatch({ type: "tone_played" });
    }
    const m = view(state, clockText);
    setText($("ex"), m.exercise);
    renderCounts(m);
    renderBand(m.band);
    for (const [id, [text, tone]] of [["st-gate", m.strip.gate], ["st-watch", m.strip.watch], ["st-rec", m.strip.rec]]) {
      setText($(id), text);
      const el = $(id);
      if (el.className !== tone) el.className = tone;
    }
    const off = $("offline");
    setText(off, m.disconnected ? "disconnected, retrying" : " ");
    off.classList.toggle("on", m.disconnected);
    video.render();
    requestAnimationFrame(frame);
  }

  sock = new BenchSocket(null, {
    onHello: (msg) => dispatch(msg),
    onState: (st) => dispatch({ type: "state", state: st }),
    onFrame: (header, jpeg) => {
      dispatch(header);
      video.onFrame(header, jpeg, state.videoOn);
    },
    onEvent: (msg) => dispatch(msg),
    onLink: (status) => dispatch({ type: "link", status }),
    onOpen: () => dispatch({ type: "socket_open" }),
    onClose: () => dispatch({ type: "socket_close" }),
  });
  requestAnimationFrame(frame);
}

if (typeof document !== "undefined" && document.body?.classList.contains("coach")) boot();
