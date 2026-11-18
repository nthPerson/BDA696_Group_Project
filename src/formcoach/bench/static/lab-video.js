// The lab console's video panel: the frame JPEG on one canvas and the skeleton on a canvas on
// top, both mirrored, no text. Paused, it shows the frozen picture; scrolled back, the skeleton
// from kept landmarks on a plain background (video frames are not kept).
import { drawSkeleton } from "./skeleton.js";

// The last 300 s of landmarks at the frame rate: 30 fps × 300 s = 9,000 entries of 33 points.
export const HISTORY_FRAMES = 9000;
const MAX_GAP_S = 0.5; // a kept skeleton is shown for a time at most this far after it
const FROZEN_MATCH_S = 0.1; // scrolled less than this from the frozen picture: show the picture

/** Frame landmarks by session-clock time (s), oldest dropped beyond `capacity`. No DOM. */
export class LandmarkHistory {
  constructor(capacity = HISTORY_FRAMES) {
    this.capacity = capacity;
    this._t = new Float64Array(capacity);
    this._items = new Array(capacity);
    this._start = 0;
    this._n = 0;
  }

  get length() { return this._n; }

  /** Keep `lm` (33 × [x, y, visibility], or null when nobody was found) and the counters' angles at time `t`. */
  push(t, lm, angles = null) {
    if (!Number.isFinite(t) || (this._n && t < this._t[(this._start + this._n - 1) % this.capacity])) return;
    const i = (this._start + this._n) % this.capacity;
    if (this._n < this.capacity) this._n++;
    else this._start = (this._start + 1) % this.capacity;
    this._t[i] = t;
    this._items[i] = { t, lm, angles };
  }

  /** The newest entry at or before `t`, if it is at most `maxGapS` older; else null. */
  at(t, maxGapS = MAX_GAP_S) {
    let a = 0;
    let b = this._n;
    while (a < b) { // first index with time > t
      const m = (a + b) >> 1;
      if (this._t[(this._start + m) % this.capacity] <= t) a = m + 1;
      else b = m;
    }
    if (a === 0) return null;
    const item = this._items[(this._start + a - 1) % this.capacity];
    return t - item.t <= maxGapS ? item : null;
  }
}

const anglesOf = (h) => ({ l: (h.counters?.l ?? h.counters?.all)?.angle ?? null, r: h.counters?.r?.angle ?? null });

/** The panel: `root` holds two canvases (video under skeleton). */
export class VideoPanel {
  constructor(root, history) {
    this.root = root;
    this.history = history;
    this.vid = root.querySelector("canvas.vid");
    this.skel = root.querySelector("canvas.skel");
    this.bitmap = null; // the picture shown (ImageBitmap) and the header it came with
    this.header = null;
    this.frozen = null; // {t} of the picture held while paused
    this._seq = 0;
    this._dirty = true;
    this._mode = "";
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => { this._dirty = true; }).observe(root);
  }

  /** A frame from the socket: its landmarks are kept; its picture is decoded unless paused. */
  onFrame(h, jpeg, { paused, videoOn }) {
    this.history.push(h.t_mono, h.lm ?? null, anglesOf(h));
    if (paused) return;
    if (!videoOn || !jpeg) {
      if (!videoOn) { // skeleton on a plain background, every frame
        this._setBitmap(null, h);
      }
      return; // with video on, a frame without a picture keeps the last picture and its skeleton
    }
    const seq = ++this._seq;
    createImageBitmap(new Blob([jpeg], { type: "image/jpeg" })).then(
      (bmp) => {
        if (seq !== this._seq || this.frozen) bmp.close(); // a newer frame won, or paused meanwhile
        else this._setBitmap(bmp, h);
      },
      () => {}, // an undecodable picture is skipped
    );
  }

  /** Pause holds the picture shown; resume lets frames through again. */
  setPaused(paused) {
    this.frozen = paused ? { t: this.header?.t_mono ?? null } : null;
    this._dirty = true;
  }

  _setBitmap(bmp, h) {
    if (this.bitmap && this.bitmap !== bmp) this.bitmap.close();
    this.bitmap = bmp;
    this.header = h;
    if (bmp) this.root.style.aspectRatio = `${bmp.width} / ${bmp.height}`;
    this._dirty = true;
  }

  /** Draw for the current window end `tEnd` (s, session clock). Cheap when nothing changed. */
  render({ paused, tEnd }) {
    let picture = this.bitmap;
    let lm = this.header?.lm ?? null;
    let angles = this.header ? anglesOf(this.header) : null;
    let mode = "live"; // live: redrawn when a new frame arrived (dirty)
    if (paused) {
      const atPicture = this.frozen?.t != null && Math.abs(tEnd - this.frozen.t) < FROZEN_MATCH_S;
      if (!atPicture) {
        const kept = this.history.at(tEnd);
        picture = null;
        lm = kept?.lm ?? null;
        angles = kept?.angles ?? null;
        mode = `back ${kept?.t ?? "none"}`;
      } else mode = "frozen";
    }
    if (!this._dirty && mode === this._mode) return;
    this._dirty = false;
    this._mode = mode;
    this._drawPicture(picture);
    this._drawSkeleton(lm, angles);
  }

  _drawPicture(bmp) {
    const c = this.vid;
    const ctx = c.getContext("2d");
    if (!bmp) {
      ctx.clearRect(0, 0, c.width, c.height);
      c.hidden = true;
      return;
    }
    c.hidden = false;
    if (c.width !== bmp.width || c.height !== bmp.height) {
      c.width = bmp.width;
      c.height = bmp.height;
    }
    ctx.setTransform(-1, 0, 0, 1, c.width, 0); // mirrored, like a mirror
    ctx.drawImage(bmp, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  _drawSkeleton(lm, angles) {
    const c = this.skel;
    const px = devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * px));
    const h = Math.max(1, Math.round(c.clientHeight * px));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (lm) drawSkeleton(ctx, lm, w, h, { mirror: true, angles });
  }
}
