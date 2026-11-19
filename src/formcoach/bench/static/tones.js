// The coach page's two tones (WebAudio). Browsers allow sound only after a click: `enable()` is
// called from the click handler; before it, `good()` and `fault()` do nothing.
let ctx = null;

/** Create or resume the one AudioContext. Call from a click. Returns false when the browser has no WebAudio. */
export function enable() {
  const Ctor = globalThis.AudioContext ?? globalThis.webkitAudioContext;
  if (!Ctor) return false;
  if (!ctx) ctx = new Ctor();
  if (ctx.state === "suspended") ctx.resume();
  return true;
}

function beep(type, hz, seconds) {
  if (!ctx || ctx.state !== "running") return;
  const t = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.value = hz;
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.25, t + 0.008);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + seconds + 0.02);
}

/** A good rep: 880 Hz for 90 ms. */
export function good() {
  beep("sine", 880, 0.09);
}

/** A rep with a fault: 220 Hz for 220 ms. */
export function fault() {
  beep("triangle", 220, 0.22);
}
