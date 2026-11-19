// Time-series ring buffers and the shared time window. Times are seconds on the laptop's
// monotonic clock (frame t_mono, IMU arrival time). Pure logic: no DOM.

/** The window every chart shows: `spanS` seconds ending at `tEnd`, with `keepS` of history. */
export class Timeline {
  constructor(spanS = 20, keepS = 300) {
    this.spanS = spanS;
    this.keepS = keepS;
    this._newest = -Infinity;
    this._paused = false;
    this._frozen = 0;
  }

  /** Called by series with each time pushed; running timelines follow the newest. */
  note(t) {
    if (t > this._newest) this._newest = t;
  }

  get newest() { return this._newest === -Infinity ? 0 : this._newest; }

  get paused() { return this._paused; }

  pause() {
    if (!this._paused) {
      this._frozen = this.newest;
      this._paused = true;
    }
  }

  resume() { this._paused = false; }

  /** Right edge of the window (s): the newest time, or the frozen position while paused. */
  get tEnd() {
    if (!this._paused) return this.newest;
    return Math.min(this.newest, Math.max(this._frozen, this.newest - this.keepS));
  }

  /** While paused, move the window by dtS seconds (negative = back in time), kept within
   * [newest - keepS, newest]. Does nothing while running. */
  scrollBy(dtS) {
    if (!this._paused) return;
    this._frozen = Math.min(this.newest, Math.max(this.newest - this.keepS, this.tEnd + dtS));
  }
}

/** Fixed-capacity ring of rows `(t, v0, v1, ...)`; the oldest rows are dropped. */
export class Series {
  /**
   * @param {string[]} names column names
   * @param {number} capacity most rows kept
   * @param {Timeline|null} timeline told of every pushed time
   */
  constructor(names, capacity, timeline = null) {
    this.names = names;
    this.capacity = capacity;
    this.timeline = timeline;
    this._t = new Float64Array(capacity);
    this._cols = names.map(() => new Float64Array(capacity));
    this._start = 0;
    this._n = 0;
  }

  get length() { return this._n; }

  get latestT() { return this._n ? this._t[(this._start + this._n - 1) % this.capacity] : -Infinity; }

  get oldestT() { return this._n ? this._t[this._start] : Infinity; }

  /** Add a row; `values[i]` belongs to `names[i]` (null/NaN is kept as NaN: a gap). */
  push(t, values) {
    const i = (this._start + this._n) % this.capacity;
    if (this._n < this.capacity) this._n++;
    else this._start = (this._start + 1) % this.capacity;
    this._t[i] = t;
    this._cols.forEach((c, k) => { c[i] = values[k] ?? NaN; });
    this.timeline?.note(t);
  }

  /**
   * Rows with `tEnd - spanS <= t <= tEnd`, oldest first, as `[t[], col0[], ...]` (NaN gaps
   * become null). Rows must have been pushed in time order (equal times are fine): the first
   * row is found by binary search, so a render reads only the rows it returns.
   */
  window(tEnd, spanS) {
    const t = [];
    const cols = this._cols.map(() => []);
    const lo = tEnd - spanS;
    const cap = this.capacity;
    // first logical index k with t >= lo
    let a = 0;
    let b = this._n;
    while (a < b) {
      const m = (a + b) >> 1;
      if (this._t[(this._start + m) % cap] < lo) a = m + 1;
      else b = m;
    }
    for (let k = a; k < this._n; k++) {
      const i = (this._start + k) % cap;
      const ti = this._t[i];
      if (ti > tEnd) break;
      t.push(ti);
      this._cols.forEach((c, j) => cols[j].push(Number.isNaN(c[i]) ? null : c[i]));
    }
    return [t, ...cols];
  }
}
