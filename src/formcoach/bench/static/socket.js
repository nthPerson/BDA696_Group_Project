// The single WebSocket to the Bench server: decoding and message routing. No DOM access at import.

/**
 * Split one binary frame message: 4-byte big-endian header length, the UTF-8 JSON header, then
 * the JPEG bytes (none when the header says has_image is false).
 * @param {ArrayBuffer} buffer
 * @returns {{header: object, jpeg: Uint8Array|null}}
 */
export function decodeFrame(buffer) {
  if (buffer.byteLength < 4) throw new Error(`frame of ${buffer.byteLength} bytes has no header length`);
  const n = new DataView(buffer).getUint32(0, false);
  if (buffer.byteLength < 4 + n) throw new Error(`frame header says ${n} bytes but only ${buffer.byteLength - 4} follow`);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
  const rest = buffer.byteLength - 4 - n;
  return { header, jpeg: rest > 0 ? new Uint8Array(buffer, 4 + n, rest) : null };
}

/** The server's socket address for the page that is open (call it in the browser only). */
export function benchUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

export const BACKOFF_MIN_S = 0.5;
export const BACKOFF_MAX_S = 5;

/**
 * Handlers (all optional): onHello(msg), onState(state) for both hello and state messages,
 * onFrame(header, jpeg), onImu(msg), onEvent(msg), onLink(status), onScan(msg), onLog(msg),
 * onError(msg), onOpen(), onClose(). Other options: `WebSocket` (constructor, default the
 * global) and `setTimer`/`clearTimer` (default setTimeout/clearTimeout), injectable for tests.
 * Reconnects after a drop with a delay of 0.5 s, doubling to 5 s, reset by a successful open.
 */
export class BenchSocket {
  constructor(url, handlers = {}) {
    const { WebSocket: Ctor, setTimer, clearTimer, ...h } = handlers;
    this.url = url ?? benchUrl();
    this.h = h;
    this._Ctor = Ctor ?? globalThis.WebSocket;
    this._setTimer = setTimer ?? ((fn, ms) => setTimeout(fn, ms));
    this._clearTimer = clearTimer ?? ((id) => clearTimeout(id));
    this._delayS = BACKOFF_MIN_S;
    this._timer = null;
    this._closed = false;
    this.ws = null;
    this._connect();
  }

  /** Send a command: JSON `{cmd, ...fields}`. Returns false (and sends nothing) when not open. */
  send(cmd, fields = {}) {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify({ cmd, ...fields }));
    return true;
  }

  /** Close for good: no reconnect. */
  close() {
    this._closed = true;
    if (this._timer !== null) this._clearTimer(this._timer);
    this._timer = null;
    if (this.ws) this.ws.close();
  }

  _connect() {
    const ws = new this._Ctor(this.url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this._delayS = BACKOFF_MIN_S;
      this.h.onOpen?.();
    };
    ws.onmessage = (ev) => this._route(ev.data);
    ws.onclose = () => {
      this.h.onClose?.();
      if (this._closed) return;
      const ms = this._delayS * 1000;
      this._delayS = Math.min(BACKOFF_MAX_S, this._delayS * 2);
      this._timer = this._setTimer(() => {
        this._timer = null;
        this._connect();
      }, ms);
    };
  }

  _route(data) {
    try {
      if (typeof data !== "string") {
        const { header, jpeg } = decodeFrame(data);
        this.h.onFrame?.(header, jpeg);
        return;
      }
      const msg = JSON.parse(data);
      switch (msg.type) {
        case "hello":
          this.h.onHello?.(msg);
          this.h.onState?.(msg.state);
          break;
        case "state": this.h.onState?.(msg.state); break;
        case "imu": this.h.onImu?.(msg); break;
        case "event": this.h.onEvent?.(msg); break;
        case "link": this.h.onLink?.(msg.status); break;
        case "scan": this.h.onScan?.(msg); break;
        case "log": this.h.onLog?.(msg); break;
        case "error": this.h.onError?.(msg); break;
        default: break; // unknown types from a newer server are ignored
      }
    } catch (exc) {
      this.h.onError?.({ type: "error", text: `bad message from the server: ${exc.message}`, cmd: null });
    }
  }
}
