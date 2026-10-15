// Records raw samples and produces the same CSV the firmware prints over USB serial
// (FC_SERIAL_CSV_HEADER), so `protocol.parse_serial_line` and existing tooling read it unchanged.
import { SERIAL_CSV_HEADER } from "./protocol.js";

export class Recorder {
  constructor() {
    this._rows = [];
    this.active = false;
  }

  get count() {
    return this._rows.length;
  }

  start() {
    this._rows = [];
    this.active = true;
  }

  add(samples) {
    if (!this.active) return;
    for (const s of samples) {
      this._rows.push(`${s.t_ms},${s.ax},${s.ay},${s.az},${s.gx},${s.gy},${s.gz},${s.flags},${s.seq}`);
    }
  }

  stop() {
    this.active = false;
    return this.csv();
  }

  csv() {
    return `${[SERIAL_CSV_HEADER, ...this._rows].join("\n")}\n`;
  }
}

/** `<device>_YYYYMMDD-HHMMSS.csv` in local time, with characters outside [A-Za-z0-9_-] replaced. */
export function csvFilename(device, date = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  const stamp =
    `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-` +
    `${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
  const safe = (device || "FormCoach").replace(/[^A-Za-z0-9_-]+/g, "_");
  return `${safe}_${stamp}.csv`;
}
