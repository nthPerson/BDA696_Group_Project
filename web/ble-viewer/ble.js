// One Web Bluetooth connection to a FormCoach wearable. Callbacks receive raw DataViews, the
// same shape the Simulator produces. Chrome/Edge only, and only in a secure context.
import { CONTROL_CHAR_UUID, IMU_CHAR_UUID, SERVICE_UUID, STATUS_CHAR_UUID } from "./protocol.js";

export function bluetoothAvailable() {
  return typeof navigator !== "undefined" && "bluetooth" in navigator && !!navigator.bluetooth;
}

export class BleLink {
  constructor(callbacks) {
    this.cb = callbacks;
    this.device = null;
    this.hasControl = false;
    this._control = null;
    this._writes = Promise.resolve();
    // One function object per link: addEventListener ignores a repeated add of the same
    // function, so a reconnect (which may hand back the same characteristic objects) never
    // doubles the packet stream.
    this._onImu = (e) => this.cb.onPacket(e.target.value);
    this._onStatus = (e) => this.cb.onStatus(e.target.value);
    this._onGone = () => {
      this._control = null;
      this.hasControl = false;
      this.cb.onDisconnect();
    };
  }

  get name() {
    return this.device?.name || "FormCoach";
  }

  get connected() {
    return !!this.device?.gatt?.connected;
  }

  /** Open the browser's device chooser (must run inside a click handler), then connect. */
  async connect() {
    const device = await navigator.bluetooth.requestDevice({ filters: [{ services: [SERVICE_UUID] }] });
    if (this.device !== device) {
      device.addEventListener("gattserverdisconnected", this._onGone);
      this.device = device;
    }
    await this._open();
  }

  /** Reconnect to the device picked earlier, without the chooser. */
  async reconnect() {
    if (!this.device) return this.connect();
    await this._open();
  }

  async disconnect() {
    if (this.device?.gatt?.connected) this.device.gatt.disconnect(); // fires gattserverdisconnected
  }

  /** Write one command byte; writes are queued so two clicks never overlap a GATT operation. */
  send(cmd) {
    const op = this._writes.then(() => {
      if (!this._control) throw new Error("not connected, or this firmware has no control characteristic");
      return this._control.writeValueWithResponse(Uint8Array.of(cmd));
    });
    this._writes = op.catch(() => {});
    return op;
  }

  async _open() {
    const server = await this.device.gatt.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    const imu = await service.getCharacteristic(IMU_CHAR_UUID);
    imu.addEventListener("characteristicvaluechanged", this._onImu);
    await imu.startNotifications();
    this._control = await service.getCharacteristic(CONTROL_CHAR_UUID).catch(() => null);
    this.hasControl = !!this._control;
    const status = await service.getCharacteristic(STATUS_CHAR_UUID).catch(() => null);
    if (status) {
      status.addEventListener("characteristicvaluechanged", this._onStatus);
      await status.startNotifications().catch(() => {});
      this.cb.onStatus(await status.readValue());
    }
  }
}
