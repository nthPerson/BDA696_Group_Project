#include "ble.h"

#include <Arduino.h>

namespace {
BleLink* g_link = nullptr;

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* server, NimBLEConnInfo& info) override {
    server->updateConnParams(info.getConnHandle(), 12, 24, 0, 400);  // 15-30 ms interval
    if (g_link) g_link->onConnected(true);
  }
  void onDisconnect(NimBLEServer* server, NimBLEConnInfo& info, int reason) override {
    if (g_link) g_link->onConnected(false);
    NimBLEDevice::startAdvertising();
  }
  void onMTUChange(uint16_t mtu, NimBLEConnInfo& info) override {
    Serial.printf("# ble mtu %u\n", mtu);
  }
};

class ControlCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* chr, NimBLEConnInfo& info) override {
    NimBLEAttValue v = chr->getValue();
    if (v.size() >= 1 && g_link) g_link->onCommand(v[0]);
  }
};
}  // namespace

void BleLink::begin(const char* name, CommandHandler handler) {
  g_link = this;
  handler_ = handler;
  NimBLEDevice::init(name);
  NimBLEDevice::setMTU(FC_REQUESTED_MTU);
  server_ = NimBLEDevice::createServer();
  static ServerCallbacks server_cb;
  server_->setCallbacks(&server_cb);
  NimBLEService* svc = server_->createService(FC_SERVICE_UUID);
  imu_ = svc->createCharacteristic(FC_IMU_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);
  control_ = svc->createCharacteristic(FC_CONTROL_CHAR_UUID,
                                       NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  static ControlCallbacks control_cb;
  control_->setCallbacks(&control_cb);
  status_ = svc->createCharacteristic(FC_STATUS_CHAR_UUID,
                                      NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  svc->start();
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(name);
  adv->addServiceUUID(FC_SERVICE_UUID);
  adv->start();
  batch_.header.n = 0;
}

void BleLink::onConnected(bool up) {
  connected_ = up;
  batch_.header.n = 0;  // never carry a partial batch across a (re)connect
  Serial.printf("# ble %s\n", up ? "connected" : "disconnected");
}

void BleLink::queue(const FcSample& s, uint16_t session_id) {
  if (!connected_) return;
  batch_.header.session_id = session_id;
  batch_.samples[batch_.header.n++] = s;
  if (batch_.header.n < FC_BATCH_SAMPLES) return;
  size_t len = FC_BATCH_HEADER_SIZE + batch_.header.n * FC_SAMPLE_SIZE;
  imu_->setValue((const uint8_t*)&batch_, len);
  imu_->notify();
  batch_.header.n = 0;
}

void BleLink::publishStatus(const FcStatus& st) {
  if (!status_) return;
  status_->setValue((const uint8_t*)&st, sizeof(st));
  if (connected_) status_->notify();
}
