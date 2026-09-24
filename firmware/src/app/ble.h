// BLE GATT service (protocol.h §3): IMU notify batches, control writes, status read/notify.
#pragma once
#include <NimBLEDevice.h>
#include "protocol.h"

class BleLink {
 public:
  using CommandHandler = void (*)(uint8_t cmd);
  void begin(const char* name, CommandHandler handler);
  bool connected() const { return connected_; }
  void queue(const FcSample& s, uint16_t session_id);  // notify every FC_BATCH_SAMPLES samples
  void publishStatus(const FcStatus& st);
  void onConnected(bool up);
  void onCommand(uint8_t cmd) { if (handler_) handler_(cmd); }

 private:
  NimBLEServer* server_ = nullptr;
  NimBLECharacteristic* imu_ = nullptr;
  NimBLECharacteristic* control_ = nullptr;
  NimBLECharacteristic* status_ = nullptr;
  CommandHandler handler_ = nullptr;
  FcBatch batch_{};
  bool connected_ = false;
};
