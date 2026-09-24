// FormCoach firmware v0.1.0 — bring-up sketch.
//
// What it does today (no IMU driver yet; Checkpoint 4 adds BMI160 + CSV streaming):
//   * prints firmware version and chip info on the serial monitor,
//   * scans the I2C bus and reports whether a BMI160 answers at 0x68 or 0x69,
//   * reads the user button on D1 (INPUT_PULLUP, pressed = LOW) with debounce,
//   * blinks the built-in LED (slow blink = advertising),
//   * advertises the FormCoach BLE service so `bleak` can already discover the board.
// This file must always compile in CI (`pio run`) — it is the firmware smoke test.

#include <Arduino.h>
#include <NimBLEDevice.h>
#include <Wire.h>

#include "protocol.h"

#ifndef FORMCOACH_FW_VERSION
#define FORMCOACH_FW_VERSION "0.0.0-dev"
#endif

namespace {

constexpr uint8_t PIN_BUTTON = D1;          // GPIO2, tactile switch to GND
constexpr uint8_t PIN_LED = LED_BUILTIN;    // GPIO21, active-low on the XIAO ESP32-S3 (verify)
constexpr bool LED_ACTIVE_LOW = true;
constexpr uint32_t DEBOUNCE_MS = 30;
constexpr uint32_t BLINK_ADVERTISING_MS = 1000;

void ledWrite(bool on) { digitalWrite(PIN_LED, (on != LED_ACTIVE_LOW) ? HIGH : LOW); }

// Scan the I2C bus once and report; the BMI160 (GY-BMI160 module) should answer at 0x69
// (SA0 high) or 0x68. Returns the address found or 0.
uint8_t scanI2C() {
  uint8_t found = 0;
  Serial.println("# i2c scan");
  for (uint8_t addr = 1; addr < 127; ++addr) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("#   device at 0x%02X%s\n", addr,
                    (addr == 0x68 || addr == 0x69) ? "  <- BMI160?" : "");
      if (addr == 0x68 || addr == 0x69) found = addr;
    }
  }
  if (!found) Serial.println("#   no BMI160 at 0x68/0x69 (check SDA=D4, SCL=D5, 3V3, GND)");
  return found;
}

String deviceName() {
  uint64_t mac = ESP.getEfuseMac();
  char buf[24];
  snprintf(buf, sizeof(buf), "%s%04X", FC_DEVICE_NAME_PREFIX, (unsigned)(mac & 0xFFFF));
  return String(buf);
}

void startAdvertising() {
  NimBLEDevice::init(deviceName().c_str());
  NimBLEServer* server = NimBLEDevice::createServer();
  server->createService(FC_SERVICE_UUID);  // started with the server (NimBLE 2.x)
  server->start();
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(FC_SERVICE_UUID);
  adv->start();
  Serial.printf("# ble advertising as %s\n", deviceName().c_str());
}

bool g_buttonState = false;  // debounced, true = pressed
bool g_buttonRaw = false;
uint32_t g_buttonChangedAt = 0;

void pollButton() {
  bool raw = digitalRead(PIN_BUTTON) == LOW;
  uint32_t now = millis();
  if (raw != g_buttonRaw) {
    g_buttonRaw = raw;
    g_buttonChangedAt = now;
  } else if (raw != g_buttonState && now - g_buttonChangedAt >= DEBOUNCE_MS) {
    g_buttonState = raw;
    Serial.printf("# button %s at %lu ms\n", raw ? "pressed" : "released", (unsigned long)now);
  }
}

}  // namespace

void setup() {
  pinMode(PIN_LED, OUTPUT);
  ledWrite(false);
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);  // wait briefly for USB CDC

  Serial.println();
  Serial.printf("# FormCoach fw %s  chip=%s rev=%d  psram=%u\n", FORMCOACH_FW_VERSION,
                ESP.getChipModel(), ESP.getChipRevision(), (unsigned)ESP.getPsramSize());
  Serial.printf("# sample=%u B batch=%u B rate=%u Hz\n", (unsigned)sizeof(FcSample),
                (unsigned)sizeof(FcBatch), FC_SAMPLE_RATE_HZ);

  Wire.begin();  // SDA=D4/GPIO5, SCL=D5/GPIO6 on the XIAO variant (verify)
  Wire.setClock(400000);
  scanI2C();

  startAdvertising();
  Serial.println(FC_SERIAL_CSV_HEADER);  // data lines follow once the IMU driver lands
}

void loop() {
  pollButton();
  ledWrite((millis() / (BLINK_ADVERTISING_MS / 2)) % 2 == 0);  // slow blink = advertising
  delay(5);
}
