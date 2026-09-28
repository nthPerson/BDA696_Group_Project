// FormCoach firmware v1.0.0 (Checkpoint 4; docs/02-system-design.md §2).
//   * BMI160 over I2C at 100 Hz, averaged to 50 Hz; 2 s ring buffer; energy gate + hysteresis
//   * BLE GATT (protocol.h): 5-sample IMU batches, 1-byte control commands, status
//   * USB serial: CSV lines (`t_ms,ax,ay,az,gx,gy,gz,flags,seq`) + one-letter commands
//   * button: short press toggles the session (session_id++), long press = 3 s calibration
//   * LED: off idle, slow blink advertising, solid connected+session, flashes on rep / fault
// Compiles in CI; nothing here has run on a board yet (see docs/STATUS.md).
#include <Arduino.h>
#include <Preferences.h>
#include <Wire.h>

#include "ble.h"
#include "bmi160.h"
#include "gate.h"
#include "protocol.h"
#include "ui.h"

#ifndef FORMCOACH_FW_VERSION
#define FORMCOACH_FW_VERSION "0.0.0-dev"
#endif

namespace {
constexpr uint8_t PIN_BUTTON = D1;        // GPIO2, tactile switch to GND
constexpr uint8_t PIN_LED = LED_BUILTIN;  // GPIO21, active-low (verified from the variant header)
constexpr bool LED_ACTIVE_LOW = true;
constexpr uint32_t SAMPLE_PERIOD_US = 10000;  // 100 Hz IMU reads, 2 averaged -> 50 Hz
constexpr uint32_t STATUS_PERIOD_MS = 1000;
constexpr uint32_t CALIB_MS = 3000;
constexpr uint32_t DEEP_SLEEP_AFTER_MS = 10UL * 60UL * 1000UL;

Bmi160 imu;
Gate gate;
BleLink ble;
Button button;
StatusLed led;
Preferences prefs;

bool imu_ok = false;
bool session_active = false;
bool calibrated = false;
uint16_t session_id = 0;
uint16_t seq = 0;
uint32_t last_sample_us = 0;
uint32_t last_status_ms = 0;
uint32_t last_activity_ms = 0;
int16_t acc_sum[3], gyr_sum[3];
uint8_t avg_count = 0;

// calibration: gravity vector (LSB) and gyro bias (LSB) captured with the device flat and still
int32_t calib_acc[3] = {0, 0, 0};
int32_t calib_gyr[3] = {0, 0, 0};
uint32_t calib_until_ms = 0;
int32_t calib_sum_acc[3], calib_sum_gyr[3];
uint32_t calib_n = 0;

String deviceName() {
  uint64_t mac = ESP.getEfuseMac();
  char buf[24];
  snprintf(buf, sizeof(buf), "%s%04X", FC_DEVICE_NAME_PREFIX, (unsigned)(mac & 0xFFFF));
  return String(buf);
}

void loadCalibration() {
  prefs.begin("formcoach", true);
  calibrated = prefs.getBool("cal", false);
  if (calibrated) {
    for (int i = 0; i < 3; ++i) {
      calib_acc[i] = prefs.getInt(("ca" + String(i)).c_str(), 0);
      calib_gyr[i] = prefs.getInt(("cg" + String(i)).c_str(), 0);
    }
  }
  prefs.end();
}

void saveCalibration() {
  prefs.begin("formcoach", false);
  prefs.putBool("cal", true);
  for (int i = 0; i < 3; ++i) {
    prefs.putInt(("ca" + String(i)).c_str(), calib_acc[i]);
    prefs.putInt(("cg" + String(i)).c_str(), calib_gyr[i]);
  }
  prefs.end();
  calibrated = true;
  Serial.printf("# calibrated g=(%ld,%ld,%ld) bias=(%ld,%ld,%ld)\n", (long)calib_acc[0],
                (long)calib_acc[1], (long)calib_acc[2], (long)calib_gyr[0], (long)calib_gyr[1],
                (long)calib_gyr[2]);
}

void startCalibration() {
  calib_until_ms = millis() + CALIB_MS;
  calib_n = 0;
  for (int i = 0; i < 3; ++i) calib_sum_acc[i] = calib_sum_gyr[i] = 0;
  Serial.println("# calibration: keep the device flat and still for 3 s");
}

void setSession(bool on) {
  if (on == session_active) return;
  session_active = on;
  if (on) ++session_id;
  led.set(on ? LedMode::Connected : (ble.connected() ? LedMode::Connected : LedMode::Advertising));
  Serial.printf("# session %s id=%u\n", on ? "start" : "stop", session_id);
  last_activity_ms = millis();
}

void handleCommand(uint8_t cmd) {
  last_activity_ms = millis();
  switch (cmd) {
    case FC_CMD_LED_GOOD_REP: led.flash(LedMode::GoodRep); break;
    case FC_CMD_LED_FAULT: led.flash(LedMode::Fault); break;
    case FC_CMD_START_SESSION: setSession(true); break;
    case FC_CMD_STOP_SESSION: setSession(false); break;
    case FC_CMD_CALIBRATE: startCalibration(); break;
    case FC_CMD_PING: Serial.println("# pong"); break;
    default: Serial.printf("# unknown command 0x%02X\n", cmd); break;
  }
}

void pollSerialCommands() {
  while (Serial.available()) {
    int c = Serial.read();
    switch (c) {
      case FC_SERIAL_CMD_LED_GOOD_REP: handleCommand(FC_CMD_LED_GOOD_REP); break;
      case FC_SERIAL_CMD_LED_FAULT: handleCommand(FC_CMD_LED_FAULT); break;
      case FC_SERIAL_CMD_START_SESSION: handleCommand(FC_CMD_START_SESSION); break;
      case FC_SERIAL_CMD_STOP_SESSION: handleCommand(FC_CMD_STOP_SESSION); break;
      case FC_SERIAL_CMD_CALIBRATE: handleCommand(FC_CMD_CALIBRATE); break;
      case FC_SERIAL_CMD_PING: handleCommand(FC_CMD_PING); break;
      default: break;  // newlines etc.
    }
  }
}

void publishStatus() {
  FcStatus st{};
  sscanf(FORMCOACH_FW_VERSION, "%hhu.%hhu.%hhu", &st.fw_major, &st.fw_minor, &st.fw_patch);
  st.session_id = session_id;
  st.gate_state = gate.state() ? 1 : 0;
  st.uptime_s = millis() / 1000;
  st.calibrated = calibrated ? 1 : 0;
  st.session_active = session_active ? 1 : 0;
  ble.publishStatus(st);
}

void emitSample(uint32_t t_ms, const int16_t* a, const int16_t* g) {
  FcSample s;
  s.t_ms = t_ms;
  s.ax = a[0]; s.ay = a[1]; s.az = a[2];
  s.gx = g[0]; s.gy = g[1]; s.gz = g[2];
  s.flags = (gate.state() ? FC_FLAG_GATE_STATE : 0) | (session_active ? FC_FLAG_SESSION_ACTIVE : 0) |
            (button.pressed() ? FC_FLAG_BUTTON_PRESSED : 0) | (calibrated ? FC_FLAG_CALIBRATED : 0);
  s.seq = seq++;
  ble.queue(s, session_id);
  Serial.printf("%lu,%d,%d,%d,%d,%d,%d,%u,%u\n", (unsigned long)s.t_ms, s.ax, s.ay, s.az, s.gx,
                s.gy, s.gz, s.flags, s.seq);
}

void sampleImu() {
  int16_t a[3], g[3];
  if (!imu.read(a, g)) return;
  if (calib_until_ms) {
    for (int i = 0; i < 3; ++i) { calib_sum_acc[i] += a[i]; calib_sum_gyr[i] += g[i]; }
    ++calib_n;
    if (millis() >= calib_until_ms) {
      for (int i = 0; i < 3; ++i) {
        calib_acc[i] = calib_sum_acc[i] / (int32_t)calib_n;
        calib_gyr[i] = calib_sum_gyr[i] / (int32_t)calib_n;
      }
      calib_until_ms = 0;
      saveCalibration();
    }
  }
  for (int i = 0; i < 3; ++i) {
    acc_sum[i] += a[i] / 2;  // average two 100 Hz reads -> one 50 Hz sample (no overflow)
    gyr_sum[i] += (g[i] - (calibrated ? calib_gyr[i] : 0)) / 2;
  }
  if (++avg_count < 2) return;
  avg_count = 0;
  int16_t aa[3] = {acc_sum[0], acc_sum[1], acc_sum[2]};
  int16_t gg[3] = {gyr_sum[0], gyr_sum[1], gyr_sum[2]};
  for (int i = 0; i < 3; ++i) acc_sum[i] = gyr_sum[i] = 0;
  bool decided = gate.push(aa, gg);
  if (decided && gate.state()) last_activity_ms = millis();
  if (decided && gate.usingModel()) {
    Serial.printf("# infer us=%lu arena=%u class=%d p_active=%.2f\n", (unsigned long)gate.lastInferUs(),
                  (unsigned)gate.arenaUsed(), gate.lastClass(), gate.lastPActive());
  }
  emitSample(millis(), aa, gg);
}
}  // namespace

void setup() {
  led.begin(PIN_LED, LED_ACTIVE_LOW);
  button.begin(PIN_BUTTON);
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);
  Serial.println();
  Serial.printf("# FormCoach fw %s  chip=%s rev=%d\n", FORMCOACH_FW_VERSION, ESP.getChipModel(),
                ESP.getChipRevision());
  Serial.printf("# sample=%u B batch=%u B status=%u B rate=%u Hz\n", (unsigned)sizeof(FcSample),
                (unsigned)sizeof(FcBatch), (unsigned)sizeof(FcStatus), FC_SAMPLE_RATE_HZ);
  Serial.printf("# flash app=%u\n", (unsigned)ESP.getSketchSize());

  Wire.begin();  // SDA=D4/GPIO5, SCL=D5/GPIO6 (pins_arduino.h of the XIAO ESP32-S3 variant)
  Wire.setClock(400000);
  uint8_t addr = Bmi160::scan(Wire);
  imu_ok = addr && imu.begin(Wire, addr);
  Serial.printf("# bmi160 %s at 0x%02X\n", imu_ok ? "ok" : "NOT FOUND", addr);
  loadCalibration();
  Serial.printf("# calibration %s\n", calibrated ? "loaded from NVS" : "none (long-press to calibrate)");

  bool model_ok = gate.begin();
  Serial.printf("# gate: %s\n", model_ok ? "int8 CNN (TFLite Micro)" : "motion-energy rule");
  ble.begin(deviceName().c_str(), handleCommand);
  Serial.printf("# ble advertising as %s\n", deviceName().c_str());
  led.set(LedMode::Advertising);
  Serial.println(FC_SERIAL_CSV_HEADER);
  last_sample_us = micros();
  last_activity_ms = millis();
}

void loop() {
  uint8_t ev = button.poll();
  if (ev == 1) setSession(!session_active);
  if (ev == 2) startCalibration();
  pollSerialCommands();

  uint32_t now_us = micros();
  if (imu_ok && now_us - last_sample_us >= SAMPLE_PERIOD_US) {
    last_sample_us += SAMPLE_PERIOD_US;
    if (now_us - last_sample_us > 5 * SAMPLE_PERIOD_US) last_sample_us = now_us;  // resync
    sampleImu();
  }

  if (millis() - last_status_ms >= STATUS_PERIOD_MS) {
    last_status_ms = millis();
    publishStatus();
    if (!session_active) led.set(ble.connected() ? LedMode::Connected : LedMode::Advertising);
  }
  led.update();

  if (!session_active && !ble.connected() && millis() - last_activity_ms > DEEP_SLEEP_AFTER_MS) {
    Serial.println("# idle 10 min: deep sleep (wake on button)");
    Serial.flush();
    esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_BUTTON, 0);
    esp_deep_sleep_start();
  }
  delay(1);
}
