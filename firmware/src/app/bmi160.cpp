#include "bmi160.h"

bool Bmi160::write8(uint8_t reg, uint8_t val) {
  wire_->beginTransmission(addr_);
  wire_->write(reg);
  wire_->write(val);
  return wire_->endTransmission() == 0;
}

bool Bmi160::readN(uint8_t reg, uint8_t* buf, size_t n) {
  wire_->beginTransmission(addr_);
  wire_->write(reg);
  if (wire_->endTransmission(false) != 0) return false;
  if (wire_->requestFrom(addr_, (uint8_t)n) != n) return false;
  for (size_t i = 0; i < n; ++i) buf[i] = wire_->read();
  return true;
}

uint8_t Bmi160::scan(TwoWire& wire) {
  for (uint8_t a : {0x69, 0x68}) {
    wire.beginTransmission(a);
    if (wire.endTransmission() == 0) return a;
  }
  return 0;
}

bool Bmi160::begin(TwoWire& wire, uint8_t address) {
  wire_ = &wire;
  addr_ = address;
  uint8_t id = 0;
  if (!readN(REG_CHIP_ID, &id, 1) || id != CHIP_ID) return false;
  write8(REG_CMD, CMD_SOFT_RESET);
  delay(60);
  // Power both sensors up FIRST. Verified on hardware (2026-10-08): CONF/RANGE writes issued
  // right after the soft reset, before the PMU commands, are dropped (ERR_REG != 0 and the
  // ranges read back as garbage), which left the accelerometer at the +-2 g default.
  bool ok = write8(REG_CMD, CMD_ACC_NORMAL);
  delay(5);
  ok &= write8(REG_CMD, CMD_GYR_NORMAL);
  delay(80);  // gyro start-up time
  ok &= write8(REG_ACC_CONF, 0x28);   // ODR 100 Hz (0x08), normal bandwidth (0x20)
  ok &= write8(REG_ACC_RANGE, 0x08);  // +-8 g  -> 4096 LSB/g
  ok &= write8(REG_GYR_CONF, 0x28);   // ODR 100 Hz
  ok &= write8(REG_GYR_RANGE, 0x01);  // +-1000 dps -> 32.8 LSB/dps
  delay(10);
  uint8_t rb[6] = {0};
  readN(0x02, rb, 2);              // ERR_REG, PMU_STATUS
  readN(REG_ACC_CONF, rb + 2, 4);  // ACC_CONF, ACC_RANGE, GYR_CONF, GYR_RANGE
  bool configured = rb[0] == 0 && rb[3] == 0x08 && rb[5] == 0x01 && (rb[1] & 0x3C) == 0x14;
  Serial.printf("# bmi160 regs err=0x%02X pmu=0x%02X acc_conf=0x%02X acc_range=0x%02X "
                "gyr_conf=0x%02X gyr_range=0x%02X %s\n",
                rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], configured ? "ok" : "MISCONFIGURED");
  return ok && configured;
}

bool Bmi160::read(int16_t* axyz, int16_t* gxyz) {
  uint8_t b[12];
  if (!readN(REG_DATA_GYR, b, 12)) return false;  // 0x0C..0x11 gyro, 0x12..0x17 accel
  for (int i = 0; i < 3; ++i) {
    gxyz[i] = (int16_t)((uint16_t)b[2 * i] | ((uint16_t)b[2 * i + 1] << 8));
    axyz[i] = (int16_t)((uint16_t)b[6 + 2 * i] | ((uint16_t)b[6 + 2 * i + 1] << 8));
  }
  return true;
}
