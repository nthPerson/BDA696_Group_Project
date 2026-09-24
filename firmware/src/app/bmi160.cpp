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
  // ODR 100 Hz (0x08), normal bandwidth (0x20) -> 0x28 for both sensors
  bool ok = write8(REG_ACC_CONF, 0x28);
  ok &= write8(REG_ACC_RANGE, 0x08);   // ±8 g  -> 4096 LSB/g
  ok &= write8(REG_GYR_CONF, 0x28);
  ok &= write8(REG_GYR_RANGE, 0x01);   // ±1000 dps -> 32.8 LSB/dps
  ok &= write8(REG_CMD, CMD_ACC_NORMAL);
  delay(5);
  ok &= write8(REG_CMD, CMD_GYR_NORMAL);
  delay(80);  // gyro start-up time
  return ok;
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
