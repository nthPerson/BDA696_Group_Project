// Minimal BMI160 register driver over I2C (ADR-0020): accel ±8 g, gyro ±1000 dps, ODR 100 Hz.
// Register map from the Bosch BMI160 datasheet (rev 1.0). Raw LSB values match protocol.h
// (4096 LSB/g, 32.8 LSB/dps) so the Python side converts identically for every source.
#pragma once
#include <Arduino.h>
#include <Wire.h>

class Bmi160 {
 public:
  static constexpr uint8_t CHIP_ID = 0xD1;
  bool begin(TwoWire& wire, uint8_t address);  // true when CHIP_ID answers and config is written
  bool read(int16_t* axyz, int16_t* gxyz);     // one burst read of gyro + accel (12 bytes)
  uint8_t address() const { return addr_; }
  static uint8_t scan(TwoWire& wire);          // 0x68 / 0x69 or 0 when absent

 private:
  enum Reg : uint8_t {
    REG_CHIP_ID = 0x00, REG_DATA_GYR = 0x0C, REG_DATA_ACC = 0x12, REG_ACC_CONF = 0x40,
    REG_ACC_RANGE = 0x41, REG_GYR_CONF = 0x42, REG_GYR_RANGE = 0x43, REG_CMD = 0x7E,
  };
  enum Cmd : uint8_t { CMD_SOFT_RESET = 0xB6, CMD_ACC_NORMAL = 0x11, CMD_GYR_NORMAL = 0x15 };
  bool write8(uint8_t reg, uint8_t val);
  bool readN(uint8_t reg, uint8_t* buf, size_t n);
  TwoWire* wire_ = nullptr;
  uint8_t addr_ = 0;
};
