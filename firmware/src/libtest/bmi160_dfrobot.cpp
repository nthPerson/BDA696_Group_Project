// Compile test only: does DFRobot_BMI160 build for the XIAO ESP32-S3 with the pinned platform?
#include <Arduino.h>
#include <DFRobot_BMI160.h>
DFRobot_BMI160 bmi160;
void setup() {
  Serial.begin(115200);
  bmi160.softReset();
  bmi160.I2cInit(0x69);
}
void loop() {
  int16_t data[6];
  bmi160.getAccelGyroData(data);
  delay(20);
}
