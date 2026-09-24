// Compile test only: does hanyazou/BMI160-Arduino (CurieIMU fork) build for the XIAO ESP32-S3?
#include <Arduino.h>
#include <BMI160Gen.h>
void setup() {
  Serial.begin(115200);
  BMI160.begin(BMI160GenClass::I2C_MODE, 0x69);
  BMI160.setAccelerometerRange(8);
  BMI160.setGyroRange(1000);
}
void loop() {
  int ax, ay, az, gx, gy, gz;
  BMI160.readMotionSensor(ax, ay, az, gx, gy, gz);
  delay(20);
}
