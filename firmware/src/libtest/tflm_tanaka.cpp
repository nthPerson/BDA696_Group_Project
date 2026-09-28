// Compile test only: does tanakamasayuki/TensorFlowLite_ESP32 build for the ESP32-S3?
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace {
constexpr int kArena = 16 * 1024;
alignas(16) uint8_t arena[kArena];
const unsigned char model_data[] = {0};
}  // namespace
void setup() {
  Serial.begin(115200);
  const tflite::Model* model = tflite::GetModel(model_data);
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  static tflite::MicroErrorReporter error_reporter;
  static tflite::MicroInterpreter interpreter(model, resolver, arena, kArena, &error_reporter);
  Serial.println(interpreter.arena_used_bytes());
}
void loop() { delay(1000); }
