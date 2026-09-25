#include "gate.h"
#include <Arduino.h>
#include <math.h>

#ifndef FC_GATE_ENERGY
#include <Chirale_TensorFlowLite.h>
#include "preprocess.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
extern const unsigned char gate_model_data[];
extern const size_t gate_model_data_len;
namespace {
constexpr int kArenaBytes = 40 * 1024;  // docs/02 §6.3 target
alignas(16) uint8_t g_arena[kArenaBytes];
tflite::MicroInterpreter* g_interp = nullptr;
TfLiteTensor* g_in = nullptr;
TfLiteTensor* g_out = nullptr;
const float kMean[6] = FC_PRE_MEAN;
const float kStd[6] = FC_PRE_STD;
}  // namespace
#endif

namespace {
constexpr float LSB_TO_MS2 = 9.80665f / 4096.0f;
constexpr float LSB_TO_RADS = (3.14159265f / 180.0f) / 32.8f;
}

bool Gate::begin() {
#ifdef FC_GATE_ENERGY
  model_ok_ = false;
  return true;
#else
  const tflite::Model* model = tflite::GetModel(gate_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("# gate: model schema %lu != %d, using energy rule\n",
                  (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddMean();
  resolver.AddExpandDims();
  resolver.AddQuantize();
  static tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("# gate: AllocateTensors failed, using energy rule");
    return false;
  }
  g_interp = &interpreter;
  g_in = interpreter.input(0);
  g_out = interpreter.output(0);
  arena_used_ = interpreter.arena_used_bytes();
  model_ok_ = (g_in->type == kTfLiteInt8) && (g_in->bytes == FC_PRE_WINDOW * FC_PRE_CHANNELS);
  Serial.printf("# gate: tflm ok arena=%u in=%u B classes=%d\n", (unsigned)arena_used_,
                (unsigned)g_in->bytes, FC_N_CLASSES);
  return model_ok_;
#endif
}

bool Gate::push(const int16_t* axyz, const int16_t* gxyz) {
  for (int i = 0; i < 3; ++i) {
    buf_[head_][i] = axyz[i];
    buf_[head_][3 + i] = gxyz[i];
  }
  head_ = (head_ + 1) % FC_WINDOW_SAMPLES;
  if (filled_ < FC_WINDOW_SAMPLES) ++filled_;
  if (++since_decision_ < FC_WINDOW_STRIDE_SAMPLES || filled_ < FC_WINDOW_SAMPLES) return false;
  since_decision_ = 0;
  bool active = decide();
  if (active == state_) {
    run_ = 0;
  } else if (++run_ >= (active ? FC_GATE_ON_WINDOWS : FC_GATE_OFF_WINDOWS)) {
    state_ = active;
    run_ = 0;
  }
  return true;
}

bool Gate::decide() {
#ifdef FC_GATE_ENERGY
  return decideEnergy();
#else
  return model_ok_ ? decideModel() : decideEnergy();
#endif
}

bool Gate::decideEnergy() {
  // var(|a|) over the window in (m/s^2)^2, same feature as the laptop energy gate
  float sum = 0.f, sum2 = 0.f;
  for (uint16_t i = 0; i < FC_WINDOW_SAMPLES; ++i) {
    float ax = buf_[i][0] * LSB_TO_MS2, ay = buf_[i][1] * LSB_TO_MS2, az = buf_[i][2] * LSB_TO_MS2;
    float m = sqrtf(ax * ax + ay * ay + az * az);
    sum += m;
    sum2 += m * m;
  }
  float mean = sum / FC_WINDOW_SAMPLES;
  last_energy_ = sum2 / FC_WINDOW_SAMPLES - mean * mean;
  return last_energy_ > FC_GATE_ENERGY_THRESHOLD_MS2SQ;
}

bool Gate::decideModel() {
#ifdef FC_GATE_ENERGY
  return decideEnergy();
#else
  uint32_t t0 = micros();
  int8_t* in = g_in->data.int8;
  // oldest sample first, same orientation as training windows
  for (uint16_t k = 0; k < FC_WINDOW_SAMPLES; ++k) {
    uint16_t i = (head_ + k) % FC_WINDOW_SAMPLES;
    for (int c = 0; c < 6; ++c) {
      float si = buf_[i][c] * (c < 3 ? LSB_TO_MS2 : LSB_TO_RADS);
      float z = (si - kMean[c]) / kStd[c];
      int q = (int)lroundf(z / FC_INPUT_SCALE) + FC_INPUT_ZERO_POINT;
      in[k * 6 + c] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
  }
  if (g_interp->Invoke() != kTfLiteOk) return decideEnergy();
  int best = 0;
  int8_t best_v = g_out->data.int8[0];
  for (int c = 1; c < FC_N_CLASSES; ++c) {
    if (g_out->data.int8[c] > best_v) { best_v = g_out->data.int8[c]; best = c; }
  }
  float p_idle = (g_out->data.int8[FC_CLASS_IDLE] - FC_OUTPUT_ZERO_POINT) * FC_OUTPUT_SCALE;
  last_class_ = best;
  last_p_active_ = 1.0f - p_idle;
  last_infer_us_ = micros() - t0;
  return best != FC_CLASS_IDLE;
#endif
}
