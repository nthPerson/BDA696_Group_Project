#include "gate.h"
#include <math.h>

namespace {
constexpr float LSB_TO_MS2 = 9.80665f / 4096.0f;
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
