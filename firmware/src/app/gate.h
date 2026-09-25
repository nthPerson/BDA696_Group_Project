// Gate on the wearable: a 2 s ring buffer at 50 Hz, a decision every 0.5 s, hysteresis
// (protocol.h constants). Default: int8 1D-CNN via TFLite Micro (model/); with
// -DFC_GATE_ENERGY=1 the v1 motion-energy rule is used instead.
#pragma once
#include <stdint.h>
#include "protocol.h"

class Gate {
 public:
  bool begin();  // allocate the TFLM interpreter; returns false (and falls back) on error
  // Push one 50 Hz sample (raw LSB). Returns true when a decision was (re)computed.
  bool push(const int16_t* axyz, const int16_t* gxyz);
  bool state() const { return state_; }
  float lastEnergy() const { return last_energy_; }
  uint32_t lastInferUs() const { return last_infer_us_; }
  int lastClass() const { return last_class_; }
  float lastPActive() const { return last_p_active_; }
  size_t arenaUsed() const { return arena_used_; }
  bool usingModel() const { return model_ok_; }

 private:
  bool decide();        // dispatches to decideEnergy / decideModel
  bool decideEnergy();
  bool decideModel();
  int16_t buf_[FC_WINDOW_SAMPLES][6];
  uint16_t head_ = 0;
  uint16_t filled_ = 0;
  uint16_t since_decision_ = 0;
  bool state_ = false;
  uint8_t run_ = 0;
  float last_energy_ = 0.f;
  uint32_t last_infer_us_ = 0;
  int last_class_ = -1;
  float last_p_active_ = 0.f;
  size_t arena_used_ = 0;
  bool model_ok_ = false;
};
