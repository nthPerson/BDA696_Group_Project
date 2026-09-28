// Gate on the wearable: a 2 s ring buffer at 50 Hz, a decision every 0.5 s, hysteresis
// (protocol.h constants). v1 uses the motion-energy rule; PR 6 swaps decide() for TFLM.
#pragma once
#include <stdint.h>
#include "protocol.h"

class Gate {
 public:
  // Push one 50 Hz sample (raw LSB). Returns true when a decision was (re)computed.
  bool push(const int16_t* axyz, const int16_t* gxyz);
  bool state() const { return state_; }
  float lastEnergy() const { return last_energy_; }

 private:
  bool decide();  // energy rule over the window
  int16_t buf_[FC_WINDOW_SAMPLES][6];
  uint16_t head_ = 0;
  uint16_t filled_ = 0;
  uint16_t since_decision_ = 0;
  bool state_ = false;
  uint8_t run_ = 0;
  float last_energy_ = 0.f;
};
