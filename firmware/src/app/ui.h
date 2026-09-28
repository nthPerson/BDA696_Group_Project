// Button (D1, active-low, debounced; short press vs 2 s long press) and status LED patterns.
#pragma once
#include <Arduino.h>

enum class LedMode : uint8_t { Off, Advertising, Connected, GoodRep, Fault };

class Button {
 public:
  void begin(uint8_t pin);
  // Returns 0 = nothing, 1 = short press released, 2 = long press (fired once at 2 s)
  uint8_t poll();
  bool pressed() const { return state_; }

 private:
  uint8_t pin_ = 0;
  bool state_ = false, raw_ = false, long_fired_ = false;
  uint32_t changed_at_ = 0, pressed_at_ = 0;
};

class StatusLed {
 public:
  void begin(uint8_t pin, bool active_low);
  void set(LedMode mode);
  void flash(LedMode mode);  // GoodRep / Fault: temporary pattern, then back to previous
  void update();

 private:
  void write(bool on);
  uint8_t pin_ = 0;
  bool active_low_ = true;
  LedMode mode_ = LedMode::Off, base_ = LedMode::Off;
  uint32_t flash_until_ = 0;
};
