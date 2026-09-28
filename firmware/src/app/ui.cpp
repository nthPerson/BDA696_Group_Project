#include "ui.h"

namespace {
constexpr uint32_t DEBOUNCE_MS = 30;
constexpr uint32_t LONG_PRESS_MS = 2000;
}  // namespace

void Button::begin(uint8_t pin) {
  pin_ = pin;
  pinMode(pin_, INPUT_PULLUP);
}

uint8_t Button::poll() {
  bool raw = digitalRead(pin_) == LOW;
  uint32_t now = millis();
  uint8_t event = 0;
  if (raw != raw_) {
    raw_ = raw;
    changed_at_ = now;
  } else if (raw != state_ && now - changed_at_ >= DEBOUNCE_MS) {
    state_ = raw;
    if (state_) {
      pressed_at_ = now;
      long_fired_ = false;
    } else if (!long_fired_) {
      event = 1;
    }
  }
  if (state_ && !long_fired_ && now - pressed_at_ >= LONG_PRESS_MS) {
    long_fired_ = true;
    event = 2;
  }
  return event;
}

void StatusLed::begin(uint8_t pin, bool active_low) {
  pin_ = pin;
  active_low_ = active_low;
  pinMode(pin_, OUTPUT);
  write(false);
}

void StatusLed::write(bool on) { digitalWrite(pin_, (on != active_low_) ? HIGH : LOW); }

void StatusLed::set(LedMode mode) { base_ = mode_ = mode; }

void StatusLed::flash(LedMode mode) {
  mode_ = mode;
  flash_until_ = millis() + (mode == LedMode::GoodRep ? 600 : 800);
}

void StatusLed::update() {
  uint32_t now = millis();
  if (flash_until_ && now >= flash_until_) {
    flash_until_ = 0;
    mode_ = base_;
  }
  uint32_t p = now % 1000;
  switch (mode_) {
    case LedMode::Off: write(false); break;
    case LedMode::Advertising: write(p < 100); break;               // slow blink
    case LedMode::Connected: write(true); break;                     // solid
    case LedMode::GoodRep: write((now % 200) < 100); break;          // triple-ish blink
    case LedMode::Fault: write((now % 250) < 60 || (now % 250 > 125 && now % 250 < 185)); break;
  }
}
