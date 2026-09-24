"""FormCoach: sensor-gated webcam form coaching.

A wrist wearable (XIAO ESP32-S3 + BMI160) reports "exercise in progress" over BLE; this
package uses that signal to gate a MediaPipe pose pipeline on the laptop, segments reps and
applies per-exercise biomechanical rules. See docs/02-system-design.md.
"""

__version__ = "0.1.0"
