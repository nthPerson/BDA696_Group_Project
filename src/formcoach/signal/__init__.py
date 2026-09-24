"""IMU signal processing.

Planned modules (Checkpoint 2): ``resample.py`` (to 50 Hz), ``filters.py`` (low-pass,
band-pass), ``gravity.py`` (gravity separation), ``windows.py`` (2 s windows, 50 % overlap,
hand-crafted features) and ``reps.py`` (peak-detection rep segmentation).

This package must never import ``formcoach.pose`` so the IMU-only fallback (PocketTrainer,
docs/02-system-design.md §9) stays buildable.
"""
