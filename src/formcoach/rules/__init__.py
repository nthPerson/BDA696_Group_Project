"""Config-driven form rules (docs/02-system-design.md §7).

Planned (Checkpoint 6): ``rules.yaml`` with per-exercise thresholds and coaching messages,
``engine.py`` that evaluates every rule on a resampled rep and returns
``{code, severity, value, threshold, message}`` records.
"""
