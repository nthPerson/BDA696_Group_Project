"""Evaluation harness (docs/02-system-design.md §8).

Every command writes Markdown tables and PNG figures under ``reports/`` and is invoked by
``make eval``. Headline numbers are always leave-one-subject-out (LOSO) on unseen subjects.
"""
