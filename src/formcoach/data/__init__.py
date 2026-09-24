"""Dataset loaders and the data manifest.

Planned modules (Checkpoint 1): ``mmfit.py``, ``recofit.py``, ``recgym.py``, ``team.py``
(one loader per source, each with ``load()`` and ``describe()``), ``schema.py`` (the common
IMUStream / Window / Pose / Rep Parquet schemas from docs/02-system-design.md §5) and
``manifest.py`` (reads/writes data/MANIFEST.md with SHA-256 checks).
"""
