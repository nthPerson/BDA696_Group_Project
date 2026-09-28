"""Dataset loaders (MM-Fit, RecoFit, RecGym, team recordings), the common IMUStream schema,
the download registry and the manifest writer. Every loader exposes ``available()``,
``describe()``, ``load_stream(...)`` and ``iter_streams()`` and returns the schema in
:mod:`formcoach.data.schema` (docs/02-system-design.md §5)."""

from __future__ import annotations

DATASETS = ("mmfit", "recofit", "recgym")


def loader(name: str):
    """Import and return the loader module for ``name``."""
    import importlib

    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {DATASETS}")
    return importlib.import_module(f"formcoach.data.{name}")
