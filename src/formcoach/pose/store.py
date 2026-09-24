"""``Pose`` Parquet (docs/02 §5): ``session, frame, t, valid`` + ``lm_{i}_{x|y|z|v}`` (image
landmarks, MediaPipe only) + ``wl_{i}_{x|y|z}`` (world/3-D joints) + ``skeleton`` (which joint
order the ``wl_`` columns follow: ``mediapipe33`` or ``h36m17`` for MM-Fit ``pose_3d``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.pose.skeletons import SKELETONS, Skeleton


@dataclass
class PoseSequence:
    session: str
    frames: np.ndarray  # (n,) int
    t: np.ndarray  # (n,) s
    world: np.ndarray  # (n, J, 3)
    skeleton: Skeleton
    valid: np.ndarray  # (n,) bool
    visibility: np.ndarray  # (n, J) in [0, 1]; ones when the source has none
    image: np.ndarray | None = None  # (n, J, 4) x, y, z, visibility (MediaPipe only)


def write_pose(
    path: Path,
    *,
    session: str,
    frames: np.ndarray,
    t: np.ndarray,
    world: np.ndarray,
    skeleton: str,
    image: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> Path:
    """Write a pose sequence; ``world`` is ``(n, J, 3)`` in the ``skeleton`` joint order."""
    if skeleton not in SKELETONS:
        raise KeyError(f"unknown skeleton {skeleton!r}; known {sorted(SKELETONS)}")
    n, j, _ = world.shape
    cols: dict[str, np.ndarray] = {
        "session": np.full(n, session, dtype=object),
        "frame": np.asarray(frames, dtype=np.int64),
        "t": np.asarray(t, dtype=np.float64),
        "valid": np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        "skeleton": np.full(n, skeleton, dtype=object),
    }
    if image is not None:
        for i in range(j):
            for k, ax in enumerate("xyzv"):
                cols[f"lm_{i}_{ax}"] = image[:, i, k].astype(np.float32)
    for i in range(j):
        for k, ax in enumerate("xyz"):
            cols[f"wl_{i}_{ax}"] = world[:, i, k].astype(np.float32)
    df = pd.DataFrame(cols)
    df["session"] = df["session"].astype("str")
    df["skeleton"] = df["skeleton"].astype("str")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_pose(path: Path) -> PoseSequence:
    df = pd.read_parquet(path)
    sk = SKELETONS[str(df["skeleton"].iloc[0])]
    j = len(sk.joints)
    n = len(df)
    world = np.empty((n, j, 3), dtype=np.float32)
    for i in range(j):
        for k, ax in enumerate("xyz"):
            world[:, i, k] = df[f"wl_{i}_{ax}"].to_numpy()
    image = None
    vis = np.ones((n, j), dtype=np.float32)
    if "lm_0_x" in df.columns:
        image = np.empty((n, j, 4), dtype=np.float32)
        for i in range(j):
            for k, ax in enumerate("xyzv"):
                image[:, i, k] = df[f"lm_{i}_{ax}"].to_numpy()
        vis = image[:, :, 3]
    return PoseSequence(
        session=str(df["session"].iloc[0]),
        frames=df["frame"].to_numpy(),
        t=df["t"].to_numpy(dtype=np.float64),
        world=world,
        skeleton=sk,
        valid=df["valid"].to_numpy(dtype=bool),
        visibility=vis,
        image=image,
    )
