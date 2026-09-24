"""Common data contracts (docs/02-system-design.md §5).

Every loader emits an ``IMUStream`` DataFrame with exactly these columns, in this order, plus
any *additional* columns a dataset needs (columns are added, never renamed):

``dataset, subject, session, device, placement, t, ax, ay, az, gx, gy, gz, exercise, rep_id,
set_id`` and the extra column ``units`` (``"si"`` or ``"normalized"``).

Units: ``t`` float seconds, strictly increasing within a stream and starting near 0 for the
session; ``ax..az`` m/s²; ``gx..gz`` rad/s. ``exercise`` is one of :data:`CANONICAL_EXERCISES`;
``rep_id`` is ``-1`` when the dataset has no per-rep boundaries; ``set_id`` is ``-1`` outside a
labelled set. ``units == "normalized"`` marks datasets that ship without physical units
(RecGym): the magnitude checks are skipped and such streams must never be mixed with SI streams
for training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

IMU_COLUMNS: tuple[str, ...] = (
    "dataset",
    "subject",
    "session",
    "device",
    "placement",
    "t",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "exercise",
    "rep_id",
    "set_id",
)
IMU_EXTRA_COLUMNS: tuple[str, ...] = ("units",)

WINDOW_COLUMNS: tuple[str, ...] = (
    "dataset",
    "subject",
    "session",
    "device",
    "window_id",
    "t_start",
    "t_end",
    "label_exercise",
    "label_active",
)  # + feat_* columns + x (fixed-shape 100x6 float32 list column)

POSE_COLUMNS_FIXED: tuple[str, ...] = ("session", "frame", "t", "valid")
# + lm_{i}_{x|y|z|v} for 33 image landmarks and wl_{i}_{x|y|z} for 33 world landmarks

REP_COLUMNS: tuple[str, ...] = (
    "session",
    "subject",
    "exercise",
    "rep_id",
    "t_start",
    "t_end",
    "duration_s",
    "source",
    "angles",
    "faults",
    "metrics",
)

CANONICAL_EXERCISES: tuple[str, ...] = ("curl", "press", "raise", "squat", "other", "idle")
ACTIVE_EXERCISES: frozenset[str] = frozenset(("curl", "press", "raise", "squat", "other"))
PLACEMENTS: tuple[str, ...] = (
    "wrist_l",
    "wrist_r",
    "wrist",  # side unknown (RecGym)
    "forearm_r",  # RecoFit master sensor
    "upper_arm",
    "pocket",
    "calf",  # RecGym "leg"
    "ear",
)
UNITS = ("si", "normalized")

SIGNAL_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")
MAX_ACCEL_MS2 = 200.0  # ±16 g plus margin; anything larger is a unit mix-up
MAX_GYRO_RADS = 40.0  # ±2000 dps plus margin


class SchemaError(ValueError):
    """Raised when a DataFrame does not follow the §5 contract."""


def is_active(exercise: str) -> bool:
    """True for every canonical label except ``idle``."""
    return exercise in ACTIVE_EXERCISES


def empty_imu_stream() -> pd.DataFrame:
    """An empty, correctly typed IMUStream."""
    return make_imu_stream(
        dataset="",
        subject="",
        session="",
        device="",
        placement="wrist_l",
        t=np.zeros(0),
        acc=np.zeros((0, 3)),
        gyr=np.zeros((0, 3)),
        exercise=np.array([], dtype=object),
    )


def make_imu_stream(
    *,
    dataset: str,
    subject: str,
    session: str,
    device: str,
    placement: str,
    t: np.ndarray,
    acc: np.ndarray,
    gyr: np.ndarray,
    exercise: Sequence[str] | np.ndarray,
    rep_id: np.ndarray | None = None,
    set_id: np.ndarray | None = None,
    units: str = "si",
) -> pd.DataFrame:
    """Assemble an IMUStream from arrays.

    ``t`` seconds (n,), ``acc`` m/s² (n, 3), ``gyr`` rad/s (n, 3), ``exercise`` canonical labels
    (n,). ``rep_id``/``set_id`` default to ``-1``. Does not validate; call
    :func:`validate_imu_stream` afterwards.
    """
    n = len(t)
    acc = np.asarray(acc, dtype=np.float32).reshape(n, 3)
    gyr = np.asarray(gyr, dtype=np.float32).reshape(n, 3)
    df = pd.DataFrame(
        {
            "dataset": pd.Series([dataset] * n, dtype="str"),
            "subject": pd.Series([subject] * n, dtype="str"),
            "session": pd.Series([session] * n, dtype="str"),
            "device": pd.Series([device] * n, dtype="str"),
            "placement": pd.Series([placement] * n, dtype="str"),
            "t": np.asarray(t, dtype=np.float64),
            "ax": acc[:, 0],
            "ay": acc[:, 1],
            "az": acc[:, 2],
            "gx": gyr[:, 0],
            "gy": gyr[:, 1],
            "gz": gyr[:, 2],
            "exercise": pd.Series(np.asarray(exercise, dtype=object), dtype="str"),
            "rep_id": (
                np.full(n, -1, dtype=np.int32)
                if rep_id is None
                else np.asarray(rep_id, dtype=np.int32)
            ),
            "set_id": (
                np.full(n, -1, dtype=np.int32)
                if set_id is None
                else np.asarray(set_id, dtype=np.int32)
            ),
            "units": pd.Series([units] * n, dtype="str"),
        }
    )
    return df


def validate_imu_stream(df: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` unless ``df`` follows the IMUStream contract.

    Checks: required columns present and in order, ``t`` float and strictly increasing,
    canonical exercise labels, known placements and units, and (for ``units == "si"``)
    plausible accelerometer / gyroscope magnitudes.
    """
    missing = [c for c in IMU_COLUMNS + IMU_EXTRA_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"IMUStream missing columns: {missing}")
    if tuple(df.columns[: len(IMU_COLUMNS)]) != IMU_COLUMNS:
        raise SchemaError(
            f"IMUStream columns must start with {IMU_COLUMNS}, got {tuple(df.columns)}"
        )
    if len(df) == 0:
        return
    t = df["t"].to_numpy()
    if not np.issubdtype(t.dtype, np.floating):
        raise SchemaError("t must be float seconds")
    if not np.all(np.isfinite(t)):
        raise SchemaError("t contains NaN/inf")
    if np.any(np.diff(t) <= 0):
        raise SchemaError("t must be strictly monotonic increasing within a stream")
    bad_ex = set(df["exercise"].unique()) - set(CANONICAL_EXERCISES)
    if bad_ex:
        raise SchemaError(f"exercise labels must be canonical {CANONICAL_EXERCISES}, got {bad_ex}")
    bad_pl = set(df["placement"].unique()) - set(PLACEMENTS)
    if bad_pl:
        raise SchemaError(f"unknown placement(s) {bad_pl}; allowed {PLACEMENTS}")
    units = set(df["units"].unique())
    if not units <= set(UNITS):
        raise SchemaError(f"units must be one of {UNITS}, got {units}")
    sig = df[list(SIGNAL_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(sig)):
        raise SchemaError("signal columns contain NaN/inf")
    if units == {"si"}:
        a_max = float(np.abs(sig[:, :3]).max())
        g_max = float(np.abs(sig[:, 3:]).max())
        if a_max > MAX_ACCEL_MS2:
            raise SchemaError(f"|accel| {a_max:.1f} exceeds {MAX_ACCEL_MS2} m/s²; units mix-up?")
        if g_max > MAX_GYRO_RADS:
            raise SchemaError(f"|gyro| {g_max:.1f} exceeds {MAX_GYRO_RADS} rad/s; units mix-up?")
