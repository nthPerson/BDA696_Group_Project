"""IMUStream schema contract (docs/02-system-design.md §5): columns, dtypes, units, monotonic t."""

import numpy as np
import pandas as pd
import pytest

from formcoach.data import schema


def _good_stream(n: int = 100) -> pd.DataFrame:
    t = np.arange(n) / 50.0
    return schema.make_imu_stream(
        dataset="fixture",
        subject="S1",
        session="0001",
        device="fake",
        placement="wrist_l",
        t=t,
        acc=np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 9.81)]),
        gyr=np.zeros((n, 3)),
        exercise=np.array(["idle"] * n),
    )


def test_columns_follow_the_design_doc_order():
    df = _good_stream()
    assert list(df.columns)[: len(schema.IMU_COLUMNS)] == list(schema.IMU_COLUMNS)
    assert schema.IMU_COLUMNS[:5] == ("dataset", "subject", "session", "device", "placement")
    assert schema.IMU_COLUMNS[5:12] == ("t", "ax", "ay", "az", "gx", "gy", "gz")
    assert schema.IMU_COLUMNS[12:] == ("exercise", "rep_id", "set_id")


def test_valid_stream_passes_and_units_default_to_si():
    df = _good_stream()
    schema.validate_imu_stream(df)
    assert (df["units"] == "si").all()
    assert df["rep_id"].dtype == np.int32
    assert (df["rep_id"] == -1).all()
    assert df["ax"].dtype == np.float32
    assert df["t"].dtype == np.float64


def test_missing_column_is_a_schema_error():
    df = _good_stream().drop(columns=["gz"])
    with pytest.raises(schema.SchemaError, match="gz"):
        schema.validate_imu_stream(df)


def test_non_monotonic_t_is_a_schema_error():
    df = _good_stream()
    df.loc[10, "t"] = df.loc[9, "t"]
    with pytest.raises(schema.SchemaError, match="monotonic"):
        schema.validate_imu_stream(df)


def test_unknown_exercise_label_is_a_schema_error():
    df = _good_stream()
    df.loc[0, "exercise"] = "bicep_curls"  # raw dataset label, not canonical
    with pytest.raises(schema.SchemaError, match="exercise"):
        schema.validate_imu_stream(df)


def test_implausible_accel_magnitude_is_a_schema_error():
    df = _good_stream()
    df.loc[0, "ax"] = 5000.0
    with pytest.raises(schema.SchemaError, match="m/s"):
        schema.validate_imu_stream(df)


def test_normalized_units_skip_the_magnitude_check():
    df = _good_stream()
    df["units"] = "normalized"
    df.loc[0, "ax"] = 5000.0
    schema.validate_imu_stream(df)  # no raise


def test_canonical_exercise_vocabulary():
    assert schema.CANONICAL_EXERCISES == ("curl", "press", "raise", "squat", "other", "idle")
    assert schema.is_active("curl") and schema.is_active("other")
    assert not schema.is_active("idle")


def test_empty_stream_validates():
    schema.validate_imu_stream(schema.empty_imu_stream())
