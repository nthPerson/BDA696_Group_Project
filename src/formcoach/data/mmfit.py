"""MM-Fit loader (Strömbäck, Huang & Radu, IMWUT 2020) → IMUStream.

Verified against the released files on 2026-09-24 (docs/04-datasets.md §1):

* ``mm-fit/wXX/wXX_<modality>.npy``; sensor arrays are ``(n, 5)``: **frame index, Unix
  timestamp in ms, x, y, z**. Smartwatch (``sw_l``, ``sw_r``) accelerometer is already in m/s²
  and gyroscope in rad/s at 100 Hz; earbud 90 Hz; phones 210/500 Hz. Some phone files carry a
  suffix (``w03_sp_l_acc_0.npy``) and 15 workouts have no left-phone data at all.
* ``wXX_pose_3d.npy`` is ``(3, frames, 18)`` (col 0 = frame, then **17 Human3.6M joints**);
  ``wXX_pose_2d.npy`` is ``(2, frames, 19)`` (**18 COCO joints**).
* ``wXX_labels.csv`` rows: ``start_frame, end_frame, reps, activity`` (no header).
* Workout → participant mapping and the authors' split come from the MM-Fit EDA notebook.

Session clock: the left RGB-D camera frame is the reference; sensor rows carry both a frame
index and a timestamp. :func:`session_clock` fits ``timestamp = t0 + ms_per_frame * frame`` on
the watch accelerometer so pose frames and sensor samples share one ``t`` axis, with ``t = 0``
at video frame 0. Duplicate sensor timestamps (a few hundred per workout) are dropped.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.data import labels, schema

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "external" / "mmfit" / "mm-fit"

WORKOUTS: tuple[str, ...] = tuple(f"w{i:02d}" for i in range(21))
WORKOUT_SUBJECT: dict[str, int] = {
    "w00": 2, "w01": 0, "w02": 1, "w03": 0, "w04": 1, "w05": 2, "w06": 0, "w07": 1, "w08": 0,
    "w09": 1, "w10": 0, "w11": 1, "w12": 3, "w13": 4, "w14": 0, "w15": 1, "w16": 5, "w17": 6,
    "w18": 7, "w19": 8, "w20": 9,
}  # fmt: skip
SPLITS: dict[str, list[str]] = {
    "train": ["w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"],
    "val": ["w14", "w15", "w19"],
    "test": ["w09", "w10", "w11"],
    "unseen_test": ["w00", "w05", "w12", "w13", "w20"],
}
IMU_DEVICES: dict[str, tuple[str, str, float]] = {
    # device key -> (placement, description, nominal rate Hz)
    "sw_l": ("wrist_l", "TicWatch Pro left wrist", 100.0),
    "sw_r": ("wrist_r", "TicWatch Pro right wrist", 100.0),
    "eb_l": ("ear", "eSense earbud left", 90.0),
    "sp_l": ("pocket", "Samsung S7 left pocket", 210.0),
    "sp_r": ("pocket", "Huawei P20 right pocket", 500.0),
}
H36M17_JOINTS: tuple[str, ...] = (
    "hip_c", "hip_l", "knee_l", "foot_l", "hip_r", "knee_r", "foot_r", "spine", "thorax",
    "neck", "head", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l",
)  # fmt: skip
COCO18_JOINTS: tuple[str, ...] = (
    "nose", "neck", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l",
    "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l", "eye_r", "eye_l", "ear_r", "ear_l",
)  # fmt: skip


def subject_id(workout: str) -> str:
    """Canonical subject code ``P<n>`` for a workout."""
    return f"P{WORKOUT_SUBJECT[workout]}"


def available(root: Path = DEFAULT_ROOT) -> bool:
    return root.is_dir() and any((root / w / f"{w}_labels.csv").exists() for w in WORKOUTS)


def list_workouts(root: Path = DEFAULT_ROOT) -> list[str]:
    return [w for w in WORKOUTS if (root / w / f"{w}_labels.csv").exists()]


def load_modality(root: Path, workout: str, key: str) -> np.ndarray | None:
    """Load ``wXX_<key>.npy`` (or the first ``wXX_<key>_*.npy`` variant); ``None`` if absent."""
    d = root / workout
    exact = d / f"{workout}_{key}.npy"
    if exact.exists():
        return np.load(exact)
    variants = sorted(d.glob(f"{workout}_{key}_*.npy"))
    if variants:
        return np.load(variants[0])
    return None


def load_labels(root: Path, workout: str) -> pd.DataFrame:
    """Set labels: ``set_id, start_frame, end_frame, reps, activity`` (raw MM-Fit names)."""
    rows = []
    with (root / workout / f"{workout}_labels.csv").open(newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row or not row[0].strip():
                continue
            rows.append((i, int(row[0]), int(row[1]), int(row[2]), row[3].strip()))
    return pd.DataFrame(rows, columns=["set_id", "start_frame", "end_frame", "reps", "activity"])


@dataclass(frozen=True)
class SessionClock:
    """Linear map between video frame index and Unix ms, fitted on a sensor stream."""

    t0_ms: float  # timestamp of frame 0
    ms_per_frame: float

    def frame_to_t(self, frame: np.ndarray | float) -> np.ndarray | float:
        """Seconds since frame 0."""
        return np.asarray(frame, dtype=np.float64) * self.ms_per_frame / 1000.0

    def ms_to_t(self, ts_ms: np.ndarray) -> np.ndarray:
        return (np.asarray(ts_ms, dtype=np.float64) - self.t0_ms) / 1000.0


def session_clock(root: Path, workout: str) -> SessionClock:
    """Fit frame→timestamp on the first available watch/earbud accelerometer stream."""
    for key in ("sw_l_acc", "sw_r_acc", "eb_l_acc", "sp_r_acc"):
        arr = load_modality(root, workout, key)
        if arr is not None and len(arr) > 10:
            break
    else:
        raise FileNotFoundError(f"{workout}: no sensor stream to fit a session clock on")
    frames = arr[:, 0].astype(np.float64)
    ts = arr[:, 1].astype(np.float64)
    slope, intercept = np.polyfit(frames, ts, 1)
    return SessionClock(t0_ms=float(intercept), ms_per_frame=float(slope))


def _merge_acc_gyr(acc: np.ndarray, gyr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align gyro onto the accelerometer rows (interpolating when timestamps differ)."""
    if acc.shape == gyr.shape and np.array_equal(acc[:, 1], gyr[:, 1]):
        return acc[:, 0], acc[:, 1], np.column_stack([acc[:, 2:5], gyr[:, 2:5]])
    order = np.argsort(gyr[:, 1], kind="stable")
    g_t = gyr[order, 1]
    g_t_u, idx = np.unique(g_t, return_index=True)
    g_xyz = gyr[order][idx, 2:5]
    gi = np.column_stack([np.interp(acc[:, 1], g_t_u, g_xyz[:, k]) for k in range(3)])
    return acc[:, 0], acc[:, 1], np.column_stack([acc[:, 2:5], gi])


def load_stream(root: Path, workout: str, device: str = "sw_l") -> pd.DataFrame:
    """IMUStream for one workout and device (``sw_l`` default; see :data:`IMU_DEVICES`).

    Extra columns: ``frame`` (video frame index of each sample), ``label_raw`` (MM-Fit activity
    name or ``non_activity``), ``units == "si"``. ``exercise`` is canonical; ``set_id`` is the
    row of ``wXX_labels.csv`` (0-based) or ``-1``; ``rep_id`` is always ``-1`` (set counts only).
    """
    if device not in IMU_DEVICES:
        raise KeyError(f"device must be one of {sorted(IMU_DEVICES)}")
    acc = load_modality(root, workout, f"{device}_acc")
    gyr = load_modality(root, workout, f"{device}_gyr")
    if acc is None or gyr is None:
        raise FileNotFoundError(f"{workout}: {device} acc/gyr not present")
    frames, ts_ms, sig = _merge_acc_gyr(acc, gyr)
    order = np.argsort(ts_ms, kind="stable")
    frames, ts_ms, sig = frames[order], ts_ms[order], sig[order]
    keep = np.concatenate([[True], np.diff(ts_ms) > 0])  # drop duplicate timestamps
    frames, ts_ms, sig = frames[keep], ts_ms[keep], sig[keep]
    clock = session_clock(root, workout)
    t = clock.ms_to_t(ts_ms)
    sets = load_labels(root, workout)
    raw = np.full(len(t), "non_activity", dtype=object)
    set_id = np.full(len(t), -1, dtype=np.int32)
    fr = frames.astype(np.int64)
    for s in sets.itertuples(index=False):
        m = (fr >= s.start_frame) & (fr <= s.end_frame)
        raw[m] = s.activity
        set_id[m] = s.set_id
    exercise = np.array([labels.canonical("mmfit", r) for r in raw], dtype=object)
    placement = IMU_DEVICES[device][0]
    df = schema.make_imu_stream(
        dataset="mmfit",
        subject=subject_id(workout),
        session=workout,
        device=device,
        placement=placement,
        t=t,
        acc=sig[:, :3],
        gyr=sig[:, 3:],
        exercise=exercise,
        set_id=set_id,
    )
    df["frame"] = fr.astype(np.int32)
    df["label_raw"] = pd.Series(raw, dtype="str")
    return df


def load_sets(root: Path, workout: str) -> pd.DataFrame:
    """Per-set table: ``workout, subject, set_id, start_frame, end_frame, t_start, t_end, reps,
    activity, exercise`` — the ground truth for `eval repcount`."""
    sets = load_labels(root, workout)
    clock = session_clock(root, workout)
    sets.insert(0, "subject", subject_id(workout))
    sets.insert(0, "workout", workout)
    sets["t_start"] = clock.frame_to_t(sets["start_frame"].to_numpy())
    sets["t_end"] = clock.frame_to_t(sets["end_frame"].to_numpy())
    sets["exercise"] = [labels.canonical("mmfit", a) for a in sets["activity"]]
    return sets


def load_pose3d(root: Path, workout: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(frames, xyz, t)``: frame indices (n,), joints ``(n, 17, 3)`` in the H3.6M order
    (:data:`H36M17_JOINTS`, camera-frame millimetres), and ``t`` seconds on the session clock."""
    arr = load_modality(root, workout, "pose_3d")
    if arr is None:
        raise FileNotFoundError(f"{workout}: pose_3d missing")
    frames = arr[0, :, 0].astype(np.int64)
    xyz = np.transpose(arr[:, :, 1:], (1, 2, 0)).astype(np.float32)  # (n, joints, 3)
    t = session_clock(root, workout).frame_to_t(frames)
    return frames, xyz, np.asarray(t)


def iter_streams(root: Path = DEFAULT_ROOT, devices: tuple[str, ...] = ("sw_l", "sw_r")):
    """Yield ``(workout, device, IMUStream)`` for every workout and available device."""
    for w in list_workouts(root):
        for d in devices:
            try:
                yield w, d, load_stream(root, w, d)
            except FileNotFoundError:
                continue


def describe(root: Path = DEFAULT_ROOT) -> dict:
    """Summary used by `formcoach data profile` and the loader's docs."""
    workouts = list_workouts(root)
    out: dict = {
        "dataset": "mmfit",
        "root": str(root),
        "workouts": len(workouts),
        "subjects": len({WORKOUT_SUBJECT[w] for w in workouts}),
        "devices": {},
        "sets_per_exercise": {},
        "reps_per_exercise": {},
        "missing_modalities": {},
        "minutes_labeled": 0.0,
    }
    per_dev: dict[str, dict] = {}
    for w in workouts:
        sets = load_sets(root, w)
        for s in sets.itertuples(index=False):
            out["sets_per_exercise"][s.exercise] = out["sets_per_exercise"].get(s.exercise, 0) + 1
            out["reps_per_exercise"][s.exercise] = out["reps_per_exercise"].get(s.exercise, 0) + s.reps
            out["minutes_labeled"] += (s.t_end - s.t_start) / 60.0
        missing = [k for k in ("sw_l", "sw_r", "eb_l", "sp_l", "sp_r", "pose_2d", "pose_3d")
                   if load_modality(root, w, k if k.startswith("pose") else f"{k}_acc") is None]  # fmt: skip
        if missing:
            out["missing_modalities"][w] = missing
        for dev in IMU_DEVICES:
            acc = load_modality(root, w, f"{dev}_acc")
            if acc is None:
                continue
            d = per_dev.setdefault(dev, {"workouts": 0, "minutes": 0.0, "rates": [], "dup_ts": 0})
            ts = acc[:, 1]
            d["workouts"] += 1
            d["minutes"] += float((ts[-1] - ts[0]) / 60000.0)
            dt = np.diff(ts)
            d["rates"].append(1000.0 / float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0)
            d["dup_ts"] += int((dt == 0).sum())
    for dev, d in per_dev.items():
        out["devices"][dev] = {
            "workouts": d["workouts"],
            "minutes": round(d["minutes"], 1),
            "rate_hz": round(float(np.median(d["rates"])), 1),
            "duplicate_timestamps": d["dup_ts"],
            "placement": IMU_DEVICES[dev][0],
        }
    out["minutes_labeled"] = round(out["minutes_labeled"], 1)
    return out
