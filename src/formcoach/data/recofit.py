"""RecoFit loader (Morris, Saponas, Guillory & Kelner, CHI 2014) → IMUStream.

Verified against the released files on 2026-09-24 (docs/04-datasets.md §2):

* Both ``.mat`` files are MATLAB v5 → ``scipy.io.loadmat(struct_as_record=False,
  squeeze_me=True)``; each loads in ~14 s and ~3 GB of RAM, so :func:`load_mat` caches them.
* ``subject_data`` has **94 rows** (subjects; 1-based ``subjectIndex`` == row + 1, and a
  ``subjectID`` such as 526). The multi-activity file has one column with a struct array of
  visits (126 in total); the single-activity file has 75 columns (``exerciseConstants.activities``)
  holding 4687 exercise recordings.
* Every recording: ``data.accelDataMatrix`` ``[t_s, x, y, z]`` in **g**, ``data.gyroDataMatrix``
  ``[t_s, x, y, z]`` in **dps**, both 50 Hz on the *master* sensor (``masterToken == "rightArm"``
  for all 4687 recordings; a ``slave*`` second device exists and is ignored here).
* Multi-file ``activityStartMatrix`` has **7 columns**: name, start_s, end_s, notes, reps
  (``-1`` when not counted), start_s (again), sequence-number struct.

Streams are built from the multi-activity file only (it contains idle time and can rebuild
every single-activity recording). Labels are canonical (:mod:`formcoach.data.labels`); junk
labels are stored as ``idle`` with the raw name in ``label_raw`` so windowing can drop them.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from formcoach.data import labels, schema

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "external" / "recofit"
MULTI_FILE = "exercise_data.50.0000_multionly.mat"
SINGLE_FILE = "exercise_data.50.0000_singleonly.mat"
G = 9.80665
DEG = np.pi / 180.0
PLACEMENT = "forearm_r"
DEVICE = "recofit_master"


def available(root: Path = DEFAULT_ROOT) -> bool:
    return (root / MULTI_FILE).exists()


@functools.lru_cache(maxsize=2)
def load_mat(path: Path) -> dict:
    """Load a RecoFit ``.mat`` file once per process (they are 1.5 GB each)."""
    from scipy.io import loadmat

    return loadmat(str(path), struct_as_record=False, squeeze_me=True)


def _records(cell) -> list:
    """A cell of ``subject_data`` is an empty array, one struct or a struct array."""
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return []
        return [r for r in cell.ravel() if hasattr(r, "data")]
    return [cell] if hasattr(cell, "data") else []


def _rows(subject_data) -> list:
    """Rows of the multi-activity ``subject_data`` (shape ``(94,)`` or ``(94, 1)``)."""
    sd = np.asarray(subject_data, dtype=object)
    if sd.ndim == 2:
        sd = sd[:, 0]
    return list(sd)


def list_visits(root: Path = DEFAULT_ROOT) -> list[tuple[int, int]]:
    """``(subject_index, visit)`` pairs, both 1-based as in ``load_exercise_data.m``."""
    mat = load_mat(root / MULTI_FILE)
    out = []
    for i, cell in enumerate(_rows(mat["subject_data"]), start=1):
        out.extend((i, v) for v in range(1, len(_records(cell)) + 1))
    return out


def _visit(root: Path, subject_index: int, visit: int):
    mat = load_mat(root / MULTI_FILE)
    recs = _records(_rows(mat["subject_data"])[subject_index - 1])
    if not 1 <= visit <= len(recs):
        raise KeyError(f"subject {subject_index} has {len(recs)} visits, asked for {visit}")
    return recs[visit - 1]


def _segments_of(rec) -> pd.DataFrame:
    asm = np.asarray(rec.activityStartMatrix, dtype=object)
    if asm.size == 0:
        return pd.DataFrame(
            columns=["segment_id", "label_raw", "exercise", "t_start", "t_end", "reps"]
        )
    asm = asm.reshape(-1, asm.shape[-1]) if asm.ndim == 2 else asm.reshape(1, -1)
    rows = []
    for k, row in enumerate(asm):
        name = str(row[0])
        reps = int(row[4]) if len(row) > 4 and np.ndim(row[4]) == 0 else -1
        rows.append((k, name, labels.canonical("recofit", name), float(row[1]), float(row[2]), reps))
    df = pd.DataFrame(
        rows, columns=["segment_id", "label_raw", "exercise", "t_start", "t_end", "reps"]
    )
    # pandas 3 would infer a `str` dtype and turn None into NaN; keep None for junk labels
    df["exercise"] = pd.Series([r[2] for r in rows], dtype=object)
    return df


def load_segments(root: Path, subject_index: int, visit: int) -> pd.DataFrame:
    """Labelled segments of one visit: ``segment_id, label_raw, exercise (None = junk), t_start,
    t_end, reps`` with times in seconds on the recording clock (``t = 0`` at the first sample)."""
    rec = _visit(root, subject_index, visit)
    segs = _segments_of(rec)
    t0 = float(np.asarray(rec.data.accelDataMatrix)[0, 0])
    segs["t_start"] -= t0
    segs["t_end"] -= t0
    return segs


def load_stream(root: Path, subject_index: int, visit: int) -> pd.DataFrame:
    """IMUStream for one visit of one subject (multi-activity file).

    ``subject`` is ``R<subjectID>``, ``session`` is ``v<visit>``, accel g→m/s², gyro dps→rad/s,
    ``t`` seconds from the first sample. Extra columns ``label_raw`` and ``units == "si"``.
    ``set_id`` is the segment index for exercise segments (junk/idle keep ``-1``).
    """
    rec = _visit(root, subject_index, visit)
    acc = np.asarray(rec.data.accelDataMatrix, dtype=np.float64)
    gyr = np.asarray(rec.data.gyroDataMatrix, dtype=np.float64)
    t = acc[:, 0] - acc[0, 0]
    if gyr.shape == acc.shape and np.allclose(gyr[:, 0], acc[:, 0]):
        g_xyz = gyr[:, 1:4]
    else:
        gt = gyr[:, 0] - acc[0, 0]
        g_xyz = np.column_stack([np.interp(t, gt, gyr[:, k]) for k in range(1, 4)])
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, a_xyz, g_xyz = t[keep], acc[keep, 1:4], g_xyz[keep]
    segs = _segments_of(rec)
    segs["t_start"] -= acc[0, 0]
    segs["t_end"] -= acc[0, 0]
    raw = np.full(len(t), "Non-Exercise", dtype=object)
    exercise = np.full(len(t), "idle", dtype=object)
    set_id = np.full(len(t), -1, dtype=np.int32)
    for s in segs.itertuples(index=False):
        m = (t >= s.t_start) & (t < s.t_end)  # half-open: a boundary sample goes to the next segment
        raw[m] = s.label_raw
        if s.exercise is None:
            exercise[m] = "idle"
        else:
            exercise[m] = s.exercise
            if s.exercise != "idle":
                set_id[m] = s.segment_id
    df = schema.make_imu_stream(
        dataset="recofit",
        subject=f"R{int(rec.subjectID)}",
        session=f"v{visit}",
        device=DEVICE,
        placement=PLACEMENT,
        t=t,
        acc=a_xyz * G,
        gyr=g_xyz * DEG,
        exercise=exercise,
        set_id=set_id,
    )
    df["label_raw"] = pd.Series(raw, dtype="str")
    return df


def iter_streams(root: Path = DEFAULT_ROOT):
    """Yield ``(subject_index, visit, IMUStream)`` for every visit."""
    for si, v in list_visits(root):
        yield si, v, load_stream(root, si, v)


def describe(root: Path = DEFAULT_ROOT) -> dict:
    """Subjects, visits, hours, minutes per canonical label (multi file) and single-file counts."""
    mat = load_mat(root / MULTI_FILE)
    rows = _rows(mat["subject_data"])
    minutes: dict[str, float] = {}
    junk_minutes = 0.0
    visits = 0
    hours = 0.0
    subjects = set()
    rates = []
    for cell in rows:
        for rec in _records(cell):
            visits += 1
            subjects.add(int(rec.subjectID))
            acc = np.asarray(rec.data.accelDataMatrix)
            hours += float(acc[-1, 0] - acc[0, 0]) / 3600.0
            dt = np.diff(acc[:200, 0])
            rates.append(1.0 / float(np.median(dt)))
            for s in _segments_of(rec).itertuples(index=False):
                mins = (s.t_end - s.t_start) / 60.0
                if s.exercise is None:
                    junk_minutes += mins
                else:
                    minutes[s.exercise] = minutes.get(s.exercise, 0.0) + mins
    out = {
        "dataset": "recofit",
        "root": str(root),
        "multi": {
            "subjects": len(subjects),
            "visits": visits,
            "hours": round(hours, 1),
            "rate_hz": round(float(np.median(rates)), 1) if rates else None,
            "minutes_per_exercise": {k: round(v, 1) for k, v in sorted(minutes.items())},
            "junk_minutes": round(junk_minutes, 1),
        },
        "activities": len(list(np.atleast_1d(mat["exerciseConstants"].activities))),
    }
    single = root / SINGLE_FILE
    if single.exists():
        smat = load_mat(single)
        sd = np.asarray(smat["subject_data"], dtype=object)
        n = sum(len(_records(c)) for c in sd.ravel())
        out["single"] = {"recordings": n, "shape": list(sd.shape)}
    return out
