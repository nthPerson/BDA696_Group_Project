"""Build the tiny raw-format fixtures under ``data/fixtures/`` (``formcoach data make-fixtures``).

Each fixture mimics the *real* on-disk format of its dataset (file names, array shapes, units,
MATLAB struct layout, CSV columns) with synthetic signals: gravity plus a 1 Hz "curl" sine
during exercise sets. They are what the loader tests and the CI smoke test run on, so a fresh
clone needs no download. Deterministic (fixed seed); every file is far below the 2 MB cap.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "data" / "fixtures"
SEED = 20260924


def _motion(t: np.ndarray, active: np.ndarray, rng, amp: float = 3.0, f: float = 1.0):
    """Gravity on z plus a sine on x while active, plus noise; returns (acc, gyr) in SI."""
    acc = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, 9.81)])
    gyr = np.zeros((len(t), 3))
    acc[:, 0] += active * amp * np.sin(2 * np.pi * f * t)
    gyr[:, 1] += active * 2.0 * np.cos(2 * np.pi * f * t)
    acc += rng.normal(0, 0.05, acc.shape)
    gyr += rng.normal(0, 0.01, gyr.shape)
    return acc, gyr


def build_mmfit(root: Path, rng) -> None:
    """One workout ``w00`` with sw_l/sw_r acc+gyr+hr (100 Hz), pose_2d/3d (30 fps), labels."""
    w = "w00"
    d = root / "mmfit" / "mm-fit" / w
    d.mkdir(parents=True, exist_ok=True)
    fps, ms_per_frame = 30.0, 1000.0 / 30.0
    n_frames = 900  # 30 s
    t0_ms = 1_562_790_000_000.0
    sets = [(150, 450, 10, "bicep_curls"), (600, 840, 8, "squats")]
    with (d / f"{w}_labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows(sets)
    frames_t = np.arange(n_frames) / fps
    active = np.zeros(n_frames)
    for s, e, _, _ in sets:
        active[s : e + 1] = 1.0
    for dev in ("sw_l", "sw_r"):
        n = n_frames * 100 // 30
        t = np.arange(n) / 100.0
        frame = np.minimum((t * fps).astype(int), n_frames - 1)
        ts = np.round(t0_ms + t * 1000.0)
        ts[100:103] = ts[100]  # duplicate timestamps, like the real watch logs
        acc, gyr = _motion(t, active[frame], rng, amp=3.0 if dev == "sw_l" else 2.0)
        np.save(d / f"{w}_{dev}_acc.npy", np.column_stack([frame, ts, acc]))
        np.save(d / f"{w}_{dev}_gyr.npy", np.column_stack([frame, ts, gyr]))
        hr_frames = np.arange(0, n_frames, 30)
        hr = np.column_stack(
            [hr_frames, t0_ms + hr_frames * ms_per_frame, 90 + 20 * active[hr_frames]]
        )
        np.save(d / f"{w}_{dev}_hr.npy", hr)
    # pose_3d: (3, frames, 18) col 0 = frame; a standing skeleton in mm with elbow flexion
    joints = np.zeros((n_frames, 17, 3))
    base = {
        0: (0, 0, 0), 1: (-100, 0, 0), 2: (-100, -450, 0), 3: (-100, -900, 0),
        4: (100, 0, 0), 5: (100, -450, 0), 6: (100, -900, 0), 7: (0, 250, 0),
        8: (0, 500, 0), 9: (0, 580, 0), 10: (0, 680, 0),
        11: (180, 480, 0), 12: (200, 180, 0), 13: (200, -100, 0),
        14: (-180, 480, 0), 15: (-200, 180, 0), 16: (-200, -100, 0),
    }  # fmt: skip
    for j, p in base.items():
        joints[:, j, :] = p
    flex = active * (0.5 - 0.5 * np.cos(2 * np.pi * 1.0 * frames_t))  # 0..1 per rep
    for wrist, elbow in ((13, 12), (16, 15)):
        joints[:, wrist, 1] = joints[:, elbow, 1] - 280 * (1 - flex) + 200 * flex
        joints[:, wrist, 2] = -280 * flex
    pose3d = np.concatenate(
        [np.broadcast_to(np.arange(n_frames)[None, :, None], (3, n_frames, 1)),
         np.transpose(joints, (2, 0, 1))], axis=2,
    )  # fmt: skip
    np.save(d / f"{w}_pose_3d.npy", pose3d + rng.normal(0, 1.0, pose3d.shape) * np.array([0, 1, 1])[:, None, None] * 0)
    pose2d = np.zeros((2, n_frames, 19))
    pose2d[:, :, 0] = np.arange(n_frames)
    pose2d[:, :, 1:] = 320 + rng.normal(0, 5, (2, n_frames, 18))
    np.save(d / f"{w}_pose_2d.npy", pose2d)


def build_recofit(root: Path, rng) -> None:
    """Multi- and single-activity ``.mat`` files with the real struct layout (2 subjects)."""
    from scipy.io import savemat

    from formcoach.data.labels import RECOFIT_ACTIVITIES

    d = root / "recofit"
    d.mkdir(parents=True, exist_ok=True)

    def visit(subject_id: int, segments, t_start=1561.13):
        dur = segments[-1][2]
        n = int(dur * 50)
        t = t_start + np.arange(n) / 50.0
        active = np.zeros(n)
        for name, s, e, reps in segments:
            if reps > 0:
                active[int(s * 50) : int(e * 50)] = 1.0
        acc, gyr = _motion(t - t_start, active, rng)
        asm = np.empty((len(segments), 7), dtype=object)
        for k, (name, s, e, reps) in enumerate(segments):
            asm[k] = [name, t_start + s, t_start + e, np.array([], dtype="<U1"), reps,
                      t_start + s, {"startSequenceNumberMaster": 0}]  # fmt: skip
        return {
            "activityName": "Activities",
            "activityStartMatrix": asm,
            "subjectID": subject_id,
            "subjectIndex": 1,
            "masterToken": "rightArm",
            "sampleRate": 50,
            "incompleteData": 0,
            "data": {
                "accelDataMatrix": np.column_stack([t, acc / 9.80665]),  # g
                "gyroDataMatrix": np.column_stack([t, gyr * 180 / np.pi]),  # dps
                "slaveAccelDataMatrix": np.zeros((0, 4)),
                "slaveGyroDataMatrix": np.zeros((0, 4)),
            },
        }

    segs1 = [("Non-Exercise", 0, 5, -1), ("Tap Left Device", 5, 6, -1), ("Bicep Curl", 6, 26, 20),
             ("Squat", 26, 41, 15)]  # fmt: skip
    segs2 = [("Non-Exercise", 0, 4, -1), ("Lateral Raise", 4, 19, 15)]
    segs3 = [("Device on Table", 0, 3, -1), ("Shoulder Press (dumbbell)", 3, 18, 15)]
    v1 = visit(526, segs1)
    v2a, v2b = visit(527, segs2), visit(527, segs3)
    v2a["subjectIndex"] = v2b["subjectIndex"] = 2
    # subject 2 has two visits -> struct array (record array with object fields)
    fields = list(v2a.keys())
    arr = np.empty(2, dtype=[(f, object) for f in fields])
    for i, v in enumerate((v2a, v2b)):
        for f in fields:
            arr[i][f] = v[f]
    subject_data = np.empty((2, 1), dtype=object)
    subject_data[0, 0] = v1
    subject_data[1, 0] = arr
    activities = np.empty(len(RECOFIT_ACTIVITIES), dtype=object)
    activities[:] = list(RECOFIT_ACTIVITIES)
    consts = {"activities": activities, "usefulActivityGroupings": np.empty((0, 2), dtype=object)}
    savemat(d / "exercise_data.50.0000_multionly.mat",
            {"subject_data": subject_data, "exerciseConstants": consts, "Fs": 50}, do_compression=True)  # fmt: skip
    # single-activity file: (2 subjects x 75 activities) cells, two non-empty
    single = np.empty((2, len(RECOFIT_ACTIVITIES)), dtype=object)
    for i in range(single.shape[0]):
        for j in range(single.shape[1]):
            single[i, j] = np.zeros((0, 0))
    rec = visit(526, [("Bicep Curl", 0, 20, 20)])
    rec["activityName"] = "Bicep Curl"
    rec["activityReps"] = 20
    del rec["activityStartMatrix"]
    single[0, RECOFIT_ACTIVITIES.index("Bicep Curl")] = rec
    rec2 = visit(527, [("Lateral Raise", 0, 15, 15)])
    rec2["activityName"] = "Lateral Raise"
    rec2["activityReps"] = 15
    del rec2["activityStartMatrix"]
    single[1, RECOFIT_ACTIVITIES.index("Lateral Raise")] = rec2
    savemat(d / "exercise_data.50.0000_singleonly.mat",
            {"subject_data": single, "exerciseConstants": consts, "Fs": 50}, do_compression=True)  # fmt: skip


def build_recgym(root: Path, rng) -> None:
    """``RecGym.csv`` with 2 subjects x 2 positions x 1 session, min-max normalised like the real file."""
    d = root / "recgym"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in (1, 2):
        for position in ("wrist", "leg"):
            n = 20 * 40  # 40 s
            t = np.arange(n) / 20.0
            workout = np.where((t >= 10) & (t < 30), "ArmCurl", "Null")
            acc, gyr = _motion(t, (workout == "ArmCurl").astype(float), rng)
            sig = np.column_stack([acc, gyr])
            norm = 0.5 + sig / 40.0  # pretend the dataset-wide min-max scaling
            cap = 0.5 + rng.normal(0, 0.02, n)
            for i in range(n):
                rows.append([subject, position, 1, *np.round(norm[i], 6), round(cap[i], 6), workout[i]])
    with (d / "RecGym.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Subject", "Position", "Session", "A_x", "A_y", "A_z", "G_x", "G_y", "G_z",
                     "C_1", "Workout"])  # fmt: skip
        wr.writerows(rows)


def build_all(root: Path = FIXTURE_ROOT) -> list[Path]:
    """Build every raw-format fixture; returns the files written."""
    rng = np.random.default_rng(SEED)
    build_mmfit(root, rng)
    build_recofit(root, rng)
    build_recgym(root, rng)
    return sorted(p for p in root.rglob("*") if p.is_file())
