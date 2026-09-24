"""Map each dataset's raw activity labels onto the canonical FormCoach vocabulary.

Canonical labels (``schema.CANONICAL_EXERCISES``): ``curl, press, raise, squat`` (the four
v1 exercises), ``other`` (any other exercise; still ``active`` for the gate) and ``idle``
(no exercise: rest, walking between stations, device on a table). ``None`` marks *junk*
labels (device taps, arm-band adjustment, notes, "Invalid") that windowing drops entirely
(:data:`RECOFIT_JUNK`); in the stream they are stored as canonical ``idle`` with the raw name
kept in ``label_raw``.

The RecoFit list below is the 75-entry ``exerciseConstants.activities`` of the released files
(2026-09-24). Which of them count as idle vs other is ADR-0015: walking is idle because the
camera must stay off between stations; static holds (plank, wall squat) are ``other`` because
the athlete is exercising.
"""

from __future__ import annotations

MMFIT_ACTIVITIES: tuple[str, ...] = (
    "squats",
    "lunges",
    "bicep_curls",
    "situps",
    "pushups",
    "tricep_extensions",
    "dumbbell_rows",
    "jumping_jacks",
    "dumbbell_shoulder_press",
    "lateral_shoulder_raises",
)
MMFIT_MAP: dict[str, str] = {
    "bicep_curls": "curl",
    "dumbbell_shoulder_press": "press",
    "lateral_shoulder_raises": "raise",
    "squats": "squat",
    "non_activity": "idle",
}

RECOFIT_ACTIVITIES: tuple[str, ...] = (
    "<Initial Activity>",
    "Arm Band Adjustment",
    "Arm straight up",
    "Band Pull-Down Row",
    "Bicep Curl",
    "Biceps Curl (band)",
    "Box Jump (on bench)",
    "Burpee",
    "Butterfly Sit-up",
    "Chest Press (rack)",
    "Crunch",
    "Device on Table",
    "Dip",
    "Dumbbell Deadlift Row",
    "Dumbbell Row (knee on bench) (label spans both arms)",
    "Dumbbell Row (knee on bench) (left arm)",
    "Dumbbell Row (knee on bench) (right arm)",
    "Dumbbell Squat (hands at side)",
    "Dynamic Stretch (at your own pace)",
    "Elliptical machine",
    "Fast Alternating Punches",
    "Invalid",
    "Jump Rope",
    "Jumping Jacks",
    "Kettlebell Swing",
    "Lateral Raise",
    "Lawnmower (label spans both arms)",
    "Lawnmower (left arm)",
    "Lawnmower (right arm)",
    "Lunge (alternating both legs, weight optional)",
    "Medicine Ball Slam",
    "Non-Exercise",
    "Note",
    "Overhead Triceps Extension",
    "Overhead Triceps Extension (label spans both arms)",
    "Plank",
    "Power Boat pose",
    "Pushup (knee or foot variation)",
    "Pushups",
    "Repetitive Stretching",
    "Rest",
    "Rowing machine",
    "Running (treadmill)",
    "Russian Twist",
    "Seated Back Fly",
    "Shoulder Press (dumbbell)",
    "Side Plank Left side",
    "Side Plank Right side",
    "Sit-up (hands positioned behind head)",
    "Sit-ups",
    "Squat",
    "Squat (arms in front of body, parallel to ground)",
    "Squat (hands behind head)",
    "Squat (kettlebell / goblet)",
    "Squat Jump",
    "Squat Rack Shoulder Press",
    "Static Stretch (at your own pace)",
    "Static stretch",
    "Tap IMU Device",
    "Tap Left Device",
    "Tap Right Device",
    "Triceps Kickback (knee on bench) (label spans both arms)",
    "Triceps Kickback (knee on bench) (left arm)",
    "Triceps Kickback (knee on bench) (right arm)",
    "Triceps extension (lying down)",
    "Triceps extension (lying down) (left arm)",
    "Triceps extension (lying down) (right arm)",
    "Two-arm Dumbbell Curl (both arms, not alternating)",
    "Unlisted Exercise",
    "V-up",
    "Walk",
    "Walking lunge",
    "Wall Ball",
    "Wall Squat",
    "Alternating Dumbbell Curl",
)
RECOFIT_CURL = {
    "Bicep Curl",
    "Two-arm Dumbbell Curl (both arms, not alternating)",
    "Alternating Dumbbell Curl",
    "Biceps Curl (band)",
}
RECOFIT_PRESS = {"Shoulder Press (dumbbell)", "Squat Rack Shoulder Press"}
RECOFIT_RAISE = {"Lateral Raise"}
RECOFIT_SQUAT = {
    "Squat",
    "Squat (arms in front of body, parallel to ground)",
    "Squat (hands behind head)",
    "Squat (kettlebell / goblet)",
    "Dumbbell Squat (hands at side)",
}
RECOFIT_IDLE = {"Non-Exercise", "Device on Table", "Rest", "Walk", "<Initial Activity>"}
RECOFIT_JUNK = {
    "Arm Band Adjustment",
    "Arm straight up",
    "Invalid",
    "Note",
    "Tap IMU Device",
    "Tap Left Device",
    "Tap Right Device",
    "Unlisted Exercise",
}

RECGYM_ACTIVITIES: tuple[str, ...] = (
    "Adductor",
    "ArmCurl",
    "BenchPress",
    "LegCurl",
    "LegPress",
    "Null",
    "Riding",
    "RopeSkipping",
    "Running",
    "Squat",
    "StairClimber",
    "Walking",
)
RECGYM_MAP = {"ArmCurl": "curl", "Squat": "squat", "Null": "idle"}


def canonical(dataset: str, raw: str) -> str | None:
    """Canonical label for a raw dataset label; ``None`` for junk. ``KeyError`` if unknown."""
    if dataset == "mmfit":
        if raw in MMFIT_MAP:
            return MMFIT_MAP[raw]
        if raw in MMFIT_ACTIVITIES:
            return "other"
    elif dataset == "recofit":
        if raw in RECOFIT_JUNK:
            return None
        if raw in RECOFIT_IDLE:
            return "idle"
        for label, group in (
            ("curl", RECOFIT_CURL),
            ("press", RECOFIT_PRESS),
            ("raise", RECOFIT_RAISE),
            ("squat", RECOFIT_SQUAT),
        ):
            if raw in group:
                return label
        if raw in RECOFIT_ACTIVITIES:
            return "other"
    elif dataset == "recgym":
        if raw in RECGYM_MAP:
            return RECGYM_MAP[raw]
        if raw in RECGYM_ACTIVITIES:
            return "other"
    else:
        raise KeyError(f"unknown dataset {dataset!r}")
    raise KeyError(f"unknown {dataset} label {raw!r}")
