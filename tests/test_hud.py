"""HUD state and text logic (docs/superpowers Task 6): counters, pill, angle labels, the
post-rep banner and word-wrapping are pure Python; ``render`` needs OpenCV and is skipped
(``pytest.importorskip``) when it is not installed."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from formcoach.app import hud
from formcoach.app.pipeline import RepMetrics


def _rep(rep_id: int, faults: list[dict], arm: str | None) -> RepMetrics:
    """A minimal RepMetrics for banner tests. ``RepMetrics`` may or may not yet declare an
    ``arm`` field (another task is adding it); either way it is a plain mutable dataclass, so
    ``setattr`` after construction works, and ``hud.py`` reads it with ``getattr(rep, "arm",
    None)``."""
    rep = RepMetrics(
        exercise="curl",
        rep_id=rep_id,
        t_start=0.0,
        t_end=1.0,
        duration_s=1.0,
        source="pose",
        angles={},
        metrics={},
        faults=faults,
    )
    rep.arm = arm
    return rep


def test_counter_text_per_arm_and_squat():
    totals = {"l": {"reps": 7, "good": 6}, "r": {"reps": 7, "good": 5}}
    assert hud.counter_text("curl", totals) == "CURL  L 7 (6 good)  R 7 (5 good)"

    squat_totals = {"all": {"reps": 7, "good": 6}}
    assert hud.counter_text("squat", squat_totals) == "SQUAT  7 (6 good)"

    zero = {"l": {"reps": 0, "good": 0}, "r": {"reps": 0, "good": 0}}
    assert hud.counter_text("curl", zero) == "CURL  L 0 (0 good)  R 0 (0 good)"


def test_pill_texts_and_dim():
    assert hud.pill(False, "ok", True, "FormCoach-8428") == ("webcam only | gate always on", False)
    assert hud.pill(True, "waiting", False, "FormCoach-8428") == (
        "looking for board... | pose paused",
        True,
    )
    assert hud.pill(True, "silent", False, "FormCoach-8428") == (
        "board silent - reconnecting",
        True,
    )
    assert hud.pill(True, "ok", True, "FormCoach-8428") == (
        "FormCoach-8428 | gate ON | pose running",
        False,
    )
    assert hud.pill(True, "ok", False, "FormCoach-8428") == ("gate OFF | pose paused", True)


def test_angle_label_handles_nan():
    assert hud.angle_label(float("nan")) == "--"
    assert hud.angle_label(None) == "--"
    assert hud.angle_label(38.4) == "38 deg"


def test_banner_names_the_arm_and_picks_the_most_severe_fault():
    now = 100.0
    rep = _rep(
        6,
        [
            {"code": "CURL_TOO_FAST", "severity": "info", "message": "slow down a little"},
            {"code": "CURL_PARTIAL_ROM", "severity": "warn", "message": "extend further"},
        ],
        "l",
    )
    banner = hud.Banner.from_rep(rep, now)
    assert banner.text.startswith("L rep 7: ")
    assert "extend further" in banner.text
    assert "(+1 more)" in banner.text
    assert banner.kind == "warn"
    assert banner.expires_at == now + hud.BANNER_S

    clean = _rep(2, [], "r")
    good = hud.Banner.from_rep(clean, now)
    assert good.text == "R rep 3: Good rep"
    assert good.kind == "good"


def test_banner_text_is_ascii():
    rep = _rep(
        0,
        [{"code": "X", "severity": "warn", "message": "elbow drift 12° · fix"}],
        "l",
    )
    banner = hud.Banner.from_rep(rep, 0.0)
    assert banner.text.isascii()


def test_wrap_lines_never_exceeds_width():
    def measure(s: str) -> int:
        return 10 * len(s)

    text = "the quick brown fox jumps over the lazy dog today"
    lines = hud.wrap_lines(text, 120, measure)
    assert all(measure(line) <= 120 for line in lines)
    assert " ".join(" ".join(lines).split()) == " ".join(text.split())

    long_word = "supercalifragilisticexpialidocious"
    solo = hud.wrap_lines(long_word, 50, measure)
    assert solo == [long_word]


def test_render_draws_on_the_frame():
    pytest.importorskip("cv2")
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    original = frame.copy()
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[hud.MEDIAPIPE33.index("shoulder_l")] = (0.4, 0.3, 0.0, 1.0)
    lm[hud.MEDIAPIPE33.index("shoulder_r")] = (0.6, 0.3, 0.0, 1.0)
    lm[hud.MEDIAPIPE33.index("elbow_l")] = (0.35, 0.5, 0.0, 1.0)
    lm[hud.MEDIAPIPE33.index("elbow_r")] = (0.65, 0.5, 0.0, 1.0)

    state = hud.HudState(
        exercise="curl",
        totals={"l": {"reps": 7, "good": 6}, "r": {"reps": 7, "good": 5}},
        angles={"l": 38.4, "r": 42.0},
        pill_text="FormCoach-8428 | gate ON | pose running",
        dim=False,
        banner=hud.Banner(text="L rep 7: Good rep", kind="good", expires_at=3.0),
        fps=24.0,
        trace={"l": [(0.0, 150.0), (0.5, 90.0), (1.0, 55.0)]},
        bands={"l": (60.0, 150.0)},
    )
    out = hud.render(frame, lm, state)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, original)
    assert np.array_equal(frame, original)  # input untouched

    white = np.full((h, w, 3), 255, dtype=np.uint8)
    dim_state = dataclasses.replace(state, dim=True)
    dimmed = hud.render(white, None, dim_state)
    patch = dimmed[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5]
    assert patch.mean() < 0.5 * 255


def test_render_without_landmarks_or_angles_does_not_crash():
    pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = hud.HudState(
        exercise="curl",
        totals={"l": {"reps": 0, "good": 0}, "r": {"reps": 0, "good": 0}},
        angles={"l": None, "r": float("nan")},
        pill_text="webcam only | gate always on",
        dim=False,
        banner=None,
        fps=0.0,
        trace={},
        bands={},
    )
    out = hud.render(frame, None, state)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not math.isnan(float(out.mean()))
