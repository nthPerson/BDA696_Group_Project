"""HUD state and text logic (docs/superpowers Task 6): counters, pill, angle labels, the
post-rep banner and word-wrapping are pure Python; ``render`` needs OpenCV and is skipped
(``pytest.importorskip``) when it is not installed."""

from __future__ import annotations

import builtins
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
        banners={"l": hud.Banner(text="L rep 7: Good rep", kind="good", expires_at=3.0)},
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


def test_render_stacks_one_banner_per_counter():
    """Both arms' banners are drawn, each in its own colour, so a fault on one arm is not hidden
    by the other arm's "Good rep" (the old single banner was replaced by whichever came last)."""
    pytest.importorskip("cv2")
    h, w = 480, 640
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    def state(banners):
        return hud.HudState(
            exercise="curl",
            totals={"l": {"reps": 3, "good": 2}, "r": {"reps": 3, "good": 3}},
            angles={},
            pill_text="webcam only | gate always on",
            dim=False,
            banners=banners,
            fps=0.0,
            trace={},
            bands={},
        )

    left = hud.Banner("L rep 3: Slow down", "warn", 3.0)
    right = hud.Banner("R rep 3: Good rep", "good", 3.0)
    only_right = hud.render(blank, None, state({"r": right}))
    both = hud.render(blank, None, state({"l": left, "r": right}))
    # the bottom-most line is where a single banner's line always sat, so the right arm's text
    # is unchanged and the left arm's warn text appears in the rows above it
    band = slice(h - 34 - 26 - 20, h - 34 - 26 + 6)  # one line (26 px) above the bottom line
    assert not np.array_equal(both[band], only_right[band])
    orange = np.all(both == hud._BANNER_COLOR["warn"], axis=2)
    assert orange[band].any() and not orange[h - 34 - 20 :].any()


def test_render_without_landmarks_or_angles_does_not_crash():
    pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = hud.HudState(
        exercise="curl",
        totals={"l": {"reps": 0, "good": 0}, "r": {"reps": 0, "good": 0}},
        angles={"l": None, "r": float("nan")},
        pill_text="webcam only | gate always on",
        dim=False,
        banners={},
        fps=0.0,
        trace={},
        bands={},
    )
    out = hud.render(frame, None, state)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not math.isnan(float(out.mean()))


def test_window_show_turns_a_display_error_into_a_friendly_runtime_error():
    class _CvError(Exception):
        pass

    class _FakeCv2:
        error = _CvError

        @staticmethod
        def imshow(_title, _frame):
            raise _CvError("no display")

    window = hud.WindowShow.__new__(hud.WindowShow)  # skip the real cv2 import
    window._cv2 = _FakeCv2()
    window.title = "FormCoach"
    with pytest.raises(RuntimeError, match="--headless"):
        window(np.zeros((4, 4, 3), np.uint8))


def test_cv2_import_error_keeps_the_real_cause(monkeypatch):
    """On WSL2 ``import cv2`` can fail with a libGL error even after ``uv sync --extra
    vision``; the install hint alone then sends the user in circles. The raised message must
    keep both the hint and the original error."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError(
                "libGL.so.1: cannot open shared object file: No such file or directory"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as exc_info:
        hud._cv2()
    msg = str(exc_info.value)
    assert "uv sync --extra vision" in msg
    assert "libGL.so.1: cannot open shared object file" in msg
