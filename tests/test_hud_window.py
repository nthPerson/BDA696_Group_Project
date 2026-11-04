"""``WindowShow`` (src/formcoach/app/hud.py): the keypress/close-window contract the live loop
relies on (``run_live`` in src/formcoach/app/live.py breaks when ``show(out) is False``).

These tests never import real OpenCV: a small fake ``cv2``-shaped object is injected the same
way tests/test_hud.py's own
``test_window_show_turns_a_display_error_into_a_friendly_runtime_error`` does it (construct with
``WindowShow.__new__`` and set ``._cv2``/``.title`` directly, skipping ``__init__``'s real
``import cv2``), so nothing here opens a window even when the ``vision`` extra is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.app import hud

FRAME = np.zeros((4, 4, 3), np.uint8)


class _CvError(Exception):
    """Stands in for ``cv2.error``."""


class _FakeCv2:
    """A minimal fake of the handful of ``cv2`` names ``WindowShow`` touches.

    ``key_sequence`` is popped one value per ``waitKey`` call (``-1`` once exhausted, matching
    real OpenCV's "no key within the wait" return). ``visible_sequence`` is popped one value per
    ``getWindowProperty`` call (``1.0`` once exhausted, i.e. "still visible").
    """

    error = _CvError
    WND_PROP_VISIBLE = 4  # arbitrary id; WindowShow only ever compares the returned value

    def __init__(self, key_sequence=(), visible_sequence=(), destroy_raises=False):
        self.key_sequence = list(key_sequence)
        self.visible_sequence = list(visible_sequence)
        self.destroy_raises = destroy_raises
        self.imshow_calls: list[tuple[str, object]] = []
        self.get_window_property_calls: list[tuple[str, object]] = []
        self.destroy_calls = 0

    def imshow(self, title, frame):
        self.imshow_calls.append((title, frame))

    def waitKey(self, _delay):  # noqa: N802 (matches cv2.waitKey's method name)
        return self.key_sequence.pop(0) if self.key_sequence else -1

    def getWindowProperty(self, title, prop):  # noqa: N802 (matches cv2.getWindowProperty)
        self.get_window_property_calls.append((title, prop))
        return self.visible_sequence.pop(0) if self.visible_sequence else 1.0

    def destroyWindow(self, _title):  # noqa: N802 (matches cv2.destroyWindow's method name)
        self.destroy_calls += 1
        if self.destroy_raises:
            raise self.error("window already gone")


def _window(fake: _FakeCv2) -> hud.WindowShow:
    """Build a ``WindowShow`` around ``fake`` without running the real ``__init__`` (no ``import
    cv2``), the same trick tests/test_hud.py uses."""
    window = hud.WindowShow.__new__(hud.WindowShow)
    window._cv2 = fake
    window.title = "FormCoach"
    return window


# ---- a normal frame ----------------------------------------------------------------------------


def test_a_normal_frame_with_no_keypress_returns_true():
    fake = _FakeCv2()
    window = _window(fake)
    assert window(FRAME) is True
    assert len(fake.imshow_calls) == 1
    title, shown_frame = fake.imshow_calls[0]
    assert title == "FormCoach" and shown_frame is FRAME
    assert fake.get_window_property_calls  # visibility was checked once no key was seen


# ---- quitting: q, Esc, or the mouse ---------------------------------------------------------


def test_pressing_q_returns_false():
    fake = _FakeCv2(key_sequence=[ord("q")])
    window = _window(fake)
    assert window(FRAME) is False
    # a quit key short-circuits before any visibility check
    assert fake.get_window_property_calls == []


def test_pressing_esc_returns_false():
    fake = _FakeCv2(key_sequence=[27])
    window = _window(fake)
    assert window(FRAME) is False
    assert fake.get_window_property_calls == []


def test_an_unrelated_key_does_not_quit():
    fake = _FakeCv2(key_sequence=[ord("a")])
    window = _window(fake)
    assert window(FRAME) is True


def test_closing_the_window_with_the_mouse_returns_false():
    # no quit key was pressed, but the window property now reports "not visible"
    fake = _FakeCv2(visible_sequence=[0.0])
    window = _window(fake)
    assert window(FRAME) is False


def test_a_window_property_error_after_closing_counts_as_not_visible():
    """Real OpenCV can raise ``cv2.error`` from ``getWindowProperty`` once the window is fully
    destroyed; that must also read as "stop", not propagate."""

    class _ErroringCv2(_FakeCv2):
        def getWindowProperty(self, title, prop):  # noqa: N802 (matches cv2.getWindowProperty)
            raise self.error("window was destroyed")

    window = _window(_ErroringCv2())
    assert window(FRAME) is False


# ---- a display error is translated, not raised raw -------------------------------------------


def test_an_imshow_failure_is_a_friendly_runtime_error():
    class _NoDisplayCv2(_FakeCv2):
        def imshow(self, _title, _frame):
            raise self.error("no display")

    window = _window(_NoDisplayCv2())
    with pytest.raises(RuntimeError, match="--headless"):
        window(FRAME)


# ---- close() -------------------------------------------------------------------------------


def test_close_destroys_the_window():
    fake = _FakeCv2()
    window = _window(fake)
    window(FRAME)
    window.close()
    assert fake.destroy_calls == 1


def test_close_is_safe_to_call_twice():
    fake = _FakeCv2()
    window = _window(fake)
    window(FRAME)
    window.close()
    window.close()  # must not raise
    assert fake.destroy_calls == 2


def test_close_is_safe_when_no_window_was_ever_shown():
    # a real cv2.destroyWindow on a title that was never shown raises cv2.error; WindowShow must
    # swallow that, not propagate it.
    fake = _FakeCv2(destroy_raises=True)
    window = _window(fake)
    window.close()  # must not raise even though destroyWindow raises every time
    window.close()  # and again
    assert fake.destroy_calls == 2
