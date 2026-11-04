"""``run_live_demo``'s ``try``/``finally`` resource cleanup (src/formcoach/app/live.py): the
window (``show``), the board link (``link``) and the frame source (``frames``) must all be
closed on every exit path, however the run ends.

tests/test_live.py covers ``run_live`` itself (the frame loop) in depth, including its own
``finally`` (log/totals on every exit, including exceptions and Ctrl-C) - that is not repeated
here. tests/test_live_cli.py covers the CLI wiring and a couple of ``run_live_demo`` smoke runs
on a real video file, but never asserts that ``show``/``link``/``frames`` were actually closed.
This file fills exactly that gap, using fakes so nothing real (OpenCV window, BLE/serial board,
camera) is ever touched.

Note on the ``(d)`` exit path the task that asked for this file described ("opening the board
fails after the window"): reading ``run_live_demo``'s ``try`` block shows the real creation order
is the *opposite* - the board link is opened first, then the window, then the frame source last
(a comment there explains why: an unstarted camera generator cannot release its device when
closed, so nothing that can still fail is allowed to run after the camera is opened). The tests
below exercise the real order: a window-open failure after the board already succeeded, and a
frame-source-open failure after both the board and the window already succeeded.
"""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.app import board as board_module
from formcoach.app import hud, live
from formcoach.pose import camera as camera_mod


@pytest.fixture(autouse=True)
def _no_real_rendering(monkeypatch):
    """These tests are about cleanup, not drawing: never reach for OpenCV (CI's main jobs run
    without the vision extra, where ``hud.render`` would raise ImportError)."""
    monkeypatch.setattr(hud, "render", lambda frame, _lm, _state, **_kw: frame)


class _FakeFrames:
    """Stands in for the generator ``open_camera``/``open_video`` return: iterable, closable, and
    optionally raises (any ``BaseException``, including ``KeyboardInterrupt``) once exhausted."""

    def __init__(self, n: int = 5, fps: float = 30.0, raise_exc: BaseException | None = None):
        self.n = n
        self.fps = fps
        self.raise_exc = raise_exc
        self.closed = 0

    def __iter__(self):
        for i in range(self.n):
            yield i, i / self.fps, np.zeros((4, 4, 3), np.uint8)
        if self.raise_exc is not None:
            raise self.raise_exc

    def close(self) -> None:
        self.closed += 1


class _ExplodingShow:
    """Stands in for ``hud.WindowShow`` when the window itself fails to open."""

    def __init__(self, title: str = "FormCoach") -> None:
        del title
        raise RuntimeError("cannot show the window (no display); run with --headless")


def _tracking_show(bucket: list):
    """A fake ``hud.WindowShow`` class whose instances append themselves to ``bucket``."""

    class _Show:
        def __init__(self, title: str = "FormCoach") -> None:
            self.title = title
            self.closed = 0
            bucket.append(self)

        def __call__(self, _frame) -> bool:
            return True

        def close(self) -> None:
            self.closed += 1

    return _Show


def _tracking_link(bucket: list):
    """A fake ``formcoach.app.board.BoardLink`` class whose instances append themselves to
    ``bucket``; the gate is always open so ``detect`` is reached on every frame."""

    class _Link:
        name = "fake-board"

        def __init__(self, source) -> None:
            self.source = source
            self.closed = 0
            bucket.append(self)

        def poll(self):
            return []

        def gate_open(self) -> bool:
            return True

        def state(self) -> str:
            return "ok"

        def send(self, _cmd) -> None:
            pass

        def close(self) -> None:
            self.closed += 1

    return _Link


def _patch_board_open(monkeypatch) -> None:
    """``_open_board`` would otherwise import ``formcoach.io.ble``/``serial_source``; replace it
    with a trivial stand-in for the opened IMU source ``BoardLink`` wraps."""
    monkeypatch.setattr(live, "_open_board", lambda kind, port: object())


# ---- (a) the run ends normally ------------------------------------------------------------


def test_normal_exit_closes_window_board_and_frames(monkeypatch):
    shows: list = []
    links: list = []
    _patch_board_open(monkeypatch)
    monkeypatch.setattr(board_module, "BoardLink", _tracking_link(links))
    monkeypatch.setattr(hud, "WindowShow", _tracking_show(shows))
    frames = _FakeFrames(n=3)
    monkeypatch.setattr(camera_mod, "open_camera", lambda index, **kw: frames)

    live.run_live_demo(
        camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
        say=lambda _s: None, detect=lambda _f, _ms: (None, None),
    )  # fmt: skip

    assert len(shows) == 1 and shows[0].closed == 1
    assert len(links) == 1 and links[0].closed == 1
    assert frames.closed == 1


# ---- (b) the detector raises mid-run --------------------------------------------------------


def test_detector_exception_still_closes_everything(monkeypatch):
    shows: list = []
    links: list = []
    _patch_board_open(monkeypatch)
    monkeypatch.setattr(board_module, "BoardLink", _tracking_link(links))
    monkeypatch.setattr(hud, "WindowShow", _tracking_show(shows))
    frames = _FakeFrames(n=5)
    monkeypatch.setattr(camera_mod, "open_camera", lambda index, **kw: frames)

    def boom(_frame, _t_ms):
        raise ValueError("model exploded")

    with pytest.raises(ValueError, match="model exploded"):
        live.run_live_demo(
            camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
            say=lambda _s: None, detect=boom,
        )  # fmt: skip

    assert shows[0].closed == 1
    assert links[0].closed == 1
    assert frames.closed == 1


# ---- (c) Ctrl-C inside the loop ---------------------------------------------------------------


def test_keyboard_interrupt_in_the_loop_closes_everything_and_does_not_propagate(monkeypatch):
    """``run_live`` itself swallows ``KeyboardInterrupt`` raised inside the frame loop and
    returns a normal ``PipelineResult`` (see tests/test_live.py
    ``test_run_live_ctrl_c_ends_normally``); ``run_live_demo`` must still have closed window,
    board and frames once that return reaches its own ``finally``."""
    shows: list = []
    links: list = []
    _patch_board_open(monkeypatch)
    monkeypatch.setattr(board_module, "BoardLink", _tracking_link(links))
    monkeypatch.setattr(hud, "WindowShow", _tracking_show(shows))
    frames = _FakeFrames(n=5)
    monkeypatch.setattr(camera_mod, "open_camera", lambda index, **kw: frames)

    calls = {"n": 0}

    def interrupt_after_two(_frame, _t_ms):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return None, None

    result = live.run_live_demo(
        camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
        say=lambda _s: None, detect=interrupt_after_two,
    )  # fmt: skip

    assert result is not None
    assert shows[0].closed == 1
    assert links[0].closed == 1
    assert frames.closed == 1


# ---- opening something fails after an earlier resource already opened ------------------------


def test_window_open_failure_still_closes_the_already_opened_board(monkeypatch):
    links: list = []
    _patch_board_open(monkeypatch)
    monkeypatch.setattr(board_module, "BoardLink", _tracking_link(links))
    monkeypatch.setattr(hud, "WindowShow", _ExplodingShow)
    camera_opened: list[int] = []
    monkeypatch.setattr(camera_mod, "open_camera", lambda index, **kw: camera_opened.append(index))

    with pytest.raises(RuntimeError, match="no display"):
        live.run_live_demo(
            camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
            say=lambda _s: None, detect=lambda _f, _ms: (None, None),
        )  # fmt: skip

    assert links[0].closed == 1
    # the camera is opened last; a failed window must stop the setup before it is ever touched
    assert camera_opened == []


def test_frame_source_open_failure_still_closes_the_already_opened_board_and_window(monkeypatch):
    shows: list = []
    links: list = []
    _patch_board_open(monkeypatch)
    monkeypatch.setattr(board_module, "BoardLink", _tracking_link(links))
    monkeypatch.setattr(hud, "WindowShow", _tracking_show(shows))

    def boom_open_camera(index, **kw):
        raise FileNotFoundError(f"cannot open camera {index}: no device responded.")

    monkeypatch.setattr(camera_mod, "open_camera", boom_open_camera)

    with pytest.raises(FileNotFoundError, match="cannot open camera 0"):
        live.run_live_demo(
            camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
            say=lambda _s: None, detect=lambda _f, _ms: (None, None),
        )  # fmt: skip

    assert shows[0].closed == 1
    assert links[0].closed == 1


# ---- close order: the camera first, and one failing close does not skip the others ----------


def _order_setup(monkeypatch):
    """Fakes that record the order their ``close`` is called in; returns ``(order, frames)``."""
    order: list[str] = []
    links: list = []
    shows: list = []
    _patch_board_open(monkeypatch)
    link_cls = _tracking_link(links)
    link_close = link_cls.close
    link_cls.close = lambda self: (order.append("link"), link_close(self))[1]
    show_cls = _tracking_show(shows)
    show_close = show_cls.close
    show_cls.close = lambda self: (order.append("show"), show_close(self))[1]
    monkeypatch.setattr(board_module, "BoardLink", link_cls)
    monkeypatch.setattr(hud, "WindowShow", show_cls)
    frames = _FakeFrames(n=3)
    frames.close = lambda: order.append("frames")
    monkeypatch.setattr(camera_mod, "open_camera", lambda index, **kw: frames)
    return order, frames


def test_the_camera_is_released_before_the_wearable_link_shuts_down(monkeypatch):
    """Closing the BLE link can take seconds (it joins its thread): the camera light must not
    stay on for that long after the user presses q."""
    order, _frames = _order_setup(monkeypatch)
    live.run_live_demo(
        camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
        say=lambda _s: None, detect=lambda _f, _ms: (None, None),
    )  # fmt: skip
    assert order[0] == "frames"
    assert sorted(order) == ["frames", "link", "show"]


def test_a_failing_close_does_not_skip_the_other_resources(monkeypatch):
    order, frames = _order_setup(monkeypatch)

    def broken_close():
        order.append("frames")
        raise OSError("camera driver hiccup")

    frames.close = broken_close
    with pytest.raises(OSError, match="camera driver hiccup"):
        live.run_live_demo(
            camera=0, video=None, board="ble", port=None, exercise="curl", headless=False,
            say=lambda _s: None, detect=lambda _f, _ms: (None, None),
        )  # fmt: skip
    assert sorted(order) == ["frames", "link", "show"]
