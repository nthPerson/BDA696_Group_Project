"""``open_camera`` against a fake capture (docs/02); no real camera is ever opened here."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.pose import camera


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


class FakeCap:
    """Stands in for ``cv2.VideoCapture``: a fixed list of frames, then ``(False, None)``."""

    def __init__(self, frames, *, opened: bool = True):
        self._frames = list(frames)
        self._opened = opened
        self._i = 0
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self):  # noqa: N802 (matches cv2.VideoCapture's method name)
        return self._opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))

    def read(self):
        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            return True, frame
        return False, None

    def release(self):
        self.released = True


def test_open_camera_yields_indexed_timed_frames():
    cap = FakeCap([_frame(1), _frame(2), _frame(3)])
    gen = camera.open_camera(0, capture_factory=lambda index: cap)

    frames = [next(gen) for _ in range(3)]

    assert [f[0] for f in frames] == [0, 1, 2]
    times = [f[1] for f in frames]
    assert times == sorted(times)
    assert all(isinstance(f[2], np.ndarray) for f in frames)
    assert frames[0][2] is not frames[1][2]  # distinct frame objects, not aliased
    assert (3, 1280) in cap.set_calls  # cv2.CAP_PROP_FRAME_WIDTH
    assert (4, 720) in cap.set_calls  # cv2.CAP_PROP_FRAME_HEIGHT
    assert not cap.released  # not released while frames remain

    with pytest.raises(RuntimeError, match="stopped sending frames"):
        next(gen)
    assert cap.released


def test_open_camera_requests_a_custom_size():
    cap = FakeCap([_frame(1)])
    gen = camera.open_camera(0, width=640, height=480, capture_factory=lambda index: cap)
    next(gen)
    assert (3, 640) in cap.set_calls
    assert (4, 480) in cap.set_calls


def test_unopenable_camera_raises_at_call_time_with_macos_hint(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    cap = FakeCap([], opened=False)

    with pytest.raises(FileNotFoundError, match="Privacy & Security") as excinfo:
        camera.open_camera(3, capture_factory=lambda index: cap)

    assert "cannot open camera 3" in str(excinfo.value)
    assert cap.released


def test_unopenable_camera_has_no_macos_hint_elsewhere(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "linux")
    cap = FakeCap([], opened=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        camera.open_camera(0, capture_factory=lambda index: cap)

    assert "Privacy & Security" not in str(excinfo.value)


def test_camera_that_never_delivers_raises():
    cap = FakeCap([])

    with pytest.raises(RuntimeError, match="gave no frames"):
        camera.open_camera(0, capture_factory=lambda index: cap)

    assert cap.released


def test_camera_dying_mid_stream_raises_runtime_error():
    cap = FakeCap([_frame(1), _frame(2)])
    gen = camera.open_camera(0, capture_factory=lambda index: cap)

    first = next(gen)
    second = next(gen)
    assert [first[0], second[0]] == [0, 1]

    with pytest.raises(RuntimeError, match=r"camera 0 stopped sending frames"):
        next(gen)
    assert cap.released


def test_open_camera_raises_before_returning_a_generator_object():
    """Validation happens at call time: the function itself raises, not a later next()."""
    cap = FakeCap([], opened=False)
    with pytest.raises(FileNotFoundError):
        result = camera.open_camera(0, capture_factory=lambda index: cap)
        # If open_camera returned a lazy generator instead of raising immediately, `result`
        # would be assigned and no error would appear until iteration started.
        next(result)
