"""Live webcam frames for the live demo (docs/superpowers/specs/2026-11-02-live-demo-design.md
§4.7). OpenCV is the ``vision`` extra (``uv sync --extra vision``); it is imported lazily inside
:func:`_open_real_capture`, which only runs when no ``capture_factory`` is injected, so this
module imports cleanly without it (as CI does).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

# cv2.CAP_PROP_FRAME_WIDTH / FRAME_HEIGHT. Hardcoded (stable OpenCV videoio property ids) so a
# fake ``capture_factory`` never has to pull in cv2 just to receive these.
_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4


def _open_real_capture(index: int) -> Any:
    import cv2

    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def open_camera(
    index: int,
    *,
    width: int = 1280,
    height: int = 720,
    capture_factory: Callable[[int], Any] | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Open webcam ``index`` and return a generator of ``(frame_index, seconds_since_open,
    frame_bgr)``.

    Opens and validates the camera immediately: an unopenable device, or one that opens but
    never delivers a frame, raises here, at call time, before any generator is returned, so a
    caller does not have to start iterating to find out the camera failed. On macOS
    (``sys.platform == "darwin"``) the AVFoundation backend is used; elsewhere, the default. The
    requested ``width``/``height`` (pixels) are a request to the driver, not a guarantee.

    ``capture_factory(index)`` is the test seam: pass a fake object exposing ``isOpened()``,
    ``set()``, ``read()`` and ``release()`` instead of opening a real camera. Production code
    should leave it ``None``.

    Raises:
        FileNotFoundError: the camera could not be opened (on macOS, the message includes the
            Privacy & Security hint).
        RuntimeError: the camera opened but the first read failed, or (once iteration starts
            consuming the returned generator) a later read fails, meaning the camera stopped
            sending frames.
    """
    factory = capture_factory if capture_factory is not None else _open_real_capture
    cap = factory(index)
    if not cap.isOpened():
        cap.release()
        hint = ""
        if sys.platform == "darwin":
            hint = (
                " On macOS: System Settings > Privacy & Security > Camera, allow your"
                " terminal app, then run again."
            )
        raise FileNotFoundError(f"cannot open camera {index}: no device responded.{hint}")

    cap.set(_CAP_PROP_FRAME_WIDTH, width)
    cap.set(_CAP_PROP_FRAME_HEIGHT, height)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            f"camera {index} opened but gave no frames (unplugged, or in use by another app?)"
        )

    return _camera_frames(cap, index, first_frame)


def _camera_frames(
    cap: Any, index: int, first_frame: np.ndarray
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frames starting with the already-read ``first_frame``; always release ``cap``."""
    t0 = time.monotonic()
    try:
        yield 0, 0.0, first_frame
        i = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera {index} stopped sending frames")
            yield i, time.monotonic() - t0, frame
            i += 1
    finally:
        cap.release()
