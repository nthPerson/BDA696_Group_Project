"""OpenCV overlay window (docs/02 §4.6 v0): skeleton, rep counter, exercise, last fault.

Needs the ``vision`` extra (``uv sync --extra vision``); everything else in ``formcoach.app``
runs headless without it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from formcoach.app.events import Event
from formcoach.io.replay import PoseFrame

EDGES = (
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "wrist_r"),
    ("shoulder_l", "shoulder_r"),
    ("hip_l", "hip_r"),
    ("shoulder_l", "hip_l"),
    ("shoulder_r", "hip_r"),
    ("hip_l", "knee_l"),
    ("knee_l", "ankle_l"),
    ("hip_r", "knee_r"),
    ("knee_r", "ankle_r"),
)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the overlay window needs OpenCV: `uv sync --extra vision` (or run with --headless)"
        ) from exc
    return cv2


class Overlay:
    def __init__(self, exercise: str, size: tuple[int, int] = (640, 480), title: str = "FormCoach"):
        self.cv2 = _cv2()
        self.exercise = exercise
        self.size = size
        self.title = title
        self.reps = 0
        self.last_fault = ""

    def on_event(self, e: Event) -> None:
        if e.kind == "rep":
            self.reps += 1
        elif e.kind == "fault":
            self.last_fault = e.payload.get("message", e.payload.get("code", ""))

    def draw(self, fr: PoseFrame) -> None:
        cv2 = self.cv2
        w, h = self.size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        pts = fr.world[:, :2].astype(np.float64)
        ok = np.isfinite(pts).all(axis=1)
        if ok.sum() > 2:
            p = pts[ok]
            lo, hi = p.min(axis=0), p.max(axis=0)
            scale = 0.8 * min(w / max(hi[0] - lo[0], 1e-6), h / max(hi[1] - lo[1], 1e-6))

            def to_px(q):
                x = int((q[0] - lo[0]) * scale + 0.1 * w)
                y = int(h - ((q[1] - lo[1]) * scale + 0.1 * h))
                return x, y

            for a, b in EDGES:
                try:
                    ia, ib = fr.skeleton.index(a), fr.skeleton.index(b)
                except KeyError:
                    continue
                if ok[ia] and ok[ib]:
                    cv2.line(img, to_px(pts[ia]), to_px(pts[ib]), (0, 200, 255), 2)
        cv2.putText(
            img,
            f"{self.exercise}  reps {self.reps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        if self.last_fault:
            cv2.putText(
                img, self.last_fault, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        cv2.imshow(self.title, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def wrap(self, frames: Iterable[PoseFrame]) -> Iterator[PoseFrame]:
        for fr in frames:
            self.draw(fr)
            yield fr
        self.cv2.destroyAllWindows()
