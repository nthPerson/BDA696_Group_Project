"""Live HUD: window state and drawing (docs/superpowers 2026-11-02 live-demo design §3, §4.4).

Over the mirrored live video: per-arm rep counters top-left, a board/gate status pill
top-right, an elbow-angle label beside each tracked elbow, an angle trace with its rep
enter/exit band in the lower third, a coaching banner after every rep, and ``fps`` in a
bottom corner.

Everything except :func:`render` and :class:`WindowShow` is pure Python with no OpenCV
dependency, so this module imports without the ``vision`` extra; ``cv2`` is imported lazily
inside those two only. OpenCV's Hershey fonts are ASCII-only, so every string drawn on the
frame is passed through an ASCII-safe replacement (``deg``, ``-``, never ``°``/``·``).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from formcoach.app.overlay import EDGES
from formcoach.pose.skeletons import MEDIAPIPE33, Skeleton

BANNER_S = 3.0  # how long a post-rep banner stays on screen (s)
MIN_DRAW_VIS = 0.5  # landmark visibility floor for drawing a skeleton point/edge

_SEVERITY_RANK = {"warn": 1, "info": 0}
_BANNER_COLOR = {"warn": (0, 165, 255), "info": (180, 120, 60), "good": (0, 200, 0)}  # BGR
_SIDE_ELBOW_JOINT = {"l": "elbow_l", "r": "elbow_r"}
_TRACE_LINE_COLOR = {"l": (255, 160, 0), "r": (0, 220, 255)}
# Distinct enter/exit band ranges, cycled by draw order; both arms share the same thresholds
# in practice, so a plain neutral colour (not per-side) keeps the common case from looking busy.
_TRACE_BAND_PALETTE = ((90, 90, 60), (70, 60, 110), (60, 110, 70))

_ASCII_REPLACEMENTS = {
    "°": "deg",  # degree sign
    "·": "-",  # middle dot
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


def _ascii_safe(text: str) -> str:
    """Replace characters OpenCV's Hershey fonts cannot draw with ASCII equivalents; anything
    else non-ASCII becomes ``?`` so the result always satisfies ``str.isascii()``."""
    for bad, good in _ASCII_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "replace").decode("ascii")


def counter_text(exercise: str, totals: dict[str, dict]) -> str:
    """One line of rep counters: per-arm (``"CURL  L 7 (6 good)  R 7 (5 good)"``) when ``totals``
    has ``"l"``/``"r"`` keys, or a single counter (``"SQUAT  7 (6 good)"``) when it has ``"all"``
    (see :meth:`formcoach.app.pipeline.Pipeline.arm_totals`)."""
    if "all" in totals:
        t = totals["all"]
        body = f"{t['reps']} ({t['good']} good)"
    else:
        parts = []
        for side, label in (("l", "L"), ("r", "R")):
            t = totals.get(side, {"reps": 0, "good": 0})
            parts.append(f"{label} {t['reps']} ({t['good']} good)")
        body = "  ".join(parts)
    return f"{exercise.upper()}  {body}"


def pill(board_configured: bool, board_state: str, gate_open: bool, name: str) -> tuple[str, bool]:
    """Status-pill text and whether the video should be dimmed (~35%), from whether a board
    was asked for, its liveness state (``"waiting"``/``"ok"``/``"silent"``, see
    :class:`formcoach.app.board.BoardLink`) and the current gate flag."""
    if not board_configured:
        return "webcam only | gate always on", False
    if board_state == "waiting":
        return "looking for board... | pose paused", True
    if board_state == "silent":
        return "board silent - reconnecting", True
    if gate_open:
        return f"{name} | gate ON | pose running", False
    return "gate OFF | pose paused", True


def angle_label(deg: float | None) -> str:
    """``"38 deg"`` for a finite angle (degrees), ``"--"`` for ``None`` or ``NaN``."""
    if deg is None:
        return "--"
    try:
        if math.isnan(deg):
            return "--"
    except TypeError:
        return "--"
    return f"{deg:.0f} deg"


@dataclass
class Banner:
    """A post-rep coaching banner; ``kind`` is ``"good"``, ``"warn"`` or ``"info"``."""

    text: str
    kind: str
    expires_at: float

    @classmethod
    def from_rep(cls, rep, now: float) -> Banner:
        """Build the banner for a just-completed rep: ``"L rep 7: <message>"`` (no arm prefix
        when ``rep.arm`` is ``None``, e.g. squat), naming the most severe fault (``warn`` beats
        ``info``; the first one wins a tie) plus ``" (+N more)"`` for the rest, or ``"Good
        rep"`` in ``kind="good"`` when there are no faults. ``now`` + :data:`BANNER_S` is the
        expiry time (s, same clock as the caller)."""
        arm = getattr(rep, "arm", None)
        prefix = f"{arm.upper()} " if arm else ""
        header = f"{prefix}rep {rep.rep_id + 1}: "
        faults = rep.faults
        if not faults:
            return cls(_ascii_safe(header + "Good rep"), "good", now + BANNER_S)
        chosen = max(faults, key=lambda f: _SEVERITY_RANK.get(f.get("severity", "info"), 0))
        extra = len(faults) - 1
        suffix = f" (+{extra} more)" if extra else ""
        text = header + chosen.get("message", "") + suffix
        kind = chosen.get("severity", "info")
        return cls(_ascii_safe(text), kind, now + BANNER_S)


def wrap_lines(text: str, max_width_px: int, measure: Callable[[str], int]) -> list[str]:
    """Greedy word-wrap: add words to the current line while ``measure(line) <= max_width_px``;
    a single word wider than the limit is kept whole, alone on its own line."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and measure(candidate) > max_width_px:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


@dataclass
class HudState:
    """Everything one HUD frame needs to draw, already computed by the live loop."""

    exercise: str
    totals: dict
    angles: dict[str, float | None]
    pill_text: str
    dim: bool
    banner: Banner | None
    fps: float
    trace: dict[str, list[tuple[float, float]]]
    bands: dict[str, tuple[float, float]]


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            f"the live HUD needs OpenCV: `uv sync --extra vision` (cv2 import failed: {exc})"
        ) from exc
    return cv2


def _put_text(cv2, frame, text, pos, font_scale, color, thickness):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
        thickness + 2, cv2.LINE_AA,
    )  # fmt: skip
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )


def _draw_skeleton(cv2, frame, lm, skeleton: Skeleton, scale: float) -> None:
    if lm is None:
        return
    h, w = frame.shape[:2]
    thickness = max(round(3 * scale), 1)
    for a, b in EDGES:
        try:
            ia, ib = skeleton.index(a), skeleton.index(b)
        except KeyError:
            continue
        if lm[ia, 3] < MIN_DRAW_VIS or lm[ib, 3] < MIN_DRAW_VIS:
            continue
        pa = (int(lm[ia, 0] * w), int(lm[ia, 1] * h))
        pb = (int(lm[ib, 0] * w), int(lm[ib, 1] * h))
        cv2.line(frame, pa, pb, (0, 200, 255), thickness)
    radius = max(round(4 * scale), 2)
    for i in range(lm.shape[0]):
        if lm[i, 3] >= MIN_DRAW_VIS:
            cv2.circle(frame, (int(lm[i, 0] * w), int(lm[i, 1] * h)), radius, (255, 255, 255), -1)


def _draw_counters(cv2, frame, state: HudState, scale: float) -> None:
    _put_text(
        cv2, frame, counter_text(state.exercise, state.totals), (int(12 * scale), int(30 * scale)),
        0.72 * scale, (255, 255, 255), max(round(2 * scale), 1),
    )  # fmt: skip


def _draw_pill(cv2, frame, state: HudState, scale: float) -> None:
    w = frame.shape[1]
    font_scale = 0.58 * scale
    thickness = max(round(2 * scale), 1)
    (tw, _th), _ = cv2.getTextSize(state.pill_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(w - tw - int(12 * scale), int(4 * scale))
    y = int(54 * scale)
    _put_text(cv2, frame, state.pill_text, (x, y), font_scale, (255, 255, 255), thickness)


def _draw_angle_labels(cv2, frame, lm, skeleton: Skeleton, state: HudState, scale: float) -> None:
    h, w = frame.shape[:2]
    thickness = max(round(2 * scale), 1)
    fallback_y = int(78 * scale)
    for i, side in enumerate(sorted(state.angles)):
        label = f"{side.upper()} {angle_label(state.angles[side])}"
        pos = None
        joint_name = _SIDE_ELBOW_JOINT.get(side)
        if lm is not None and joint_name is not None and skeleton.has(joint_name):
            idx = skeleton.index(joint_name)
            if lm[idx, 3] >= MIN_DRAW_VIS:
                # offset up and to the side of the elbow so the label clears both the upper-arm
                # and forearm skeleton lines instead of sitting on top of either.
                x = int(np.clip(lm[idx, 0] * w + 16 * scale, 0, w - 1))
                y = int(np.clip(lm[idx, 1] * h - 14 * scale, 14 * scale, h - 1))
                pos = (x, y)
        if pos is None:
            pos = (int(12 * scale), fallback_y + i * int(26 * scale))
        _put_text(cv2, frame, label, pos, 0.55 * scale, (0, 255, 255), thickness)


def _draw_trace(cv2, frame, state: HudState, scale: float) -> None:
    if not state.trace:
        return
    h, w = frame.shape[:2]
    pad = int(20 * scale)
    plot_h = int(h * 0.20)
    y1 = h - int(h * 0.16)
    y0 = y1 - plot_h
    x0, x1 = pad, w - pad
    if x1 <= x0 or y1 <= y0:
        return

    values: list[float] = []
    for pts in state.trace.values():
        values.extend(v for _, v in pts if np.isfinite(v))
    for lo, hi in state.bands.values():
        values.extend([lo, hi])
    if not values:
        return
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    margin = 0.1 * (vmax - vmin)
    vmin, vmax = vmin - margin, vmax + margin

    def ymap(v: float) -> int:
        return int(y1 - (v - vmin) / (vmax - vmin) * (y1 - y0))

    panel = frame.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), (35, 35, 35), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, dst=frame)

    # Arms sharing the same thresholds (the common case) would otherwise stack two semi
    # transparent tints into a muddy colour, so draw each distinct (lo, hi) range once.
    drawn_ranges: dict[tuple[float, float], int] = {}
    for enter, exit_ in state.bands.values():
        lo, hi = sorted((enter, exit_))
        key = (round(lo, 3), round(hi, 3))
        if key not in drawn_ranges:
            drawn_ranges[key] = len(drawn_ranges)
    for (lo, hi), i in drawn_ranges.items():
        color = _TRACE_BAND_PALETTE[i % len(_TRACE_BAND_PALETTE)]
        ya, yb = sorted((ymap(lo), ymap(hi)))
        band = frame[ya:yb, x0:x1]
        if band.size:
            tint = np.full_like(band, color)
            cv2.addWeighted(tint, 0.35, band, 0.65, 0, dst=band)

    for label, pts in state.trace.items():
        if len(pts) < 2:
            continue
        t_vals = [t for t, _ in pts]
        t0, t1 = min(t_vals), max(t_vals)
        span = max(t1 - t0, 1e-6)
        color = _TRACE_LINE_COLOR.get(label, (255, 255, 255))
        thickness = max(round(2 * scale), 1)
        prev = None
        for t, v in pts:
            if not np.isfinite(v):
                prev = None
                continue
            x = int(x0 + (t - t0) / span * (x1 - x0))
            y = ymap(v)
            if prev is not None:
                cv2.line(frame, prev, (x, y), color, thickness)
            prev = (x, y)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 1)


def _draw_banner(cv2, frame, state: HudState, scale: float) -> None:
    if state.banner is None:
        return
    h, w = frame.shape[:2]
    font_scale = 0.62 * scale
    thickness = max(round(2 * scale), 1)
    max_width = max(w - int(24 * scale), 1)

    def measure(s: str) -> int:
        (tw, _th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return tw

    lines = wrap_lines(state.banner.text, max_width, measure)
    color = _BANNER_COLOR.get(state.banner.kind, (255, 255, 255))
    line_h = int(26 * scale)
    bottom_margin = int(34 * scale)
    y = h - bottom_margin - (len(lines) - 1) * line_h
    for line in lines:
        _put_text(cv2, frame, line, (int(12 * scale), y), font_scale, color, thickness)
        y += line_h


def _draw_fps(cv2, frame, state: HudState, scale: float) -> None:
    h, w = frame.shape[:2]
    text = f"fps {state.fps:.0f}"
    font_scale = 0.5 * scale
    thickness = max(round(1 * scale), 1)
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pos = (max(w - tw - int(10 * scale), 0), h - int(10 * scale))
    _put_text(cv2, frame, text, pos, font_scale, (200, 200, 200), thickness)


def render(
    frame_bgr: np.ndarray,
    image_lm: np.ndarray | None,
    state: HudState,
    *,
    skeleton: Skeleton = MEDIAPIPE33,
) -> np.ndarray:
    """Draw one HUD frame over ``frame_bgr`` (``(H, W, 3)`` uint8 BGR) and return a new mirrored
    frame; ``frame_bgr`` is never modified. ``image_lm`` is ``(J, 4)`` normalised x, y, z,
    visibility image-plane landmarks (or ``None`` when no person was detected)."""
    cv2 = _cv2()
    h = frame_bgr.shape[0]
    frame = np.ascontiguousarray(frame_bgr[:, ::-1, :])
    frame = (frame.astype(np.float32) * 0.35).astype(np.uint8) if state.dim else frame.copy()

    lm = None
    if image_lm is not None:
        lm = np.array(image_lm, dtype=np.float64, copy=True)
        lm[:, 0] = 1.0 - lm[:, 0]  # mirror x to match the flipped frame

    scale = max(h, 1) / 480.0

    _draw_skeleton(cv2, frame, lm, skeleton, scale)
    _draw_counters(cv2, frame, state, scale)
    _draw_pill(cv2, frame, state, scale)
    _draw_angle_labels(cv2, frame, lm, skeleton, state, scale)
    _draw_trace(cv2, frame, state, scale)
    _draw_banner(cv2, frame, state, scale)
    _draw_fps(cv2, frame, state, scale)
    return frame


class WindowShow:
    """Opens (lazily, on first call) an OpenCV window and shows frames in it."""

    def __init__(self, title: str = "FormCoach"):
        self._cv2 = _cv2()
        self.title = title

    def __call__(self, frame_bgr: np.ndarray) -> bool:
        """Show one frame; ``False`` once the user presses ``q``/Esc or closes the window."""
        cv2 = self._cv2
        try:
            cv2.imshow(self.title, frame_bgr)
        except cv2.error as exc:
            raise RuntimeError(
                f"cannot show the window ({exc}); no display? run with --headless"
            ) from exc
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return False
        return visible >= 1

    def close(self) -> None:
        with contextlib.suppress(self._cv2.error):  # pragma: no cover - window already gone
            self._cv2.destroyWindow(self.title)
