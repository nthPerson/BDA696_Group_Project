"""``formcoach pose extract``: video → MediaPipe PoseLandmarker → ``pose.parquet`` (docs/02 §4.3).

MediaPipe and OpenCV are the ``vision`` extra (``uv sync --extra vision``) and are imported
lazily inside :func:`open_video` / :func:`make_landmarker`; :func:`extract_frames` is pure
Python over an injected landmarker so it is unit-tested without them. The output follows the
Pose schema with ``skeleton == "mediapipe33"``: 33 image landmarks (``lm_i_x/y/z/v``) and 33
world landmarks (``wl_i_x/y/z``, metres, hip-centred). Extraction is checkpointed: an existing
output with a matching ``.done`` marker is skipped.

Model files: ``pose_landmarker_{lite,full,heavy}.task`` under ``models/`` (gitignored);
``make setup`` does not download them yet — :func:`model_path` prints the URL to fetch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from formcoach.pose.store import write_pose

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_URLS = {
    v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
    for v in ("lite", "full", "heavy")
}
N_LANDMARKS = 33


@dataclass
class FrameResult:
    """One frame's landmarks: ``image`` (33, 4) x/y normalised, z, visibility; ``world`` (33, 3)
    metres. ``None`` arrays mean no person was detected."""

    frame: int
    t: float
    image: np.ndarray | None
    world: np.ndarray | None


def model_path(variant: str = "lite") -> Path:
    if variant not in MODEL_URLS:
        raise ValueError(
            f"unknown pose model variant {variant!r}; choose from {sorted(MODEL_URLS)}"
        )
    p = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it:\n"
            f"  uv run formcoach pose fetch-model --model {variant}\n"
            f"  (or) curl -L -o {p} {MODEL_URLS[variant]}\n"
            "(then re-run; the file is gitignored)"
        )
    return p


def fetch_model(
    variant: str = "lite",
    *,
    dest_dir: Path | None = None,
    get: Callable[..., object] | None = None,
) -> Path:
    """Download the MediaPipe PoseLandmarker ``variant`` model into ``dest_dir`` (default
    :data:`MODELS_DIR`), in Python with ``requests`` (a core dependency, not the ``vision``
    extra), so it works the same on Windows, macOS and Linux without ``curl``.

    Returns the existing path immediately, without downloading, if the model file is already
    there. Streams the response to a ``<file>.partial`` file next to the destination and renames
    it into place only once the download finishes, so an interrupted or failed download never
    leaves a corrupt model file behind (the ``.partial`` is removed and the error re-raised).

    Args:
        variant: one of :data:`MODEL_URLS` (``"lite"``, ``"full"``, ``"heavy"``).
        dest_dir: directory to download into; defaults to :data:`MODELS_DIR`.
        get: injected in place of ``requests.get`` for tests, so no network call happens; must
            accept ``(url, stream=True, timeout=...)`` and return an object with
            ``raise_for_status()`` and ``iter_content(chunk_size=...)``, like a
            ``requests.Response``. The real call times out after 10 s to connect or 60 s without
            data, so a stalled connection fails instead of hanging ``make setup-live``.

    Raises:
        ValueError: ``variant`` is not a known model name.
    """
    if variant not in MODEL_URLS:
        raise ValueError(
            f"unknown pose model variant {variant!r}; choose from {sorted(MODEL_URLS)}"
        )
    dest_dir = Path(dest_dir) if dest_dir is not None else MODELS_DIR
    dest = dest_dir / f"pose_landmarker_{variant}.task"
    if dest.exists():
        return dest

    get = get or requests.get
    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        response = get(MODEL_URLS[variant], stream=True, timeout=(10, 60))
        response.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        partial.rename(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return dest


def make_landmarker(variant: str = "lite", path: Path | None = None):
    """A MediaPipe ``PoseLandmarker`` in VIDEO mode (needs ``uv sync --extra vision``)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        raise ImportError(
            f"pose extraction needs MediaPipe: `uv sync --extra vision` (import failed: {exc})"
        ) from exc
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(path or model_path(variant))),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(frame_bgr: np.ndarray, t_ms: int):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr[:, :, ::-1].copy())
        res = landmarker.detect_for_video(image, t_ms)
        if not res.pose_landmarks:
            return None, None
        lm = res.pose_landmarks[0]
        wl = res.pose_world_landmarks[0]
        img = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
        world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
        return img, world

    return detect


def open_video(path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t_seconds, frame_bgr)`` with OpenCV (``vision`` extra)."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            f"video reading needs OpenCV: `uv sync --extra vision` (cv2 import failed: {exc})"
        ) from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, i / fps, frame
            i += 1
    finally:
        cap.release()


def extract_frames(
    frames: Iterable[tuple[int, float, np.ndarray]],
    detect: Callable[[np.ndarray, int], tuple[np.ndarray | None, np.ndarray | None]],
) -> list[FrameResult]:
    """Run ``detect(frame_bgr, t_ms)`` on every frame; frames with no detection get ``None``."""
    out = []
    for idx, t, frame in frames:
        img, world = detect(frame, round(t * 1000))
        out.append(FrameResult(idx, float(t), img, world))
    return out


def results_to_parquet(results: list[FrameResult], out: Path, session: str) -> Path:
    n = len(results)
    image = np.full((n, N_LANDMARKS, 4), np.nan, dtype=np.float32)
    world = np.full((n, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for k, r in enumerate(results):
        if r.image is not None and r.world is not None:
            image[k], world[k], valid[k] = r.image, r.world, True
    return write_pose(
        out,
        session=session,
        frames=np.array([r.frame for r in results], dtype=np.int64),
        t=np.array([r.t for r in results], dtype=np.float64),
        world=world,
        skeleton="mediapipe33",
        image=image,
        valid=valid,
    )


def extract_video(
    video: Path, out: Path | None = None, variant: str = "lite", force: bool = False
) -> Path:
    """Video → ``pose.parquet`` next to it (or ``out``); skipped when ``.done`` exists."""
    video = Path(video)
    out = out or video.with_name(video.stem + "_pose.parquet")
    marker = out.with_suffix(out.suffix + ".done")
    if marker.exists() and out.exists() and not force:
        return out
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")
    detect = make_landmarker(variant)
    results = extract_frames(open_video(video), detect)
    results_to_parquet(results, out, session=video.stem)
    marker.write_text(f"{variant}\n", encoding="utf-8")
    return out
