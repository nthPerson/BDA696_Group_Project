"""pose extract plumbing without MediaPipe: a fake detector drives extract_frames."""

from __future__ import annotations

import builtins
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from formcoach.pose import landmarker, store


def _fake_detect(frame, t_ms):
    if t_ms == 2000:  # one frame with no person
        return None, None
    img = np.zeros((33, 4), dtype=np.float32)
    img[:, 3] = 0.95
    world = np.zeros((33, 3), dtype=np.float32)
    world[11] = (-0.2, 0.5, 0)
    world[12] = (0.2, 0.5, 0)
    world[23] = (-0.1, 0, 0)
    world[24] = (0.1, 0, 0)
    return img, world


def test_extract_frames_to_parquet_round_trip(tmp_path):
    frames = [(i, i / 1.0, np.zeros((4, 4, 3), dtype=np.uint8)) for i in range(4)]
    results = landmarker.extract_frames(frames, _fake_detect)
    assert [r.image is None for r in results] == [False, False, True, False]
    out = landmarker.results_to_parquet(results, tmp_path / "pose.parquet", session="v")
    seq = store.read_pose(out)
    assert seq.skeleton.name == "mediapipe33" and seq.world.shape == (4, 33, 3)
    assert seq.valid.tolist() == [True, True, False, True]
    assert np.isnan(seq.world[2]).all() and seq.visibility[0, 0] == pytest.approx(0.95)


def test_missing_model_file_gives_download_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(landmarker, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="curl -L -o"):
        landmarker.model_path("lite")


def test_real_landmarker_starts_on_this_machine():
    """The real MediaPipe graph starts and runs one frame. mediapipe 1.0.1 aborted here on Apple
    Silicon (ADR-0027); the abort is a C++ crash, not an exception, hence the subprocess.
    Skipped without the ``vision`` extra or the Lite model (CI has neither)."""
    if importlib.util.find_spec("mediapipe") is None:
        pytest.skip("needs `uv sync --extra vision`")
    if not (landmarker.MODELS_DIR / "pose_landmarker_lite.task").exists():
        pytest.skip("needs models/pose_landmarker_lite.task")
    code = (
        "import numpy as np\n"
        "from formcoach.pose.landmarker import make_landmarker\n"
        "img, world = make_landmarker('lite')(np.zeros((480, 640, 3), np.uint8), 0)\n"
        "assert img is None and world is None\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_extract_video_missing_video_or_extra(tmp_path):
    with pytest.raises((FileNotFoundError, ImportError)):
        landmarker.extract_video(tmp_path / "nope.mp4")


def _fake_import_raising(name_to_fail: str, message: str):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == name_to_fail:
            raise ImportError(message)
        return real_import(name, *args, **kwargs)

    return fake_import


def test_make_landmarker_import_error_keeps_the_real_cause(monkeypatch):
    """A WSL2 ``libGL`` failure importing mediapipe must not be hidden behind the install hint
    alone (it already ran ``uv sync --extra vision``)."""
    monkeypatch.setattr(
        builtins,
        "__import__",
        _fake_import_raising(
            "mediapipe", "libGL.so.1: cannot open shared object file: No such file or directory"
        ),
    )
    with pytest.raises(ImportError) as exc_info:
        landmarker.make_landmarker("lite")
    msg = str(exc_info.value)
    assert "uv sync --extra vision" in msg
    assert "libGL.so.1: cannot open shared object file" in msg


def test_open_video_import_error_keeps_the_real_cause(tmp_path, monkeypatch):
    monkeypatch.setattr(
        builtins,
        "__import__",
        _fake_import_raising(
            "cv2", "libGL.so.1: cannot open shared object file: No such file or directory"
        ),
    )
    with pytest.raises(ImportError) as exc_info:
        next(landmarker.open_video(tmp_path / "nope.mp4"))
    msg = str(exc_info.value)
    assert "uv sync --extra vision" in msg
    assert "libGL.so.1: cannot open shared object file" in msg


def test_pose_extract_cli_fails_cleanly_without_video():
    from typer.testing import CliRunner

    from formcoach.cli import app

    result = CliRunner().invoke(app, ["pose", "extract", "--video", "does-not-exist.mp4"])
    assert result.exit_code == 2
    assert "not found" in result.output or "extra vision" in result.output
