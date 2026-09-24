"""pose extract plumbing without MediaPipe: a fake detector drives extract_frames."""

from __future__ import annotations

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


def test_extract_video_missing_video_or_extra(tmp_path):
    with pytest.raises((FileNotFoundError, ImportError)):
        landmarker.extract_video(tmp_path / "nope.mp4")


def test_pose_extract_cli_fails_cleanly_without_video():
    from typer.testing import CliRunner

    from formcoach.cli import app

    result = CliRunner().invoke(app, ["pose", "extract", "--video", "does-not-exist.mp4"])
    assert result.exit_code == 2
    assert "not found" in result.output or "extra vision" in result.output
