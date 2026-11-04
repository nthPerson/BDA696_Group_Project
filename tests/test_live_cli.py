"""`formcoach demo --camera/--video/--board` and `pose fetch-model` (plan Task 9)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from formcoach.app import live
from formcoach.cli import app

runner = CliRunner()


class _Recorder:
    """Stands in for ``live.run_live_demo``; remembers its keyword arguments."""

    def __init__(self, raises: Exception | None = None):
        self.kwargs: dict = {}
        self.raises = raises

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises


@pytest.fixture
def fake_live(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(live, "run_live_demo", rec)
    return rec


def test_demo_help_lists_the_live_flags():
    out = runner.invoke(app, ["demo", "--help"]).output
    for flag in ("--camera", "--board", "--video", "--no-log", "--model"):
        assert flag in out, flag
    assert "--side" not in out


def test_replay_demo_is_unchanged():
    result = runner.invoke(app, ["demo", "--source", "replay", "--headless"])
    assert result.exit_code == 0
    assert "frames processed" in result.output and "STUB" not in result.output


def test_the_ble_stub_points_at_the_live_command():
    result = runner.invoke(app, ["demo", "--source", "ble"])
    assert result.exit_code == 0
    assert "--camera" in result.output and "--board" in result.output


def test_live_flags_are_passed_through(fake_live):
    result = runner.invoke(
        app,
        [
            "demo",
            "--camera",
            "1",
            "--headless",
            "--exercise",
            "press",
            "--board",
            "ble",
            "--no-log",
        ],
    )
    assert result.exit_code == 0, result.output
    kw = fake_live.kwargs
    assert kw["camera"] == 1 and kw["video"] is None
    assert kw["board"] == "ble" and kw["exercise"] == "press"
    assert kw["headless"] is True and kw["out_dir"] is None and kw["model"] == "lite"
    assert "wrote" not in result.output


def test_live_defaults_to_curl_and_no_board(fake_live):
    result = runner.invoke(app, ["demo", "--camera", "0", "--headless", "--no-log"])
    assert result.exit_code == 0, result.output
    assert fake_live.kwargs["exercise"] == "curl" and fake_live.kwargs["board"] is None


def test_default_log_dir_is_runs_live_timestamp(fake_live):
    result = runner.invoke(app, ["demo", "--camera", "0", "--headless"])
    assert result.exit_code == 0, result.output
    out_dir = fake_live.kwargs["out_dir"]
    assert isinstance(out_dir, Path) and out_dir.parent == Path("runs") / "live"
    assert re.fullmatch(r"\d{8}-\d{6}", out_dir.name)

    result = runner.invoke(app, ["demo", "--camera", "0", "--headless", "--out", "x"])
    assert result.exit_code == 0, result.output
    assert fake_live.kwargs["out_dir"] == Path("x")


def test_out_and_no_log_together_are_rejected(fake_live):
    result = runner.invoke(app, ["demo", "--camera", "0", "--out", "x", "--no-log"])
    assert result.exit_code == 2 and "--no-log" in result.output
    assert fake_live.kwargs == {}


def test_unknown_exercise_is_a_friendly_error():
    result = runner.invoke(app, ["demo", "--camera", "0", "--headless", "--exercise", "deadlift"])
    assert result.exit_code == 2
    assert "exercise" in result.output and "Traceback" not in result.output


def test_source_ble_with_camera_is_rejected_with_a_hint(fake_live):
    result = runner.invoke(app, ["demo", "--camera", "0", "--source", "ble"])
    assert result.exit_code == 2
    assert "--board" in result.output
    assert fake_live.kwargs == {}


def test_unopenable_camera_is_a_friendly_error(monkeypatch):
    rec = _Recorder(raises=FileNotFoundError("cannot open camera 9: no device responded."))
    monkeypatch.setattr(live, "run_live_demo", rec)
    result = runner.invoke(app, ["demo", "--camera", "9", "--headless", "--no-log"])
    assert result.exit_code == 2
    assert "cannot open camera 9" in result.output and "Traceback" not in result.output


def test_a_missing_vision_extra_is_a_friendly_error(monkeypatch):
    rec = _Recorder(raises=ImportError("the live HUD needs OpenCV: `uv sync --extra vision`"))
    monkeypatch.setattr(live, "run_live_demo", rec)
    result = runner.invoke(app, ["demo", "--camera", "0", "--no-log"])
    assert result.exit_code == 2 and "uv sync --extra vision" in result.output


def test_a_missing_video_is_a_friendly_error(tmp_path):
    result = runner.invoke(app, ["demo", "--video", str(tmp_path / "nope.mp4"), "--headless"])
    assert result.exit_code == 2
    assert "nope.mp4" in result.output and "Traceback" not in result.output


def test_fetch_model_reports_the_path(monkeypatch, tmp_path):
    from formcoach.pose import landmarker

    target = tmp_path / "pose_landmarker_lite.task"
    monkeypatch.setattr(landmarker, "fetch_model", lambda variant="lite": target)
    result = runner.invoke(app, ["pose", "fetch-model"])
    assert result.exit_code == 0 and "pose_landmarker_lite.task" in result.output


def test_fetch_model_rejects_an_unknown_variant():
    result = runner.invoke(app, ["pose", "fetch-model", "--model", "huge"])
    assert result.exit_code == 2 and "huge" in result.output


# ---- end to end on a video file --------------------------------------------------------------


def _blank_clip(path: Path, seconds: float = 3.0, fps: int = 30) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for _ in range(int(seconds * fps)):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    return path


def _assert_zeroed_session(out: Path) -> None:
    session = json.loads((out / "session.json").read_text())
    zero = {"reps": 0, "good": 0, "flagged": 0, "faults_by_code": {}}
    assert session["arms"] == {"l": zero, "r": zero}
    assert session["live"] is True and session["frames_total"] >= 80
    assert (out / "events.parquet").exists() and (out / "frames.parquet").exists()


def test_video_headless_smoke_with_injected_detector(tmp_path):
    clip = _blank_clip(tmp_path / "blank.mp4")
    out = tmp_path / "run"
    said: list[str] = []
    live.run_live_demo(
        camera=None, video=clip, board=None, port=None, exercise="curl", headless=True,
        out_dir=out, say=said.append, detect=lambda _f, _ms: (None, None),
    )  # fmt: skip
    _assert_zeroed_session(out)
    assert any(s.startswith("L: 0 reps") for s in said)


def test_video_headless_smoke_with_the_real_model(tmp_path):
    pytest.importorskip("mediapipe")
    from formcoach.pose import landmarker

    if not (landmarker.MODELS_DIR / "pose_landmarker_lite.task").exists():
        pytest.skip("models/pose_landmarker_lite.task not downloaded (make setup-live)")
    clip = _blank_clip(tmp_path / "blank.mp4")
    out = tmp_path / "run"
    live.run_live_demo(
        camera=None, video=clip, board=None, port=None, exercise="curl", headless=True,
        out_dir=out, say=lambda _s: None,
    )  # fmt: skip
    _assert_zeroed_session(out)


def test_cli_end_to_end_on_a_video(tmp_path):
    clip = _blank_clip(tmp_path / "blank.mp4")
    out = tmp_path / "run"
    result = runner.invoke(
        app, ["demo", "--video", str(clip), "--headless", "--out", str(out), "--exercise", "squat"]
    )
    if "pose_landmarker_lite.task" in result.output and result.exit_code == 2:
        pytest.skip("pose model not downloaded")
    assert result.exit_code == 0, result.output
    assert "all: 0 reps" in result.output and "wrote" in result.output
    assert json.loads((out / "session.json").read_text())["arms"]["all"]["reps"] == 0


def test_a_bad_board_setting_never_opens_the_camera(monkeypatch):
    """--board serial without --port fails before the camera is touched (an unstarted camera
    generator cannot release its device, so the camera is opened last)."""
    from formcoach.pose import camera as camera_mod

    opened: list[int] = []
    monkeypatch.setattr(camera_mod, "_open_real_capture", lambda i: opened.append(i))
    with pytest.raises(FileNotFoundError, match="--port"):
        live.run_live_demo(
            camera=0, video=None, board="serial", port=None, exercise="curl", headless=True,
            say=lambda _s: None, detect=lambda _f, _ms: (None, None),
        )  # fmt: skip
    assert opened == []


def test_an_unknown_model_is_a_friendly_error_before_anything_opens():
    result = runner.invoke(app, ["demo", "--camera", "0", "--model", "huge", "--no-log"])
    assert result.exit_code == 2
    assert "huge" in result.output and "Traceback" not in result.output


def test_live_only_flags_without_camera_or_video_are_rejected():
    for extra in (["--board", "ble"], ["--no-log"], ["--model", "full"]):
        result = runner.invoke(app, ["demo", "--headless", *extra])
        assert result.exit_code == 2, extra
        assert "--camera" in result.output and "--video" in result.output
