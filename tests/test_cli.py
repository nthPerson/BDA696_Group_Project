"""Smoke tests for the `formcoach` CLI: help works, stubs run and say what they will do."""

import builtins

from typer.testing import CliRunner

from formcoach import __version__
from formcoach.cli import app

runner = CliRunner()


def _fake_import_raising(name_to_fail: str, message: str):
    """A ``builtins.__import__`` replacement that fails only ``import <name_to_fail>`` with a
    realistic ``ImportError`` message, so the real cause behind an install-hint ImportError can
    be checked without actually uninstalling anything."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == name_to_fail:
            raise ImportError(message)
        return real_import(name, *args, **kwargs)

    return fake_import


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("data", "features", "pose", "train", "eval", "session", "demo", "record"):
        assert group in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_demo_replay_headless_is_the_demo_of_record():
    result = runner.invoke(app, ["demo", "--source", "replay", "--headless"])
    assert result.exit_code == 0
    assert "replay" in result.output
    assert "STUB" not in result.output  # Checkpoint 3: the real pipeline runs on the fixture


def test_every_checkpoint_command_exists():
    commands = []
    for cmd in commands:
        result = runner.invoke(app, cmd)
        assert result.exit_code == 0, (cmd, result.output)
        assert "Will:" in result.output, cmd


def test_bad_dataset_is_rejected():
    result = runner.invoke(app, ["data", "fetch", "--dataset", "nope"])
    assert result.exit_code != 0


def test_data_fetch_verify_only_reports_missing_files(tmp_path):
    result = runner.invoke(
        app, ["data", "fetch", "--dataset", "recgym", "--root", str(tmp_path), "--verify-only"]
    )
    assert result.exit_code == 1
    assert "missing" in result.output
    assert not any(tmp_path.rglob("*.zip"))  # nothing downloaded


def test_eval_all_lists_its_steps_in_help():
    result = runner.invoke(app, ["eval", "all", "--help"])
    assert result.exit_code == 0 and "quick" in result.output


# ---- install-hint ImportErrors must keep the real cause (WSL2 libGL etc.) --------------------


def test_demo_replay_overlay_cv2_import_error_keeps_the_real_cause(monkeypatch):
    """``demo`` without ``--headless`` builds the OpenCV overlay window; on WSL2 ``import cv2``
    can fail with a libGL error even after ``uv sync --extra vision``."""
    monkeypatch.setattr(
        builtins,
        "__import__",
        _fake_import_raising(
            "cv2", "libGL.so.1: cannot open shared object file: No such file or directory"
        ),
    )
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 2
    assert "uv sync --extra vision" in result.output
    assert "libGL.so.1: cannot open shared object file" in result.output


def test_record_serial_import_error_keeps_the_real_cause(monkeypatch):
    monkeypatch.setattr(
        builtins,
        "__import__",
        _fake_import_raising("serial", "No module named 'serial.tools.list_ports_common'"),
    )
    result = runner.invoke(
        app, ["record", "--source", "serial", "--port", "COM99", "--duration", "1"]
    )
    assert result.exit_code == 2
    assert "uv sync --extra device" in result.output
    assert "No module named 'serial.tools.list_ports_common'" in result.output


def test_record_ble_import_error_keeps_the_real_cause(monkeypatch):
    monkeypatch.setattr(
        builtins,
        "__import__",
        _fake_import_raising(
            "bleak", "libdbus-1.so.3: cannot open shared object file: No such file or directory"
        ),
    )
    result = runner.invoke(app, ["record", "--source", "ble", "--duration", "1"])
    assert result.exit_code == 2
    assert "uv sync --extra device" in result.output
    assert "libdbus-1.so.3: cannot open shared object file" in result.output
