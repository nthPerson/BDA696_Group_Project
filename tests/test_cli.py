"""Smoke tests for the `formcoach` CLI: help works, stubs run and say what they will do."""

from typer.testing import CliRunner

from formcoach import __version__
from formcoach.cli import app

runner = CliRunner()


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
    commands = [
        ["train", "gate"],
        ["eval", "all"],
        ["eval", "rules"],
        ["eval", "gating", "--gate", "energy"],
    ]
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
