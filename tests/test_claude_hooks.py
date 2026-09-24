"""The Claude Code hooks under .claude/hooks/ enforce CLAUDE.md rules 4 and 7 mechanically.

They are plain stdlib Python so they run through ``uv run --no-sync python`` on every OS
without the project's dependencies installed. These tests drive them the way Claude Code does:
JSON on stdin, JSON (or plain text) on stdout.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / ".claude" / "hooks" / "data_guard.py"
SESSION_START = ROOT / ".claude" / "hooks" / "session_start.py"


def run_guard(command: str, tool_name: str = "Bash") -> dict:
    payload = json.dumps({"tool_name": tool_name, "tool_input": {"command": command}})
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        input=payload,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def decision(command: str) -> str | None:
    out = run_guard(command)
    return out.get("hookSpecificOutput", {}).get("permissionDecision")


@pytest.mark.parametrize(
    "command",
    [
        "git add data/team/S1/0001/imu.parquet",
        "git add -f data/external/mmfit/w00.zip",
        "git add --force ./data/processed/recofit/",
        "git commit -m 'oops' data/team/S2/video.mp4",
        "git add recording.mp4",
        "cd docs && git add ../data/team/S3",
        "git status && git add data/team/*",
        "rm -rf data/team/S1",
        "rm data/MANIFEST.md",
        "sudo rm -r data/external/recofit",
        "find data/processed -name '*.parquet' -delete",
        "git clean -fdx",
        "git clean -X -f",
        "rm data/fixtures/mmfit/w00_small.npy",  # fixtures may be committed, never deleted
    ],
)
def test_blocks_data_hygiene_violations(command):
    assert decision(command) == "deny", command


@pytest.mark.parametrize(
    "command",
    [
        "git add data/fixtures/mmfit/w00_small.npy",
        "git add data/MANIFEST.md src/formcoach/data/mmfit.py",
        "git add -A && git commit -m 'feat: loader'",
        "git commit -m 'docs: status'",
        "rm -rf .pytest_cache firmware/.pio",
        "rm /tmp/scratch/data.txt",
        "git clean -fd",
        "ls data/team",
        "uv run formcoach data fetch --dataset mmfit",
        "cat data/external/mmfit/w00/w00_labels.csv | head",
        "echo 'never rm -rf data/team' > notes.txt",
    ],
)
def test_allows_ordinary_commands(command):
    assert decision(command) is None, command


def test_reason_names_the_rule_and_robert():
    out = run_guard("git add data/team/S1")
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "CLAUDE.md" in reason
    assert "Robert" in reason


def test_ignores_other_tools_and_garbage_input():
    assert run_guard("git add data/team/S1", tool_name="Read") == {}
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        input="not json",
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_session_start_prints_status_and_branch():
    proc = subprocess.run(
        [sys.executable, str(SESSION_START)],
        input="{}",
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Current phase" in out
    assert "branch:" in out
    assert "## 20" in out  # the newest dated STATUS entry heading
    assert out.count("\n") < 80  # keep the injected context small
