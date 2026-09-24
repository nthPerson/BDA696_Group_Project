"""Claude Code PreToolUse hook: block shell commands that would violate CLAUDE.md rules 4 and 7.

Rule 4 (data hygiene): raw data under data/external/, processed data under data/processed/,
team recordings under data/team/ and every video file are never committed. Rule 7: ask Robert
before deleting data.

The hook receives the tool call as JSON on stdin and answers with a JSON permission decision
on stdout. It only ever *denies*; anything it does not recognise is left to the normal
permission flow. Stdlib only, so it runs via ``uv run --no-sync python`` on a fresh clone on
Linux, macOS and Windows. ``tests/test_claude_hooks.py`` pins the behaviour.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

PROTECTED_DIRS = re.compile(r"(^|[\\/])data[\\/](external|processed|team)([\\/*]|$)")
ANY_DATA_DIR = re.compile(r"(^|[\\/])data([\\/*]|$)")
VIDEO = re.compile(r"\.mp4$", re.IGNORECASE)
RAW_EXTENSIONS = re.compile(r"\.(npy|mat|parquet|zip)$", re.IGNORECASE)
FIXTURES = re.compile(r"(^|[\\/])data[\\/]fixtures([\\/]|$)")
SEGMENT_BREAKS = {"&&", "||", ";", "|", "&"}

RULE_TEXT = (
    "Blocked by .claude/hooks/data_guard.py (CLAUDE.md working rules 4 and 7): {what}. "
    "Raw datasets, processed data, team recordings and video are never committed, and data is "
    "never deleted without asking Robert first. If this is intended, ask Robert and run it "
    "yourself outside Claude Code."
)


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:  # unbalanced quotes: fall back to whitespace split
        return command.split()


def _segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in SEGMENT_BREAKS:
            segments.append([])
        else:
            segments[-1].append(tok)
    return [s for s in segments if s]


def _is_protected_path(tok: str) -> bool:
    if FIXTURES.search(tok):
        return False
    return bool(PROTECTED_DIRS.search(tok) or VIDEO.search(tok) or RAW_EXTENSIONS.search(tok))


def _strip_prefix(seg: list[str]) -> list[str]:
    """Drop leading ``sudo``/``env``/``cd x &&``-style wrappers so ``git``/``rm`` is first."""
    while seg and seg[0] in {"sudo", "env", "nice", "time"}:
        seg = seg[1:]
    return seg


def check(command: str) -> str | None:
    """Return a reason to deny ``command``, or ``None`` if it is fine."""
    for seg in _segments(_tokens(command)):
        seg = _strip_prefix(seg)
        if not seg:
            continue
        head = seg[0]
        args = seg[1:]
        if head == "git" and args:
            sub = args[0]
            if sub in {"add", "commit"} and any(_is_protected_path(t) for t in args[1:]):
                return f"`git {sub}` names a raw/processed/team data file or a video"
            if sub == "clean" and any(
                t.startswith("-") and not t.startswith("--") and ("x" in t or "X" in t)
                for t in args[1:]
            ):
                return "`git clean -x/-X` would delete the gitignored data/ directories"
        if head in {"rm", "rmdir", "shred", "trash"} and any(
            ANY_DATA_DIR.search(t) or _is_protected_path(t) for t in args
        ):
            return f"`{head}` targets something under data/ or a raw data file"
        if head == "find" and "-delete" in args and any(ANY_DATA_DIR.search(t) for t in args):
            return "`find ... -delete` targets something under data/"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if not isinstance(payload, dict) or payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(command, str):
        return 0
    reason = check(command)
    if reason is None:
        return 0
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": RULE_TEXT.format(what=reason),
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
