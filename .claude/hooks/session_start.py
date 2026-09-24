"""Claude Code SessionStart hook: put the current FormCoach state into context.

CLAUDE.md says every session starts by reading the top of docs/STATUS.md and working on a
branch. This prints the branch (with a warning when it is ``main``), the count of uncommitted
changes and the newest STATUS entry, so the first turn does not have to spend tool calls on it.
Plain text on stdout is added to the model's context by Claude Code. Stdlib only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_STATUS_LINES = 45


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _status_head(path: Path) -> list[str]:
    """Header lines up to the first rule plus the first dated ``## `` entry, truncated."""
    if not path.exists():
        return [f"(no {path} found)"]
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen_entry = False
    for line in lines:
        if line.startswith("## "):
            if seen_entry:
                break
            seen_entry = True
        elif line.startswith("**Current") or line.startswith("**Hardware"):
            out.append(line)
            continue
        elif not seen_entry:
            continue
        out.append(line)
    if len(out) > MAX_STATUS_LINES:
        out = [*out[:MAX_STATUS_LINES], "... (truncated; read docs/STATUS.md for the rest)"]
    return out


def main() -> int:
    # Windows consoles default to cp1252; STATUS.md and our banner contain UTF-8 punctuation.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(_git("rev-parse", "--show-toplevel") or ".")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "(not a git repo)"
    dirty = _git("status", "--porcelain")
    n_dirty = len(dirty.splitlines()) if dirty else 0
    print("=== FormCoach session start (.claude/hooks/session_start.py) ===")
    print(f"branch: {branch} · uncommitted changes: {n_dirty}")
    if branch == "main":
        print(
            "WARNING: you are on `main`. CLAUDE.md rule 2: create a feat/ fix/ data/ docs/ fw/ "
            "chore/ branch before changing anything."
        )
    print("--- docs/STATUS.md (top entry) ---")
    print("\n".join(_status_head(root / "docs" / "STATUS.md")))
    print("--- end STATUS; next: docs/05-roadmap.md §2 or docs/00-START-HERE.md checkpoint ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
