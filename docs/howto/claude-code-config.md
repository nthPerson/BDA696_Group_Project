# How to: the team's Claude Code configuration (hooks that enforce CLAUDE.md)

**Owner:** Bryce · **Built by:** Claude Code with Robert on 2026-09-24 · **PR:** #2
**Checkpoint:** 0 (`docs/00-START-HERE.md`) — repo tooling built alongside the Checkpoint 0 skeleton, not a listed deliverable
**State:** works (verified in this session; nothing partial)

## 1. Run it (verified 2026-09-24 on Linux/WSL2)

Run the hook test suite (also part of `make test`, since `testpaths = ["tests"]` in `pyproject.toml`):

```
uv run --no-sync pytest tests/test_claude_hooks.py -v
```
```
collected 27 items
tests/test_claude_hooks.py ...........................                   [100%]
============================== 27 passed in 0.40s ==============================
```

Feed the data-hygiene guard a blocked command, exactly as Claude Code invokes it on every `Bash` tool call:

```
echo '{"tool_name":"Bash","tool_input":{"command":"git add data/team/S1/0001/imu.parquet"}}' \
  | uv run --no-sync python .claude/hooks/data_guard.py
```
```
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Blocked by .claude/hooks/data_guard.py (CLAUDE.md working rules 4 and 7): `git add` names a raw/processed/team data file or a video. Raw datasets, processed data, team recordings and video are never committed, and data is never deleted without asking Robert first. If this is intended, ask Robert and run it yourself outside Claude Code."}}
```

Feed it an ordinary command — no output means "allow, fall through to normal permissions":

```
echo '{"tool_name":"Bash","tool_input":{"command":"uv run formcoach data fetch --dataset mmfit"}}' \
  | uv run --no-sync python .claude/hooks/data_guard.py
```
(prints nothing, exits 0)

Run the session-start hook the way Claude Code runs it at the top of every session:

```
echo '{}' | uv run --no-sync python .claude/hooks/session_start.py
```
```
=== FormCoach session start (.claude/hooks/session_start.py) ===
branch: chore/claude-code-config · uncommitted changes: 3
--- docs/STATUS.md (top entry) ---
**Current phase:** Phase 1 (Sep 23 – Oct 6) · **Current checkpoint:** 0 done → 1 next
...
--- end STATUS; next: docs/05-roadmap.md §2 or docs/00-START-HERE.md checkpoint ---
```

## 2. Where things live

| Path | What it is |
|---|---|
| `.claude/settings.json` | Registers two hooks: `SessionStart` → `session_start.py`; `PreToolUse` (matcher `Bash`) → `data_guard.py`. Both are invoked as `uv run --no-sync python "$CLAUDE_PROJECT_DIR/.claude/hooks/<name>.py"` (absolute, ADR-0010) so they keep working after a Bash call has changed directory. The commands in §1 use the relative form and must be run from the repo root |
| `.claude/hooks/data_guard.py` | `PreToolUse` hook. Reads the tool call JSON on stdin, calls `check(command: str) -> str \| None`, and if it returns a reason, writes a `permissionDecision: "deny"` JSON to stdout via `main()` |
| `.claude/hooks/session_start.py` | `SessionStart` hook. No JSON output — prints plain text (branch, dirty-file count, `docs/STATUS.md` top entry) via `main()`; Claude Code adds stdout to context |
| `tests/test_claude_hooks.py` | pytest suite; drives both hooks as subprocesses with JSON on stdin, exactly as Claude Code does |

## 3. How it works (10 lines max)

`data_guard.py` only ever **denies**, never allows explicitly — anything it doesn't recognize falls through to Claude Code's normal permission prompts. It tokenizes the `Bash` command with `shlex.split`, splits on `&& || ; | &` into segments (`_segments`), strips `sudo`/`env`/`nice`/`time` prefixes, and pattern-matches `git add|commit` naming a path under `data/{external,processed,team}` or a raw extension (`PROTECTED_DIRS`, `RAW_EXTENSIONS`, `VIDEO`), `git clean -x/-X`, `rm`/`rmdir`/`shred`/`trash` targeting anything under `data/`, and `find ... -delete` under `data/`. `data/fixtures/` is explicitly exempted (`FIXTURES`) since CLAUDE.md rule 4 allows those to be committed. This enforces CLAUDE.md rule 4 (data hygiene) and rule 7 (ask Robert before deleting data). `session_start.py` just runs `git rev-parse`/`git status`, warns if `branch == "main"` (rule 2: work on a branch), and prints the top dated entry of `docs/STATUS.md` (capped at `MAX_STATUS_LINES = 45`) so the first turn of a session doesn't need to spend a tool call reading it.

## 4. Tests

`uv run pytest tests/test_claude_hooks.py` (27 tests, included automatically in `make test`/`make lint test`):
- `test_blocks_data_hygiene_violations` — 13 commands that must be denied (data/team, data/external, data/processed, `*.mp4`, `git clean -fdx`/`-X -f`, `rm`, `find -delete`, including through `sudo`, `cd ... &&`, and compound `&&`/`;` commands).
- `test_allows_ordinary_commands` — 11 commands that must fall through (`data/fixtures/`, `data/MANIFEST.md`, ordinary `git commit`/`rm` outside `data/`, a `formcoach data fetch` invocation, `cat`, and a command that merely *mentions* `rm -rf data/team` in an echoed string).
- `test_reason_names_the_rule_and_robert` — the denial message cites `CLAUDE.md` and `Robert`.
- `test_ignores_other_tools_and_garbage_input` — non-`Bash` tool calls and invalid JSON produce no output.
- `test_session_start_prints_status_and_branch` — output contains `Current phase`, `branch:`, a dated `## 20` STATUS heading, and stays under 80 lines.

Update together: if `PROTECTED_DIRS`/`RAW_EXTENSIONS`/`VIDEO` patterns in `data_guard.py` change, add a case to both parametrized lists in `test_claude_hooks.py`. If CLAUDE.md rule 4 or 7 changes (new protected path, new exception), update the regexes and the tests in the same PR.

## 5. Could not verify / open questions

- No **(verify)** items from the design docs are touched by this config — it only encodes CLAUDE.md working rules 4 and 7, not third-party hardware/data formats.
- Not yet exercised: how Claude Code's real hook runner behaves on Windows (`uv run --no-sync python .claude/hooks/...`); only Linux/WSL2 was available this session. The scripts are stdlib-only and path-agnostic (`re` patterns use `[\\/]` for both slash styles), so this should work unchanged, but Bryce should confirm on his OS.
- Hooks were built on Linux/WSL2; CI runs the tests on Windows and macOS too (a cp1252
  console-encoding crash was found and fixed that way). A live Windows session is untested.

## 6. Next steps for Bryce

1. Open/merge the PR for `chore/claude-code-config` (CLAUDE.md rule 2: small reviewable increment, conventional commit `chore:`).
2. Confirm the hooks fire for you: start a fresh Claude Code session in this repo and check the session-start banner appears; try a `Bash` tool call that touches `data/team/` and confirm it's denied.
3. If you add a new protected data path or a new destructive command pattern later, extend `data_guard.py`'s regexes/`check()` and add matching cases to `tests/test_claude_hooks.py` in the same change — see §4.
4. Per `docs/05-roadmap.md`, docs/demo are your area — this config protects the data hygiene rules the rest of the team's Makefile targets (`make data`, `make record`) depend on, so keep it green in CI as new targets are added.
