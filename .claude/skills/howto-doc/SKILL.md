---
name: howto-doc
description: Use when Claude has done (or is finishing) a task that belongs to a FormCoach teammate — Ruby (data), Rochelle (vision & rules), Christian (embedded ML & eval), Bryce (app, docs, demo) — and must hand it over via docs/howto/<topic>.md, or when a teammate asks for a "how do I run/take over X" document.
---

# Writing a `docs/howto/<topic>.md` hand-over

CLAUDE.md: when Claude does a teammate's task it writes docstrings, a `docs/howto/<topic>.md`
and a STATUS entry naming what they should read. The reader has **less hardware/ML
experience than Robert, will not debug toolchains, and may be on Windows**. The document is
a set of commands they can run, not an essay.

## The document is exactly these parts, in this order

```markdown
# How to: <what it does, in plain words>

**Owner:** <teammate> · **Built by:** Claude Code with <who> on <YYYY-MM-DD> · **PR:** #<n>
**Checkpoint:** <n> (`docs/00-START-HERE.md`) · **State:** <works on fixture | works on real data | partial>

## 1. Run it (verified <YYYY-MM-DD> on <OS>)
<one fenced block per command, each followed by the first lines of its real output>

## 2. Where things live
| Path | What it is |
<code, tests, fixture, CLI command, Makefile target, report it writes>

## 3. How it works (10 lines max)
<the one design idea and the units/schema it must respect, with the doc section it comes from>

## 4. Tests
<`uv run pytest tests/test_<x>.py`; what each test pins; what to update together with what>

## 5. Could not verify / open questions
<every **(verify)** item touched and what was found or not; anything assumed>

## 6. Next steps for <owner>
<numbered, in roadmap order, each pointing at the STATUS/roadmap line it comes from>
```

Then add to the same PR:

- **STATUS entry** paragraph: "`<owner>`: read `docs/howto/<topic>.md` first, then `<file>`."
- **DECISIONS entry** if any alternative was chosen or any **(verify)** item differed from the docs.
- `docs/howto/` link from the README documentation table if it is the first howto.

## Rules that make the document trustworthy

- **Every command in §1 was run before the doc was written, in the same session, and the
  output shown is pasted from that run.** A command that was not run is not in the doc.
- **Every path in §2 exists** (`ls` it). Every function named exists with that signature
  (`grep -n "def name" <file>`). Signatures are copied from the code, never typed from memory.
- The stack is the one in CLAUDE.md: pandas, NumPy, SciPy, pyarrow, typer. Do not describe
  the code with another library's types.
- File name: `docs/howto/<topic>.md`, lowercase, hyphens, named after the *thing*
  (`mmfit-loader.md`, `pose-extraction.md`, `flashing-firmware.md`).
- Length: under 120 lines. Move anything longer into the code's docstrings and link to it.
- Windows: show the `uv run formcoach ...` form next to any `make` target; serial ports are `COMx`.
- Hardware topic: paste `docs/03-hardware.md` §4 (battery safety) verbatim under a
  "## Battery safety" heading before §1. No exceptions (CLAUDE.md rule 9).

## Common mistakes

| Mistake | Fix |
|---|---|
| Describing planned code as if it existed | §1 shows real output; if it cannot run yet, State = partial and §1 says what does run |
| Inventing API names or types | copy from `grep -n "def " <file>`; pandas not polars |
| Omitting the STATUS line that names the reader | the reader finds the doc through STATUS, not by browsing |
| Long narrative of what Claude did | that belongs in the STATUS entry; the howto is for running and continuing |
| No "next steps" | the owner's first question is "what do I do now?" |
