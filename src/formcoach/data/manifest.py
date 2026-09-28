"""Read and update the Markdown tables in ``data/MANIFEST.md``.

The manifest is a hand-readable file, so this module edits it in place: it finds the table
under a ``## <section>`` heading, parses the header row, and upserts one row keyed by one or
more columns. Everything outside that table is preserved byte for byte. Placeholder rows whose
key column is ``_pending_`` for the same dataset are replaced by the first real row.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

PENDING = "_pending_"


def _escape(cell: str) -> str:
    return str(cell).replace("|", "\\|")


def _unescape(cell: str) -> str:
    return cell.replace("\\|", "|").strip()


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells = re.split(r"(?<!\\)\|", body)
    return [_unescape(c) for c in cells]


def _locate_table(lines: list[str], section: str) -> tuple[int, int, int]:
    """Return (header_line, first_data_line, end_line_exclusive) of the table under ``section``."""
    heading = re.compile(rf"^##\s+{re.escape(section)}\s*$")
    start = next((i for i, ln in enumerate(lines) if heading.match(ln)), None)
    if start is None:
        raise KeyError(f"section '## {section}' not found")
    header = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("|")), None
    )
    if header is None or header + 1 >= len(lines) or not lines[header + 1].lstrip().startswith("|"):
        raise KeyError(f"no table under '## {section}'")
    end = header + 2
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    return header, header + 2, end


def read_rows(path: Path, section: str) -> list[dict[str, str]]:
    """Rows of the table under ``## section`` as dicts keyed by header cell."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header, first, end = _locate_table(lines, section)
    cols = _split_row(lines[header])
    rows = []
    for ln in lines[first:end]:
        cells = _split_row(ln)
        cells += [""] * (len(cols) - len(cells))
        rows.append(dict(zip(cols, cells, strict=False)))
    return rows


def upsert_row(path: Path, section: str, key: Sequence[str], row: dict[str, str]) -> None:
    """Insert or replace one row (matched on the ``key`` columns) in the table under ``section``.

    A placeholder row with the same first key value and ``_pending_`` in the second key column
    (or in the first key column when only one key is given) is replaced. Missing cells are
    written blank. The rest of the file is untouched.
    """
    text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    header, first, end = _locate_table(lines, section)
    cols = _split_row(lines[header])
    unknown = set(row) - set(cols)
    if unknown:
        raise KeyError(f"columns {unknown} not in table header {cols}")
    new_line = "| " + " | ".join(_escape(row.get(c, "")) for c in cols) + " |"
    key = list(key)
    target = None
    for i in range(first, end):
        cells = dict(zip(cols, _split_row(lines[i]), strict=False))
        if all(cells.get(k, "") == row.get(k, "") for k in key):
            target = i
            break
        is_placeholder = (
            cells.get(key[0], "") == row.get(key[0], "") and cells.get(key[-1], "") == PENDING
            if len(key) > 1
            else cells.get(key[0], "") == PENDING
        )
        if is_placeholder and target is None:
            target = i
    if target is None:
        lines.insert(end, new_line)
    else:
        lines[target] = new_line
    out = newline.join(lines)
    if text.endswith(("\n", "\r\n")):
        out += newline
    path.write_text(out, encoding="utf-8")
