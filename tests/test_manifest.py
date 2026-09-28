"""data/MANIFEST.md is machine-updated by `formcoach data fetch`; rows must be idempotent."""

from pathlib import Path

from formcoach.data import manifest

TEMPLATE = """# Data manifest

Intro text.

## External datasets

| Dataset | File | Source URL | SHA-256 | Size | License | Fetched | By | Notes |
|---|---|---|---|---|---|---|---|---|
| mmfit | _pending_ | https://mmfit.github.io/ | | | MIT | | | video separate |

## Fixtures (committed, small)

| File | Origin | Purpose | Size |
|---|---|---|---|
| _pending_ | | 30-second replay fixture | < 2 MB |

## Citations

- Some citation.
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "MANIFEST.md"
    p.write_text(TEMPLATE, encoding="utf-8")
    return p


def test_read_rows_parses_the_external_table(tmp_path):
    p = _write(tmp_path)
    rows = manifest.read_rows(p, "External datasets")
    assert rows[0]["Dataset"] == "mmfit"
    assert rows[0]["File"] == "_pending_"
    assert rows[0]["Notes"] == "video separate"


def test_upsert_replaces_pending_row_and_is_idempotent(tmp_path):
    p = _write(tmp_path)
    row = {
        "Dataset": "mmfit",
        "File": "mm-fit.zip",
        "Source URL": "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip",
        "SHA-256": "abc",
        "Size": "1,742,309,258",
        "License": "MIT",
        "Fetched": "2026-09-24",
        "By": "robert",
        "Notes": "sensor + pose",
    }
    manifest.upsert_row(p, "External datasets", key=("Dataset", "File"), row=row)
    manifest.upsert_row(p, "External datasets", key=("Dataset", "File"), row=row)
    rows = manifest.read_rows(p, "External datasets")
    assert [r["File"] for r in rows] == ["mm-fit.zip"]  # _pending_ placeholder replaced
    text = p.read_text(encoding="utf-8")
    assert text.count("| mm-fit.zip |") == 1  # the URL also contains the name
    assert "## Citations" in text and "Intro text." in text  # rest of the file untouched


def test_upsert_appends_new_dataset_rows(tmp_path):
    p = _write(tmp_path)
    manifest.upsert_row(
        p,
        "External datasets",
        key=("Dataset", "File"),
        row={"Dataset": "recgym", "File": "RecGym.csv", "SHA-256": "x"},
    )
    rows = manifest.read_rows(p, "External datasets")
    assert {r["Dataset"] for r in rows} == {"mmfit", "recgym"}
    assert rows[-1]["Size"] == ""  # missing cells are blank, not KeyError


def test_pipes_in_cells_are_escaped(tmp_path):
    p = _write(tmp_path)
    manifest.upsert_row(
        p,
        "External datasets",
        key=("Dataset", "File"),
        row={"Dataset": "x", "File": "a|b", "Notes": "c|d"},
    )
    rows = manifest.read_rows(p, "External datasets")
    assert rows[-1]["File"] == "a|b" and rows[-1]["Notes"] == "c|d"
