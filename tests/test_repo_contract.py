"""Guards for the working rules in CLAUDE.md that are cheap to check mechanically."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MAKE_TARGETS = {
    "setup",
    "lint",
    "test",
    "data",
    "features",
    "train-gate",
    "eval",
    "demo",
    "record",
}


def test_makefile_has_every_workflow_target():
    targets = set(re.findall(r"^([a-z][a-z0-9-]*):", (ROOT / "Makefile").read_text(), re.M))
    missing = REQUIRED_MAKE_TARGETS - targets
    assert not missing, f"Makefile is missing targets: {sorted(missing)}"


def test_data_dirs_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in (
        "data/external/*",
        "data/processed/*",
        "data/team/*",
        "*.mp4",
        "*.npy",
        "*.mat",
        "*.parquet",
        "!data/fixtures/**",
    ):
        assert pattern in ignore, pattern


def test_expected_layout_exists():
    for rel in (
        "src/formcoach/cli.py",
        "src/formcoach/data",
        "src/formcoach/signal",
        "src/formcoach/pose",
        "src/formcoach/rules",
        "src/formcoach/models",
        "src/formcoach/io",
        "src/formcoach/app",
        "src/formcoach/eval",
        "firmware/platformio.ini",
        "firmware/src/main.cpp",
        "firmware/include/protocol.h",
        "data/MANIFEST.md",
        "docs/STATUS.md",
        "docs/DECISIONS.md",
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
        "uv.lock",
    ):
        assert (ROOT / rel).exists(), rel


def test_no_video_or_raw_data_is_tracked():
    """Belt-and-braces: nothing outside fixtures with a raw-data extension may exist in tree."""
    bad = [
        p
        for p in ROOT.rglob("*")
        if p.suffix in {".mp4", ".npy", ".mat", ".parquet"}
        and "data/fixtures" not in p.as_posix()
        and ".venv" not in p.parts
        and ".pio" not in p.parts
        and not p.as_posix().startswith((ROOT / "data").as_posix())
    ]
    assert not bad, bad
