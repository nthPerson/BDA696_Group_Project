"""`data convert` (raw -> IMUStream Parquet) and `data profile` (-> reports/data_profile.md)."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from formcoach.data import convert, profile, schema
from formcoach.data.fixtures import FIXTURE_ROOT

ROOTS = {
    "mmfit": FIXTURE_ROOT / "mmfit" / "mm-fit",
    "recofit": FIXTURE_ROOT / "recofit",
    "recgym": FIXTURE_ROOT / "recgym",
}


def test_convert_writes_one_parquet_per_stream_and_skips_existing(tmp_path):
    out = tmp_path / "processed"
    written = convert.convert_dataset("mmfit", ROOTS["mmfit"], out)
    names = sorted(p.name for p in written)
    assert names == ["P2-w00-sw_l.parquet", "P2-w00-sw_r.parquet"]
    df = pd.read_parquet(written[0])
    schema.validate_imu_stream(df)
    assert pq.read_schema(written[0]).metadata is not None
    again = convert.convert_dataset("mmfit", ROOTS["mmfit"], out)
    assert again == []  # nothing rewritten
    forced = convert.convert_dataset("mmfit", ROOTS["mmfit"], out, force=True)
    assert len(forced) == 2


def test_convert_all_three_datasets_and_list_streams(tmp_path):
    out = tmp_path / "processed"
    for name, root in ROOTS.items():
        convert.convert_dataset(name, root, out)
    streams = convert.list_streams(out)
    assert {s.dataset for s in streams} == {"mmfit", "recofit", "recgym"}
    assert {s.subject for s in streams if s.dataset == "recofit"} == {"R526", "R527"}
    rg = [s for s in streams if s.dataset == "recgym"]
    assert len(rg) == 4 and all(s.path.exists() for s in rg)
    df = pd.read_parquet(rg[0].path)
    assert (df["units"] == "normalized").all()


def test_profile_writes_report_and_figures(tmp_path):
    out_md = tmp_path / "reports" / "data_profile.md"
    profile.write_profile(ROOTS, out_md)
    text = out_md.read_text(encoding="utf-8")
    assert text.startswith("# Data profile")
    for section in ("## MM-Fit", "## RecoFit", "## RecGym", "## Minutes per canonical label"):
        assert section in text, section
    assert "| curl |" in text
    assert "seed" in text.lower() and "formcoach data profile" in text
    figs = sorted((tmp_path / "reports" / "figures").glob("data_profile_*.png"))
    assert len(figs) >= 2
