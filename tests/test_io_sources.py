"""Serial / BLE decoding through fakes that emit protocol.py packets, the session recorder and
`session check`. No pyserial or bleak needed."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from formcoach.data import schema
from formcoach.io import ble, fake, protocol, recorder, serial_source
from formcoach.io.source import SISample


def _raw(i: int, t_ms: int | None = None, flags: int = 0) -> protocol.Sample:
    return protocol.Sample(t_ms=t_ms if t_ms is not None else 20 * i, ax=0, ay=0, az=4096,
                           gx=33, gy=0, gz=0, flags=flags, seq=i & 0xFFFF)  # fmt: skip


def test_serial_line_decoder_skips_junk_and_counts_drops():
    lines = [protocol.SERIAL_CSV_HEADER, "# FormCoach fw 1.0.0"]
    lines += [",".join(str(v) for v in (20 * i, 0, 0, 4096, 33, 0, 0, 0, i)) for i in range(5)]
    lines += ["1,2,3", "garbage,,,,,,,,", ""]
    lines += [",".join(str(v) for v in (20 * i, 0, 0, 4096, 33, 0, 0, 1, i)) for i in range(7, 10)]
    dec = serial_source.LineDecoder()
    out = [s for ln in lines for s in [dec.feed(ln)] if s is not None]
    assert len(out) == 8
    assert isinstance(out[0], SISample)
    assert out[0].t == pytest.approx(0.0) and out[1].t == pytest.approx(0.02)
    assert out[0].az == pytest.approx(9.80665) and out[0].gx == pytest.approx(
        33 / 32.8 * np.pi / 180, rel=1e-3
    )
    assert out[-1].gate_state is True
    assert dec.stats.bad_lines == 2 and dec.stats.comment_lines == 2
    assert dec.stats.dropped == 2  # seq jumped 4 -> 7
    assert dec.stats.samples == 8


def test_ble_batch_decoder_handles_truncated_and_wrapping_seq():
    dec = ble.BatchDecoder()
    buf = protocol.pack_batch(1, [_raw(i) for i in range(5)])
    out = dec.feed(buf)
    assert len(out) == 5 and out[0].session_id == 1 if hasattr(out[0], "session_id") else True
    assert dec.feed(buf[:-3]) == []  # truncated: dropped, counted, no raise
    assert dec.stats.bad_packets == 1
    out2 = dec.feed(protocol.pack_batch(1, [_raw(10), _raw(11)]))
    assert len(out2) == 2 and dec.stats.dropped == 5  # seq 4 -> 10 skipped 5..9
    dec2 = ble.BatchDecoder()
    dec2.feed(protocol.pack_batch(1, [_raw(65535)]))
    dec2.feed(protocol.pack_batch(1, [_raw(65536)]))
    assert dec2.stats.dropped == 0  # 65535 -> 0 is a clean wrap
    dec2.feed(protocol.pack_batch(1, [_raw(3)]))
    assert dec2.stats.dropped == 2  # 0 -> 3 skipped 1, 2
    dec3 = ble.BatchDecoder()
    dec3.feed(protocol.pack_batch(1, [_raw(40000)]))
    dec3.feed(protocol.pack_batch(1, [_raw(0)]))
    assert dec3.stats.dropped == 0  # a jump of >= 32768 is a device restart, not drops


def test_fake_source_streams_protocol_packets():
    src = fake.FakeSource(duration_s=2.0, fs=50, active_from_s=1.0)
    samples = list(src.iter_samples())
    assert len(samples) == 100
    assert samples[0].t == pytest.approx(0.0) and samples[-1].t == pytest.approx(1.98)
    assert not samples[10].gate_state and samples[-1].gate_state
    assert 8 < samples[0].a < 12


def test_recorder_writes_session_dir_in_the_schema(tmp_path):
    rec = recorder.SessionRecorder(subject="S1", exercise="curl", root=tmp_path / "team",
                                   source="fake", firmware="1.0.0", notes="unit test")  # fmt: skip
    src = fake.FakeSource(duration_s=1.0, fs=50)
    with rec:
        for s in src.iter_samples():
            rec.add(s)
    assert rec.session_dir.parent.name == "S1"
    df = pd.read_parquet(rec.session_dir / "imu.parquet")
    schema.validate_imu_stream(df)
    assert df["dataset"].iloc[0] == "team" and df["subject"].iloc[0] == "S1"
    assert df["exercise"].iloc[0] == "curl" and df["placement"].iloc[0] == "wrist_l"
    assert {"flags", "seq"} <= set(df.columns) and len(df) == 50
    meta = json.loads((rec.session_dir / "meta.json").read_text())
    assert meta["subject"] == "S1" and meta["exercise"] == "curl" and meta["n_samples"] == 50
    assert meta["firmware"] == "1.0.0" and meta["source"] == "fake" and "started" in meta
    assert "video.mp4" not in {p.name for p in rec.session_dir.iterdir()}


def test_session_check_reports_drops_and_gate_timeline(tmp_path):
    from formcoach.io import session_check

    rec = recorder.SessionRecorder(
        subject="S2", exercise="squat", root=tmp_path / "team", source="fake"
    )
    with rec:
        for i, s in enumerate(
            fake.FakeSource(duration_s=4.0, fs=50, active_from_s=2.0).iter_samples()
        ):
            if 60 <= i < 65:
                continue  # simulate 5 dropped samples
            rec.add(s)
    report = session_check.check(rec.session_dir)
    assert report["n_samples"] == 195 and report["dropped"] == 5
    assert report["duration_s"] == pytest.approx(3.98, abs=0.05)
    assert 40 < report["gate_open_pct"] < 60
    assert report["rate_hz"] == pytest.approx(50, abs=2)
    text = session_check.format_report(report)
    assert "dropped" in text and "gate open" in text


def test_session_check_on_the_replay_fixture():
    from formcoach.io import session_check
    from formcoach.io.replay import FIXTURE_SESSION

    report = session_check.check(FIXTURE_SESSION)
    assert report["n_samples"] > 1000 and report["dropped"] is None  # no seq column in MM-Fit
    assert report["pose_frames"] > 800 and report["expected_reps"] == 10


def test_record_cli_with_fake_source(tmp_path):
    from typer.testing import CliRunner

    from formcoach.cli import app

    result = CliRunner().invoke(
        app, ["record", "--source", "fake", "--subject", "S3", "--exercise", "raise",
              "--duration", "1", "--root", str(tmp_path)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    dirs = list((tmp_path / "S3").iterdir())
    assert len(dirs) == 1 and (dirs[0] / "imu.parquet").exists()
    check = CliRunner().invoke(app, ["session", "check", str(dirs[0])])
    assert check.exit_code == 0 and "samples" in check.output


def test_record_cli_without_device_extra_is_a_clear_error(monkeypatch):
    from typer.testing import CliRunner

    from formcoach.cli import app

    def no_pyserial():
        raise ImportError("serial input needs pyserial: `uv sync --extra device`")

    monkeypatch.setattr(serial_source, "_pyserial", no_pyserial)
    result = CliRunner().invoke(
        app, ["record", "--source", "serial", "--port", "COM99", "--duration", "1"]
    )
    assert result.exit_code == 2
    assert "extra device" in result.output or "pyserial" in result.output
