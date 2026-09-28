"""Replay source, pose store, gate logic, headless pipeline and the committed 30 s session.

The fixture session under data/fixtures/replay/mmfit_w00_curls/ is the demo of record
(`formcoach demo --source replay --headless`) and the CI smoke test."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from formcoach.app import gate as gate_mod
from formcoach.app import pipeline
from formcoach.data.fixtures import FIXTURE_ROOT
from formcoach.io import replay, source
from formcoach.io.protocol import GATE_OFF_WINDOWS, GATE_ON_WINDOWS
from formcoach.pose import store

SESSION = FIXTURE_ROOT / "replay" / "mmfit_w00_curls"


# ---- pose store --------------------------------------------------------------------------------
def test_pose_parquet_round_trip(tmp_path):
    n, j = 5, 17
    xyz = np.random.default_rng(0).normal(size=(n, j, 3)).astype(np.float32)
    frames = np.arange(100, 100 + n)
    t = frames / 30.0
    p = tmp_path / "pose.parquet"
    store.write_pose(p, session="s", frames=frames, t=t, world=xyz, skeleton="h36m17")
    seq = store.read_pose(p)
    assert seq.skeleton.name == "h36m17" and seq.world.shape == (n, j, 3)
    assert np.allclose(seq.world, xyz) and seq.valid.all() and seq.frames[0] == 100
    cols = pd.read_parquet(p).columns
    assert {"session", "frame", "t", "valid", "wl_0_x", "wl_16_z", "skeleton"} <= set(cols)
    assert "lm_0_x" not in cols  # no image landmarks for a 3-D-only source


def test_pose_parquet_with_image_landmarks_and_visibility(tmp_path):
    n = 3
    world = np.zeros((n, 33, 3))
    image = np.zeros((n, 33, 4))
    image[:, :, 3] = 0.9
    p = tmp_path / "pose.parquet"
    store.write_pose(
        p,
        session="s",
        frames=np.arange(n),
        t=np.arange(n) / 30,
        world=world,
        skeleton="mediapipe33",
        image=image,
        valid=np.array([True, False, True]),
    )
    seq = store.read_pose(p)
    assert seq.visibility.shape == (n, 33) and seq.visibility[0, 0] == pytest.approx(0.9)
    assert seq.valid.tolist() == [True, False, True]
    assert "lm_32_v" in pd.read_parquet(p).columns


# ---- replay source -----------------------------------------------------------------------------
def test_replay_source_yields_si_samples_in_order_and_fast():
    src = replay.ReplaySource(SESSION, speed=0.0)
    samples = list(src.iter_samples())
    assert len(samples) > 1000
    assert isinstance(samples[0], source.SISample)
    t = np.array([s.t for s in samples])
    assert (np.diff(t) > 0).all()
    assert 8 < np.median([abs(s.a) for s in samples]) < 12  # m/s² magnitude
    assert src.meta["exercise"] == "curl" and src.meta["subject"] == "P2"
    assert src.meta["expected_reps"] == 10


def test_replay_source_from_dataframe_and_pose_frames():
    df = pd.read_parquet(SESSION / "imu.parquet")
    src = replay.ReplaySource.from_stream(df.iloc[:200])
    assert len(list(src.iter_samples())) == 200
    frames = replay.PoseReplay(SESSION / "pose.parquet")
    first = next(iter(frames))
    assert first.world.shape == (17, 3) and first.t >= 0


# ---- gates -------------------------------------------------------------------------------------
def test_hysteresis_opens_after_on_windows_and_closes_after_off_windows():
    h = gate_mod.Hysteresis(on_windows=GATE_ON_WINDOWS, off_windows=GATE_OFF_WINDOWS)
    states = [h.update(True) for _ in range(GATE_ON_WINDOWS)]
    assert states[-1] is True and states[0] is False
    states = [h.update(False) for _ in range(GATE_OFF_WINDOWS)]
    assert states[-2] is True and states[-1] is False


def _samples(n_s: float, active: bool, fs: int = 50):
    t = np.arange(int(n_s * fs)) / fs
    amp = 5.0 if active else 0.02  # along gravity so |a| itself oscillates
    for i, ti in enumerate(t):
        yield source.SISample(t=float(ti), ax=0.0, ay=0.0, az=9.81 + amp * np.sin(2 * np.pi * ti),
                              gx=0.0, gy=0.0, gz=0.0, flags=0, seq=i)  # fmt: skip


def test_energy_gate_opens_on_motion_and_closes_when_still():
    g = gate_mod.EnergyGate(threshold=0.5)
    opened = [g.update(s) for s in _samples(4.0, active=True)]
    assert opened[-1] is True and any(o is False for o in opened[:50])
    closed = [g.update(s) for s in _samples(10.0, active=False)]
    assert closed[-1] is False
    assert gate_mod.AlwaysOnGate().update(next(_samples(1, False))) is True
    dev = gate_mod.DeviceGate()
    s = next(_samples(1, False))
    assert dev.update(s) is False
    assert dev.update(dataclasses.replace(s, flags=1)) is True


def test_make_gate_names():
    assert isinstance(gate_mod.make_gate("always_on"), gate_mod.AlwaysOnGate)
    assert isinstance(gate_mod.make_gate("energy"), gate_mod.EnergyGate)
    assert isinstance(gate_mod.make_gate("device"), gate_mod.DeviceGate)
    with pytest.raises(ValueError):
        gate_mod.make_gate("nope")


# ---- pipeline on the fixture session ----------------------------------------------------------
def test_headless_pipeline_counts_the_curls_on_the_fixture(tmp_path):
    result = pipeline.run_replay(SESSION, gate="always_on", headless=True, out_dir=tmp_path / "run")
    reps = [e for e in result.events if e.kind == "rep"]
    assert 8 <= len(reps) <= 12, [e.payload for e in reps]
    # trailing pose frames after the last IMU sample carry no gate information: skipped
    assert result.frames_total - result.frames_processed <= 1
    assert (tmp_path / "run" / "events.parquet").exists()
    assert (tmp_path / "run" / "frames.parquet").exists()
    summary = json.loads((tmp_path / "run" / "session.json").read_text())
    assert summary["reps"] == len(reps) and summary["gate"] == "always_on"
    r = reps[0].payload
    assert r["exercise"] == "curl" and 0.3 < r["duration_s"] < 6
    assert (
        "elbow_min" in r["metrics"] and r["metrics"]["elbow_min"] < 110
    )  # MM-Fit lifted pose: ~95


def test_energy_gate_skips_frames_but_finds_the_same_reps(tmp_path):
    on = pipeline.run_replay(SESSION, gate="always_on", headless=True, out_dir=tmp_path / "a")
    en = pipeline.run_replay(SESSION, gate="energy", headless=True, out_dir=tmp_path / "b")
    assert en.frames_processed < en.frames_total
    n_on = sum(e.kind == "rep" for e in on.events)
    n_en = sum(e.kind == "rep" for e in en.events)
    assert abs(n_on - n_en) <= 2
    kinds = {e.kind for e in en.events}
    assert {"gate_open", "gate_close"} & kinds


def test_demo_cli_replay_headless_runs_the_real_pipeline():
    from typer.testing import CliRunner

    from formcoach.cli import app

    result = CliRunner().invoke(app, ["demo", "--source", "replay", "--headless"])
    assert result.exit_code == 0, result.output
    assert "STUB" not in result.output
    assert "rep" in result.output and "frames processed" in result.output


def test_event_log_with_no_events_still_writes_files(tmp_path):
    from formcoach.app.events import EventLog

    log = EventLog()
    log.frame(0.0, False, False)
    log.write(tmp_path, {"reps": 0})
    ev = pd.read_parquet(tmp_path / "events.parquet")
    assert len(ev) == 0 and list(ev.columns) == ["kind", "t", "payload"]
    assert len(pd.read_parquet(tmp_path / "frames.parquet")) == 1


def test_pipeline_bridges_short_invalid_gaps_and_survives_long_ones(tmp_path):
    """docs/02 §4.3: gaps <= 3 frames are bridged; longer gaps give NaN angles, no crash."""
    import shutil

    from formcoach.pose import store

    sess = tmp_path / "sess"
    shutil.copytree(SESSION, sess)
    seq = store.read_pose(sess / "pose.parquet")
    valid = seq.valid.copy()
    valid[200:203] = False  # 3-frame gap: bridged
    valid[500:510] = False  # 10-frame gap: invalid
    store.write_pose(sess / "pose.parquet", session=seq.session, frames=seq.frames, t=seq.t,
                     world=seq.world, skeleton=seq.skeleton.name, valid=valid)  # fmt: skip
    res = pipeline.run_replay(sess, gate="always_on", headless=True, out_dir=tmp_path / "out")
    reps = [e for e in res.events if e.kind == "rep"]
    assert 8 <= len(reps) <= 12
    # online bridging cannot know a gap's length in advance: the first 3 frames of every gap
    # are bridged (3 + 3), the remaining 7 of the long gap are invalid
    assert res.summary["frames_bridged"] == 6 and res.summary["frames_invalid"] == 7
