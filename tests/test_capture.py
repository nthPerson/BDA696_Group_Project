"""Capture (Bench plan Task 6): a take held in memory, written as a session folder on Keep, with
the wearable's samples and the camera's frames on one clock. Fake frames, a synthetic detector,
a fake watch on an injected clock; everything is written under ``tmp_path``."""

from __future__ import annotations

import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from formcoach.app.capture import SETS_HEADER, TakeLabels
from formcoach.app.session import LiveSession
from formcoach.io import protocol
from formcoach.io.source import SISample
from formcoach.pose.store import read_pose
from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

from .session_fakes import BLANK, FakeBoard, synthetic_detect

T0 = 1000.0  # the injected laptop clock's reading at frame 0 (s)
BOOT = 990.0  # laptop time (s) at which the fake watch's device clock read 0
WALL = datetime(2026, 11, 18, 14, 15, 0)
FPS = 30.0


class Clock:
    """A settable laptop clock (s), shared by the session and the fake watch."""

    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class Watch(FakeBoard):
    """BoardLink-shaped wearable on the test clock: a 50 Hz device clock that started at laptop
    time ``boot``; each poll delivers every sample taken since the last one, stamped
    ``t_arrival = clock()``. ``gate(t)`` and ``silent(t)`` take laptop seconds since :data:`T0`;
    samples taken while silent are lost (their ``seq`` is skipped)."""

    def __init__(self, clock, *, gate=lambda t: True, silent=lambda t: False, boot=BOOT):
        super().__init__()
        self.clock, self.gate_at, self.silent, self.boot = clock, gate, silent, boot
        self.seq = round((clock() - boot) * 50)
        self.last_arrival: float | None = None

    def poll(self):
        now = self.clock()
        out = []
        while self.boot + self.seq / 50.0 <= now + 1e-9:
            t_dev = self.seq / 50.0
            if not self.silent(self.boot + t_dev - T0):
                flags = protocol.Flag.GATE_STATE if self.gate_at(now - T0) else 0
                out.append(SISample(t_dev, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, flags, self.seq, now))
                self.last_arrival = now
            self.seq += 1
        return out

    def alive(self):
        return self.last_arrival is not None and self.clock() - self.last_arrival <= 2.0

    def gate_open(self):
        return self.alive() and self.gate_at(self.clock() - T0)

    def state(self):
        if self.last_arrival is None:
            return "waiting"
        return "ok" if self.alive() else "silent"


class Rig:
    """A session on a fake clock: frame ``i`` has frame time ``i / FPS`` and laptop time
    ``T0 + i / FPS``."""

    def __init__(self, *, exercise="curl", detect=synthetic_detect, board=None, **kw):
        self.clock = Clock()
        if callable(board):
            board = board(self.clock)
        self.session = LiveSession(
            exercise=exercise, detect=detect, board=board, clock=self.clock, now=lambda: WALL, **kw
        )
        self.i = 0
        self.snaps: list = []
        self.events: list = []
        self.session.subscribe(on_snapshot=self.snaps.append, on_event=self.events.append)

    @property
    def t(self) -> float:
        """Laptop seconds since :data:`T0` of the next frame."""
        return self.i / FPS

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds * FPS)):
            self.clock.t = T0 + self.i / FPS
            self.session.step(self.i, self.i / FPS, BLANK)
            self.i += 1

    def capture_events(self) -> list:
        return [e for e in self.events if e.kind == "capture"]


def labels(**kw) -> TakeLabels:
    return TakeLabels(**{"subject": "S1", "exercise": "curl", **kw})


def read_events(folder) -> pd.DataFrame:
    ev = pd.read_parquet(folder / "events.parquet")
    ev["payload"] = ev["payload"].map(json.loads)
    return ev


def test_take_writes_all_files_with_one_time_base(tmp_path):
    rig = Rig(board=Watch, rules=RuleEngine.from_yaml(WEBCAM_RULES), rules_name="webcam")
    rig.run(2.0)
    rig.session.start_capture(labels(notes="first"))
    n_before = len(rig.snaps)
    rig.run(12.0)
    recorded = rig.snaps[n_before:]
    summary = rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=4, root=tmp_path)

    folder = tmp_path / "S1" / "20261118-141500"
    assert out["dir"] == str(folder)
    for name in ("imu.parquet", "pose.parquet", "events.parquet", "frames.parquet", "meta.json"):
        assert (folder / name).is_file(), name
    assert not list(folder.glob("*.mp4"))  # never video

    imu = pd.read_parquet(folder / "imu.parquet")
    pose = read_pose(folder / "pose.parquet")
    frames = pd.read_parquet(folder / "frames.parquet")
    assert abs(imu["t"].iloc[0]) < 0.1 and abs(pose.t[0]) < 0.1
    assert {"flags", "seq", "t_arrival", "t_device"} <= set(imu.columns)
    np.testing.assert_allclose(pose.t, frames["t"])
    assert len(pose.t) == len(recorded) == summary["frames"]

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    # one time base: a sample's t is never later than its arrival (both relative to the take)
    t0 = meta["clock"]["t0_mono"]
    assert (imu["t"] <= imu["t_arrival"] - t0 + 1e-6).all()
    np.testing.assert_allclose(pose.t, [s.t_mono - t0 for s in recorded])
    assert meta["clock"]["method"] == "envelope"
    assert abs(meta["clock"]["offset_s"] - BOOT) < 0.04
    assert meta["bench"] == {"version": 1}
    assert meta["labels"]["notes"] == "first" and meta["arm"] == "both"
    assert meta["expected_reps"] == 4
    assert meta["gate_source"] == "device" and meta["rules"] == "webcam"
    assert meta["overrides"] == {}
    assert meta["device"] == "FormCoach-TEST"
    for key in ("subject", "exercise", "placement", "session_id", "n_samples", "dropped"):
        assert key in meta, key
    assert meta["n_samples"] == len(imu) and meta["summary"]["frames"] == len(recorded)

    # landmarks are stored exactly as the live pipeline saw them (float32 detector output)
    np.testing.assert_array_equal(pose.world[-1], recorded[-1].world_landmarks)
    assert out["check"]["n_samples"] == len(imu)


def test_pose_runs_on_every_frame_while_recording_even_with_the_gate_closed(tmp_path):
    def gate(t):  # laptop seconds since T0; the take starts at 0
        return not 10.0 <= t < 20.0

    rig = Rig(board=lambda c: Watch(c, gate=gate))
    rig.session.start_capture(labels())
    rig.run(30.0)
    rig.session.stop_capture()
    folder = rig.session.keep_capture(true_reps=12, root=tmp_path)["dir"]

    pose = read_pose(tmp_path / "S1" / "20261118-141500" / "pose.parquet")
    frames = pd.read_parquet(tmp_path / "S1" / "20261118-141500" / "frames.parquet")
    assert pose.valid.all() and len(pose.t) == 900
    closed = (frames["t"] >= 10.05) & (frames["t"] < 19.95)
    assert not frames.loc[closed, "processed"].any()
    assert not frames.loc[closed, "gate_state"].any()
    assert frames.loc[frames["t"] < 9.9, "processed"].all()
    assert (frames["processed"] == frames["gate_state"]).all()
    assert {"ang_l", "enter_l", "exit_l", "armed_l", "vis_l", "ang_r", "vis_r"} <= set(frames)
    assert frames.loc[closed, "ang_l"].isna().all()  # unprocessed frames carry no diagnostics

    reps = read_events(tmp_path / "S1" / "20261118-141500").query("kind == 'rep'")
    assert len(reps) > 0
    assert not ((reps["t"] > 10.0) & (reps["t"] < 20.0)).any()
    assert folder.endswith("20261118-141500")


def test_counts_start_at_zero_for_each_take():
    rig = Rig()  # no watch: always_on
    rig.run(6.0)
    assert rig.session.pipeline.arm_totals()["l"]["reps"] > 0

    for _ in range(2):
        rig.events.clear()
        rig.session.start_capture(labels())
        rig.run(8.0)
        summary = rig.session.stop_capture()
        reps = [e for e in rig.events if e.kind == "rep"]
        assert reps, "the take counted nothing"
        for arm in ("l", "r"):
            ids = [e.payload["rep_id"] for e in reps if e.payload["arm"] == arm]
            assert ids == list(range(len(ids)))
            assert summary["reps_by_arm"][arm] == len(ids)
        rig.session.discard_capture()


def test_keep_writes_and_discard_writes_nothing(tmp_path):
    rig = Rig(board=Watch)
    rig.session.start_capture(labels())
    rig.run(3.0)
    assert not any(tmp_path.iterdir())  # nothing is written while recording ...
    rig.session.stop_capture()
    assert not any(tmp_path.iterdir())  # ... or in review
    rig.session.discard_capture()
    assert rig.session.capture_state == "idle"
    assert not any(tmp_path.iterdir())
    assert [e.payload["state"] for e in rig.capture_events()] == ["recording", "review", "idle"]

    rig.session.start_capture(labels())
    rig.run(3.0)
    rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=None, root=tmp_path)
    assert rig.session.capture_state == "idle"
    assert sorted(p.name for p in (tmp_path / "S1").iterdir()) == ["20261118-141500", "sets.csv"]
    assert isinstance(out["check"], dict) and out["warnings"] == []
    assert rig.capture_events()[-1].payload["state"] == "idle"
    assert rig.capture_events()[-1].payload["dir"] == out["dir"]


def test_wrong_state_and_wrong_exercise_are_refused(tmp_path):
    rig = Rig()
    s = rig.session
    for action in (s.stop_capture, s.discard_capture):
        with pytest.raises(RuntimeError):
            action()
    with pytest.raises(RuntimeError):
        s.keep_capture(true_reps=1, root=tmp_path)
    with pytest.raises(ValueError):
        s.start_capture(labels(exercise="squat"))
    assert s.capture_state == "idle"

    s.start_capture(labels(), delay_s=5.0)
    assert s.capture_state == "countdown"
    for state in ("countdown", "recording", "review"):
        if state == "recording":
            rig.run(5.5)
        elif state == "review":
            s.stop_capture()
        assert s.capture_state == state
        with pytest.raises(RuntimeError):
            s.start_capture(labels())
        with pytest.raises(RuntimeError):
            s.set_exercise("press")
        with pytest.raises(RuntimeError):
            s.reset_counters()
        if state != "review":
            with pytest.raises(RuntimeError):
                s.keep_capture(true_reps=1, root=tmp_path)
            with pytest.raises(RuntimeError):
                s.discard_capture()
    with pytest.raises(RuntimeError):
        s.stop_capture()  # already in review
    s.discard_capture()

    # stopping during the countdown goes straight back to idle, with nothing kept
    s.start_capture(labels(), delay_s=5.0)
    assert s.stop_capture() == {}
    assert s.capture_state == "idle"
    assert not any(tmp_path.iterdir())
    s.set_exercise("press")  # allowed again once idle


def test_auto_stop_five_seconds_after_the_gate_closes():
    def gate(t):  # closed before the set (does not count), open 6-12 s, closed after
        return 6.0 <= t < 12.0

    rig = Rig(board=lambda c: Watch(c, gate=gate))
    rig.session.start_capture(labels(), auto_stop=True)
    rig.run(16.5)
    assert rig.session.capture_state == "recording"
    rig.run(1.0)
    assert rig.session.capture_state == "review"
    review = rig.capture_events()[-1]
    assert review.payload["state"] == "review"
    assert 16.9 <= review.payload["summary"]["duration_s"] <= 17.1

    # ignored with the always-open gate
    rig = Rig()
    rig.session.start_capture(labels(), auto_stop=True)
    rig.run(10.0)
    assert rig.session.capture_state == "recording"


def test_countdown_delays_the_first_recorded_frame():
    rig = Rig()
    rig.run(1.0)
    started = rig.clock()
    rig.session.start_capture(labels(), delay_s=3.0)
    rig.run(2.9)
    assert rig.session.capture_state == "countdown"
    rig.run(2.1)  # 5 s after start_capture: recording for the last 2 s
    assert rig.session.capture_state == "recording"
    summary = rig.session.stop_capture()
    assert summary["frames"] in (60, 61)
    states = [e.payload for e in rig.capture_events()]
    assert states[0] == {"state": "countdown", "delay_s": 3.0}
    assert states[1]["state"] == "recording"
    assert started + 3.0 <= states[1]["t_mono"] < started + 3.0 + 1.0 / FPS


def test_take_with_no_watch_writes_an_empty_imu_file_and_a_valid_folder(tmp_path):
    rig = Rig()
    rig.session.start_capture(labels())
    rig.run(3.0)
    summary = rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=1, root=tmp_path)
    folder = tmp_path / "S1" / "20261118-141500"
    imu = pd.read_parquet(folder / "imu.parquet")
    assert len(imu) == 0
    assert {"t", "ax", "flags", "seq", "t_arrival", "t_device"} <= set(imu.columns)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["clock"]["method"] == "none" and meta["n_samples"] == 0
    assert summary["imu_samples"] == 0 and summary["rate_hz"] == 0.0
    assert len(read_pose(folder / "pose.parquet").t) == 90
    assert out["check"]["n_samples"] == 0 and out["check"]["gate_open_pct"] is None
    json.dumps(out, allow_nan=False)  # the result goes to a browser as JSON


def test_watch_silent_mid_take_is_recorded_not_fatal(tmp_path):
    rig = Rig(board=lambda c: Watch(c, silent=lambda t: 4.0 <= t < 7.0))
    rig.session.start_capture(labels())
    rig.run(10.0)
    assert any(s.link_state == "silent" for s in rig.snaps)
    summary = rig.session.stop_capture()
    assert summary["frames"] == 300
    assert summary["dropped"] == 150  # 3 s at 50 Hz
    out = rig.session.keep_capture(true_reps=3, root=tmp_path)
    imu = pd.read_parquet(tmp_path / "S1" / "20261118-141500" / "imu.parquet")
    assert not ((imu["t"] > 4.05) & (imu["t"] < 6.95)).any()
    assert imu["t"].max() > 9.0
    assert any("dropped" in p for p in out["check"]["problems"])


def test_sets_csv_gets_a_line_and_an_arm_column(tmp_path):
    rig = Rig()
    rig.session.start_capture(labels(arm="l", form="partial", notes="lost count, rep 7"))
    rig.run(1.0)
    rig.session.stop_capture()
    rig.session.keep_capture(true_reps=10, root=tmp_path)
    with (tmp_path / "S1" / "sets.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["session_id", "exercise", "form", "reps", "notes", "arm"],
        ["20261118-141500", "curl", "partial", "10", "lost count, rep 7", "l"],
    ]

    # an old five-column file is upgraded in place
    old = tmp_path / "S2" / "sets.csv"
    old.parent.mkdir()
    old.write_text(
        "session_id,exercise,form,reps,notes\n20261106-141500,curl,clean,10,camera 2.5 m\n",
        encoding="utf-8",
    )
    rig.session.start_capture(labels(subject="S2"))
    rig.run(1.0)
    rig.session.stop_capture()
    rig.session.keep_capture(true_reps=None, root=tmp_path)
    with old.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["session_id", "exercise", "form", "reps", "notes", "arm"],
        ["20261106-141500", "curl", "clean", "10", "camera 2.5 m", ""],
        ["20261118-141500", "curl", "clean", "", "", "both"],
    ]


def test_cue_shown_latency_is_stored_in_the_summary():
    rig = Rig()
    rig.session.start_capture(labels())
    done = []
    while not done:
        rig.run(1.0 / FPS)
        snap = rig.snaps[-1]
        done = [(r.arm, r.rep_id, snap.t_mono) for r in snap.completed]
    arm, rep_id, t_frame = done[0]
    rig.session.note_cue_shown(arm, rep_id, t_frame + 0.120)
    cue = [e for e in rig.events if e.kind == "cue_shown"]
    assert len(cue) == 1
    assert cue[0].payload["arm"] == arm and cue[0].payload["rep_id"] == rep_id
    assert cue[0].payload["latency_ms"] == pytest.approx(120.0)
    rig.session.note_cue_shown(arm, rep_id + 99, t_frame)  # unknown rep: ignored
    summary = rig.session.stop_capture()
    assert summary["latency_ms_median"] == pytest.approx(120.0)


def test_idle_take_records_imu_and_pose_without_counters(tmp_path):
    calls = []

    def detect(frame, t_ms):
        calls.append(t_ms)
        return synthetic_detect(frame, t_ms)

    rig = Rig(exercise="idle", detect=detect, board=lambda c: Watch(c, gate=lambda t: False))
    assert rig.session.pipeline is None
    rig.session.start_capture(labels(exercise="idle"))
    rig.run(4.0)
    summary = rig.session.stop_capture()
    assert len(calls) == 120  # pose on every frame although the gate never opened
    assert summary["reps_by_arm"] == {} and summary["gate_open_pct"] == 0.0
    rig.session.keep_capture(true_reps=None, root=tmp_path)
    folder = tmp_path / "S1" / "20261118-141500"
    assert read_pose(folder / "pose.parquet").valid.all()
    assert len(pd.read_parquet(folder / "imu.parquet")) > 150
    frames = pd.read_parquet(folder / "frames.parquet")
    assert list(frames.columns) == ["t", "processed", "gate_state", "t_source"]
    assert not read_events(folder)["kind"].isin(["rep", "fault"]).any()


def test_labels_reject_unknown_form_and_arm():
    with pytest.raises(ValueError, match="form"):
        TakeLabels(subject="S1", exercise="curl", form="sloppy")
    with pytest.raises(ValueError, match="arm"):
        TakeLabels(subject="S1", exercise="curl", arm="left")
    with pytest.raises(ValueError, match="exercise"):
        TakeLabels(subject="S1", exercise="jog")
    for subject in ("", "../S1", "S1/x"):
        with pytest.raises(ValueError, match="subject"):
            TakeLabels(subject=subject, exercise="curl")
    ok = TakeLabels(subject="S3", exercise="squat", form="shallow", arm="both")
    assert ok.notes == ""


def test_frames_without_a_person_are_stored_invalid(tmp_path):
    def detect(frame, t_ms):
        return (None, None) if 1000 <= t_ms < 1500 else synthetic_detect(frame, t_ms)

    rig = Rig(detect=detect)
    rig.session.start_capture(labels())
    rig.run(2.0)
    rig.session.stop_capture()
    rig.session.keep_capture(true_reps=None, root=tmp_path)
    pose = read_pose(tmp_path / "S1" / "20261118-141500" / "pose.parquet")
    hidden = (pose.t >= 1.0) & (pose.t < 1.5)
    assert hidden.any() and not pose.valid[hidden].any() and pose.valid[~hidden].all()
    assert np.isnan(pose.world[hidden]).all()


def test_a_written_take_is_kept_when_sets_csv_cannot_be_updated(tmp_path):
    (tmp_path / "S1" / "sets.csv").mkdir(parents=True)  # opening it fails on every OS
    rig = Rig()
    rig.session.start_capture(labels())
    rig.run(1.0)
    rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=3, root=tmp_path)
    folder = tmp_path / "S1" / "20261118-141500"
    assert out["dir"] == str(folder) and (folder / "meta.json").is_file()
    assert rig.session.capture_state == "idle"
    assert len(out["warnings"]) == 1 and "sets.csv" in out["warnings"][0]
    assert rig.capture_events()[-1].payload["warnings"] == out["warnings"]
    assert not list((tmp_path / "S1").glob(".*"))  # no partial folder or temp file left
    with pytest.raises(RuntimeError):
        rig.session.discard_capture()  # the take is no longer held

    # the next take is recorded and kept as usual (another second on the wall clock)
    rig.session.now = lambda: datetime(2026, 11, 18, 14, 16, 0)
    rig.session.start_capture(labels())
    rig.run(1.0)
    rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=3, root=tmp_path)
    assert (tmp_path / "S1" / "20261118-141600" / "meta.json").is_file()
    assert rig.session.capture_state == "idle"
    assert "sets.csv" in out["warnings"][0]


def test_a_written_take_is_kept_when_the_session_check_fails(tmp_path, monkeypatch):
    def broken(_folder):
        raise KeyError("flags")

    monkeypatch.setattr("formcoach.app.session.session_check.check", broken)
    rig = Rig()
    rig.session.start_capture(labels())
    rig.run(1.0)
    rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=3, root=tmp_path)
    assert (tmp_path / "S1" / "20261118-141500" / "meta.json").is_file()
    assert (tmp_path / "S1" / "sets.csv").is_file()
    assert rig.session.capture_state == "idle"
    assert out["warnings"] == ["session check failed: KeyError: 'flags'"]
    assert out["check"] == {"problems": out["warnings"]}
    assert rig.capture_events()[-1].payload["warnings"] == out["warnings"]


def test_a_failed_folder_write_keeps_the_take_in_review(tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise OSError("disk full")

    rig = Rig()
    rig.session.start_capture(labels())
    rig.run(1.0)
    rig.session.stop_capture()
    monkeypatch.setattr("formcoach.app.capture.write_pose", broken)
    with pytest.raises(OSError, match="disk full"):
        rig.session.keep_capture(true_reps=3, root=tmp_path)
    assert rig.session.capture_state == "review"
    assert not any((tmp_path / "S1").iterdir())  # nothing left on disk
    monkeypatch.undo()
    out = rig.session.keep_capture(true_reps=3, root=tmp_path)  # Keep again works
    assert out["warnings"] == [] and rig.session.capture_state == "idle"


# ---- final review fixes (C1-C4, I3, I4) --------------------------------------------------------


class RestartingWatch(Watch):
    """A :class:`Watch` whose device clock restarts (the kit reboots, or another kit takes over
    in the same link) when the laptop clock passes ``T0 + restart_at``: device time goes back to
    0.5 s and ``seq`` to 25, and keeps counting from there."""

    def __init__(self, clock, *, restart_at: float, **kw):
        super().__init__(clock, **kw)
        self.restart_at = restart_at
        self.restarted = False

    def poll(self):
        if not self.restarted and self.clock() - T0 >= self.restart_at:
            self.restarted = True
            self.boot = self.clock() - 0.5  # the device clock reads 0.5 s now
            self.seq = 25
        return super().poll()


def test_a_device_clock_restart_mid_take_keeps_the_take(tmp_path):
    from formcoach.app import pipeline

    engine = RuleEngine.from_yaml(WEBCAM_RULES)
    rig = Rig(board=lambda c: RestartingWatch(c, restart_at=5.0), rules=engine)
    rig.session.start_capture(labels())
    rig.run(10.0)
    totals = rig.session.pipeline.arm_totals()
    summary = rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=None, root=tmp_path)

    folder = tmp_path / "S1" / "20261118-141500"
    imu = pd.read_parquet(folder / "imu.parquet")
    assert len(imu) >= 490  # both stretches, 10 s at 50 Hz
    assert (np.diff(imu["t"].to_numpy()) > 0).all()
    assert imu["t_device"].iloc[0] > 9.9 and imu["t_device"].iloc[-1] < 6.0
    np.testing.assert_allclose(imu["t"].iloc[-1], 10.0 - 1 / FPS, atol=0.05)
    restarted = [w for w in out["warnings"] if "device clock restarted" in w]
    assert len(restarted) == 1 and "t=5.0" in restarted[0]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    segments = meta["clock"]["segments"]
    assert len(segments) == 2 and [s["method"] for s in segments] == ["envelope"] * 2
    assert abs(segments[0]["offset_s"] - BOOT) < 0.04
    assert abs(segments[1]["offset_s"] - (T0 + 4.5)) < 0.04
    assert meta["clock"]["offset_s"] in (segments[0]["offset_s"], segments[1]["offset_s"])
    assert summary["dropped"] == 0 and 49.0 < summary["rate_hz"] < 51.0

    again = pipeline.run_replay(folder)  # as recorded
    assert again.summary["as_recorded"] is True and again.summary["arms"] == totals
    assert pipeline.run_replay(folder, gate="energy").summary["gate"] == "energy"


def test_imu_that_fails_validation_is_left_out_with_a_warning(tmp_path, monkeypatch):
    from formcoach.data import schema

    real = schema.validate_imu_stream

    def refuse_samples(df):
        if len(df):
            raise schema.SchemaError("t must be strictly monotonic increasing within a stream")
        real(df)

    monkeypatch.setattr(schema, "validate_imu_stream", refuse_samples)
    rig = Rig(board=Watch)
    rig.session.start_capture(labels())
    rig.run(2.0)
    rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=None, root=tmp_path)

    folder = tmp_path / "S1" / "20261118-141500"
    assert rig.session.capture_state == "idle" and out["dir"] == str(folder)
    imu = pd.read_parquet(folder / "imu.parquet")
    assert len(imu) == 0 and {"t", "flags", "seq", "t_device"} <= set(imu.columns)
    assert len(read_pose(folder / "pose.parquet").t) == 60
    assert len(pd.read_parquet(folder / "frames.parquet")) == 60
    assert any("IMU samples" in w and "monotonic" in w for w in out["warnings"])
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_samples"] == 0


def test_the_watch_cannot_change_while_a_take_is_recording(tmp_path):
    rig = Rig(board=Watch)
    s = rig.session
    first = s.board
    s.start_capture(labels(), delay_s=1.0)
    s.set_board(first)  # a countdown records nothing yet: allowed
    rig.run(1.5)
    assert s.capture_state == "recording"
    for board in (None, FakeBoard()):
        with pytest.raises(RuntimeError, match="recording"):
            s.set_board(board)
    assert s.board is first
    s.stop_capture()
    s.set_board(None)  # in review: allowed again
    assert s.board is None


def test_subject_labels_are_safe_folder_names_on_every_os():
    for bad in ("S1?", "S1|", "C:x", "S 1", "CON", "con", "Lpt1", "COM9", "nul", "aux", "prn",
                "s" * 33, ".hidden", "S1."):  # fmt: skip
        with pytest.raises(ValueError, match="subject"):
            TakeLabels(subject=bad, exercise="curl")
    for good in ("S1", "S10", "test_a-b", "COM10", "CONSOLE"):
        assert TakeLabels(subject=good, exercise="curl").subject == good


def test_a_bad_subject_is_refused_before_anything_records(tmp_path):
    rig = Rig()
    with pytest.raises(ValueError, match="subject"):
        rig.session.start_capture(TakeLabels(subject="CON", exercise="curl"))
    assert rig.session.capture_state == "idle"


def test_a_cue_confirmed_twice_counts_once_and_every_latency_is_kept(tmp_path):
    rig = Rig()
    rig.session.start_capture(labels())
    done: list = []
    while len(done) < 2:
        rig.run(1.0 / FPS)
        snap = rig.snaps[-1]
        done += [(r.arm, r.rep_id, snap.t_mono) for r in snap.completed]
    (arm0, id0, t0), (arm1, id1, t1) = done[:2]
    rig.session.note_cue_shown(arm0, id0, t0 + 0.100)
    rig.session.note_cue_shown(arm0, id0, t0 + 0.900)  # a second coach page: ignored
    rig.session.note_cue_shown(arm1, id1, t1 + 0.200)
    cues = [e.payload for e in rig.events if e.kind == "cue_shown"]
    assert [(c["arm"], c["rep_id"]) for c in cues] == [(arm0, id0), (arm1, id1)]
    summary = rig.session.stop_capture()
    assert summary["latency_ms_median"] == pytest.approx(150.0)
    rig.session.keep_capture(true_reps=None, root=tmp_path)
    meta = json.loads((tmp_path / "S1" / "20261118-141500" / "meta.json").read_text("utf-8"))
    assert meta["latency"] == [
        {"arm": arm0, "rep_id": id0, "latency_ms": pytest.approx(100.0)},
        {"arm": arm1, "rep_id": id1, "latency_ms": pytest.approx(200.0)},
    ]


def test_a_take_without_cue_latencies_records_an_empty_list(tmp_path):
    rig = Rig()
    rig.session.start_capture(labels())
    rig.run(1.0)
    rig.session.stop_capture()
    rig.session.keep_capture(true_reps=None, root=tmp_path)
    meta = json.loads((tmp_path / "S1" / "20261118-141500" / "meta.json").read_text("utf-8"))
    assert meta["latency"] == []


def test_a_sets_csv_the_csv_module_cannot_read_becomes_a_warning(tmp_path):
    sets = tmp_path / "S1" / "sets.csv"
    sets.parent.mkdir()
    # a field over csv.field_size_limit() raises csv.Error, which is neither OSError nor
    # ValueError (a NUL byte no longer does on Python 3.11)
    sets.write_text(",".join(SETS_HEADER) + "\n" + "x" * 200_000 + "\n", encoding="utf-8")
    rig = Rig()
    rig.session.start_capture(labels())
    rig.run(1.0)
    rig.session.stop_capture()
    out = rig.session.keep_capture(true_reps=3, root=tmp_path)
    assert rig.session.capture_state == "idle"
    assert (tmp_path / "S1" / "20261118-141500" / "meta.json").is_file()
    assert len(out["warnings"]) == 1 and "sets.csv" in out["warnings"][0]


def test_without_keep_log_the_session_log_stays_bounded():
    from .session_fakes import must_not_be_called

    s = LiveSession(
        exercise="curl", detect=must_not_be_called, gate_source="manual", keep_log=False
    )
    for i in range(20_000):
        s.step(i, i / FPS, BLANK)
    assert s.log.frames_total == 20_000 and s.log.frames_processed == 0
    assert len(s.log.frames) <= 1_000
    for _ in range(20_000):
        s.emit("info", text="x")
    assert len(s.log.events) == 10_000
    assert s.finish().summary["frames_total"] == 20_000

    kept = LiveSession(exercise="curl", detect=must_not_be_called, gate_source="manual")
    for i in range(30):
        kept.step(i, i / FPS, BLANK)
    assert len(kept.log.frames) == 30  # run_live's default keeps every row


def test_a_take_recorded_without_keep_log_writes_every_frame(tmp_path):
    rig = Rig(keep_log=False)
    rig.session.start_capture(labels())
    rig.run(3.0)
    rig.session.stop_capture()
    rig.session.keep_capture(true_reps=None, root=tmp_path)
    folder = tmp_path / "S1" / "20261118-141500"
    assert len(pd.read_parquet(folder / "frames.parquet")) == 90
    assert len(read_pose(folder / "pose.parquet").t) == 90
    assert (read_events(folder)["kind"] == "rep").any()
