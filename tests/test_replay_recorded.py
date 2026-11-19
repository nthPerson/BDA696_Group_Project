"""Replay "as recorded" and the per-arm session check (Bench plan Task 7).

Takes are recorded through the real capture code: a :class:`FakeScene` (synthetic person and
camera) and a :class:`FakeWatch` (synthetic wearable, read synchronously) on one injected clock,
written under ``tmp_path``. No camera, no Bluetooth, no threads."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from formcoach.app import pipeline, replay_recorded
from formcoach.app.capture import TakeLabels
from formcoach.app.replay_recorded import is_bench_session, replay_as_recorded
from formcoach.app.session import LiveSession
from formcoach.cli import app
from formcoach.eval import gating
from formcoach.io import session_check
from formcoach.io.fake import FakeWatch
from formcoach.io.replay import FIXTURE_SESSION
from formcoach.pose.synthetic import FakeScene
from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

T0 = 500.0  # the injected laptop clock at the first frame (s)
WALL = datetime(2026, 11, 18, 15, 0, 0)
FPS = 30.0
WEBCAM = WEBCAM_RULES.name
PARTIAL_ROM_MIN = "rule.curl.CURL_PARTIAL_ROM.elbow_min"  # 115 deg in the webcam rules
TOO_FAST = "rule.curl.CURL_TOO_FAST.duration_s"  # 1.0 s


class Clock:
    """The one laptop clock (s); FakeScene paces itself by sleeping on it."""

    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(s, 0.0)


class WatchBoard:
    """BoardLink-shaped and synchronous: :meth:`poll` delivers the :class:`FakeWatch` samples
    whose device time has come on the shared clock, stamped with their arrival time."""

    name = FakeWatch.device_name

    def __init__(self, clock: Clock, schedule):
        self.clock = clock
        self._samples = FakeWatch(schedule=schedule, clock=clock, sleep=lambda s: None)
        self._it = None
        self._t0 = 0.0
        self._next = None
        self._last = None
        self._last_arrival: float | None = None

    def poll(self):
        now = self.clock()
        if self._it is None:
            self._it, self._t0 = self._samples.iter_samples(), now
        out = []
        while True:
            if self._next is None:
                self._next = next(self._it)
            if self._t0 + self._next.t > now + 1e-9:
                break
            out.append(replace(self._next, t_arrival=now))
            self._next = None
        if out:
            self._last, self._last_arrival = out[-1], now
        return out

    def alive(self) -> bool:
        return self._last_arrival is not None and self.clock() - self._last_arrival <= 2.0

    def gate_open(self) -> bool:
        return self.alive() and self._last.gate_state

    def state(self) -> str:
        if self._last_arrival is None:
            return "waiting"
        return "ok" if self.alive() else "silent"

    def send(self, cmd) -> None:
        pass


@dataclass
class Live:
    """What the live run gave: the take's folder, the per-arm totals and fault codes at stop,
    the session-check report and the session itself."""

    dir: Path
    summary_arms: dict
    faults: list
    check: dict
    session: LiveSession


def _codes(reps) -> list:
    """``(arm, rep_id, fault codes)`` of every rep, sorted."""
    return sorted((r.arm, r.rep_id, tuple(sorted(f["code"] for f in r.faults))) for r in reps)


def fault_codes(result) -> list:
    return _codes(result.reps)


def fault_codes_live(live: Live) -> list:
    return live.faults


def busy(t: float) -> bool:
    """The person curls from 2 s to 34 s of the run."""
    return 2.0 <= t < 34.0


def record_take(
    root: Path,
    *,
    seconds: float = 40.0,
    start_at: float = 1.0,
    exercise: str = "curl",
    scene_schedule=busy,
    watch_schedule=busy,
    watch: bool = True,
    rules_name: str = WEBCAM,
    rules_file: Path = WEBCAM_RULES,
    arm: str = "both",
    true_reps: int | None = None,
    actions: dict | None = None,
    now: datetime = WALL,
) -> Live:
    """Run a live session on a FakeScene (and a FakeWatch unless ``watch`` is false, which
    leaves the gate always on), record from ``start_at`` s to the end and keep the take under
    ``root``. ``actions`` maps run seconds to ``fn(session)`` called just before that frame.
    The session judges with ``rules_file`` and records it as ``rules_name``."""
    clock = Clock()
    scene = FakeScene(
        exercise="curl" if exercise == "idle" else exercise, schedule=scene_schedule,
        clock=clock, sleep=clock.sleep, duration_s=seconds,
    )  # fmt: skip
    session = LiveSession(
        exercise=exercise, detect=scene.detect, rules=RuleEngine.from_yaml(rules_file),
        rules_name=rules_name, board=WatchBoard(clock, watch_schedule) if watch else None,
        clock=clock, now=lambda: now,
    )  # fmt: skip
    due = {round(s * FPS): fn for s, fn in (actions or {}).items()}
    start = round(start_at * FPS)
    for i, t, frame in scene.frames():
        if i == start:
            session.start_capture(TakeLabels(subject="S1", exercise=exercise, arm=arm))
        if i in due:
            due[i](session)
        session.step(i, t, frame)
    pipe = session.pipeline
    totals = pipe.arm_totals() if pipe is not None else {}
    faults = _codes(r for track in pipe.tracks for r in track.reps) if pipe is not None else []
    session.stop_capture()
    out = session.keep_capture(true_reps=true_reps, root=root)
    return Live(Path(out["dir"]), totals, faults, out["check"], session)


def read_events(folder: Path) -> pd.DataFrame:
    ev = pd.read_parquet(folder / "events.parquet")
    ev["payload"] = ev["payload"].map(json.loads)
    return ev


@pytest.fixture(scope="module")
def take(tmp_path_factory) -> Live:
    """One 40 s two-arm curl take, watch gate (``device``), recording from 1 s."""
    return record_take(tmp_path_factory.mktemp("takes"))


# ---- replay as recorded ------------------------------------------------------------------------


def test_replay_as_recorded_matches_the_live_counts_and_faults(take):
    live = take
    again = replay_as_recorded(live.dir)
    assert again.summary["arms"] == live.summary_arms
    assert fault_codes(again) == fault_codes_live(live)
    assert live.summary_arms["l"]["reps"] >= 8 and live.summary_arms["r"]["reps"] >= 8
    assert again.summary["as_recorded"] is True and again.summary["notes"] == []
    assert again.summary["gate"] == "recorded"

    # reported on the take clock: the reps sit where the live take logged them
    logged = read_events(live.dir).query("kind == 'rep'")
    replayed = [e for e in again.events if e.kind == "rep"]
    assert len(replayed) == len(logged)
    np.testing.assert_allclose([e.t for e in replayed], logged["t"], atol=1e-6)
    np.testing.assert_allclose(
        [e.payload["t_start"] for e in replayed], [p["t_start"] for p in logged["payload"]],
        atol=1e-6,
    )  # fmt: skip
    np.testing.assert_allclose(sorted(r.t_end for r in again.reps), sorted(logged["t"]), atol=1e-6)
    # and on the take clock, which started 1 s into the frame source's clock
    assert all(e.payload["t_end"] == pytest.approx(e.t) for e in replayed)
    frames = pd.read_parquet(live.dir / "frames.parquet")
    np.testing.assert_allclose(frames["t_source"] - frames["t"], 1.0, atol=1e-6)


def test_overrides_stored_in_the_take_are_applied_on_replay(tmp_path):
    def flag_every_rep(s):  # set before recording: stored in meta.json's overrides
        s.set_param(PARTIAL_ROM_MIN, 40.0)  # synthetic curls bottom out at 50 deg

    live = record_take(
        tmp_path,
        start_at=1.0,
        actions={
            0.5: flag_every_rep,
            # during the take: params_changed events (re-judged live, so the final values count)
            12.0: lambda s: s.set_param(TOO_FAST, 2.5),
            20.0: lambda s: s.set_param("seg.min_range_deg", 40.0),
        },
    )
    meta = json.loads((live.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["overrides"] == {PARTIAL_ROM_MIN: 40.0}
    changed = read_events(live.dir).query("kind == 'params_changed'")
    assert [p["key"] for p in changed["payload"]] == [TOO_FAST, "seg.min_range_deg"]

    again = replay_as_recorded(live.dir)
    assert again.summary["arms"] == live.summary_arms
    assert fault_codes(again) == fault_codes_live(live)
    codes = {c for _, _, cs in fault_codes(again) for c in cs}
    assert {"CURL_PARTIAL_ROM", "CURL_TOO_FAST"} <= codes
    assert all("CURL_PARTIAL_ROM" in cs for _, _, cs in fault_codes(again))

    # without the take's overrides the same reps are judged differently
    plain = replay_as_recorded(live.dir, rules=RuleEngine.from_yaml(WEBCAM_RULES))
    assert [r["reps"] for r in plain.summary["arms"].values()] == [
        r["reps"] for r in again.summary["arms"].values()
    ]
    assert not any("CURL_PARTIAL_ROM" in cs for _, _, cs in fault_codes(plain))


def test_unknown_rules_file_falls_back_to_the_webcam_rules_with_a_note(tmp_path):
    live = record_take(tmp_path, seconds=12.0, rules_name="my-tuned-rules.yaml")
    meta = json.loads((live.dir / "meta.json").read_text(encoding="utf-8"))
    assert isinstance(meta.pop("rules_doc"), dict)  # a take from before rules_doc was stored:
    (live.dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")  # found by name
    again = replay_as_recorded(live.dir)
    assert again.summary["arms"] == live.summary_arms  # the session judged with webcam rules
    assert len(again.summary["notes"]) == 1
    assert "my-tuned-rules.yaml" in again.summary["notes"][0]
    assert WEBCAM in again.summary["notes"][0]


def test_always_on_gate_counts_at_least_as_many_reps_as_recorded(tmp_path):
    # the watch stops flagging movement from 12 s to 22 s while the person keeps curling
    live = record_take(tmp_path, watch_schedule=lambda t: busy(t) and not 12.0 <= t < 22.0)
    recorded = replay_as_recorded(live.dir)
    always = replay_as_recorded(live.dir, gate="always_on")
    assert recorded.summary["arms"] == live.summary_arms
    for arm in ("l", "r"):
        assert always.summary["arms"][arm]["reps"] >= recorded.summary["arms"][arm]["reps"]
    assert always.summary["reps"] > recorded.summary["reps"]
    assert always.frames_processed == always.frames_total
    assert recorded.frames_processed < recorded.frames_total
    assert always.summary["gate"] == "always_on"


def test_energy_gate_is_recomputed_from_the_imu_file(take, tmp_path):
    energy = replay_as_recorded(take.dir, gate="energy")
    assert energy.summary["gate"] == "energy"
    assert 0 < energy.frames_processed < energy.frames_total
    assert energy.summary["reps"] > 0
    opens = [e for e in energy.events if e.kind == "gate_open"]
    assert opens and opens[0].t > 2.0  # the wrist starts moving 1 s into the take
    assert energy.summary["imu_seconds"] > 35.0

    # the same take with the wrist still in the IMU file: the gate never opens
    still = tmp_path / "still"
    shutil.copytree(take.dir, still)
    imu = pd.read_parquet(still / "imu.parquet")
    imu[["ax", "ay", "gx", "gy", "gz"]] = 0.0
    imu["az"] = 9.81
    imu.to_parquet(still / "imu.parquet", index=False)
    quiet = replay_as_recorded(still, gate="energy")
    assert quiet.frames_processed == 0 and quiet.summary["reps"] == 0
    # gate_kwargs reach the gate: a threshold nothing reaches keeps it closed as well
    high = replay_as_recorded(take.dir, gate="energy", gate_kwargs={"threshold": 1e9})
    assert high.frames_processed == 0


def test_take_without_imu_replays_with_recorded_and_always_on(tmp_path):
    live = record_take(tmp_path, seconds=15.0, watch=False)
    assert len(pd.read_parquet(live.dir / "imu.parquet")) == 0
    for gate in ("recorded", "always_on"):
        again = replay_as_recorded(live.dir, gate=gate)
        assert again.summary["arms"] == live.summary_arms
        assert again.summary["imu_seconds"] == 0.0
    for gate in ("energy", "device"):
        with pytest.raises(ValueError, match="no IMU samples"):
            replay_as_recorded(live.dir, gate=gate)
    with pytest.raises(ValueError, match="unknown gate"):
        replay_as_recorded(live.dir, gate="manual")


def test_idle_take_replays_without_reps(tmp_path):
    live = record_take(tmp_path, seconds=10.0, exercise="idle")
    for gate in ("recorded", "always_on", "device"):
        again = replay_as_recorded(live.dir, gate=gate)
        assert again.reps == [] and again.summary["reps"] == 0
        assert again.summary["arms"] == {} and again.summary["exercise"] == "idle"
        assert again.frames_total == 270
    assert replay_as_recorded(live.dir, gate="always_on").frames_processed == 270


def test_run_replay_dispatches_only_for_bench_sessions(take, monkeypatch):
    assert not is_bench_session(FIXTURE_SESSION)
    assert is_bench_session(take.dir)

    calls = []
    real = replay_recorded.replay_as_recorded

    def spy(session, **kw):
        calls.append(kw)
        return real(session, **kw)

    monkeypatch.setattr(replay_recorded, "replay_as_recorded", spy)
    old = pipeline.run_replay(FIXTURE_SESSION)  # the committed fixture: the old path
    assert calls == [] and old.summary["gate"] == "always_on" and "as_recorded" not in old.summary
    assert "reps_imu" in old.summary

    bench = pipeline.run_replay(take.dir)
    assert calls[-1]["gate"] == "recorded"
    assert bench.summary["arms"] == take.summary_arms
    pipeline.run_replay(take.dir, gate="energy")
    assert calls[-1]["gate"] == "energy"

    # forced off: a take goes down the old path (one averaged track, no per-arm totals)
    n = len(calls)
    off = pipeline.run_replay(take.dir, as_recorded=False)
    assert len(calls) == n and "arms" not in off.summary and off.summary["gate"] == "always_on"
    with pytest.raises(ValueError, match="recorded"):
        pipeline.run_replay(FIXTURE_SESSION, gate="recorded")


def test_eval_gating_accepts_a_bench_take(take, tmp_path):
    out = tmp_path / "gating.md"
    table = gating.evaluate([take.dir], gates=("always_on", "device"), out=out)
    assert list(table["gate"]) == ["always_on", "device"]
    assert "error" not in table.columns
    rows = table.set_index("gate")
    assert rows.loc["always_on", "frames_processed_pct"] == 100.0
    assert 0 < rows.loc["device", "frames_processed_pct"] < 100.0
    assert rows.loc["always_on", "reps"] >= rows.loc["device", "reps"] > 0
    assert out.is_file() and take.dir.name in out.read_text(encoding="utf-8")


# ---- final review fixes: a take's own rules (I1, I2) ------------------------------------------


def _custom_rules(path: Path) -> Path:
    """The webcam rules with CURL_PARTIAL_ROM's elbow_min limit at 40 deg (synthetic curls
    bottom out at 50, so every rep is flagged), written to ``path``."""
    import yaml

    doc = yaml.safe_load(WEBCAM_RULES.read_text(encoding="utf-8"))
    rule = next(r for r in doc["exercises"]["curl"]["rules"] if r["code"] == "CURL_PARTIAL_ROM")
    next(c for c in rule["when"] if c["metric"] == "elbow_min")["threshold"] = 40.0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_a_take_recorded_with_a_custom_rules_file_replays_with_its_live_faults(tmp_path):
    custom = _custom_rules(tmp_path / "rules" / "my-curl-rules.yaml")
    # a new name, and a copy that shares a packaged name: neither is looked up by name
    for name in ("my-curl-rules.yaml", WEBCAM):
        live = record_take(tmp_path / name, seconds=15.0, rules_file=custom, rules_name=name)
        assert live.faults and all("CURL_PARTIAL_ROM" in cs for _, _, cs in live.faults)
        again = replay_as_recorded(live.dir)
        assert again.summary["notes"] == []
        assert again.summary["arms"] == live.summary_arms
        assert fault_codes(again) == fault_codes_live(live)


def test_eval_gating_skips_imu_gates_for_a_take_without_imu(tmp_path):
    live = record_take(tmp_path, seconds=15.0, watch=False)
    table = gating.evaluate([live.dir], out=tmp_path / "gating.md")  # default gates
    rows = table.set_index("gate")
    assert list(table["gate"]) == ["always_on", "energy"]
    assert rows.loc["always_on", "reps"] > 0
    assert "no IMU samples" in rows.loc["energy", "error"]
    every = gating.evaluate(
        [live.dir], gates=("always_on", "energy", "laptop", "device"), out=tmp_path / "g2.md"
    )
    assert every["error"].notna().sum() == 3

    # a take the replay refuses (pose and frames disagree) is an error row per gate, not a crash
    broken = tmp_path / "broken"
    shutil.copytree(live.dir, broken)
    frames = pd.read_parquet(broken / "frames.parquet")
    frames.iloc[:-1].to_parquet(broken / "frames.parquet", index=False)
    table = gating.evaluate([broken], gates=("always_on",), out=tmp_path / "g3.md")
    assert "frames" in table.loc[0, "error"]


def test_eval_gating_judges_a_take_with_its_own_rules_unless_rules_are_given(tmp_path, monkeypatch):
    from formcoach.rules.engine import DEFAULT_RULES

    live = record_take(
        tmp_path, seconds=15.0, rules_file=_custom_rules(tmp_path / "custom.yaml"),
        rules_name="custom.yaml",
    )  # fmt: skip
    codes: list[list[str]] = []
    real = gating.run_session

    def spy(session, gate, rules):
        row = real(session, gate, rules)
        codes.append(row["_faults"])
        return row

    monkeypatch.setattr(gating, "run_session", spy)
    own = gating.evaluate([live.dir], gates=("device",), out=tmp_path / "own.md")
    given = gating.evaluate(
        [live.dir], gates=("device",), out=tmp_path / "given.md", rules_path=DEFAULT_RULES
    )
    reps = int(own.loc[0, "reps"])
    assert reps > 0 and codes[0].count("CURL_PARTIAL_ROM") == reps  # the take's rules
    assert "CURL_PARTIAL_ROM" not in codes[1]  # rules.yaml, as --rules asked
    assert int(given.loc[0, "reps"]) == reps


# ---- session check -----------------------------------------------------------------------------


def _relabel(folder: Path, **meta_keys) -> None:
    path = folder / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(meta_keys)
    path.write_text(json.dumps(meta), encoding="utf-8")


def test_session_check_compares_per_arm_counts(take, tmp_path):
    n = take.summary_arms["l"]["reps"]
    assert take.summary_arms["r"]["reps"] == n  # both arms curl together
    folder = tmp_path / "take"
    shutil.copytree(take.dir, folder)

    _relabel(folder, arm="both", expected_reps=n)  # both arms n of n
    report = session_check.check(folder)
    assert report["reps_by_arm"] == {"l": n, "r": n}
    assert report["reps"] == 2 * n  # every rep event, as before
    assert report["problems"] == []

    _relabel(folder, arm="l", expected_reps=n + 2)  # one labelled arm, miscounted
    report = session_check.check(folder)
    assert len(report["problems"]) == 1
    assert "left" in report["problems"][0] and f"expected {n + 2}" in report["problems"][0]

    _relabel(folder, arm="both", expected_reps=n + 2)  # each arm is checked
    problems = session_check.check(folder)["problems"]
    assert len(problems) == 2 and "left" in problems[0] and "right" in problems[1]
    assert "by arm" in session_check.format_report(session_check.check(folder))


def test_session_check_accepts_a_take_without_imu(tmp_path, recwarn):
    live = record_take(tmp_path, seconds=15.0, watch=False)
    n = live.summary_arms["l"]["reps"]
    _relabel(live.dir, expected_reps=n)
    report = session_check.check(live.dir)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert report["n_samples"] == 0 and report["gate_open_pct"] is None
    assert report["notes"] == ["no IMU samples"]
    assert report["problems"] == []
    json.dumps(report, allow_nan=False)
    assert live.check["notes"] == ["no IMU samples"]  # what Keep showed


def test_session_check_legacy_session_unchanged():
    report = session_check.check(FIXTURE_SESSION)
    assert report == {
        "session": str(FIXTURE_SESSION),
        "subject": "P2",
        "exercise": "curl",
        "n_samples": 3089,
        "duration_s": pytest.approx(29.984),
        "rate_hz": pytest.approx(103.02161152614676),
        "dropped": None,
        "gate_open_pct": None,
        "pose_frames": 898,
        "frames_processed_pct": None,
        "reps": None,
        "expected_reps": 10,
        "faults": None,
        "problems": [],
    }
    assert session_check.format_report(report).splitlines()[1:] == [
        "subject  P2   exercise curl",
        "samples  3089   duration 30.0 s   rate 103.0 Hz",
        "dropped  n/a   gate open n/a",
        "pose     frames 898   processed n/a",
        "reps     n/a (expected 10)   faults n/a",
        "problems none",
    ]


# ---- CLI ---------------------------------------------------------------------------------------


def test_cli_demo_replay_of_a_take_prints_per_arm_totals(take):
    runner = CliRunner()
    for extra in ([], ["--headless"]):
        result = runner.invoke(
            app, ["demo", "--source", "replay", "--session", str(take.dir), *extra]
        )
        assert result.exit_code == 0, result.output
        out = re.sub(r"\s+", " ", result.output)
        assert "as recorded" in out
        for arm in ("l", "r"):
            assert f"{arm.upper()}: {take.summary_arms[arm]['reps']} reps" in out

    forced = runner.invoke(
        app,
        ["demo", "--source", "replay", "--session", str(take.dir), "--headless",
         "--no-as-recorded"],
    )  # fmt: skip
    assert forced.exit_code == 0, forced.output
    assert "as recorded" not in forced.output and "(imu)" in forced.output

    bad = runner.invoke(app, ["demo", "--source", "replay", "--headless", "--gate", "recorded"])
    assert bad.exit_code == 2 and "recorded" in bad.output
