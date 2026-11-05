"""The live loop (plan Task 8): fake frames + a fake detector, no camera, no model, no cv2."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from formcoach.app import hud, live
from formcoach.io import protocol
from formcoach.rules.engine import RuleEngine

from .synthetic_pose import curl_deg, pose_frame

FPS = 30.0
BLANK = np.zeros((48, 64, 3), np.uint8)


def frames(seconds: float, board=None):
    """``(i, t, frame)`` at 30 fps; ``board.now`` follows the frame time for fake boards."""
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if board is not None:
            board.now = t
        yield i, t, BLANK


def make_detect(left=curl_deg, right=curl_deg, *, missing=(), vis=(1.0, 1.0)):
    """A detector keyed by frame time: ``left``/``right`` map t (s) -> elbow degrees; frames whose
    t falls in a ``missing`` ``(a, b)`` window return ``(None, None)`` (no person)."""

    def detect(_frame, t_ms):
        t = t_ms / 1000.0
        if any(a <= t < b for a, b in missing):
            return None, None
        pf = pose_frame(0, t, left(t), right(t), vis_l=vis[0], vis_r=vis[1])
        img = np.zeros((33, 4), np.float32)
        img[:, 0] = 0.5 + 0.5 * pf.world[:, 0]
        img[:, 1] = 0.5 - 0.5 * pf.world[:, 1]
        img[:, 3] = pf.visibility
        return img, pf.world.astype(np.float32)

    return detect


def idle(_t):
    return 170.0


# ---- helpers ---------------------------------------------------------------------------------


def test_pose_frame_from_detection_valid_and_missing():
    img = np.random.default_rng(0).random((33, 4)).astype(np.float32)
    world = np.random.default_rng(1).random((33, 3)).astype(np.float32)
    ok = live.pose_frame_from_detection(7, 1.5, img, world)
    assert ok.valid and ok.frame == 7 and ok.t == 1.5
    assert ok.world.shape == (33, 3) and np.allclose(ok.world, world)
    assert np.allclose(ok.visibility, img[:, 3]) and ok.image is not None

    for args in ((None, None), (img, None), (None, world)):
        bad = live.pose_frame_from_detection(8, 1.6, *args)
        assert not bad.valid and bad.image is None
        assert bad.world.shape == (33, 3) and np.isnan(bad.world).all()
        assert bad.visibility.shape == (33,) and not bad.visibility.any()


def test_looks_black():
    assert live.looks_black([0.5] * 30)
    assert not live.looks_black([0.5] * 29)
    assert not live.looks_black([0.5] * 29 + [50.0])
    assert not live.looks_black([])


def test_totals_lines_per_arm_and_squat():
    totals = {
        "l": {"reps": 7, "good": 6, "flagged": 1, "faults_by_code": {"CURL_TOO_FAST": 1}},
        "r": {"reps": 1, "good": 1, "flagged": 0, "faults_by_code": {}},
    }
    left, right = live.totals_lines("curl", totals)
    assert left == "L: 7 reps (6 good, 1 flagged)  CURL_TOO_FAST x1"
    assert right == "R: 1 rep (1 good, 0 flagged)"
    squat = live.totals_lines(
        "squat", {"all": {"reps": 3, "good": 3, "flagged": 0, "faults_by_code": {}}}
    )
    assert squat == ["all: 3 reps (3 good, 0 flagged)"]


# ---- counting and logging --------------------------------------------------------------------


def test_run_live_counts_each_arm_and_logs(tmp_path):
    out = tmp_path / "run"
    res = live.run_live(
        frames(25), make_detect(), exercise="curl", rules=RuleEngine.from_yaml(), out_dir=out,
        say=lambda _s: None,
    )  # fmt: skip
    arms = res.summary["arms"]
    assert 8 <= arms["l"]["reps"] <= 10 and 8 <= arms["r"]["reps"] <= 10
    assert res.summary["live"] is True and res.summary["gate"] == "always_on"
    assert res.summary["board"] is None and res.summary["fps_mean"] == pytest.approx(30.0, abs=0.5)
    for name in ("events.parquet", "frames.parquet", "session.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "session.json").read_text())["arms"] == arms
    assert res.out_dir == out
    reps = [e for e in res.events if e.kind == "rep"]
    assert {e.payload["arm"] for e in reps} == {"l", "r"}


def test_run_live_logs_the_counter_state_for_both_arms(tmp_path):
    """The live frames.parquet carries what each arm's segmenter saw, so a run that missed reps
    can be explained from the log (the idle arm stays unarmed, the moving arm arms)."""
    out = tmp_path / "run"
    live.run_live(
        frames(25), make_detect(right=idle), exercise="curl", rules=RuleEngine.from_yaml(),
        out_dir=out, say=lambda _s: None,
    )  # fmt: skip
    fr = pd.read_parquet(out / "frames.parquet")
    for col in ("ang_l", "enter_l", "exit_l", "armed_l", "vis_l", "armed_r", "vis_r"):
        assert col in fr.columns, col
    assert fr["armed_l"].iloc[-1] == 1.0
    assert (fr["armed_r"] == 0.0).all()


def test_run_live_only_one_arm_leaves_the_other_at_zero():
    res = live.run_live(
        frames(25), make_detect(right=idle), exercise="curl", rules=RuleEngine.from_yaml(),
        say=lambda _s: None,
    )  # fmt: skip
    arms = res.summary["arms"]
    assert 8 <= arms["l"]["reps"] <= 10
    assert arms["r"] == {"reps": 0, "good": 0, "flagged": 0, "faults_by_code": {}}
    codes = {f["code"] for f in (f for e in res.events if e.kind == "fault" for f in [e.payload])}
    assert "CURL_ASYMMETRY" not in codes


def test_run_live_prints_totals_at_exit():
    said: list[str] = []
    live.run_live(frames(10), make_detect(), exercise="curl", say=said.append)
    assert any(s.startswith("L: ") for s in said) and any(s.startswith("R: ") for s in said)


def test_run_live_writes_log_and_totals_when_frames_raise(tmp_path):
    def dying():
        yield from frames(5)
        raise RuntimeError("camera 1 stopped sending frames")

    said: list[str] = []
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="stopped sending frames"):
        live.run_live(dying(), make_detect(), exercise="curl", out_dir=out, say=said.append)
    assert (out / "session.json").exists()
    assert any(s.startswith("L: ") for s in said)


def test_run_live_writes_no_log_when_it_fails_before_the_first_frame(tmp_path):
    """A video that cannot be opened never yields a frame: there is nothing to log, so no empty
    session directory is left behind and no zero-rep totals are printed."""

    def unopenable():
        raise FileNotFoundError("cannot open video clip.mp4")
        yield  # makes this a generator, like landmarker.open_video

    said: list[str] = []
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="cannot open video"):
        live.run_live(unopenable(), make_detect(), exercise="curl", out_dir=out, say=said.append)
    assert not out.exists()
    assert said == []


def test_run_live_writes_log_when_the_detector_raises(tmp_path):
    def boom(_frame, _t_ms):
        raise ValueError("model exploded")

    out = tmp_path / "run"
    with pytest.raises(ValueError, match="model exploded"):
        live.run_live(frames(3), boom, exercise="curl", out_dir=out, say=lambda _s: None)
    assert (out / "session.json").exists()


def test_run_live_ctrl_c_ends_normally(tmp_path):
    def interrupted():
        yield from frames(2)
        raise KeyboardInterrupt

    out = tmp_path / "run"
    res = live.run_live(
        interrupted(), make_detect(), exercise="curl", out_dir=out, say=lambda _s: None
    )
    assert res.frames_total == 60 and (out / "session.json").exists()


def test_run_live_quits_when_show_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(hud, "render", lambda frame, _lm, _state, **_kw: frame)
    seen = []

    def show(frame):
        seen.append(frame)
        return len(seen) < 10

    out = tmp_path / "run"
    res = live.run_live(
        frames(5), make_detect(), exercise="curl", show=show, out_dir=out, say=lambda _s: None
    )
    assert res.frames_total == 10 and len(seen) == 10
    assert (out / "session.json").exists()


def test_run_live_zero_reps_still_writes_zeroed_arms(tmp_path):
    out = tmp_path / "run"
    res = live.run_live(
        frames(5), make_detect(left=idle, right=idle), exercise="curl", out_dir=out,
        say=lambda _s: None,
    )  # fmt: skip
    zero = {"reps": 0, "good": 0, "flagged": 0, "faults_by_code": {}}
    assert res.summary["arms"] == {"l": zero, "r": zero}
    session = json.loads((out / "session.json").read_text())
    assert session["arms"] == {"l": zero, "r": zero}
    assert (out / "events.parquet").exists() and (out / "frames.parquet").exists()


def test_run_live_survives_a_long_no_person_gap_and_resumes():
    res = live.run_live(
        frames(25), make_detect(missing=[(8.0, 16.0)]), exercise="curl",
        rules=RuleEngine.from_yaml(), say=lambda _s: None,
    )  # fmt: skip
    assert 0 < res.summary["frames_invalid"] <= res.summary["frames_total"]
    arms = res.summary["arms"]
    assert arms["l"]["reps"] >= 6 and arms["r"]["reps"] >= 6
    late = [e for e in res.events if e.kind == "rep" and e.t > 17.0]
    assert late, "counting must resume after the person comes back"


def test_hidden_arm_counts_nothing_and_the_other_keeps_going():
    res = live.run_live(
        frames(20), make_detect(vis=(1.0, 0.2)), exercise="curl", rules=RuleEngine.from_yaml(),
        say=lambda _s: None,
    )  # fmt: skip
    assert res.summary["arms"]["r"]["reps"] == 0
    assert res.summary["arms"]["l"]["reps"] >= 6


def test_black_camera_hint_is_said_once():
    said: list[str] = []
    black = (np.zeros((48, 64, 3), np.uint8) for _ in range(40))
    live.run_live(
        ((i, i / FPS, f) for i, f in enumerate(black)), make_detect(), exercise="curl",
        say=said.append, black_hint="camera 0 looks black",
    )  # fmt: skip
    assert said.count("camera 0 looks black") == 1


def test_a_lit_camera_gets_no_hint():
    said: list[str] = []
    lit = ((i, i / FPS, np.full((48, 64, 3), 90, np.uint8)) for i in range(40))
    live.run_live(
        lit, make_detect(), exercise="curl", say=said.append, black_hint="camera 0 looks black"
    )
    assert "camera 0 looks black" not in said


# ---- the wearable ----------------------------------------------------------------------------


class FakeBoard:
    """BoardLink-shaped: gate closed for 10 s, open 10-20 s, then silent (gate closed)."""

    name = "FormCoach-8428"

    def __init__(self, open_between=(10.0, 20.0), send_raises=False):
        self.now = 0.0
        self.open_between = open_between
        self.send_raises = send_raises
        self.sent: list[protocol.Command] = []

    def poll(self):
        return []

    def gate_open(self):
        a, b = self.open_between
        return a <= self.now < b

    def state(self):
        if self.now < self.open_between[0]:
            return "waiting"
        return "ok" if self.gate_open() else "silent"

    def send(self, cmd):
        if self.send_raises:
            raise OSError("link lost")
        self.sent.append(cmd)


def test_board_gate_follows_the_flag_and_goes_silent():
    board = FakeBoard()
    res = live.run_live(
        frames(30, board), make_detect(), exercise="curl", board=board, say=lambda _s: None
    )
    assert res.frames_processed < res.frames_total
    assert res.frames_processed == pytest.approx(10 * FPS, abs=2)
    kinds = [e.kind for e in res.events if e.kind.startswith("gate_")]
    assert kinds == ["gate_open", "gate_close"]
    assert res.summary["gate"] == "device" and res.summary["board"] == "FormCoach-8428"


class _Fault:
    def as_dict(self):
        return {"code": "TEST_FAULT", "severity": "warn", "message": "fix it"}


class FlagSecondRep:
    """Rules stand-in: the second rep of every arm gets a fault, all others are clean."""

    def evaluate(self, rm, context=None):
        return [_Fault()] if rm.rep_id == 1 else []


def test_a_headless_run_says_what_the_wearable_is_doing():
    """No window means no status pill, so the same text is printed when it changes."""
    board = FakeBoard()
    said: list[str] = []
    live.run_live(frames(30, board), make_detect(), exercise="curl", board=board, say=said.append)
    assert [s for s in said if s.startswith("wearable:")] == [
        "wearable: looking for board... | pose paused",
        "wearable: FormCoach-8428 | gate ON | pose running",
        "wearable: board silent - reconnecting",
    ]


def test_no_wearable_lines_without_a_board_or_with_a_window(monkeypatch):
    monkeypatch.setattr(hud, "render", lambda frame, _lm, _state, **_kw: frame)
    said: list[str] = []
    live.run_live(frames(5), make_detect(), exercise="curl", say=said.append)
    board = FakeBoard()
    live.run_live(
        frames(5, board), make_detect(), exercise="curl", board=board, show=lambda _f: True,
        say=said.append,
    )  # fmt: skip
    assert not [s for s in said if s.startswith("wearable:")]


def test_led_commands_follow_rep_quality():
    board = FakeBoard(open_between=(0.0, 1e9))
    res = live.run_live(  # the clock follows frame time: the fault-flash guard is in wall seconds
        frames(25, board), make_detect(right=idle), exercise="curl", rules=FlagSecondRep(),
        board=board, say=lambda _s: None, clock=lambda: board.now,
    )  # fmt: skip
    reps = res.summary["arms"]["l"]["reps"]
    assert reps >= 8
    assert board.sent.count(protocol.Command.LED_FAULT) == 1
    assert board.sent.count(protocol.Command.LED_GOOD_REP) == reps - 1


FIRMWARE_FAULT_FLASH_S = 0.8  # firmware/src/app/ui.cpp: a fault flash lasts 800 ms


class TimedBoard(FakeBoard):
    """A always-open FakeBoard that also records the (wall) time of every command it is sent."""

    def __init__(self):
        super().__init__(open_between=(0.0, 1e9))
        self.log: list[tuple[float, protocol.Command]] = []

    def send(self, cmd):
        self.log.append((self.now, cmd))
        super().send(cmd)


class FlagLeftSecondRep:
    """Rules stand-in: only the LEFT arm's second rep gets a fault."""

    def evaluate(self, rm, context=None):
        return [_Fault()] if rm.arm == "l" and rm.rep_id == 1 else []


@pytest.mark.parametrize(
    "right_lag_frames", [-1, 0, 1], ids=["right-1-early", "same", "right-1-late"]
)
def test_a_clean_arm_finishing_with_a_faulty_one_hides_neither_banner_nor_led(
    monkeypatch, right_lag_frames
):
    """Both arms curl together; the left arm's second rep has a fault, the right arm's is clean.
    Whether the right arm finishes on the same frame or one frame either side, the left arm's
    fault text stays on screen next to the right arm's "Good rep", and the device is not sent a
    good-rep command while the fault flash (0.8 s) is still showing."""
    states: list[hud.HudState] = []

    def fake_render(frame, _lm, state, **_kw):
        states.append(state)
        return frame

    monkeypatch.setattr(hud, "render", fake_render)
    board = TimedBoard()
    shift = right_lag_frames / FPS
    live.run_live(
        frames(12, board), make_detect(right=lambda t: curl_deg(t - shift)), exercise="curl",
        rules=FlagLeftSecondRep(), board=board, show=lambda _f: True, say=lambda _s: None,
        clock=lambda: board.now,
    )  # fmt: skip

    faults = [t for t, cmd in board.log if cmd == protocol.Command.LED_FAULT]
    assert len(faults) == 1
    flash = [cmd for t, cmd in board.log if faults[0] <= t < faults[0] + FIRMWARE_FAULT_FLASH_S]
    assert flash == [protocol.Command.LED_FAULT], "a good-rep command would cancel the fault flash"

    both = [
        s for s in states
        if s.banners.get("l") is not None and s.banners["l"].text == "L rep 2: fix it"
        and s.banners.get("r") is not None and s.banners["r"].text == "R rep 2: Good rep"
    ]  # fmt: skip
    assert both, "the left arm's fault must be on screen together with the right arm's banner"
    shown = [s for s in states if "l" in s.banners and s.banners["l"].text == "L rep 2: fix it"]
    # it stays until the left arm's own next rep (2.5 s later) replaces it; with one shared banner
    # the right arm's banner took its place after at most a frame
    assert len(shown) >= 2 * FPS, "the fault banner must stay up on its own"


def test_a_good_rep_command_is_sent_again_once_the_fault_flash_is_over():
    board = TimedBoard()
    live.run_live(
        frames(25, board), make_detect(right=idle), exercise="curl", rules=FlagLeftSecondRep(),
        board=board, say=lambda _s: None, clock=lambda: board.now,
    )  # fmt: skip
    (fault_t,) = [t for t, cmd in board.log if cmd == protocol.Command.LED_FAULT]
    later = [t for t, cmd in board.log if cmd == protocol.Command.LED_GOOD_REP and t > fault_t]
    assert later and min(later) - fault_t >= FIRMWARE_FAULT_FLASH_S


def test_a_board_whose_send_raises_does_not_stop_the_loop():
    board = FakeBoard(open_between=(0.0, 1e9), send_raises=True)
    res = live.run_live(
        frames(15, board), make_detect(), exercise="curl", board=board, say=lambda _s: None
    )
    assert res.summary["arms"]["l"]["reps"] >= 4


# ---- the window ------------------------------------------------------------------------------


def test_hud_state_wiring(monkeypatch):
    states: list[hud.HudState] = []

    def fake_render(frame, _lm, state, **_kw):
        states.append(state)
        return frame

    monkeypatch.setattr(hud, "render", fake_render)
    live.run_live(
        frames(12), make_detect(), exercise="curl", show=lambda _f: True, say=lambda _s: None,
        rules=RuleEngine.from_yaml(),
    )  # fmt: skip
    assert len(states) == 12 * 30
    last = states[-1]
    assert last.pill_text == "webcam only | gate always on" and not last.dim
    assert last.totals["l"]["reps"] >= 3
    assert set(last.angles) == {"l", "r"}
    assert all(v is None or np.isfinite(v) for s in states for v in s.angles.values())
    assert last.bands["l"][0] < last.bands["l"][1]
    shown = [b.text for s in states for b in s.banners.values()]
    assert shown and shown[0].startswith(("L rep 1", "R rep 1"))


def test_headless_run_never_renders(monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("render must not be called without a window")

    monkeypatch.setattr(hud, "render", explode)
    res = live.run_live(frames(3), make_detect(), exercise="curl", say=lambda _s: None)
    assert res.frames_total == 90


def test_the_hud_renders_real_frames_when_opencv_is_present():
    pytest.importorskip("cv2")
    out: list[np.ndarray] = []
    live.run_live(
        ((i, t, np.full((240, 320, 3), 60, np.uint8)) for i, t, _ in frames(4)),
        make_detect(), exercise="curl", show=lambda f: out.append(f) or True,
        say=lambda _s: None,
    )  # fmt: skip
    assert len(out) == 120 and out[0].shape == (240, 320, 3) and out[0].dtype == np.uint8


# ---- the live demo's default rules (ADR-0033) ------------------------------------------------


def _demo(tmp_path, name, **kw):
    """``run_live_demo`` over a fake camera and detector, headless, logging to ``tmp_path/name``."""
    from formcoach.pose import camera as camera_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(camera_mod, "open_camera", lambda index, **_kw: frames(2))
        return live.run_live_demo(
            camera=0, video=None, board=None, port=None, exercise="curl", headless=True,
            out_dir=tmp_path / name, say=lambda _s: None, detect=make_detect(), **kw,
        )  # fmt: skip


def test_the_live_demo_defaults_to_the_webcam_rules_and_logs_which_file(tmp_path):
    from formcoach.rules.engine import DEFAULT_RULES

    _demo(tmp_path, "default")
    _demo(tmp_path, "explicit", rules=DEFAULT_RULES)
    assert json.loads((tmp_path / "default/session.json").read_text())["rules"] == (
        "rules.mediapipe-webcam.yaml"
    )
    assert json.loads((tmp_path / "explicit/session.json").read_text())["rules"] == "rules.yaml"


def _judged(top, bottom, rules_path):
    """Both arms curl ``top``..``bottom`` degrees for 25 s; returns the left arm's totals."""
    f = _curl(top, bottom)
    res = live.run_live(
        frames(25), make_detect(left=f, right=f), exercise="curl",
        rules=RuleEngine.from_yaml(rules_path), say=lambda _s: None,
    )  # fmt: skip
    return res.summary["arms"]["l"]


def test_a_webcam_range_curl_is_good_under_the_webcam_rules_but_flagged_by_the_defaults():
    """A curl that a webcam reads as ~88-150 deg (the real runs of 2026-11-04) is flagged as a
    partial range by the design defaults and is a good rep under the live webcam rules."""
    from formcoach.rules.engine import DEFAULT_RULES, WEBCAM_RULES

    webcam = _judged(88, 150, WEBCAM_RULES)
    assert webcam["reps"] >= 8 and webcam["good"] == webcam["reps"]
    default = _judged(88, 150, DEFAULT_RULES)
    assert default["faults_by_code"] == {"CURL_PARTIAL_ROM": default["reps"]}


def test_a_rep_that_barely_bends_the_elbow_is_still_flagged_by_the_webcam_rules():
    from formcoach.rules.engine import WEBCAM_RULES

    arm = _judged(125, 170, WEBCAM_RULES)
    assert arm["reps"] >= 8 and arm["faults_by_code"]["CURL_PARTIAL_ROM"] == arm["reps"]


# ---- Pipeline accessors the HUD relies on ----------------------------------------------------


def test_pipeline_latest_angles_and_thresholds():
    from formcoach.app.pipeline import Pipeline

    pipe = Pipeline(exercise="curl", per_arm=True)
    assert all(np.isnan(v) for v in pipe.latest_angles().values())
    pipe.set_gate(True, 0.0)
    pipe.on_frame(pose_frame(0, 0.0, 90.0, 120.0))
    angles = pipe.latest_angles()
    assert angles["l"] == pytest.approx(90.0, abs=0.5)
    assert angles["r"] == pytest.approx(120.0, abs=0.5)
    th = pipe.thresholds()
    assert th["l"] == (60.0, 150.0) and th["r"] == (60.0, 150.0)
    pipe.set_gate(False, 1.0)  # a closed gate clears the buffers
    assert all(np.isnan(v) for v in pipe.latest_angles().values())


# ---- the metrics window: clean curls must be able to be "good" -------------------------------


def _curl(top, bottom, period=2.5):
    return lambda t: curl_deg(t, top=top, bottom=bottom, period_s=period)


@pytest.mark.parametrize("top,bottom,period", [(50, 150, 2.5), (35, 165, 2.5), (40, 160, 4.0)])
def test_clean_curls_are_good_reps_not_partial_rom(top, bottom, period):
    f = _curl(top, bottom, period)
    res = live.run_live(
        frames(25), make_detect(left=f, right=f), exercise="curl", rules=RuleEngine.from_yaml(),
        say=lambda _s: None,
    )  # fmt: skip
    for arm in ("l", "r"):
        a = res.summary["arms"][arm]
        assert a["reps"] >= 5 and a["good"] == a["reps"], (arm, a)
        assert a["faults_by_code"] == {}


@pytest.mark.parametrize("top,bottom", [(90, 130), (40, 100)])
def test_partial_curls_are_still_flagged(top, bottom):
    f = _curl(top, bottom)
    res = live.run_live(
        frames(25), make_detect(left=f, right=f), exercise="curl", rules=RuleEngine.from_yaml(),
        say=lambda _s: None,
    )  # fmt: skip
    a = res.summary["arms"]["l"]
    assert a["reps"] >= 8 and a["faults_by_code"]["CURL_PARTIAL_ROM"] == a["reps"]


def test_padding_the_metrics_window_changes_no_count_or_timing():
    from formcoach.app.pipeline import Pipeline

    def run(pad):
        pipe = Pipeline(exercise="curl", per_arm=True, pad_rep_start_s=pad)
        pipe.set_gate(True, 0.0)
        for i in range(int(25 * FPS)):
            t = i / FPS
            pipe.on_frame(pose_frame(i, t, curl_deg(t, top=35, bottom=165), idle(t)))
        return [r for r in pipe.finish(compute_imu=False).reps if r.arm == "l"]

    plain, padded = run(0.0), run(1.0)
    assert len(plain) == len(padded) >= 8
    for a, b in zip(plain, padded, strict=True):
        assert (a.t_start, a.t_end, a.duration_s) == (b.t_start, b.t_end, b.duration_s)
        assert b.metrics["elbow_max"] > a.metrics["elbow_max"] + 15
        # the same rep resampled to 30 steps over a longer window: the min moves a little
        assert b.metrics["elbow_min"] == pytest.approx(a.metrics["elbow_min"], abs=1.0)


# ---- the cleanup path ------------------------------------------------------------------------


def test_a_failing_log_write_does_not_mask_the_loops_own_error(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the log directory should go")

    def dying():
        yield from frames(2)
        raise RuntimeError("camera 1 stopped sending frames")

    said: list[str] = []
    with pytest.raises(RuntimeError, match="stopped sending frames"):
        live.run_live(dying(), make_detect(), exercise="curl", out_dir=blocker, say=said.append)
    assert any(s.startswith("L: ") for s in said), "totals are still printed"
    assert any("could not write the session log" in s for s in said)


def test_a_failing_log_write_on_a_good_run_is_reported_after_the_totals(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    said: list[str] = []
    with pytest.raises(OSError):
        live.run_live(frames(2), make_detect(), exercise="curl", out_dir=blocker, say=said.append)
    assert any(s.startswith("R: ") for s in said)
