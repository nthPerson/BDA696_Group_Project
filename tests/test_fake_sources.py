"""The no-hardware sources: FakeScene (camera + pose), FakeWatch (IMU) and the firmware's
``# infer`` log-line parser."""

from __future__ import annotations

import numpy as np
import pytest

from formcoach.app.session import LiveSession
from formcoach.io import protocol
from formcoach.io.fake import FakeWatch, default_schedule
from formcoach.io.serial_source import InferInfo, LineDecoder, parse_infer_line
from formcoach.pose.angles import joint_angles
from formcoach.pose.skeletons import MEDIAPIPE33
from formcoach.pose.synthetic import FakeScene
from formcoach.pose.synthetic import default_schedule as scene_default_schedule


class FakeClock:
    """A clock that only moves when ``sleep`` is called."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_default_schedule_is_shared_and_follows_the_30_second_rule():
    assert scene_default_schedule is default_schedule
    assert [default_schedule(t) for t in (0, 4.99, 5, 24.99, 25, 35)] == [
        False, False, True, True, False, True,
    ]  # fmt: skip


def test_fake_scene_curls_are_counted_by_a_live_session():
    clock = FakeClock()
    scene = FakeScene(exercise="curl", clock=clock, sleep=clock.sleep, duration_s=30)
    session = LiveSession(exercise="curl", detect=scene.detect, clock=clock)
    session.run(scene.frames())
    result = session.finish()
    per_arm = {arm: sum(r.arm == arm for r in result.reps) for arm in ("l", "r")}
    assert per_arm == {"l": 8, "r": 8}
    assert result.frames_total == 900


@pytest.mark.parametrize("exercise,keys", [("press", "lr"), ("raise", "lr"), ("squat", ("all",))])
def test_fake_scene_other_exercises_are_counted(exercise, keys):
    clock = FakeClock()
    scene = FakeScene(exercise=exercise, clock=clock, sleep=clock.sleep, duration_s=30)
    session = LiveSession(exercise=exercise, detect=scene.detect, clock=clock)
    session.run(scene.frames())
    reps = session.finish().reps
    for key in keys:
        assert sum(r.arm == (None if key == "all" else key) for r in reps) == 8, key


def test_fake_scene_landmarks_are_normalised_and_visible():
    clock = FakeClock()
    scene = FakeScene(clock=clock, sleep=clock.sleep, duration_s=30)
    for t_ms in (0, 7000, 8000, 8625, 40000):
        clock.now = 100.0 + t_ms / 1000
        img, world = scene.detect(None, t_ms)
        assert img.shape == (33, 4) and world.shape == (33, 3)
        assert img[:, :2].min() >= 0.0 and img[:, :2].max() <= 1.0
        assert img[:, 3].min() >= 0.9
        ys = img[[23, 24, 27, 28], 1]  # hips, ankles: figure spans the frame, roughly centred
        assert 0.1 < img[11, 1] < 0.4 and ys.max() < 0.9
        assert abs(img[:, 0].mean() - 0.5) < 0.05


def elbow_and_shoulder_deg(world: np.ndarray) -> tuple[float, float, float]:
    """Left elbow, left shoulder-abduction and left knee angles (degrees) of one figure."""
    a = joint_angles(world[None].astype(float), MEDIAPIPE33, np.array([0.0, 1.0, 0.0]))
    return float(a["elbow_l"][0]), float(a["shoulder_abd_l"][0]), float(a["knee_l"][0])


def test_fake_scene_stands_still_off_schedule_and_moves_on_it():
    clock = FakeClock()
    scene = FakeScene(clock=clock, sleep=clock.sleep)
    scene._t0 = 0.0  # as if frames() had started at clock 0
    elbow = {}
    for label, now, t_ms in (("rest", 1, 1000), ("up", 6, 6000), ("down", 6, 7250)):
        clock.now = float(now)
        elbow[label] = elbow_and_shoulder_deg(scene.detect(None, t_ms)[1])[0]
    assert elbow["rest"] == pytest.approx(150.0, abs=0.1)  # standing: the curl's start position
    assert elbow["up"] == pytest.approx(150.0, abs=0.1)  # start of the movement: arm extended
    assert elbow["down"] == pytest.approx(50.0, abs=0.1)  # half a period later: curled


def test_fake_scene_world_angles_match_the_counting_joint():
    clock = FakeClock()
    clock.now = 6.0
    for exercise, index, far in (("raise", 1, 95.0), ("squat", 2, 80.0), ("press", 0, 60.0)):
        scene = FakeScene(exercise=exercise, clock=clock, sleep=clock.sleep)
        scene._t0 = 0.0
        scene.detect(None, 6000)  # starts the movement
        world = scene.detect(None, 7250)[1]  # half a period on: far end
        assert elbow_and_shoulder_deg(world)[index] == pytest.approx(far, abs=0.1), exercise


def test_fake_scene_frames_are_paced_and_end_at_duration():
    clock = FakeClock()
    scene = FakeScene(fps=10, size=(36, 64), clock=clock, sleep=clock.sleep, duration_s=2.0)
    frames = list(scene.frames())
    assert [f[0] for f in frames] == list(range(20))
    assert frames[3][1] == pytest.approx(0.3) and frames[0][2].shape == (36, 64, 3)
    assert not frames[0][2].any()
    assert clock.now - 100.0 == pytest.approx(1.9)


def test_fake_scene_rejects_unknown_exercise():
    with pytest.raises(ValueError, match="exercise"):
        FakeScene(exercise="deadlift")


def test_fake_watch_is_paced_and_endless_until_closed():
    clock = FakeClock()
    watch = FakeWatch(clock=clock, sleep=clock.sleep)
    it = watch.iter_samples()
    got = [next(it) for _ in range(100)]
    assert clock.now - 100.0 == pytest.approx(2.0)
    assert [s.seq for s in got] == list(range(100))
    assert all(s.t_arrival is not None for s in got)
    assert got[99].t == pytest.approx(1.98, abs=0.01)
    assert watch.stats.samples == 100 and watch.stats.dropped == 0
    next(it)  # still going: endless
    watch.close()
    assert list(watch.iter_samples()) == []  # closed stays closed


def test_fake_watch_close_ends_a_running_iterator():
    clock = FakeClock()
    watch = FakeWatch(clock=clock, sleep=clock.sleep)
    n = 0
    for _ in watch.iter_samples():
        n += 1
        if n == 12:
            watch.close()
    assert 12 <= n <= 15  # the rest of the batch in flight, then it stops


def test_fake_watch_duration_ends_the_stream():
    clock = FakeClock()
    watch = FakeWatch(clock=clock, sleep=clock.sleep, duration_s=1.0)
    assert len(list(watch.iter_samples())) == 50


def test_fake_watch_gate_flag_follows_the_schedule_with_hysteresis():
    clock = FakeClock()
    # exercising from 5 s to 15 s only
    watch = FakeWatch(schedule=lambda t: 5 <= t < 15, clock=clock, sleep=clock.sleep, duration_s=30)
    flags = {}
    for s in watch.iter_samples():
        flags[round(s.t, 2)] = bool(s.flags & protocol.Flag.GATE_STATE)
    # a batch is 0.1 s, so the edges are good to one batch
    assert not any(v for t, v in flags.items() if t < 5.85)  # nothing before 1 s of exercise
    assert all(flags[t] for t in (6.1, 10.0, 14.9, 15.5, 20.8))  # on, held 6 s past the end
    assert not any(v for t, v in flags.items() if t >= 21.0)  # then cleared


def test_fake_watch_signal_is_energetic_only_while_exercising():
    clock = FakeClock()
    watch = FakeWatch(schedule=lambda t: 5 <= t < 15, clock=clock, sleep=clock.sleep, duration_s=20)
    a = np.array([[s.t, s.a] for s in watch.iter_samples()])
    rest, work = a[a[:, 0] < 4, 1], a[(a[:, 0] > 6) & (a[:, 0] < 14), 1]
    assert rest.var() < 0.01
    assert work.var() > protocol.GATE_ENERGY_THRESHOLD_MS2SQ


def test_fake_watch_records_commands_and_has_a_status():
    clock = FakeClock()
    watch = FakeWatch(clock=clock, sleep=clock.sleep)
    assert watch.device_name == "FormCoach-FAKE"
    watch.send(protocol.Command.LED_GOOD_REP)
    watch.send(protocol.Command.PING)
    assert watch.sent == [protocol.Command.LED_GOOD_REP, protocol.Command.PING]
    clock.now += 12.0
    st = watch.status
    assert st.firmware == "0.0.0" and st.calibrated == 1 and st.uptime_s == 12
    assert protocol.unpack_status(protocol.pack_status(st)) == st


def test_parse_infer_line():
    assert parse_infer_line("# infer us=13390 arena=6196 class=1 p_active=0.93") == InferInfo(
        13390, 6196, 1, 0.93
    )
    assert parse_infer_line("# pong") is None
    assert parse_infer_line("# infer us=oops") is None
    assert parse_infer_line("1,2,3") is None


def test_line_decoder_keeps_the_last_infer_and_still_counts_comments():
    d = LineDecoder()
    assert d.last_infer is None and d.infer_count == 0
    assert d.feed("# infer us=100 arena=10 class=0 p_active=0.10") is None
    assert d.feed("# infer us=200 arena=10 class=1 p_active=0.80") is None
    assert d.feed("# pong") is None
    assert d.last_infer == InferInfo(200, 10, 1, 0.80)
    assert d.infer_count == 2 and d.stats.comment_lines == 3 and d.stats.bad_lines == 0
    assert d.feed("100,1,2,3,4,5,6,0,7") is not None  # data lines are unaffected
