"""The Bench server (Bench plan Task 8): hub, controller, Starlette app, runner and the
``formcoach bench`` command.

The pieces come from :func:`formcoach.bench.runner.build` with ``--fake`` (a synthetic person
and a fake watch) on a manual clock: the test thread steps the session itself, one frame per
1/30 s of clock, and the fake watch's pump thread wakes as the clock advances. Pages are
Starlette ``TestClient`` WebSockets; every receive is bounded (:func:`recv`), so a broken
server fails a test instead of hanging it. No camera, port or radio is opened, except one
local port in the CLI's port-in-use check.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from formcoach.bench import protocol
from formcoach.cli import app as cli_app
from formcoach.io.protocol import Command

FPS = 30.0
# the server answers only to local host names (TestClient's default is "testserver"), and
# websocket_connect ignores base_url, so the socket URL is given whole
WS_URL = "ws://localhost/ws"
REP_STEPS = 330  # 11 s of fake scene: it starts curling at 5 s, the watch gate opens at 6 s


def _need_starlette():
    pytest.importorskip("starlette")
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    return functools.partial(TestClient, base_url="http://localhost")


class ManualClock:
    """A laptop clock (s) the test advances; ``sleep`` blocks until the clock has advanced
    that far (or the clock is closed), so paced fakes follow the test's steps."""

    def __init__(self, t: float = 1000.0):
        self.t = t
        self.closed = False
        self._cv = threading.Condition()

    def __call__(self) -> float:
        with self._cv:
            return self.t

    def advance(self, dt: float) -> None:
        with self._cv:
            self.t += dt
            self._cv.notify_all()

    def sleep(self, dt: float) -> None:
        with self._cv:
            target = self.t + dt - 1e-9
            self._cv.wait_for(lambda: self.closed or self.t >= target, timeout=5.0)

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()


class Rig:
    """``build(fake=True)`` on a :class:`ManualClock`, stepped by the test."""

    def __init__(self, tmp_path: Path, **kw):
        from formcoach.bench import runner

        self.clock = ManualClock()
        self.team = tmp_path / "team"
        self.bench = runner.build(
            camera=None, video=None, fake=True, exercise=kw.pop("exercise", "curl"),
            team_root=self.team, rules_dir_base=tmp_path, kits_file=tmp_path / "kits.json",
            clock=self.clock,
            sleep=self.clock.sleep, **kw,
        )  # fmt: skip
        self.session = self.bench.session
        self.frames = iter(self.bench.frames)
        self.steps = 0
        self.published: list[dict] = []  # every ordered message the controller publishes
        publish = self.bench.hub.publish

        def record(msg: dict) -> None:
            self.published.append(msg)
            publish(msg)

        self.bench.hub.publish = record

    def count(self, type_: str, kind: str | None = None) -> int:
        """Published messages of ``type_`` (and event ``kind``) so far."""
        return sum(
            m["type"] == type_ and (kind is None or m.get("kind") == kind)
            for m in tuple(self.published)
        )

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            index, t, frame = next(self.frames)
            self.session.step(index, t, frame)
            self.clock.advance(1.0 / FPS)
            self.steps += 1

    def step_until(self, done, limit: int = 60) -> None:
        """Step until ``done()`` (a command sent over the socket reaches the session at the
        start of a step after the server has queued it); fail after ``limit`` steps."""
        for _ in range(limit):
            if done():
                return
            self.step()
            time.sleep(0.002)  # let the server's thread queue a command just sent
        assert done(), f"not done after {limit} steps"

    def close(self) -> None:
        self.clock.close()
        self.bench.cleanup()


@pytest.fixture
def rig(tmp_path):
    _need_starlette()
    r = Rig(tmp_path)
    yield r
    r.close()


def recv(ws, timeout: float = 5.0) -> dict:
    """The next message: text as a dict; a binary frame as its header with ``"_jpeg"`` (the
    image bytes or ``None``) added. Fails the test after ``timeout`` seconds."""
    box: list = []

    def get() -> None:
        try:
            box.append(ws.receive())
        except BaseException as exc:
            box.append(exc)

    thread = threading.Thread(target=get, daemon=True)
    thread.start()
    thread.join(timeout)
    if not box:
        pytest.fail(f"no message from the server within {timeout} s")
    msg = box[0]
    if isinstance(msg, BaseException):
        raise msg
    if msg["type"] == "websocket.close":
        raise ConnectionError(f"socket closed with code {msg.get('code')}")
    if msg.get("bytes") is not None:
        header, jpeg = protocol.decode_frame(msg["bytes"])
        header["_jpeg"] = jpeg
        return header
    return json.loads(msg["text"])


def until(ws, pred, limit: int = 3000, seen: list | None = None) -> dict:
    """Receive until ``pred(msg)``; every message read goes into ``seen``."""
    for _ in range(limit):
        msg = recv(ws)
        if seen is not None:
            seen.append(msg)
        if pred(msg):
            return msg
    pytest.fail(f"no matching message in {limit} messages")


def published_link_state(rig: Rig) -> str | None:
    """The state in the latest ``link`` message published."""
    links = [m for m in tuple(rig.published) if m["type"] == "link"]
    return links[-1]["status"]["state"] if links else None


def is_state(msg, **want) -> bool:
    if msg["type"] not in ("state", "hello"):
        return False
    return all(msg["state"].get(k) == v for k, v in want.items())


def test_ws_sends_hello_with_state_then_frames_imu_and_a_rep_event(rig):
    TestClient = _need_starlette()
    has_cv2 = importlib.util.find_spec("cv2") is not None
    with TestClient(rig.bench.app) as client, client.websocket_connect(WS_URL) as ws:
        hello = recv(ws)
        assert hello["type"] == "hello" and hello["v"] == protocol.SCHEMA_VERSION
        st = hello["state"]
        assert set(st) == set(protocol.STATE_KEYS)
        assert st["source"] == "fake" and st["exercise"] == "curl"
        assert st["rules"] == "rules.mediapipe-webcam.yaml"
        assert st["gate_source"] == "device" and st["tuned_exercises"] == ["curl"]
        assert st["link"]["kind"] == "fake" and st["link"]["id"] == "fake"
        assert st["video"] == {"on": True, "width": 640}
        assert st["capture"]["state"] == "idle" and "clean" in st["forms"]
        assert st["params"] and all(p["key"] and "min" in p for p in st["params"])
        assert any(p["key"].startswith("rule.curl.") for p in st["params"])

        rig.step(REP_STEPS)
        seen: list[dict] = []
        rep = until(ws, lambda m: m["type"] == "event" and m["kind"] == "rep", seen=seen)

    assert rep["payload"]["arm"] in ("l", "r") and rep["t"] > 5.0
    frames = [m for m in seen if m["type"] == "frame"]
    assert frames, "no video frame arrived"
    assert all(set(m) - {"_jpeg"} == set(frames[0]) - {"_jpeg"} for m in frames)
    with_pose = [m for m in frames if m["lm"] is not None]
    assert with_pose and len(with_pose[0]["lm"]) == 33
    assert all(m["has_image"] is has_cv2 for m in frames)
    if has_cv2:
        assert frames[0]["_jpeg"][:2] == b"\xff\xd8"
    imu = [m for m in seen if m["type"] == "imu"]
    assert imu and all(len(m["t"]) == len(m["ax"]) == len(m["seq"]) for m in imu)
    assert all(t is not None and t >= 1000.0 for m in imu for t in m["t"])
    assert any(m["type"] == "link" and m["status"]["state"] == "ok" for m in seen)
    assert any(m["type"] == "event" and m["kind"] == "gate_open" for m in seen)


def test_two_clients_both_receive_and_a_stalled_client_does_not_slow_the_session(rig):
    TestClient = _need_starlette()
    from formcoach.bench.hub import ClientOverflowError

    hub = rig.bench.hub

    def step_times(n: int) -> float:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            rig.step()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    with (
        TestClient(rig.bench.app) as client,
        client.websocket_connect(WS_URL) as a,
        client.websocket_connect(WS_URL) as b,
    ):
        assert recv(a)["type"] == "hello" and recv(b)["type"] == "hello"
        rig.step(5)
        for ws in (a, b):
            assert until(ws, lambda m: m["type"] == "frame")["i"] >= 0

        before = step_times(60)
        # a page that never reads: registered on the server's loop, never asks for a message
        hub.max_queue = 200  # new clients only; a and b keep the default cap
        stalled = client.portal.call(hub.register)
        assert hub.client_count() == 3
        during = step_times(60)
        assert during < 3 * before + 0.002, (before, during)

        for i in range(300):  # more than the stalled client's cap
            hub.publish(protocol.log("info", f"m{i}", 0.0))
        got = []
        until(a, lambda m: m["type"] == "log" and m["text"] == "m299", seen=got)
        ours = [m["text"] for m in got if m["type"] == "log" and m["text"][1:].isdigit()]
        assert ours == [f"m{i}" for i in range(300)]  # all of them, in order
        until(b, lambda m: m["type"] == "log" and m["text"] == "m299")

        assert stalled.overflowed and hub.client_count() == 2  # dropped, queue freed
        with pytest.raises(ClientOverflowError):
            client.portal.call(stalled.next)
        rig.step(3)  # the others are still served
        assert until(a, lambda m: m["type"] == "frame")["i"] >= rig.steps - 3
        assert until(b, lambda m: m["type"] == "frame")["i"] >= rig.steps - 3


def test_new_client_gets_current_state_at_once(rig):
    TestClient = _need_starlette()
    with TestClient(rig.bench.app) as client:
        with client.websocket_connect(WS_URL) as a:
            recv(a)
            a.send_json({"cmd": "set_exercise", "exercise": "press"})
            rig.step_until(lambda: rig.session.exercise == "press")
            until(a, lambda m: is_state(m, exercise="press"))
            # no step in between: the second page's hello already has it
            with client.websocket_connect(WS_URL) as b:
                hello = recv(b)
        assert hello["type"] == "hello" and hello["state"]["exercise"] == "press"
        assert all(not p["key"].startswith("rule.curl.") for p in hello["state"]["params"])


def test_commands_change_the_session_and_publish_state(rig):
    TestClient = _need_starlette()
    s = rig.session
    with TestClient(rig.bench.app) as client, client.websocket_connect(WS_URL) as ws:
        hello = recv(ws)
        key = next(p["key"] for p in hello["state"]["params"] if p["key"].startswith("rule."))

        ws.send_json({"cmd": "set_gate_source", "source": "manual"})
        rig.step_until(lambda: s.gate_source == "manual")
        until(ws, lambda m: is_state(m, gate_source="manual"))

        ws.send_json({"cmd": "set_manual_gate", "open": True})
        rig.step_until(lambda: rig.bench.controller.state()["manual_gate"])
        until(ws, lambda m: is_state(m, manual_gate=True))
        rig.step(1)
        assert until(ws, lambda m: m["type"] == "frame")["gate"] is True

        ws.send_json({"cmd": "set_param", "key": key, "value": 0.01})
        rig.step_until(lambda: key in s.overrides.values)
        st = until(ws, lambda m: is_state(m) and key in m["state"]["values"])
        assert st["state"]["values"][key] == s.overrides.values[key]

        ws.send_json({"cmd": "reset_params"})
        rig.step_until(lambda: not s.overrides.values)
        until(ws, lambda m: is_state(m, values={}))

        ws.send_json({"cmd": "reset_counters"})
        rig.step_until(lambda: rig.count("event", "counters_reset"))
        until(ws, lambda m: m["type"] == "event" and m["kind"] == "counters_reset")

        ws.send_json({"cmd": "set_exercise", "exercise": "squat"})
        rig.step_until(lambda: s.exercise == "squat")
        st = until(ws, lambda m: is_state(m, exercise="squat"))
        assert any(p["key"].startswith("rule.squat.") for p in st["state"]["params"])

        saved = rig.team.parent / "tuned.yaml"  # the rig's start directory is tmp_path
        ws.send_json({"cmd": "save_rules", "path": "tuned.yaml"})
        rig.step_until(saved.is_file)
        ev = until(ws, lambda m: m["type"] == "event" and m["kind"] == "rules_saved")
        assert ev["payload"] == {"path": "tuned.yaml", "overwritten": False}

        # the watch: scan, label, identify, disconnect, connect (link worker + session thread)
        links = rig.bench.links
        ws.send_json({"cmd": "scan", "kind": "fake"})
        scan = until(ws, lambda m: m["type"] == "scan")
        assert scan["devices"][0]["id"] == "fake" and scan["devices"][0]["kit"] is None
        ws.send_json({"cmd": "label_kit", "id": "fake", "kit": "K3"})
        scan = until(ws, lambda m: m["type"] == "scan")
        assert scan["devices"][0]["kit"] == "K3"
        rig.step_until(lambda: s.capture_context.get("kit") == "K3")
        until(ws, lambda m: is_state(m) and m["state"]["link"]["kit"] == "K3")
        ws.send_json({"cmd": "device", "action": "identify"})
        rig.step_until(lambda: Command.LED_GOOD_REP in links.board.source.sent)

        ws.send_json({"cmd": "disconnect"})
        # step until the link is closed (closing joins the fake watch, which waits on the clock)
        rig.step_until(lambda: s.board is None and published_link_state(rig) == "none")
        until(ws, lambda m: m["type"] == "link" and m["status"]["state"] == "none")
        ws.send_json({"cmd": "connect", "kind": "fake", "id": "fake"})
        rig.step_until(lambda: s.board is not None and s.board is links.board)
        st = until(ws, lambda m: is_state(m) and m["state"]["link"]["kind"] == "fake")
        assert st["state"]["link"]["kit"] == "K3"
        ws.send_json({"cmd": "connect", "kind": "carrier-pigeon", "id": None})
        err = until(ws, lambda m: m["type"] == "error")
        assert err["cmd"] == "connect" and "carrier-pigeon" in err["text"]
        assert s.board is links.board is not None  # a refused connect keeps the watch
    assert "rule" in saved.read_text()


def test_bad_command_gets_an_error_message_and_the_session_keeps_running(rig):
    TestClient = _need_starlette()
    with TestClient(rig.bench.app) as client, client.websocket_connect(WS_URL) as ws:
        recv(ws)
        ws.send_text("this is not json")
        err = until(ws, lambda m: m["type"] == "error")
        assert "JSON" in err["text"] and err["cmd"] is None
        ws.send_json({"cmd": "set_gate_source"})
        err = until(ws, lambda m: m["type"] == "error")
        assert err["cmd"] == "set_gate_source" and "source" in err["text"]

        ws.send_json({"cmd": "set_param", "key": "rule.curl.NOPE.x", "value": 1})
        ws.send_json({"cmd": "set_exercise", "exercise": "yoga"})
        ws.send_json({"cmd": "capture_keep", "true_reps": 3})  # nothing in review
        ws.send_json({"cmd": "set_video", "on": "yes"})
        ws.send_json({"cmd": "device", "action": "self_destruct"})
        rig.step_until(lambda: rig.count("error") >= 5)
        seen: list[dict] = []
        for _ in range(5):
            until(ws, lambda m: m["type"] == "error", seen=seen)
        errors = {m["cmd"]: m["text"] for m in seen if m["type"] == "error"}
        assert "unknown parameter" in errors["set_param"]
        assert "yoga" in errors["set_exercise"]
        assert "no take to keep" in errors["capture_keep"]
        assert "true or false" in errors["set_video"]
        assert "self_destruct" in errors["device"]
        assert any(is_state(m) for m in seen)  # a failed session command still sends state

        before = rig.steps
        rig.step(10)  # the session runs on
        frame = until(ws, lambda m: m["type"] == "frame" and m["i"] >= before + 9)
        assert frame["exercise"] == "curl"


def test_capture_flow_over_the_socket_writes_a_take(rig):
    TestClient = _need_starlette()
    s = rig.session
    with TestClient(rig.bench.app) as client, client.websocket_connect(WS_URL) as ws:
        recv(ws)
        labels = {"subject": "S1", "exercise": "curl", "form": "clean", "arm": "both"}
        ws.send_json({"cmd": "capture_start", "labels": labels, "delay_s": 0, "auto_stop": False})
        rig.step_until(lambda: s.capture_state == "recording")
        st = until(ws, lambda m: is_state(m) and m["state"]["capture"]["state"] == "recording")
        assert st["state"]["capture"]["labels"]["subject"] == "S1"
        rig.step(60)

        ws.send_json({"cmd": "capture_stop"})
        rig.step_until(lambda: s.capture_state == "review")
        st = until(ws, lambda m: is_state(m) and m["state"]["capture"]["state"] == "review")
        assert st["state"]["capture"]["summary"] is not None

        ws.send_json({"cmd": "capture_keep", "true_reps": 3})
        rig.step_until(lambda: s.capture_state == "idle")
        st = until(ws, lambda m: is_state(m) and m["state"]["capture"]["last"] is not None)

    last = st["state"]["capture"]["last"]
    folder = Path(last["dir"])
    assert folder.is_dir() and folder.parent == rig.team / "S1"
    assert (folder / "meta.json").is_file() and isinstance(last["check"], dict)
    assert last["warnings"] == []
    meta = json.loads((folder / "meta.json").read_text())
    assert "rules.mediapipe-webcam.yaml" in json.dumps(meta)
    assert st["state"]["capture"]["state"] == "idle" and st["state"]["capture"]["summary"] is None


def test_source_ended_is_reported_and_server_stays_up(rig):
    TestClient = _need_starlette()
    from formcoach.bench.runner import serve_frames

    said: list[str] = []
    with TestClient(rig.bench.app) as client, client.websocket_connect(WS_URL) as ws:
        recv(ws)
        # queued for the session thread, which never takes another step: the source is empty
        rig.bench.controller.handle({"cmd": "set_exercise", "exercise": "raise"})
        rig.bench.frames = []
        run = threading.Thread(target=serve_frames, args=(rig.bench,), kwargs={"say": said.append})
        run.start()
        run.join(timeout=5)
        assert not run.is_alive()
        until(ws, lambda m: is_state(m, source="ended"))
        assert rig.session.exercise == "raise"  # run when the source ended

        # commands still work, now without frames
        ws.send_json({"cmd": "set_exercise", "exercise": "press"})
        until(ws, lambda m: is_state(m, exercise="press", source="ended"))
        assert client.get("/lab").status_code == 200
        with client.websocket_connect(WS_URL) as late:
            assert recv(late)["state"]["source"] == "ended"
    assert any("ended" in line for line in said)


def test_video_off_sends_frames_without_images(rig):
    TestClient = _need_starlette()
    cv2 = pytest.importorskip("cv2")
    published: list[bytes] = []
    hub = rig.bench.hub
    publish_frame = hub.publish_frame

    def record(data: bytes) -> None:
        published.append(data)
        publish_frame(data)

    hub.publish_frame = record
    rig.session.set_gate_source("always_on")  # pose on every frame
    with TestClient(rig.bench.app) as client, client.websocket_connect(WS_URL) as ws:
        recv(ws)
        rig.step(3)
        frame = until(ws, lambda m: m["type"] == "frame")
        assert frame["has_image"] is True and frame["lm"] is not None
        img = cv2.imdecode(np.frombuffer(frame["_jpeg"], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 640

        rig.bench.controller.video_max_fps = 10.0  # at most 10 images a second
        published.clear()
        rig.step(30)
        images = [protocol.decode_frame(d)[1] for d in published]
        assert len(images) == 30 and sum(j is not None for j in images) == 10

        ws.send_json({"cmd": "set_video", "on": False})
        rig.step_until(lambda: not rig.bench.controller.state()["video"]["on"])
        published.clear()
        rig.step(5)
        headers = [protocol.decode_frame(d) for d in published]
        assert len(headers) == 5
        assert all(h["has_image"] is False and jpeg is None for h, jpeg in headers)
        assert all(h["lm"] is not None and len(h["lm"]) == 33 for h, _ in headers)
        st = until(ws, lambda m: is_state(m) and m["state"]["video"]["on"] is False)
        assert st["state"]["video"] == {"on": False, "width": 640}


def test_static_pages_and_vendor_are_served_with_js_mime_and_no_store(rig, tmp_path):
    TestClient = _need_starlette()
    from formcoach.bench.server import VENDOR_DIR, create_app

    with TestClient(rig.bench.app) as client:
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == "/lab"
        for page in ("lab", "coach", "review"):
            r = client.get(f"/{page}")
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
            assert r.headers["cache-control"] == "no-store"
            assert page == "lab" or "not built yet" in r.text  # the lab console is built (Task 10)
        r = client.get("/static/coach.html")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
        if (VENDOR_DIR / "uPlot.iife.min.js").exists():
            r = client.get("/vendor/uPlot.iife.min.js")
            assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
            assert r.headers["content-type"].startswith("text/javascript")
            r = client.get("/vendor/uPlot.min.css")
            assert r.headers["content-type"].startswith("text/css")
        assert client.get("/static/nope.js").status_code == 404
        assert client.get("/static/..%2Fserver.py").status_code == 404
        assert client.get("/vendor/..%2F..%2Fapp.js").status_code == 404

    static = tmp_path / "static"
    static.mkdir()
    (static / "lab.html").write_text("<!doctype html><title>lab</title>")
    (static / "app.js").write_text("export const x = 1;\n")
    (static / "app.css").write_text("body {}\n")
    app = create_app(
        rig.bench.hub, rig.bench.controller, static_dir=static, vendor_dir=tmp_path / "missing"
    )
    with TestClient(app) as client:
        r = client.get("/static/app.js")
        assert r.headers["content-type"].startswith("text/javascript")
        assert r.headers["cache-control"] == "no-store"
        assert client.get("/static/app.css").headers["content-type"].startswith("text/css")
        assert client.get("/vendor/uPlot.iife.min.js").status_code == 404  # no vendor dir
        assert client.get("/lab").status_code == 200


def test_cli_bench_rejects_two_sources_and_reports_missing_extra(monkeypatch):
    runner = CliRunner()
    r = runner.invoke(cli_app, ["bench", "--fake", "--camera", "0", "--no-open"])
    assert r.exit_code == 2 and "exactly one" in r.output and "--camera and --fake" in r.output
    r = runner.invoke(cli_app, ["bench", "--no-open"])
    assert r.exit_code == 2 and "exactly one" in r.output
    r = runner.invoke(cli_app, ["bench", "--fake", "--port", "COM5", "--no-open"])
    assert r.exit_code == 2 and "--board" in r.output

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "uvicorn", None)  # as if the bench extra were not installed
        r = runner.invoke(cli_app, ["bench", "--fake", "--no-open"])
    assert r.exit_code == 2 and "uv sync --extra bench" in r.output

    if not all(importlib.util.find_spec(n) for n in ("starlette", "uvicorn", "websockets")):
        return
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        r = runner.invoke(cli_app, ["bench", "--fake", "--no-open", "--http-port", str(port)])
    assert r.exit_code == 2 and "--http-port" in r.output and str(port) in r.output


def test_bench_package_is_not_imported_by_the_core():
    code = (
        "import sys\n"
        "import formcoach.app.session, formcoach.app.live, formcoach.app.pipeline\n"
        "import formcoach.app.link, formcoach.app.capture, formcoach.app.tuning\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m.split('.')[0] in ('starlette', 'uvicorn', 'websockets')\n"
        "             or m.startswith('formcoach.bench'))\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_ws_refuses_foreign_origins_and_http_refuses_foreign_hosts(rig):
    _need_starlette()
    from starlette.testclient import TestClient, WebSocketDenialResponse
    from starlette.websockets import WebSocketDisconnect

    from formcoach.bench.server import origin_allowed

    hub = rig.bench.hub
    with TestClient(rig.bench.app, base_url="http://localhost:8010") as client:
        for origin in (
            "http://evil.example",
            "http://localhost:8011",  # another local port: another program's page
            "https://localhost:8010",
            "null",
        ):
            ws_kw = {"headers": {"origin": origin}}
            with (
                pytest.raises(WebSocketDisconnect) as info,  # closed instead of a hello
                client.websocket_connect("ws://localhost:8010/ws", **ws_kw),
            ):
                pytest.fail(f"{origin} was accepted")
            assert info.value.code == 1008
            assert hub.client_count() == 0
        for headers in ({"origin": "http://localhost:8010"}, {}):  # same port, or no browser
            with client.websocket_connect("ws://localhost:8010/ws", headers=headers) as ws:
                assert recv(ws)["type"] == "hello"

        r = client.get("/lab", headers={"host": "evil.example"})
        assert r.status_code == 400 and "not built yet" not in r.text
        for host in ("localhost:8010", "127.0.0.1:8010", "[::1]:8010"):
            assert client.get("/lab", headers={"host": host}).status_code == 200
        with (
            pytest.raises(WebSocketDenialResponse) as denied,  # HTTP 400, never upgraded
            client.websocket_connect("ws://evil.example:8010/ws"),
        ):
            pytest.fail("a foreign Host was accepted")
        assert denied.value.status_code == 400
        assert hub.client_count() == 0

    assert origin_allowed("http://127.0.0.1:8010", "127.0.0.1:8010")
    assert origin_allowed("http://[::1]:8010", "[::1]:8010")
    assert origin_allowed("http://localhost", "localhost:80")
    assert not origin_allowed("http://localhost:99999", "localhost:8010")  # bad port
    assert not origin_allowed("http://localhost.evil.example:8010", "localhost:8010")


def test_save_rules_is_confined_to_the_start_directory(rig, tmp_path):
    TestClient = _need_starlette()
    s = rig.session
    outside = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    existing = tmp_path / "kept.yaml"
    existing.write_text("# mine\n")
    (tmp_path / "sub").mkdir()
    link = tmp_path / "sub" / "out"
    try:
        link.symlink_to(tmp_path.parent, target_is_directory=True)
    except OSError:  # no symlinks (Windows without the privilege)
        link = None

    def save(ws, **fields) -> dict:
        n = rig.count("error") + rig.count("event", "rules_saved")
        ws.send_json({"cmd": "save_rules", **fields})
        rig.step_until(lambda: rig.count("error") + rig.count("event", "rules_saved") > n)
        return until(ws, lambda m: m["type"] == "error" or m.get("kind") == "rules_saved")

    with (
        TestClient(rig.bench.app) as client,
        client.websocket_connect("ws://localhost:8010/ws") as ws,
    ):
        recv(ws)
        s.set_param(next(iter(s.overrides.params("curl"))).key, 0.01)
        refused = {
            f"../{outside.name}": "leaves the start directory",
            str(tmp_path / "abs.yaml"): "relative",
            "C:/rules/x.yaml": "relative",
            "notes.txt": ".yaml",
            "kept.yaml": "already exists",
        }
        if link is not None:
            refused["sub/out/x.yaml"] = "leaves the start directory"
        for path, why in refused.items():
            msg = save(ws, path=path)
            assert msg["type"] == "error" and msg["cmd"] == "save_rules", path
            assert why in msg["text"], (path, msg["text"])
        assert not outside.exists() and not (tmp_path / "abs.yaml").exists()
        assert not (tmp_path / "notes.txt").exists() and existing.read_text() == "# mine\n"
        if link is not None:
            assert not (tmp_path.parent / "x.yaml").exists()

        msg = save(ws, path="sub/tuned.yml")
        assert msg["payload"] == {"path": "sub/tuned.yml", "overwritten": False}
        assert "rule" in (tmp_path / "sub" / "tuned.yml").read_text()
        msg = save(ws, path="kept.yaml", overwrite=True)
        assert msg["payload"] == {"path": "kept.yaml", "overwritten": True}
        assert existing.read_text() != "# mine\n"
        msg = save(ws, path="kept.yaml", overwrite="yes")
        assert msg["type"] == "error" and "true or false" in msg["text"]
