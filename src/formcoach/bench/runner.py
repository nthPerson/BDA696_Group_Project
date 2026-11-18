"""``formcoach bench``: build the session, link, hub, controller and app, then serve them
(Bench design §5.3).

:func:`build` assembles everything without binding a port or starting a thread (tests call
it); :func:`serve_frames` runs the session over the frame source on the calling thread;
:func:`run_bench` is the thin wrapper the CLI calls: it binds ``127.0.0.1:<http_port>`` first (a
port in use fails before a camera is touched), builds, starts the session thread and runs
uvicorn on the main thread, which owns Ctrl-C. On exit the session is stopped and the camera
released (on the session thread) before the watch link is closed.

The frame source is opened the way :func:`formcoach.app.live.run_live_demo` opens it, with the
same checks and messages: ``--video`` (paced to real time), ``--camera N`` (opened last, since a
camera generator that never started cannot release the device) or ``--fake`` (a
:class:`~formcoach.pose.synthetic.FakeScene` doing the exercise, plus a connected
:class:`~formcoach.io.fake.FakeWatch`). When the source ends the server keeps serving and
``state.source`` becomes ``"ended"``.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
import webbrowser
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from formcoach.app.link import FAKE_ID, KITS_FILE, LinkManager
from formcoach.app.session import EXERCISES, LiveSession
from formcoach.bench import DEFAULT_HTTP_PORT, HOST, JPEG_QUALITY, VIDEO_MAX_FPS, VIDEO_WIDTH
from formcoach.bench.controller import Controller
from formcoach.bench.hub import Hub
from formcoach.io.recorder import TEAM_ROOT

BOARDS = ("ble", "serial", "fake")
INSTALL_HINT = "uv sync --extra bench"
JOIN_S = 3.0  # how long exit waits for the session thread (s)


class PortInUseError(OSError):
    """The HTTP port could not be bound."""


def require_bench_extra() -> None:
    """Check that Starlette, uvicorn and websockets import.

    Raises:
        ImportError: one is missing; the message gives the install command.
    """
    try:
        import starlette  # noqa: F401
        import uvicorn  # noqa: F401
        import websockets  # noqa: F401
    except ImportError as exc:
        missing = exc.name or "a package"
        raise ImportError(
            f"formcoach bench needs the bench extra ({missing} is not installed): {INSTALL_HINT}"
        ) from exc


def bind_socket(port: int, host: str = HOST) -> socket.socket:
    """A listening TCP socket on ``host:port`` for uvicorn.

    Raises:
        PortInUseError: the port is taken (or cannot be bound); the message says what to do.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # On Windows SO_REUSEADDR would let us bind a port another program holds, so a busy port
    # would not fail; elsewhere it only allows a quick restart (as in app/viewer.py).
    if os.name != "nt":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.listen(128)
    except OSError as exc:
        sock.close()
        raise PortInUseError(
            f"cannot serve on {host}:{port} ({exc.strerror or exc}); is another formcoach bench "
            f"running? Stop it or pass --http-port {port + 1}"
        ) from exc
    return sock


@dataclass
class Bench:
    """The assembled pieces. ``frames`` yields ``(index, t_seconds, frame_bgr)``; ``source`` is
    ``"camera N"``, ``"video NAME"`` or ``"fake"``; ``cleanup`` closes the frame source, the
    controller and the link (call it after the session thread has stopped)."""

    session: LiveSession
    links: LinkManager
    hub: Hub
    controller: Controller
    app: object
    frames: Iterable
    source: str
    cleanup: Callable[[], None]
    stopping: threading.Event = field(default_factory=threading.Event)

    def stop(self) -> None:
        """Make :func:`serve_frames` return after the current frame, without reporting the
        source as ended. Thread-safe."""
        self.stopping.set()
        self.session.stop()


def build(
    *,
    camera: int | None,
    video: Path | None,
    fake: bool,
    exercise: str,
    model: str = "lite",
    rules: Path | None = None,
    board: str | None = None,
    port: str | None = None,
    video_width: int = VIDEO_WIDTH,
    team_root: Path = TEAM_ROOT,
    rules_dir_base: Path | None = None,
    kits_file: Path = KITS_FILE,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Bench:
    """Assemble the Bench; binds nothing and starts no session thread.

    Exactly one of ``camera`` (webcam index), ``video`` (a file) and ``fake`` picks the frame
    source. ``exercise`` is one of the session's exercises (``idle`` included); ``model`` the
    MediaPipe variant (not used with ``fake``); ``rules`` a rules YAML (default: the packaged
    ``rules.mediapipe-webcam.yaml``; the session records its file name). ``board``
    (``ble``/``serial``/``fake``, default ``fake`` with ``fake``) is connected at once, at
    ``port`` (BLE address or serial port). ``video_width`` in pixels; kept takes go under
    ``team_root``; ``save_rules`` writes only inside ``rules_dir_base`` (default: the current
    directory, where the bench was started); kit labels live in ``kits_file``. ``clock`` (s)
    and ``sleep`` drive the session, the link, the fake scene and the fake watch (tests
    inject them).

    Raises:
        ValueError: not exactly one source, unknown exercise, model or board, or a rules file
            that is malformed or lacks ``exercise``.
        FileNotFoundError: the rules, model, video, camera or serial port is missing.
        ImportError: the ``vision`` (OpenCV, MediaPipe) or ``device`` extra is not installed.
        RuntimeError: the camera opened but gave no frames.
    """
    from formcoach.bench.server import create_app
    from formcoach.rules.engine import WEBCAM_RULES, RuleEngine

    if sum((camera is not None, video is not None, bool(fake))) != 1:
        raise ValueError("pass exactly one of --camera N, --video PATH or --fake")
    if exercise not in EXERCISES:
        raise ValueError(f"unknown exercise {exercise!r}; choose one of {', '.join(EXERCISES)}")
    if not fake:
        from formcoach.pose import landmarker

        if model not in landmarker.MODEL_URLS:
            raise ValueError(
                f"unknown pose model {model!r}; choose from {sorted(landmarker.MODEL_URLS)}"
            )
    if board is not None and board not in BOARDS:
        raise ValueError(f"unknown board {board!r}; choose {', '.join(BOARDS)}")
    if video is not None and not Path(video).exists():
        raise FileNotFoundError(f"cannot open video {video}: no such file")

    rules_file = Path(rules) if rules is not None else WEBCAM_RULES
    try:
        engine = RuleEngine.from_yaml(rules_file)
    except (yaml.YAMLError, KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"cannot load rules file {rules_file}: {exc}") from exc
    if exercise != "idle" and exercise not in engine.exercises:
        where = rules if rules is not None else f"the packaged {WEBCAM_RULES.name}"
        raise ValueError(
            f"{where} has no rules for exercise {exercise!r}; it defines {sorted(engine.exercises)}"
        )

    scene = None
    if fake:
        from formcoach.pose.synthetic import FakeScene

        # the synthetic person keeps doing this exercise whatever the console selects
        scene = FakeScene(
            exercise=exercise if exercise != "idle" else "curl", clock=clock, sleep=sleep
        )
        detect = scene.detect
    else:
        detect = landmarker.make_landmarker(model)

    def fake_watch(_id: str | None):
        from formcoach.io.fake import FakeWatch

        return FakeWatch(clock=clock, sleep=sleep)

    links = LinkManager(kits_file=kits_file, clock=clock, factories={"fake": fake_watch})
    controller = None
    frames = None
    try:
        kind = board or ("fake" if fake else None)
        if kind is not None:
            links.connect(kind, FAKE_ID if kind == "fake" else port)
        session = LiveSession(
            exercise=exercise,
            detect=detect,
            rules=engine,
            board=links.board,
            clock=clock,
            rules_name=rules_file.name,
        )
        if video is not None:
            source = f"video {Path(video).name}"
        elif camera is not None:
            source = f"camera {camera}"
        else:
            source = "fake"
        session.capture_context.update(
            model="synthetic" if fake else model, camera=camera, source=source
        )
        hub = Hub()
        controller = Controller(
            session, links, hub, video_width=video_width, jpeg_quality=JPEG_QUALITY,
            video_max_fps=VIDEO_MAX_FPS, team_root=team_root,
            rules_dir_base=rules_dir_base if rules_dir_base is not None else Path.cwd(),
            clock=clock,
        )  # fmt: skip
        app = create_app(hub, controller)
        # the camera goes last: nothing that can fail may run after it is opened
        if scene is not None:
            frames = scene.frames()
        elif video is not None:
            from formcoach.app.live import _paced

            frames = _paced(landmarker.open_video(Path(video)), clock=clock, sleep=sleep)
        else:
            from formcoach.pose import camera as camera_mod

            frames = camera_mod.open_camera(int(camera))
    except BaseException:
        if controller is not None:
            controller.close()
        links.disconnect()
        raise

    def cleanup() -> None:
        # the camera first (its light goes off at once), the link last (closing it can take a
        # second); each step runs even if an earlier one raises
        with contextlib.ExitStack() as stack:
            stack.callback(links.disconnect)
            stack.callback(controller.close)
            close = getattr(frames, "close", None)
            if close is not None:
                stack.callback(_quiet_close, close)

    return Bench(session, links, hub, controller, app, frames, source, cleanup)


def _quiet_close(close: Callable[[], None]) -> None:
    with contextlib.suppress(Exception):  # e.g. a generator still running on another thread
        close()


def serve_frames(bench: Bench, *, say: Callable[[str], None] = print) -> None:
    """Run the session over ``bench.frames`` on this thread until they end, fail or
    :meth:`Bench.stop` is called; then close the frame source here (releasing the camera). If
    the source ended or failed on its own, the controller reports it (``state.source`` becomes
    ``"ended"``) and the server keeps serving."""
    error = None
    try:
        bench.session.run(bench.frames)
    except Exception as exc:
        error = str(exc) or type(exc).__name__
    finally:
        close = getattr(bench.frames, "close", None)
        if close is not None:
            _quiet_close(close)
    if bench.stopping.is_set():
        return
    if error is not None:
        say(f"the frame source stopped: {error}")
    say("the frame source ended; the pages stay up (Ctrl+C to quit)")
    bench.controller.source_ended(error)


def run_bench(
    *,
    camera: int | None,
    video: Path | None,
    fake: bool,
    exercise: str,
    model: str,
    rules: Path | None,
    board: str | None,
    port: str | None,
    http_port: int = DEFAULT_HTTP_PORT,
    open_browser: bool = True,
    video_width: int = VIDEO_WIDTH,
    say: Callable[[str], None] = print,
) -> None:
    """Serve the Bench on ``http://localhost:<http_port>/lab`` until Ctrl-C (arguments as in
    :func:`build`; ``open_browser`` opens the lab console).

    Raises:
        ImportError: the ``bench`` extra (or a source's extra) is missing.
        PortInUseError: ``http_port`` is taken.
        ValueError, FileNotFoundError, RuntimeError: as :func:`build`.
    """
    require_bench_extra()
    import uvicorn

    sock = bind_socket(http_port)
    try:
        bench = build(
            camera=camera, video=video, fake=fake, exercise=exercise, model=model, rules=rules,
            board=board, port=port, video_width=video_width,
        )  # fmt: skip
    except BaseException:
        sock.close()
        raise
    url = f"http://localhost:{http_port}/lab"
    thread = threading.Thread(
        target=serve_frames, args=(bench,), kwargs={"say": say}, name="bench-session", daemon=True
    )
    thread.start()
    config = uvicorn.Config(
        bench.app, log_level="warning", lifespan="on", timeout_graceful_shutdown=1
    )
    server = uvicorn.Server(config)
    say(f"FormCoach Bench ({bench.source}): {url}  (coach page: /coach; Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
    finally:
        bench.stop()
        thread.join(timeout=JOIN_S)
        bench.cleanup()
        sock.close()
        say("bench stopped")
