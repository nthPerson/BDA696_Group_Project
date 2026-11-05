"""The live demo loop: camera (or video) frames -> pose -> per-arm reps -> rules -> HUD.

``run_live`` is the frame-driven loop. It feeds frames through an injectable ``detect`` callable
(the real MediaPipe landmarker in production, a fake in tests) into a
:class:`formcoach.app.pipeline.Pipeline` built ``per_arm=True``, so each arm is counted and judged
on its own without any flag: both arms together, alternating, or one arm at a time all work from
one command (spec §4.3). An optional :class:`formcoach.app.board.BoardLink` supplies the wearable's
gate (flag AND alive) and receives the good-rep / fault LED commands; without a board the gate is
always open. The loop never writes a camera frame or a video to disk: only ``events.parquet``,
``frames.parquet`` and ``session.json`` (spec §4.6).

``run_live_demo`` assembles the real pieces (camera or video, landmarker, rules, board, window)
for the CLI (``formcoach demo --camera N``) and always releases them on the way out.

Timing: frame times ``t`` are seconds on the frames' own clock (camera: since open; video:
``i / fps``); the banner expiry and the FPS readout use the injectable ``clock`` (wall seconds).
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import yaml

from formcoach.app import hud
from formcoach.app.events import Event, EventLog
from formcoach.app.pipeline import Pipeline, PipelineResult
from formcoach.io import protocol
from formcoach.io.replay import PoseFrame
from formcoach.pose.skeletons import MEDIAPIPE33

BLACK_FRAMES = 30  # a camera is called black after this many frames, all with mean < BLACK_MEAN
BLACK_MEAN = 2.0  # mean pixel value (0-255) below which a frame counts as black
TRACE_S = 8.0  # seconds of angle history drawn in the HUD trace
MAX_BUFFER_FRAMES = 3600  # per-arm angle buffer cap (about two minutes at 30 fps)
REP_PAD_S = 1.0  # metrics window reaches this far back to the rest position (see Pipeline)
FPS_WINDOW = 30  # frames averaged for the on-screen FPS readout
FAULT_FLASH_S = 0.8  # firmware fault flash (ui.cpp): a newer LED command cancels it, so a good-rep
# command is held back for this long after a fault one (wall seconds on the injectable clock)

DetectFn = Callable[[np.ndarray, int], "tuple[np.ndarray | None, np.ndarray | None]"]


def pose_frame_from_detection(
    index: int, t: float, img: np.ndarray | None, world: np.ndarray | None
) -> PoseFrame:
    """Wrap one detector result as a :class:`PoseFrame` (MediaPipe-33 skeleton).

    ``img`` is ``(33, 4)`` normalised image x, y, z and visibility; ``world`` is ``(33, 3)``
    metres. Either being ``None`` (no person found) gives an invalid frame: ``valid=False``,
    ``world`` all NaN, ``visibility`` zeros, ``image`` ``None``. ``t`` is seconds.
    """
    if img is None or world is None:
        return PoseFrame(
            index, float(t), np.full((33, 3), np.nan), np.zeros(33), False, MEDIAPIPE33, None
        )
    img = np.asarray(img)
    return PoseFrame(
        index,
        float(t),
        np.asarray(world, dtype=np.float64),
        img[:, 3].astype(np.float64),
        True,
        MEDIAPIPE33,
        img,
    )


def looks_black(frame_means: Sequence[float]) -> bool:
    """``True`` once at least :data:`BLACK_FRAMES` frames were seen and every one of the first
    :data:`BLACK_FRAMES` has a mean pixel value below :data:`BLACK_MEAN` (0-255). A camera that
    delivers only black frames is usually the iPhone Continuity Camera or a covered lens."""
    first = list(frame_means)[:BLACK_FRAMES]
    return len(first) >= BLACK_FRAMES and all(m < BLACK_MEAN for m in first)


def totals_lines(exercise: str, totals: dict[str, dict]) -> list[str]:
    """One printable line per counter from :meth:`Pipeline.arm_totals`, e.g.
    ``"L: 7 reps (6 good, 1 flagged)  CURL_TOO_FAST x1"``. Squat has the single counter
    ``"all"``. ``exercise`` is accepted for symmetry with :func:`hud.counter_text`."""
    del exercise
    lines = []
    for key in ("l", "r", "all"):
        t = totals.get(key)
        if t is None:
            continue
        label = key.upper() if key != "all" else "all"
        reps = t["reps"]
        line = f"{label}: {reps} {'rep' if reps == 1 else 'reps'} "
        line += f"({t['good']} good, {t['flagged']} flagged)"
        codes = sorted(t.get("faults_by_code", {}).items(), key=lambda kv: (-kv[1], kv[0]))
        if codes:
            line += "  " + "  ".join(f"{code} x{n}" for code, n in codes)
        lines.append(line)
    return lines


def run_live(
    frames: Iterable[tuple[int, float, np.ndarray]],
    detect: DetectFn,
    *,
    exercise: str,
    rules=None,
    board=None,
    show: Callable[[np.ndarray], bool] | None = None,
    on_event: Callable[[Event], None] | None = None,
    say: Callable[[str], None] = print,
    out_dir: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    rep_config: dict | None = None,
    black_hint: str | None = None,
    extra_summary: dict | None = None,
) -> PipelineResult:
    """Run the live loop over ``frames`` until they run out, ``show`` returns ``False`` (``q``,
    Esc or the window closed) or Ctrl-C.

    ``frames`` yields ``(frame_index, t_seconds, frame_bgr)``. ``detect(frame_bgr, t_ms)``
    returns ``(image_landmarks (33, 4), world_landmarks (33, 3))`` or ``(None, None)`` when no
    person is found; ``t_ms`` is strictly increasing. ``board`` is a
    :class:`formcoach.app.board.BoardLink` (or anything with ``poll``/``gate_open``/``state``/
    ``send``/``name``): the gate follows ``board.gate_open()``. Reps completed on one frame send
    ONE LED command: ``Command.LED_FAULT`` if any of them has a fault, else
    ``Command.LED_GOOD_REP``, and a good-rep command is not sent within :data:`FAULT_FLASH_S`
    of a fault one (it would cancel the device's fault flash). ``send`` errors are ignored.
    ``show`` receives each rendered HUD frame; leave it ``None`` for a headless run (OpenCV is
    then never touched). ``on_event`` sees every ``rep``/``fault``/``gate_*`` event as it happens.

    The totals are printed through ``say`` and, when ``out_dir`` is given, ``events.parquet``,
    ``frames.parquet`` and ``session.json`` are written (even for zero reps) however the loop
    ends, including when ``frames`` or ``detect`` raises: the exception is re-raised only after
    the log and totals are out. The one exception: if ``frames`` raises before yielding anything
    (a video that cannot be opened) nothing ran, so no log or totals are written. Without a
    window (``show`` is ``None``) and with a ``board``, the board's status is printed through
    ``say`` as ``"wearable: <status>"`` whenever it changes. Returns a :class:`PipelineResult`
    whose ``summary`` carries ``arms`` (per-arm totals), ``live``, ``board``, ``fps_mean``
    (frames/s) and ``duration_s``.
    """
    log = EventLog()
    pipe = Pipeline(
        exercise=exercise,
        rules=rules,
        log=log,
        on_event=on_event,
        rep_config=rep_config,
        per_arm=True,
        max_buffer_frames=MAX_BUFFER_FRAMES,
        pad_rep_start_s=REP_PAD_S,
        gate_name="device" if board is not None else "always_on",
    )

    totals = pipe.arm_totals()
    banners: dict[str, hud.Banner] = {}  # one per counter ("l", "r", "all"), each expires alone
    fault_led_at: float | None = None  # when LED_FAULT was last sent (clock seconds)
    trace: dict[str, deque[tuple[float, float]]] = {k: deque() for k in totals}
    stamps: deque[float] = deque(maxlen=FPS_WINDOW)
    means: list[float] = []
    hinted = False
    last_status = ""
    last_ms = -1
    t_first: float | None = None
    t_last: float | None = None
    n_frames = 0

    def send_led(cmd: protocol.Command) -> None:
        if board is not None:
            with contextlib.suppress(Exception):
                board.send(cmd)

    failed = False
    try:
        for index, t, frame in frames:
            if t_first is None:
                t_first = t
            t_last = t
            n_frames += 1
            stamps.append(clock())

            if black_hint is not None and not hinted and len(means) < BLACK_FRAMES:
                means.append(float(np.mean(frame)))
                if looks_black(means):
                    say(black_hint)
                    hinted = True

            if board is not None:
                board.poll()
                gate = bool(board.gate_open())
                if show is None:  # no window, so no status pill: print it when it changes
                    status, _dim = hud.pill(True, board.state(), gate, board.name)
                    if status != last_status:
                        last_status = status
                        say(f"wearable: {status}")
            else:
                gate = True
            pipe.set_gate(gate, t)

            img_lm = None
            if gate:
                t_ms = max(int(t * 1000), last_ms + 1)
                last_ms = t_ms
                img_lm, world = detect(frame, t_ms)
                pf = pose_frame_from_detection(index, t, img_lm, world)
                completed = pipe.on_frame(pf)
            else:
                pipe.skip_frame(t)
                completed = []

            if completed:
                now = clock()
                for rep in completed:  # each counter keeps its own banner: a later arm in this
                    banners[getattr(rep, "arm", None) or "all"] = hud.Banner.from_rep(rep, now)
                totals = pipe.arm_totals()
                # ONE command per frame, so arms finishing together cannot overwrite each other's
                # LED flash: any fault wins. A good-rep command is skipped while a fault flash is
                # still showing (an arm finishing a frame or two later would cancel it).
                if any(rep.faults for rep in completed):
                    send_led(protocol.Command.LED_FAULT)
                    fault_led_at = now
                elif fault_led_at is None or now - fault_led_at >= FAULT_FLASH_S:
                    send_led(protocol.Command.LED_GOOD_REP)

            if show is None:
                continue

            angles = pipe.latest_angles()
            for key, deg in angles.items():
                pts = trace[key]
                if gate:
                    pts.append((t, deg))
                while pts and t - pts[0][0] > TRACE_S:
                    pts.popleft()
            now = clock()
            for key in [k for k, b in banners.items() if now >= b.expires_at]:
                del banners[key]
            pill_text, dim = hud.pill(
                board is not None,
                board.state() if board is not None else "ok",
                gate,
                board.name if board is not None else "",
            )
            fps = 0.0
            if len(stamps) > 1 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            state = hud.HudState(
                exercise=exercise,
                totals=totals,
                angles={k: (None if np.isnan(v) else float(v)) for k, v in angles.items()},
                pill_text=pill_text,
                dim=dim,
                banners=dict(banners),  # a copy: the HUD state must not change after the frame
                fps=fps,
                trace={k: list(v) for k, v in trace.items()},
                bands=pipe.thresholds(),
            )
            out = hud.render(frame, img_lm, state)
            if show(out) is False:
                break
    except KeyboardInterrupt:
        pass
    except BaseException:
        failed = True
        raise
    finally:
        duration = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
        result = pipe.finish(compute_imu=False, end_t=t_last)
        result.summary["live"] = True
        result.summary["board"] = board.name if board is not None else None
        result.summary["fps_mean"] = round((n_frames - 1) / duration, 1) if duration > 0 else 0.0
        result.summary["duration_s"] = round(duration, 2)
        if extra_summary:
            result.summary.update(extra_summary)
        # a source that failed before its first frame (an unopenable video) ran nothing: no
        # zero-rep totals and no empty session directory, just the error
        nothing_ran = failed and n_frames == 0
        if not nothing_ran:
            for line in totals_lines(exercise, result.summary["arms"]):
                say(line)
        if out_dir is not None and not nothing_ran:
            try:
                log.write(out_dir, result.summary)
                result.out_dir = out_dir
            except OSError as exc:
                if not failed:
                    raise  # the run itself was fine; the caller must hear the log was lost
                # the loop already failed: report the lost log but let that error propagate
                say(f"could not write the session log to {out_dir}: {exc}")
    return result


def _paced(
    frames: Iterator[tuple[int, float, np.ndarray]],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``frames`` no faster than their own timestamps (real time), so a video shown in a
    window plays at its natural speed instead of as fast as it decodes."""
    start = clock()
    try:
        for index, t, frame in frames:
            lag = start + t - clock()
            if lag > 0:
                sleep(lag)
            yield index, t, frame
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()


def _open_board(kind: str, port: str | None):
    """The wearable as an IMU source: ``"ble"`` (first ``FormCoach-XXXX`` found, or ``port`` as an
    address) or ``"serial"`` (``port`` required). Needs the ``device`` extra."""
    if kind == "ble":
        from formcoach.io.ble import BLESource

        return BLESource(address=port)
    if kind == "serial":
        from formcoach.io.serial_source import SerialSource, list_ports

        if not port:
            raise FileNotFoundError(
                f"--port is required for --board serial (visible ports: {list_ports() or 'none'})"
            )
        return SerialSource(port)
    raise ValueError(f"unknown board {kind!r}; choose ble or serial")


def run_live_demo(
    *,
    camera: int | None,
    video: Path | None,
    board: str | None,
    port: str | None,
    exercise: str,
    model: str = "lite",
    rules: Path | None = None,
    headless: bool = False,
    out_dir: Path | None = None,
    say: Callable[[str], None] = print,
    on_event: Callable[[Event], None] | None = None,
    detect: DetectFn | None = None,
) -> PipelineResult:
    """Assemble and run the live demo: ``video`` (a file, played through the same loop) or
    ``camera`` (an OpenCV index), the MediaPipe ``model`` variant, the packaged rules or the
    ``rules`` YAML, an optional wearable (``board`` ``"ble"``/``"serial"`` on ``port``) and a
    window unless ``headless``. ``detect`` replaces the real landmarker (tests).

    If both ``video`` and ``camera`` are given the video wins and no camera is opened (the CLI
    rejects the combination before it gets here).

    Raises:
        ValueError: unknown ``exercise`` or ``board``, neither ``camera`` nor ``video``, the
            ``rules`` file is malformed (bad YAML, a rule without ``when``, empty, or a
            top-level list that cannot be read as the rules document), or the rules file (or
            the packaged ``rules.yaml``) does not define ``exercise``. Raised before the
            landmarker is built, so a hand-edited rules file never gets as far as opening the
            camera or loading the pose model.
        FileNotFoundError: the rules, model, video, camera or serial port is missing (messages
            say what to do next).
        ImportError: the ``vision`` (OpenCV, MediaPipe) or ``device`` (bleak, pyserial) extra is
            not installed.
        RuntimeError: the camera stopped sending frames (the log is written first, unless not a
            single frame had arrived).
    """
    from formcoach.pose import landmarker
    from formcoach.pose.reps import DEFAULT_REP_CONFIG
    from formcoach.rules.engine import RuleEngine

    if exercise not in DEFAULT_REP_CONFIG:
        raise ValueError(
            f"unknown exercise {exercise!r}; choose one of {', '.join(sorted(DEFAULT_REP_CONFIG))}"
        )
    if model not in landmarker.MODEL_URLS:
        raise ValueError(
            f"unknown pose model {model!r}; choose from {sorted(landmarker.MODEL_URLS)}"
        )
    if camera is None and video is None:
        raise ValueError("pass --camera N (a webcam) or --video PATH (a recording)")
    if board not in (None, "ble", "serial"):
        raise ValueError(f"unknown board {board!r}; choose ble or serial")
    if video is not None and not Path(video).exists():
        raise FileNotFoundError(f"cannot open video {video}: no such file")

    try:
        engine = RuleEngine.from_yaml(rules)
    except (yaml.YAMLError, KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"cannot load rules file {rules}: {exc}") from exc
    if exercise not in engine.exercises:
        where = rules if rules is not None else "the packaged rules.yaml"
        raise ValueError(
            f"{where} has no rules for exercise {exercise!r}; it defines {sorted(engine.exercises)}"
        )
    detector = detect if detect is not None else landmarker.make_landmarker(model)

    frames = None
    link = None
    show = None
    try:
        # the camera goes last: an unstarted camera generator cannot release its device when
        # closed, so nothing that can fail (board, window) may run after it is opened
        if board is not None:
            from formcoach.app.board import BoardLink

            link = BoardLink(_open_board(board, port))
        if not headless:
            show = hud.WindowShow()
        if video is not None:
            frames = landmarker.open_video(Path(video))
            if not headless:
                frames = _paced(frames)
            source, hint = f"video {Path(video).name}", None
        else:
            from formcoach.pose import camera as camera_mod

            frames = camera_mod.open_camera(int(camera))
            source = f"camera {camera}"
            hint = (
                f"camera {camera} looks black (iPhone Continuity Camera?) - "
                f"try --camera {camera + 1}"
            )
        return run_live(
            frames,
            detector,
            exercise=exercise,
            rules=engine,
            board=link,
            show=show,
            on_event=on_event,
            say=say,
            out_dir=out_dir,
            black_hint=hint,
            extra_summary={"source": source, "model": model},
        )
    finally:
        # The camera is released first: closing the wearable link can take seconds (it joins its
        # reader thread), and the camera light should go off when the user quits, not after.
        # ExitStack runs callbacks last-in-first-out and still runs the rest if one raises.
        with contextlib.ExitStack() as cleanup:
            if link is not None:
                cleanup.callback(link.close)
            if show is not None:
                cleanup.callback(show.close)
            close = getattr(frames, "close", None)
            if close is not None:
                cleanup.callback(close)
