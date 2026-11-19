"""Capture (Bench design §7): one exercise set ("a take") held in memory, then written as a
session folder.

A :class:`Take` collects, from the moment recording starts, the wearable's samples, one record
per camera frame (time, landmarks or "no person", the gate the pipeline used, the counters'
diagnostics) and every session event. Nothing touches the disk until :meth:`Take.write`, which
the session calls only when the operator keeps the take. The folder follows
:class:`formcoach.io.recorder.SessionRecorder`'s layout, ``<root>/<subject>/<YYYYMMDD-HHMMSS>/``:

- ``imu.parquet``: IMUStream (``schema.make_imu_stream``) plus ``flags``, ``seq``,
  ``t_arrival`` (laptop clock, s, as received) and ``t_device`` (device clock, s);
- ``pose.parquet``: one row per frame (``pose.store.write_pose``, ``mediapipe33``), image and
  world landmarks; frames with nobody found are ``valid=False`` with NaN landmarks;
- ``events.parquet`` and ``frames.parquet`` (``EventLog.write_tables``): ``frames`` carries
  ``t, processed, gate_state``, the counters' ``ang_*``/``enter_*``/``exit_*``/``armed_*``/
  ``vis_*`` on processed frames and ``t_source`` (the frame source's own time, s, which the
  pipeline was given);
- ``meta.json``: the ``SessionRecorder`` keys, the labels, the clock fit (per device-clock
  segment), the capture context, the take's summary and every cue latency (``latency``);
- one line appended to ``<root>/<subject>/sets.csv``.

**One time base.** ``t`` in every file is seconds since the take's ``t0_mono`` on the session's
laptop clock: frames by their ``Snapshot.t_mono``, IMU samples by mapping device time onto the
laptop clock (:func:`formcoach.io.clock.fit_device_clock`), events by their time on the laptop
clock (see :meth:`Take.add_event`).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from formcoach import __version__
from formcoach.app.events import Event, EventLog
from formcoach.data import schema
from formcoach.io.clock import ClockFit, fit_device_clock, to_laptop
from formcoach.io.source import SISample
from formcoach.pose.store import write_pose

if TYPE_CHECKING:
    from formcoach.app.session import Snapshot

FORMS = ("clean", "partial", "no_lockout", "bent_elbow", "shallow")
ARMS = ("both", "l", "r")
SETS_HEADER = ["session_id", "exercise", "form", "reps", "notes", "arm"]
PLACEMENT = "wrist_l"
N_LANDMARKS = 33
# context keys a take records in meta.json (the session fills them, see LiveSession)
CONTEXT_KEYS = (
    "rules", "rules_doc", "overrides", "gate_source", "model", "camera", "firmware", "device",
    "kit", "source", "frame_source",
)  # fmt: skip
SUBJECT_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
WINDOWS_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"] + [f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)]
)
_SHIFTED_PAYLOAD_KEYS = ("t_start", "t_end")  # frame-clock times inside event payloads (s)


@dataclass
class TakeLabels:
    """What the operator says about a take before recording it.

    ``subject`` a folder name such as ``"S1"``: 1-32 letters, digits, ``_`` or ``-``, and not
    a name Windows reserves for a device (``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``-``COM9``,
    ``LPT1``-``LPT9``, any case), so the take can be written on every OS; ``exercise`` one of
    the session's exercises (``curl``, ``press``, ``raise``, ``squat``, ``idle``); ``form`` one
    of :data:`FORMS`; ``arm`` the working arm, one of :data:`ARMS`; ``notes`` free text.

    Raises:
        ValueError: any field outside its allowed values, or ``subject`` not such a name.
    """

    subject: str
    exercise: str
    form: str = "clean"
    arm: str = "both"
    notes: str = ""

    def __post_init__(self) -> None:
        from formcoach.app.session import EXERCISES  # session imports this module

        s = self.subject
        if (
            not isinstance(s, str)
            or not SUBJECT_RE.fullmatch(s)
            or s.upper() in WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(
                f"subject must be a folder name of 1-32 letters, digits, '_' or '-' such as "
                f"'S1' (and not a Windows device name such as CON), got {s!r}"
            )
        if self.exercise not in EXERCISES:
            raise ValueError(f"unknown exercise {self.exercise!r}; choose from {EXERCISES}")
        if self.form not in FORMS:
            raise ValueError(f"unknown form {self.form!r}; choose from {FORMS}")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {ARMS}")


@dataclass(frozen=True)
class _ImuTable:
    """A take's samples ready to write: ``rows`` ``(n, 10)`` (t_device, ax, ay, az, gx, gy, gz,
    flags, seq, t_arrival), ``t`` their laptop-clock times (s, strictly increasing),
    ``segment`` each row's device-clock segment, and per segment its ``fits`` and ``starts``
    (laptop time, s, of its first sample)."""

    rows: np.ndarray
    t: np.ndarray
    segment: np.ndarray
    fits: list[ClockFit]
    starts: list[float]


@dataclass(slots=True)
class _Frame:
    index: int
    t: float  # s since t0_mono
    image: np.ndarray | None  # (33, 4) or None (nobody found)
    world: np.ndarray | None  # (33, 3) m or None
    gate: bool


class Take:
    """One exercise set recorded in memory (see the module doc).

    ``t0_mono`` is the session clock's reading (s) when recording started; every ``t`` written is
    relative to it. ``context`` holds what ``meta.json`` records about the setup: ``rules`` (the
    rules file's name), ``rules_doc`` (the rules document the session's base engine was built
    from, which replay prefers to the name), ``overrides`` (the tuning in force,
    ``{key: value}``), ``gate_source``, ``model`` (pose model), ``camera``, ``firmware``,
    ``device`` (watch name or address), ``kit`` (K1…K6), ``source`` (the watch link's kind:
    ``ble``/``serial``/``fake``/``none``) and ``frame_source`` (``camera 0``, ``video NAME``,
    ``fake``); missing keys are ``None``. ``started`` is the local wall-clock time recording
    started (default now); it names the folder.

    **A device clock that restarts** (the kit reboots and the link reconnects, or another kit
    takes over the link) is a backward jump of device time. The samples are split there into
    segments, each segment's clock is fitted on its own, and a sample is kept only if its
    laptop time is after every sample kept before it; ``meta.json`` ``clock`` holds the longest
    segment's fit with every segment's in ``segments``, and :attr:`warnings` says where the
    clock restarted.
    """

    def __init__(
        self,
        labels: TakeLabels,
        *,
        t0_mono: float,
        context: dict,
        started: datetime | None = None,
    ):
        self.labels = labels
        self.t0_mono = float(t0_mono)
        self.context = {k: context.get(k) for k in CONTEXT_KEYS}
        self.started = started if started is not None else datetime.now()
        self.session_id = self.started.strftime("%Y%m%d-%H%M%S")
        self._imu: list[tuple] = []  # t_device, ax, ay, az, gx, gy, gz, flags, seq, t_arrival
        self._frames: list[_Frame] = []
        self._log = EventLog()  # this take's events (rebased) and frame rows
        self._latency: list[dict] = []
        self._totals: dict[str, dict] = {}  # the latest snapshot's per-counter totals
        self.warnings: list[str] = []  # what :meth:`write` could not do after the folder was in

    # ---- collecting --------------------------------------------------------------------------

    def add_imu(self, samples: Sequence[SISample]) -> None:
        """Keep wearable samples (SI units; ``t`` device seconds, ``t_arrival`` laptop seconds on
        the session clock, ``None`` when the source did not stamp it)."""
        for s in samples:
            arrival = float("nan") if s.t_arrival is None else float(s.t_arrival)
            self._imu.append((s.t, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq, arrival))

    def add_frame(self, snap: Snapshot) -> None:
        """Keep one frame: its time (``snap.t_mono``), landmarks (or "nobody found" when
        ``world_landmarks`` is ``None``), the gate the pipeline used and, when the pipeline
        processed the frame (the gate was open), the counters' diagnostics. The live pipeline
        processes exactly the frames its gate is open on, so ``processed`` is ``snap.gate``."""
        t = float(snap.t_mono) - self.t0_mono
        found = snap.world_landmarks is not None and snap.image_landmarks is not None
        self._frames.append(
            _Frame(
                int(snap.index),
                t,
                np.array(snap.image_landmarks) if found else None,
                np.array(snap.world_landmarks) if found else None,
                bool(snap.gate),
            )
        )
        extra: dict[str, float] = {}
        if snap.gate:
            extra = _diagnostics(snap.counters)
        extra["t_source"] = float(snap.t)
        self._log.frame(t, bool(snap.gate), bool(snap.gate), **extra)
        self._totals = snap.totals

    def add_event(self, event: Event, t_mono: float) -> None:
        """Keep one session event. ``t_mono`` is the event's time on the session clock (s); the
        take stores ``t = t_mono - t0_mono`` and moves the frame-clock times in its payload
        (``t_start``, ``t_end``) by the same amount, so they line up with ``pose.parquet``."""
        shift = float(t_mono) - event.t - self.t0_mono
        payload = dict(event.payload)
        for key in _SHIFTED_PAYLOAD_KEYS:
            value = payload.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                payload[key] = float(value) + shift
        self._log.events.append(Event(event.kind, float(t_mono) - self.t0_mono, payload))

    def note_latency(self, arm: str | None, rep_id: int, latency_ms: float) -> None:
        """Keep the cue latency (ms) of one rep: from the frame that completed it to the coach
        page confirming the cue was drawn."""
        self._latency.append({"arm": arm, "rep_id": int(rep_id), "latency_ms": float(latency_ms)})

    # ---- reporting ---------------------------------------------------------------------------

    def _imu_table(self) -> _ImuTable:
        """The samples to write, on the laptop clock (see "A device clock that restarts" in the
        class doc): repeats of a device time are dropped, a backward jump starts a segment,
        each segment gets its own clock fit, and only samples later than every earlier kept
        one are kept, so ``t`` is strictly increasing."""
        arr = np.array(self._imu, dtype=np.float64) if self._imu else np.zeros((0, 10))
        if not len(arr):
            return _ImuTable(arr, np.zeros(0), np.zeros(0, dtype=np.int64), [], [])
        step = np.diff(arr[:, 0])
        arr = arr[np.concatenate([[True], step != 0])]  # a repeated device time: dropped
        segment = np.concatenate([[0], np.cumsum(np.diff(arr[:, 0]) < 0)])
        t = np.empty(len(arr))
        fits: list[ClockFit] = []
        starts: list[float] = []
        for k in range(int(segment[-1]) + 1):
            m = segment == k
            fit = fit_device_clock(arr[m, 0], arr[m, 9])
            t[m] = to_laptop(fit, arr[m, 0])
            fits.append(fit)
            starts.append(float(t[m][0]))
        before = np.concatenate([[-np.inf], np.maximum.accumulate(t)[:-1]])
        keep = t > before
        return _ImuTable(arr[keep], t[keep], segment[keep], fits, starts)

    def summary(self) -> dict:
        """The take in numbers: ``duration_s`` (s from the start to the last frame), ``frames``,
        ``pose_frames`` (frames with a person found), ``imu_samples`` (those that will be
        written), ``rate_hz`` (samples per second of device time), ``dropped`` (samples missing
        from the ``seq`` counter), ``reps_by_arm`` (``{"l"|"r"|"all": n}``, ``{}`` for
        ``idle``), ``faults`` (``{counter: {code: n}}``, as currently judged),
        ``gate_open_pct`` (% of frames with the gate open) and ``latency_ms_median`` (median
        cue latency in ms, ``None`` without any). With a restarted device clock, ``rate_hz``
        and ``dropped`` are summed over its segments."""
        imu = self._imu_table()
        intervals, span, dropped = 0, 0.0, 0
        for k in np.unique(imu.segment):
            rows = imu.rows[imu.segment == k]
            if len(rows) > 1:
                intervals += len(rows) - 1
                span += float(rows[-1, 0] - rows[0, 0])
                dropped += _dropped(rows[:, 8])
        frames = self._frames
        return {
            "duration_s": round(max(frames[-1].t, 0.0), 3) if frames else 0.0,
            "frames": len(frames),
            "pose_frames": sum(f.world is not None for f in frames),
            "imu_samples": len(imu.rows),
            "rate_hz": round(intervals / span, 2) if span > 0 else 0.0,
            "dropped": dropped,
            "reps_by_arm": {k: int(v["reps"]) for k, v in self._totals.items()},
            "faults": {k: dict(v["faults_by_code"]) for k, v in self._totals.items()},
            "gate_open_pct": (
                round(100.0 * sum(f.gate for f in frames) / len(frames), 1) if frames else 0.0
            ),
            "latency_ms_median": (
                float(np.median([x["latency_ms"] for x in self._latency]))
                if self._latency
                else None
            ),
        }

    # ---- writing -----------------------------------------------------------------------------

    def write(self, root: Path, *, true_reps: int | None) -> Path:
        """Write the take to ``<root>/<subject>/<YYYYMMDD-HHMMSS>/`` and append its line to
        ``<root>/<subject>/sets.csv``; returns the folder. ``true_reps`` is the operator's rep
        count *per arm* (``None`` when not given): for ``arm == "both"`` the reps done with each
        arm. The files are written to a hidden temporary folder first and renamed into place, so
        a failed write leaves no partial take behind.

        Once the folder is in place the take is written, whatever happens to ``sets.csv``:
        if its line cannot be added (the file is locked or not writable, has a header this
        code does not know, or cannot be read as CSV) the reason is added to :attr:`warnings`
        as a sentence and the method still returns the folder. A restarted device clock is a
        warning too, and IMU samples that still fail the IMUStream checks are left out
        (``imu.parquet`` with no rows) with a warning rather than losing the take.

        Raises:
            FileExistsError: the folder already exists (nothing is written).
            OSError, ValueError: the folder could not be written or renamed into place (nothing
                is left).
        """
        subject_dir = Path(root) / self.labels.subject
        final = subject_dir / self.session_id
        if final.exists():
            raise FileExistsError(f"{final} already exists")
        tmp = subject_dir / f".{self.session_id}.partial"
        if tmp.exists():  # left by an earlier failed write of this same take
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        self.warnings = []  # a failed earlier write must not repeat its warnings
        try:
            imu, n_written = self._write_imu(tmp)
            self._write_pose(tmp)
            self._log.write_tables(tmp)
            meta = self._meta(imu, n_written, true_reps)
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str), "utf-8")
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        sets_path = subject_dir / "sets.csv"
        try:
            rows = _read_sets(sets_path)
            _append_set(sets_path, rows, self.session_id, self.labels, true_reps)
        except Exception as exc:  # the take is in place: never lose it over sets.csv
            self.warnings.append(f"could not update {sets_path}: {exc}")
        return final

    def _imu_frame(self, rows: np.ndarray, t: np.ndarray):
        """The IMUStream table of ``rows`` (``(n, 10)``) at laptop times ``t`` (s)."""
        df = schema.make_imu_stream(
            dataset="team",
            subject=self.labels.subject,
            session=self.session_id,
            device="formcoach",
            placement=PLACEMENT,
            t=t - self.t0_mono,
            acc=rows[:, 1:4],
            gyr=rows[:, 4:7],
            exercise=np.full(len(rows), self.labels.exercise, dtype=object),
        )
        df["flags"] = rows[:, 7].astype(np.int16)
        df["seq"] = rows[:, 8].astype(np.int32)
        df["t_arrival"] = rows[:, 9]
        df["t_device"] = rows[:, 0]
        return df

    def _write_imu(self, folder: Path) -> tuple[_ImuTable, int]:
        """Write ``imu.parquet``; returns the IMU table and the number of rows written."""
        imu = self._imu_table()
        for start in imu.starts[1:]:
            self.warnings.append(
                f"device clock restarted at t={start - self.t0_mono:.2f} s (the watch rebooted "
                "or another watch took over); its samples were fitted to the laptop clock "
                "separately"
            )
        df = self._imu_frame(imu.rows, imu.t)
        try:
            schema.validate_imu_stream(df)
        except schema.SchemaError as exc:
            self.warnings.append(
                f"the IMU samples were left out ({exc}); imu.parquet has no rows, the pose, "
                "events and frames are kept"
            )
            df = self._imu_frame(imu.rows[:0], imu.t[:0])
            schema.validate_imu_stream(df)
        df.to_parquet(folder / "imu.parquet", index=False)
        return imu, len(df)

    def _write_pose(self, folder: Path) -> None:
        n = len(self._frames)
        image = np.full((n, N_LANDMARKS, 4), np.nan, dtype=np.float32)
        world = np.full((n, N_LANDMARKS, 3), np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        for i, f in enumerate(self._frames):
            if f.world is not None:
                image[i] = f.image
                world[i] = f.world
                valid[i] = True
        write_pose(
            folder / "pose.parquet",
            session=self.session_id,
            frames=np.array([f.index for f in self._frames], dtype=np.int64),
            t=np.array([f.t for f in self._frames], dtype=np.float64),
            world=world,
            skeleton="mediapipe33",
            image=image,
            valid=valid,
        )

    def _clock_meta(self, imu: _ImuTable) -> dict:
        """``meta.json`` ``clock``: the fit of the segment with the most samples kept (all
        zeros without samples), ``t0_mono``, and ``segments``: every segment's fit with
        ``t_start``, its first sample's time (s, take clock)."""
        if imu.fits:
            counts = np.bincount(imu.segment, minlength=len(imu.fits))
            main = imu.fits[int(np.argmax(counts))]
        else:
            main = fit_device_clock(np.zeros(0), np.zeros(0))
        segments = [
            {"t_start": start - self.t0_mono, **asdict(fit)}
            for fit, start in zip(imu.fits, imu.starts, strict=True)
        ]
        return {**asdict(main), "t0_mono": self.t0_mono, "segments": segments}

    def _meta(self, imu: _ImuTable, n_written: int, true_reps: int | None) -> dict:
        summary = self.summary()
        ctx = self.context
        return {
            # the SessionRecorder keys
            "subject": self.labels.subject,
            "exercise": self.labels.exercise,
            "placement": PLACEMENT,
            "source": ctx["source"],
            "firmware": ctx["firmware"],
            "device_name": ctx["device"],
            "camera": ctx["camera"],
            "notes": self.labels.notes,
            "laptop": platform.node(),
            "os": platform.platform(),
            "formcoach": __version__,
            "session_id": self.session_id,
            "started": self.started.isoformat(timespec="seconds"),
            "ended": (self.started + timedelta(seconds=summary["duration_s"])).isoformat(
                timespec="seconds"
            ),
            "n_samples": n_written,
            "duration_s": summary["duration_s"],
            "dropped": summary["dropped"],
            # the Bench's own keys
            "bench": {"version": 1},
            "labels": asdict(self.labels),
            "arm": self.labels.arm,
            "expected_reps": true_reps,
            "clock": self._clock_meta(imu),
            **ctx,
            "overrides": dict(ctx["overrides"] or {}),
            "summary": summary,
            "latency": list(self._latency),  # every confirmed cue: [{arm, rep_id, latency_ms}]
        }


def _diagnostics(counters: dict[str, dict]) -> dict[str, float]:
    """The live log's per-counter frame columns (``Pipeline._frame_diagnostics``) from a
    snapshot's ``counters``: ``ang_k`` (deg, NaN if none), ``enter_k``/``exit_k`` (deg),
    ``armed_k`` (1.0/0.0) and, for arm counters, ``vis_k`` (0-1; 0.0 when nobody was found,
    as the pipeline's zero visibilities give)."""
    out: dict[str, float] = {}
    for k, c in counters.items():
        out[f"ang_{k}"] = float("nan") if c["angle"] is None else float(c["angle"])
        out[f"enter_{k}"] = float(c["enter"])
        out[f"exit_{k}"] = float(c["exit"])
        out[f"armed_{k}"] = 1.0 if c["armed"] else 0.0
        if k != "all":
            out[f"vis_{k}"] = 0.0 if c["visibility"] is None else float(c["visibility"])
    return out


def _dropped(seq: np.ndarray) -> int:
    """Samples missing from a 16-bit ``seq`` counter (as ``SessionRecorder`` counts them)."""
    if len(seq) < 2:
        return 0
    gaps = (np.diff(seq.astype(np.int64)) - 1) % 65536
    return int(gaps.clip(0, 32767).sum())


def _read_sets(path: Path) -> list[list[str]]:
    """The rows of ``sets.csv`` (header first; ``[]`` when there is no file).

    Raises:
        ValueError: the header is neither :data:`SETS_HEADER` nor its older five-column form.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    if rows and rows[0] not in (SETS_HEADER, SETS_HEADER[:-1]):
        raise ValueError(f"{path} has an unexpected header {rows[0]}; expected {SETS_HEADER}")
    return rows


def _append_set(
    path: Path, rows: list[list[str]], session_id: str, labels: TakeLabels, true_reps: int | None
) -> None:
    """Append one line to ``sets.csv`` (header :data:`SETS_HEADER`; ``rows`` its current
    content from :func:`_read_sets`), creating it, or first upgrading an older five-column file
    in place (its rows get an empty ``arm``)."""
    row = [
        session_id,
        labels.exercise,
        labels.form,
        "" if true_reps is None else str(int(true_reps)),
        labels.notes,
        labels.arm,
    ]
    if rows and rows[0] == SETS_HEADER:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerow(row)
        return
    body = [r + [""] * (len(SETS_HEADER) - len(r)) for r in rows[1:]]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows([SETS_HEADER, *body, row])
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
