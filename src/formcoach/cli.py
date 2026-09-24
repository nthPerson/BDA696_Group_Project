"""``formcoach`` command-line entry point (typer).

Every Makefile workflow target maps to one command here so teammates without ``make``
(plain PowerShell on Windows) can run the same thing with ``uv run formcoach ...``.

Commands that are not implemented yet are *stubs*: they print exactly what they will do and
which checkpoint in docs/00-START-HERE.md delivers them, then exit 0. Replace the stub body,
keep the signature, add a test.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from formcoach import __version__

console = Console()
app = typer.Typer(
    name="formcoach",
    help="Sensor-gated webcam form coaching (BDA 696, SDSU Fall 2026).",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
data_app = typer.Typer(help="Fetch, verify and profile datasets.", no_args_is_help=True)
features_app = typer.Typer(help="Build IMU windows and features.", no_args_is_help=True)
pose_app = typer.Typer(help="Pose extraction from video.", no_args_is_help=True)
train_app = typer.Typer(help="Train and export models.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation harness (writes reports/).", no_args_is_help=True)
session_app = typer.Typer(help="Inspect recorded sessions.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(pose_app, name="pose")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(session_app, name="session")


def stub(command: str, checkpoint: int, what: str) -> None:
    """Print a clearly-labelled not-implemented notice and return (exit code 0)."""
    console.print(f"[bold yellow]STUB[/] [cyan]{command}[/] (Checkpoint {checkpoint})")
    console.print(f"  Will: {what}")
    console.print("  See docs/00-START-HERE.md and docs/STATUS.md for progress.")


class Dataset(StrEnum):
    mmfit = "mmfit"
    recofit = "recofit"
    recgym = "recgym"


class Source(StrEnum):
    replay = "replay"
    serial = "serial"
    ble = "ble"


class Gate(StrEnum):
    always_on = "always_on"
    energy = "energy"
    laptop = "laptop"
    device = "device"


def _version(value: bool) -> None:
    if value:
        console.print(f"formcoach {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """FormCoach CLI."""


# ---- data ------------------------------------------------------------------------------------
@data_app.command("fetch")
def data_fetch(
    dataset: Annotated[Dataset, typer.Option(help="Which public dataset to download.")],
    with_video: Annotated[
        bool, typer.Option(help="MM-Fit only: also download RGB video for --session workouts.")
    ] = False,
    session: Annotated[
        list[str] | None, typer.Option(help="MM-Fit workout ids for --with-video, e.g. w00.")
    ] = None,
    root: Annotated[
        Path | None, typer.Option(help="Download root (default: data/external).")
    ] = None,
    verify_only: Annotated[
        bool, typer.Option(help="Only check files already on disk; never download.")
    ] = False,
    by: Annotated[
        str | None, typer.Option(help="Name written to the manifest 'By' column.")
    ] = None,
) -> None:
    """Download a dataset into data/external/<name>/, verify SHA-256, update MANIFEST.md.

    Resumable: interrupted downloads continue from the `.partial` file. Files that do not match
    the registry are reported, never deleted (rule 7). See docs/howto/datasets.md.
    """
    from formcoach.data import fetch

    manifest_path = fetch.default_manifest_path() if root is None else None
    try:
        results = fetch.fetch_dataset(
            dataset.value,
            root,
            with_video=with_video,
            sessions=session or ["w00"],
            manifest_path=manifest_path,
            by=by,
            verify_only=verify_only,
        )
    except fetch.IntegrityError as exc:
        console.print(f"[bold red]integrity error:[/] {exc}")
        raise typer.Exit(2) from exc
    missing = 0
    for r in results:
        colour = {"downloaded": "green", "verified": "cyan", "missing": "red"}[r.status]
        extra = " (extracted)" if r.extracted else ""
        console.print(f"[{colour}]{r.status:10s}[/] {r.path}{extra}")
        missing += r.status == "missing"
    if manifest_path is not None:
        console.print(f"manifest: {manifest_path}")
    if missing:
        console.print(
            f"[yellow]{missing} file(s) missing; run without --verify-only to download.[/]"
        )
        raise typer.Exit(1)


@data_app.command("profile")
def data_profile(
    dataset: Annotated[
        Dataset | None, typer.Option(help="Profile one dataset (default: all present).")
    ] = None,
) -> None:
    """Write reports/data_profile.md: subjects, minutes, class balance, sampling checks."""
    stub(
        "data profile",
        1,
        f"describe {dataset.value if dataset else 'every downloaded dataset'} and write "
        "reports/data_profile.md with tables and figures.",
    )


# ---- features --------------------------------------------------------------------------------
@features_app.command("build")
def features_build(
    dataset: Annotated[Dataset | None, typer.Option(help="Limit to one dataset.")] = None,
) -> None:
    """Resample to 50 Hz, filter, window (2 s / 50 %), featurize -> data/processed/."""
    stub(
        "features build",
        2,
        f"build windows + features for {dataset.value if dataset else 'all datasets'} into "
        "data/processed/<dataset>/<subject>-<session>.parquet.",
    )


# ---- pose ------------------------------------------------------------------------------------
@pose_app.command("extract")
def pose_extract(
    video: Annotated[Path, typer.Option(help="Video file to process.")],
    model: Annotated[str, typer.Option(help="lite | full | heavy")] = "lite",
) -> None:
    """Run MediaPipe PoseLandmarker over a video -> pose.parquet next to it (checkpointed)."""
    stub("pose extract", 3, f"extract {model} pose landmarks from {video} into pose.parquet.")


# ---- train -----------------------------------------------------------------------------------
@train_app.command("gate")
def train_gate(
    model: Annotated[str, typer.Option(help="rf | cnn")] = "cnn",
    export_int8: Annotated[bool, typer.Option(help="Also export int8 TFLite + .cc")] = True,
) -> None:
    """Train the gate / exercise model on RecoFit + MM-Fit windows (LOSO validation)."""
    stub(
        "train gate",
        5 if model == "cnn" else 2,
        f"train the {model} gate model, report LOSO metrics"
        + (", export int8 TFLite to firmware/model/." if export_int8 else "."),
    )


# ---- eval ------------------------------------------------------------------------------------
@eval_app.command("all")
def eval_all() -> None:
    """Regenerate every table and figure under reports/ (what `make eval` runs)."""
    stub("eval all", 6, "run loso, repcount, rules, gating, pose-ablation, latency, device.")


@eval_app.command("loso")
def eval_loso(
    model: Annotated[str, typer.Option(help="energy | rf | cnn | cnn-int8")] = "rf",
    dataset: Annotated[Dataset, typer.Option()] = Dataset.recofit,
) -> None:
    """Leave-one-subject-out accuracy / macro-F1 / confusion matrix."""
    stub("eval loso", 2, f"LOSO-evaluate {model} on {dataset.value} -> reports/loso_{model}.md.")


@eval_app.command("repcount")
def eval_repcount(source: Annotated[str, typer.Option(help="imu|pose|fused|peaks")] = "peaks"):
    """Rep-count MAE against MM-Fit labels."""
    stub("eval repcount", 2, f"rep-count MAE for {source} on MM-Fit -> reports/repcount.md.")


@eval_app.command("rules")
def eval_rules() -> None:
    """Correct-form pass rate + synthetic perturbation detection on MM-Fit pose."""
    stub("eval rules", 6, "validate rules.yaml on MM-Fit correct-form reps and perturbations.")


@eval_app.command("gating")
def eval_gating(gate: Annotated[Gate, typer.Option()] = Gate.always_on) -> None:
    """Gated vs always-on: frames processed, CPU %, wall time, identical events check."""
    stub("eval gating", 6, f"benchmark gate={gate.value} on recorded sessions.")


@eval_app.command("pose-ablation")
def eval_pose_ablation() -> None:
    """MediaPipe Lite / Full / Heavy (YOLO11n optional): agreement, FPS, CPU."""
    stub("eval pose-ablation", 6, "compare pose model variants.")


@eval_app.command("latency")
def eval_latency() -> None:
    """Distribution of rep-end -> feedback latency."""
    stub("eval latency", 6, "measure end-to-end feedback latency on live/replay sessions.")


@eval_app.command("device")
def eval_device(log: Annotated[Path | None, typer.Option(help="Firmware log file")] = None):
    """Parse a firmware log: inference ms, arena bytes, flash bytes."""
    stub("eval device", 5, f"parse on-device metrics from {log or 'the serial log'}.")


@eval_app.command("transfer")
def eval_transfer() -> None:
    """Public-trained models evaluated on data/team/ recordings."""
    stub("eval transfer", 6, "evaluate sensor transfer on team validation recordings.")


# ---- demo / record / session ------------------------------------------------------------------
@app.command("demo")
def demo(
    source: Annotated[Source, typer.Option(help="replay | serial | ble")] = Source.replay,
    session: Annotated[
        str | None, typer.Option(help="Replay session id (dataset stream or data/team dir).")
    ] = None,
    gate: Annotated[Gate, typer.Option(help="Gate implementation to use.")] = Gate.device,
    headless: Annotated[bool, typer.Option(help="No window; print rep/fault events.")] = False,
    port: Annotated[str | None, typer.Option(help="Serial port for --source serial.")] = None,
) -> None:
    """Run the full pipeline: IMU source -> gate -> pose -> reps -> rules -> overlay.

    `--source replay` needs no hardware and must always work; it is the demo of record.
    """
    where = f"session {session}" if session else "the bundled 30-second fixture"
    mode = "headless (events to stdout)" if headless else "with the OpenCV overlay window"
    stub(
        f"demo --source {source.value}",
        3 if source is Source.replay else 4,
        f"replay {where} from {source.value}{f' on {port}' if port else ''} through gate="
        f"{gate.value}, pose, rep segmentation and rules, {mode}.",
    )


@app.command("record")
def record(
    source: Annotated[Source, typer.Option(help="serial | ble")] = Source.serial,
    port: Annotated[str | None, typer.Option(help="e.g. /dev/ttyACM0, COM5")] = None,
    subject: Annotated[str, typer.Option(help="Subject code S1..S5")] = "S1",
    exercise: Annotated[str, typer.Option(help="curl | press | raise | squat")] = "curl",
    camera: Annotated[int | None, typer.Option(help="Webcam index; omit for IMU only.")] = None,
) -> None:
    """Record a session to data/team/<subject>/<session_id>/ (imu.parquet, meta.json, ...)."""
    stub(
        f"record --source {source.value}",
        4,
        f"log {exercise} for {subject} from {source.value}"
        f"{f' on {port}' if port else ''}"
        f"{f' + camera {camera}' if camera is not None else ''} into data/team/{subject}/.",
    )


@session_app.command("check")
def session_check(session_dir: Annotated[Path, typer.Argument(help="data/team/<S#>/<id>")]):
    """Quality check: sample drops, gate timeline, frames processed, reps vs expected."""
    stub("session check", 4, f"summarize {session_dir} and flag problems.")


if __name__ == "__main__":  # pragma: no cover
    app()
