"""Draw Figure 1 of the report (system architecture) to architecture.png.

Run from the repository root:  uv run python docs/report/figures/make_architecture.py
The figure is a diagram, not a result: it contains no measured numbers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX_W, BOX_H, STEP = 1.42, 0.9, 1.6
LANE_FILL = {"wearable": "#fdf1dc", "laptop": "#e3eef9"}
BOX_EDGE = "#333333"

WEARABLE = [
    "BMI160 IMU\n50 Hz, 6 axes",
    "2 s ring buffer\n100 x 6 samples",
    "int8 1D-CNN gate\nTFLite Micro\nevery 0.5 s",
    "Hysteresis\nopen 1 s\nclose 6 s",
    "BLE notify\nIMU samples\n+ gate flag",
]
LAPTOP = [
    "IMU source\nBLE / serial\n/ replay",
    "Gate logic",
    "Webcam capture\nonly while\ngate is open",
    "MediaPipe\npose\n33 landmarks",
    "Normalize,\njoint angles",
    "Repetition\nsegmentation",
    "Rules engine\nrules.yaml",
    "Overlay,\nevent log,\nLED code",
]


def _row(ax, labels, x0, y):
    """Draw one row of boxes joined by arrows; return the box centres."""
    centres = []
    for i, text in enumerate(labels):
        x = x0 + i * STEP
        ax.add_patch(
            FancyBboxPatch(
                (x, y), BOX_W, BOX_H, boxstyle="round,pad=0.03", fc="white", ec=BOX_EDGE, lw=1.0
            )
        )
        ax.text(x + BOX_W / 2, y + BOX_H / 2, text, ha="center", va="center", fontsize=7.2)
        centres.append((x + BOX_W / 2, y + BOX_H / 2))
        if i:
            _arrow(ax, (x - (STEP - BOX_W), y + BOX_H / 2), (x, y + BOX_H / 2))
    return centres


def _arrow(ax, start, end, **kw):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#333333", **kw
        )
    )


def main() -> None:
    width = len(LAPTOP) * STEP + 0.3
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, width)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    for name, y, title in (
        ("wearable", 2.45, "Wearable: Seeed XIAO ESP32-S3 (firmware)"),
        ("laptop", 0.15, "Laptop: formcoach Python package (one process, CPU only)"),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y),
                width - 0.1,
                1.6,
                boxstyle="round,pad=0.02",
                fc=LANE_FILL[name],
                ec="none",
            )
        )
        ax.text(0.2, y + 1.42, title, fontsize=9, fontweight="bold", va="center")

    wear = _row(ax, WEARABLE, 0.25, 2.6)
    lap = _row(ax, LAPTOP, 0.25, 0.3)

    # BLE link: last wearable box down to the laptop's IMU source.
    link_y = 2.2
    ax.plot([wear[-1][0], wear[-1][0], lap[0][0]], [2.6, link_y, link_y], color="#333333", lw=1.0)
    _arrow(ax, (lap[0][0], link_y), (lap[0][0], 0.3 + BOX_H))
    ax.text(
        (wear[-1][0] + lap[0][0]) / 2,
        link_y - 0.13,
        "Bluetooth Low Energy (USB serial as fallback)",
        fontsize=7.5,
        ha="center",
        va="top",
    )

    # Feedback: LED code from the laptop back to the wearable.
    _arrow(ax, (lap[-1][0], 0.3 + BOX_H), (lap[-1][0], 2.45), linestyle="dashed")
    ax.text(lap[-1][0] - 0.1, 2.08, "LED feedback code", fontsize=7.5, ha="right")

    out = Path(__file__).with_name("architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
