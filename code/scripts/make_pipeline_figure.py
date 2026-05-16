"""make_pipeline_figure.py — emit F1 (pipeline diagram) for the FloodLite manuscript.

Four-stage horizontal pipeline:
    1. Train teacher         (UNet + EfficientNet-B0)
    2. Train students        (3 backbones x 4 KD configs)
    3. INT8 PTQ              (ONNX static QDQ, Percentile 99.999% calibration)
    4. Benchmark             (M2 Pro ORT-CPU + browser WASM)

Output: figures/F1_pipeline.pdf (vector) + .png (raster).

Usage:
    python3 scripts/make_pipeline_figure.py --out_dir ../manuscript/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Stage colors -- muted, colorblind-safe, journal-friendly.
COLOR_TEACHER  = "#5b8def"   # blue
COLOR_STUDENTS = "#7bc97b"   # green
COLOR_QUANT    = "#f0a35a"   # orange
COLOR_BENCH    = "#c084e0"   # purple
COLOR_EDGE     = "#1f1f1f"
COLOR_NOTE     = "#555555"


def _tint(hex_color: str, alpha: float = 0.18) -> tuple[float, float, float, float]:
    """Return a translucent fill colour for the body of each stage box."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


STAGES = [
    {
        "title":  "1. Train teacher",
        "body":   ["UNet $+$ EfficientNet-B0", "6.25 M params",
                   "FSSD, 35 epochs, T4 GPU", "Task loss only"],
        "color":  COLOR_TEACHER,
    },
    {
        "title":  "2. Train students",
        "body":   ["MobileNetV3-S  (3.59 M)", "EfficientNet-Lite0  (5.61 M)",
                   "MobileViT-XXS  (3.09 M)",
                   r"$\times$ \{none, resp, feat, comb\} KD"],
        "color":  COLOR_STUDENTS,
    },
    {
        "title":  "3. INT8 quantization",
        "body":   ["ONNX static QDQ", "Percentile 99.999\\% calib.",
                   "Conv/Gemm/MatMul per-ch.", "no-KD students only"],
        "color":  COLOR_QUANT,
    },
    {
        "title":  "4. Benchmark",
        "body":   ["M2 Pro: ORT CPU EP", "Browser: ORT-Web WASM SIMD",
                   "20 warm $+$ 200 timed runs", "same .onnx, two providers"],
        "color":  COLOR_BENCH,
    },
]


def _draw_stage(ax, x, y, w, h, stage):
    """Single rounded box: tinted fill, dark edge, bold coloured title + body lines."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.16",
        linewidth=1.2, edgecolor=stage["color"],
        facecolor=_tint(stage["color"], alpha=0.13),
        zorder=2,
    )
    ax.add_patch(box)

    title_y = y + h - 0.32
    ax.text(x + w / 2, title_y, stage["title"],
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color=stage["color"], zorder=4)

    # Coloured separator under the title.
    ax.plot([x + 0.18, x + w - 0.18], [title_y - 0.18, title_y - 0.18],
            color=stage["color"], linewidth=1.0, alpha=0.55, zorder=3)

    # Body lines.
    body_top = title_y - 0.42
    body_bottom = y + 0.18
    n = len(stage["body"])
    if n == 1:
        ys = [(body_top + body_bottom) / 2]
    else:
        ys = [body_top - i * (body_top - body_bottom) / (n - 1) for i in range(n)]
    for line, ly in zip(stage["body"], ys):
        ax.text(x + w / 2, ly, line,
                ha="center", va="center", fontsize=9.0,
                color="#222222", zorder=4)


def _draw_arrow(ax, x0, y, x1, label=None):
    arrow = FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.6, color=COLOR_EDGE, zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        # Below the arrow so the label cannot clash with the title text of
        # the boxes on either side.
        ax.text((x0 + x1) / 2, y - 0.16, label,
                ha="center", va="top", fontsize=8.5,
                color=COLOR_NOTE, style="italic", zorder=2)


def make_pipeline(out_path: Path):
    fig, ax = plt.subplots(figsize=(13.0, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Header band.
    ax.text(6.5, 3.35, "FloodLite training-and-deployment pipeline",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="#111111")
    ax.text(6.5, 3.07,
            "3-fold cross-validation on FSSD; one ONNX artifact per "
            "(student, fold, precision), benchmarked under two execution providers.",
            ha="center", va="center", fontsize=8.8, color=COLOR_NOTE)

    # Layout: 4 stages, equally spaced. Boxes narrower than half the gap so
    # the arrow labels between consecutive boxes don't run into the next box.
    n = len(STAGES)
    box_w = 2.45
    box_h = 2.10
    box_y = 0.55
    gap = (13 - n * box_w) / (n + 1)

    x_centres = []
    for i, stage in enumerate(STAGES):
        x = gap + i * (box_w + gap)
        _draw_stage(ax, x, box_y, box_w, box_h, stage)
        x_centres.append((x, x + box_w))

    # Arrows between consecutive stages.
    arrow_y = box_y + box_h / 2
    transition_labels = [
        "frozen",        # teacher -> students
        "best ckpt",     # students -> quant
        ".onnx",         # quant -> bench
    ]
    for i in range(n - 1):
        right_of_left = x_centres[i][1]
        left_of_right = x_centres[i + 1][0]
        # Inset the arrow ends only slightly so the arrow itself is clearly
        # visible (not just the arrowhead).
        _draw_arrow(ax, right_of_left + 0.04, arrow_y, left_of_right - 0.04,
                    label=transition_labels[i])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[F1] wrote {out_path}")
    print(f"[F1] wrote {out_path.with_suffix('.png')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=Path("../manuscript/figures"),
                    help="Output directory (default: ../manuscript/figures)")
    ap.add_argument("--name", default="F1_pipeline",
                    help="Filename stem; .pdf and .png will be written")
    args = ap.parse_args()
    make_pipeline(args.out_dir / f"{args.name}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
