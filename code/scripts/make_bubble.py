"""make_bubble.py -- consolidated multi-objective bubble figure (F5) for FloodLite.

One figure, four objectives:
  x = Apple M2 Pro ORT-CPU p50 latency (ms)   [Table 5]
  y = IoU                                      [Tables 2 (FP32) / 4 (INT8)]
  bubble area proportional to parameters (M)   [Table 1]
  colour = ONNX disk size (MB)                 [Table 1]  -> shows INT8 shrink
  marker: FP32 = circle, INT8 = diamond

All values are the committed manuscript numbers (no new experiments); sources
noted per point below. Headless-safe (Agg).

Usage:
    python3 scripts/make_bubble.py --out_dir ../manuscript/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# name, m2_p50_ms, iou, params_M, onnx_MB, precision, note
POINTS = [
    ("Teacher (EffB0)",   37.6, 0.9257, 6.25, 22.3, "FP32", None),
    ("MNV3-S",            19.5, 0.9269, 3.59, 13.8, "FP32", None),
    ("MNV3-S",            13.7, 0.476,  3.59,  3.7, "INT8", "PTQ collapse"),
    ("EffLite0",          34.2, 0.9277, 5.61, 19.8, "FP32", None),
    ("EffLite0",          17.3, 0.885,  5.61,  5.2, "INT8", "deployable"),
    ("MobileViT-XXS",     28.3, 0.9367, 3.09, 12.0, "FP32", "max-acc"),
    ("MobileViT-XXS",     21.6, 0.475,  3.09,  3.5, "INT8", "PTQ collapse"),
]


def build(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    sizes_mb = [p[4] for p in POINTS]
    vmin, vmax = min(sizes_mb), max(sizes_mb)
    cmap = plt.get_cmap("viridis")

    def area(params_M: float) -> float:
        # bubble area ~ params; scale for visibility
        return 90.0 * params_M

    for name, x, y, params, mb, prec, note in POINTS:
        marker = "o" if prec == "FP32" else "D"
        color = cmap((mb - vmin) / (vmax - vmin))
        ax.scatter(x, y, s=area(params), c=[color], marker=marker,
                   edgecolors="black", linewidths=0.8, alpha=0.9, zorder=3)
        # highlight the two recommended operating points with a ring
        if note in ("deployable", "max-acc"):
            ax.scatter(x, y, s=area(params) + 340, facecolors="none",
                       edgecolors="#d62728", linewidths=2.0, zorder=4)

    # per-point labels (staggered to avoid collisions)
    lbl = dict(fontsize=8.5, zorder=6)
    ax.annotate("MNV3-S FP32", (19.5, 0.9269), textcoords="offset points",
                xytext=(0, 14), ha="center", **lbl)
    ax.annotate("MobileViT-XXS FP32\n(max accuracy)", (28.3, 0.9367),
                textcoords="offset points", xytext=(0, 12), ha="center", **lbl)
    ax.annotate("EffLite0 FP32", (34.2, 0.9277), textcoords="offset points",
                xytext=(0, -22), ha="center", **lbl)
    ax.annotate("Teacher (EffB0)", (37.6, 0.9257), textcoords="offset points",
                xytext=(0, 12), ha="center", **lbl)
    ax.annotate("EffLite0 INT8\n(deployable)", (17.3, 0.885),
                textcoords="offset points", xytext=(14, 4), ha="left", **lbl)
    ax.annotate("MNV3-S & MobileViT-XXS INT8: PTQ collapse (Table 4)",
                (26.5, 0.585), ha="center", fontsize=8.5, color="#555555",
                style="italic", zorder=6)

    ax.set_xlabel("Apple M2 Pro ORT-CPU latency, p50 (ms) --- lower is better")
    ax.set_ylabel("IoU --- higher is better")
    ax.set_ylim(0.42, 0.99)
    ax.set_xlim(10.5, 41.5)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)

    # colourbar for ONNX size
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("ONNX disk size (MB)")

    # legends: precision/recommended (lower right) + bubble-size key (mid right)
    marker_leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=9, label="FP32"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=9, label="INT8"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#d62728", markersize=13, markeredgewidth=2,
               label="recommended"),
    ]
    size_leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbb",
               markeredgecolor="black", markersize=np.sqrt(area(p)) / 3.2,
               label=f"{p:.0f} M params")
        for p in (3.0, 6.0)
    ]
    leg1 = ax.legend(handles=marker_leg, loc="lower right", fontsize=8.5,
                     framealpha=0.95, title="precision")
    ax.add_artist(leg1)
    ax.legend(handles=size_leg, loc="center right", fontsize=8.5,
              framealpha=0.95, title="bubble area = params", labelspacing=1.9,
              borderpad=1.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"F5_bubble.{ext}", dpi=300, bbox_inches="tight")
    print(f"Wrote {out_dir / 'F5_bubble.png'} and .pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../manuscript/figures")
    args = ap.parse_args()
    build(Path(args.out_dir))


if __name__ == "__main__":
    main()
