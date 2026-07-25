"""F5 --- KD gain vs training-data fraction (review Mod4).

Reads runs/lowdata/lowdata_kd.json (from run_lowdata_kd.py) and plots the KD gain
IoU(comb) - IoU(none) against the training-data fraction, per student and pooled.
The expected story: gain ~ 0 at frac = 1.0 (the paper's saturated negative result)
rising above 0 as data shrinks (KD helps when the benchmark is not saturated).

Usage:
    python scripts/make_lowdata_figure.py --lowdata runs/lowdata/lowdata_kd.json \
        --out_dir figures
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHORT = {"mobilenetv3_small": "MNV3-S", "efficientnet_lite0": "EffLite0",
         "mobilevit_xxs": "MViT-XXS"}
COLORS = {"mobilenetv3_small": "#4daf4a", "efficientnet_lite0": "#ff7f00",
          "mobilevit_xxs": "#984ea3"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowdata", default="runs/lowdata/lowdata_kd.json")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--kd", default="comb", help="KD config to contrast against 'none'.")
    args = ap.parse_args()

    blob = json.loads(Path(args.lowdata).read_text())
    results = blob["results"]
    fracs = sorted({float(f) for f in results}, )
    students = blob.get("meta", {}).get("students") or sorted(
        {s for f in results.values() for fo in f.values() for s in fo})

    def gains_for(student, frac):
        out = []
        for fold, cell in results.get(f"{frac}", {}).items():
            c = cell.get(student, {})
            if "none" in c and args.kd in c:
                out.append(c[args.kd]["iou"] - c["none"]["iou"])
        return out

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="#888", lw=1, ls="--", zorder=1)

    # per-student lines (mean +/- std over folds)
    for s in students:
        xs, ys, es = [], [], []
        for fr in fracs:
            g = gains_for(s, fr)
            if g:
                xs.append(fr); ys.append(float(np.mean(g))); es.append(float(np.std(g)))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.5,
                        color=COLORS.get(s, "#333"), label=SHORT.get(s, s), alpha=0.9)

    # pooled mean across students+folds (thick black)
    px, py = [], []
    for fr in fracs:
        allg = [v for s in students for v in gains_for(s, fr)]
        if allg:
            px.append(fr); py.append(float(np.mean(allg)))
    if px:
        ax.plot(px, py, marker="s", ms=8, lw=2.5, color="black",
                label="pooled mean", zorder=5)

    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.set_xlabel("Training-data fraction (of FSSD train split)", fontsize=11)
    ax.set_ylabel(f"KD gain: IoU({args.kd}) - IoU(none)", fontsize=11)
    ax.set_title("F5 - Knowledge distillation helps only when data is scarce",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=9, loc="best")
    # annotate the two regimes
    ax.annotate("saturated:\nKD ~ 0 (paper's result)", xy=(1.0, 0.0),
                xytext=(0.55, 0.4), textcoords="axes fraction", fontsize=8.5,
                color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999"))

    fig.tight_layout()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"F5_lowdata_kd.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("Saved", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
