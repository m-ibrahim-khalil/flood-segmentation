"""Statistical comparisons across folds and within fold-0 pixels.

Outputs:
  - 3-fold paired Wilcoxon for: no-KD vs comb-KD (per student); teacher vs best.
  - Fold-0 paired pixel-bootstrap (n=1000 resamples) for the same comparisons.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import wilcoxon

from floodlite.data import make_loaders
from floodlite.models import make_teacher, make_student, STUDENT_BACKBONES
from floodlite.metrics import compute_metrics


def per_image_iou(model, loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            for i in range(logits.size(0)):
                m = compute_metrics(logits[i:i+1], y[i:i+1])
                ious.append(m.iou)
    return np.array(ious)


def bootstrap_p(deltas, n_boot=1000, seed=42):
    """Paired bootstrap p-value: P(mean(delta_resample) <= 0)."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.array([rng.choice(deltas, n, replace=True).mean() for _ in range(n_boot)])
    return float((means <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Single results.json with fold_0/fold_1/fold_2 keys "
                         "(produced by run_full_experiment.py)")
    ap.add_argument("--data_root", default=None,
                    help="FSSD root for the per-image bootstrap. If omitted, only "
                         "the 3-fold Wilcoxon is computed.")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", default="runs/3fold/stats.json")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())

    # ----- 3-fold Wilcoxon -----
    wilc = {}
    for sname in STUDENT_BACKBONES:
        none_iou, comb_iou = [], []
        for fold in (0, 1, 2):
            students = results[f"fold_{fold}"]["students"]
            none_iou.append(students[f"{sname}_none"]["iou"])
            comb_iou.append(students[f"{sname}_comb"]["iou"])
        try:
            stat, p = wilcoxon(none_iou, comb_iou)
            wilc[f"{sname}_none_vs_comb"] = {
                "n": 3, "p_value": float(p), "stat": float(stat),
                "mean_delta": float(np.mean(np.array(none_iou) - np.array(comb_iou))),
                "note": "n=3 -> minimum two-sided p ≈ 0.25 (low power)",
            }
        except ValueError as e:
            wilc[f"{sname}_none_vs_comb"] = {"n": 3, "error": str(e)}

    # ----- Fold-0 paired pixel-bootstrap (optional, needs FSSD locally) -----
    boot = None
    if args.data_root and Path(args.data_root).exists():
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _, va = make_loaders(args.data_root, fold=0, batch_size=8)
            boot = {}
            for sname in STUDENT_BACKBONES:
                s_none = make_student(sname).to(device)
                s_none.load_state_dict(torch.load(
                    Path(args.ckpt_dir) / f"{sname}_none_fold0.pt", map_location=device))
                s_comb = make_student(sname).to(device)
                s_comb.load_state_dict(torch.load(
                    Path(args.ckpt_dir) / f"{sname}_comb_fold0.pt", map_location=device))
                none_per = per_image_iou(s_none, va, device)
                comb_per = per_image_iou(s_comb, va, device)
                deltas = none_per - comb_per
                boot[f"{sname}_none_vs_comb"] = {
                    "n": int(len(deltas)),
                    "mean_delta": float(deltas.mean()),
                    "p_one_sided_no_kd_better": bootstrap_p(deltas),
                }
        except Exception as e:
            boot = {"error": f"bootstrap skipped: {e}"}
    else:
        boot = {"skipped_reason": "no --data_root provided or path missing"}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = {"wilcoxon_3fold": wilc, "bootstrap_fold0": boot}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print()
    print("=== Wilcoxon (3-fold paired) ===")
    for k, v in wilc.items():
        if "error" in v:
            print(f"  {k}: {v['error']}")
        else:
            print(f"  {k}: mean Δ={v['mean_delta']:+.4f}  p={v['p_value']:.4f}")
    if boot and "skipped_reason" not in boot and "error" not in boot:
        print()
        print("=== Bootstrap (fold-0 per-image, n=1000) ===")
        for k, v in boot.items():
            print(f"  {k}: mean Δ={v['mean_delta']:+.4f}  p_one_sided={v['p_one_sided_no_kd_better']:.4f}")


if __name__ == "__main__":
    main()
