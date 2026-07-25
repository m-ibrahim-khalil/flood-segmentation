"""Powered per-image paired bootstrap for the negative-KD claim (review Mod1).

The manuscript's KD and teacher-vs-student claims currently rest on n=3
fold-means (paired Wilcoxon, minimum reportable two-sided p = 0.25). This script
replaces those 3 observations with per-image IoU pooled across all three folds
(~398 held-out images), giving real statistical power:

  * For each architecture, the paired per-image difference no-KD - {resp, feat,
    comb}, with a percentile bootstrap 95% CI and a one-sided p for "KD is
    actually better".
  * The headline contrast MobileViT-XXS(no-KD) - teacher, same treatment.

Per-image IoU is a biased estimator of the pooled-pixel IoU reported in Table 2
(empty-mask images degenerate to 0), but that bias is image-specific and cancels
in the *paired* difference, which is the quantity of interest here. CPU-only,
num_workers=0 (macOS-safe). ~16 min on an M2 Pro.

Usage
-----
    cd code/
    python scripts/stats_bootstrap.py --data_root /path/to/FSSD_parent \
        --ckpt_dir runs/3fold --out runs/3fold/stats_bootstrap.json
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from floodlite.data import make_loaders
from floodlite.models import make_teacher, make_student, STUDENT_BACKBONES
from floodlite.metrics import compute_metrics

FOLDS = (0, 1, 2)
KD_CONFIGS = ("none", "resp", "feat", "comb")


@torch.no_grad()
def per_image_iou(model, loader, device) -> np.ndarray:
    model.eval()
    ious = []
    for x, y in loader:
        logits = model(x.to(device))
        y = y.to(device)
        for i in range(logits.size(0)):
            ious.append(compute_metrics(logits[i:i + 1], y[i:i + 1]).iou)
    return np.asarray(ious, dtype=np.float64)


def boot(deltas: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    """Percentile bootstrap on the mean paired difference.

    Convention: delta = A - B, positive => A better than B.
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    return {
        "n_images": int(n),
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        # one-sided probability that B is actually better than A (delta<0):
        "p_B_better": float((means < 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True, help="Parent dir containing dataset/{train,val}/...")
    ap.add_argument("--ckpt_dir", default="runs/3fold")
    ap.add_argument("--out", default="runs/3fold/stats_bootstrap.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    device = "cpu"
    ckpt = Path(args.ckpt_dir)
    t_start = time.perf_counter()

    # per_iou[fold][name] -> np.ndarray of per-image IoU (image order fixed per fold)
    per_iou: dict[int, dict[str, np.ndarray]] = {}
    for fold in FOLDS:
        _, va = make_loaders(args.data_root, fold=fold, batch_size=8, num_workers=0)
        per_iou[fold] = {}
        # teacher
        t = make_teacher().to(device)
        t.load_state_dict(torch.load(ckpt / f"teacher_fold{fold}.pt", map_location=device))
        per_iou[fold]["teacher"] = per_image_iou(t, va, device)
        print(f"[fold {fold}] teacher done ({len(per_iou[fold]['teacher'])} imgs, "
              f"{time.perf_counter()-t_start:.0f}s)", flush=True)
        del t
        # students x KD
        for sname in STUDENT_BACKBONES:
            for kd in KD_CONFIGS:
                ck = ckpt / f"{sname}_{kd}_fold{fold}.pt"
                if not ck.exists():
                    print(f"[fold {fold}] MISSING {ck.name}", flush=True)
                    continue
                m = make_student(sname).to(device)
                m.load_state_dict(torch.load(ck, map_location=device))
                per_iou[fold][f"{sname}_{kd}"] = per_image_iou(m, va, device)
                del m
            print(f"[fold {fold}] {sname} all-KD done ({time.perf_counter()-t_start:.0f}s)", flush=True)

    def pooled(name: str) -> np.ndarray:
        return np.concatenate([per_iou[f][name] for f in FOLDS])

    out: dict = {
        "meta": {
            "description": "Per-image paired bootstrap pooled across folds 0-2 (review Mod1).",
            "n_boot": args.n_boot,
            "note": "delta convention A-B; positive => A better. Per-image IoU is "
                    "biased vs pooled-pixel IoU but cancels in the paired difference.",
        },
        "kd_contrasts": {},
        "headline_best_vs_teacher": {},
    }

    # ---- KD contrasts: none - {resp,feat,comb}, per architecture ----
    for sname in STUDENT_BACKBONES:
        none = pooled(f"{sname}_none")
        out["kd_contrasts"][sname] = {}
        for kd in ("resp", "feat", "comb"):
            key = f"{sname}_{kd}"
            if all(key in per_iou[f] for f in FOLDS):
                out["kd_contrasts"][sname][f"none_minus_{kd}"] = boot(
                    none - pooled(key), n_boot=args.n_boot)

    # ---- Headline: best student (MobileViT-XXS none) - teacher ----
    if all("mobilevit_xxs_none" in per_iou[f] for f in FOLDS):
        out["headline_best_vs_teacher"]["mobilevit_xxs_none_minus_teacher"] = boot(
            pooled("mobilevit_xxs_none") - pooled("teacher"), n_boot=args.n_boot)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # ---- console summary ----
    print("\n================  PER-IMAGE PAIRED BOOTSTRAP  ================")
    print("KD contrasts (delta = no-KD - KD; positive => no-KD better, i.e. KD does not help):")
    for sname, d in out["kd_contrasts"].items():
        for k, v in d.items():
            flag = "KD WORSE (CI>0)" if v["ci95_low"] > 0 else (
                   "KD BETTER (CI<0)" if v["ci95_high"] < 0 else "no sig. diff")
            print(f"  {sname:20s} {k:16s} n={v['n_images']:3d}  "
                  f"Δ={v['mean_delta']:+.4f}  95%CI[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}]  "
                  f"p(KD better)={v['p_B_better']:.3f}  -> {flag}")
    print("\nHeadline (delta = MobileViT-XXS no-KD - teacher; positive => student better):")
    for k, v in out["headline_best_vs_teacher"].items():
        print(f"  {k}: n={v['n_images']}  Δ={v['mean_delta']:+.4f}  "
              f"95%CI[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}]  p(teacher better)={v['p_B_better']:.3f}")
    print(f"\nWrote {args.out}  (total {time.perf_counter()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
