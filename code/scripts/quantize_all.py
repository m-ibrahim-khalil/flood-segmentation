"""Quantize all (student × fold × no-KD) checkpoints to INT8 and report IoU drop."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from floodlite.data import make_loaders
from floodlite.models import make_student, STUDENT_BACKBONES
from floodlite.quantize import quantize_int8, model_size_mb
from floodlite.metrics import compute_metrics


def evaluate_cpu(model, loader):
    model.eval()
    ms = []
    with torch.no_grad():
        for x, y in loader:
            ms.append(compute_metrics(model(x), y))
    n = len(ms)
    return {k: sum(getattr(m, k) for m in ms) / n
            for k in ("accuracy", "precision", "recall", "f1", "iou")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="Dir holding {sname}_none_fold{N}.pt")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs/3fold/quantized.json")
    args = ap.parse_args()

    results = {}
    for fold in args.folds:
        tr, va = make_loaders(args.data_root, fold=fold, batch_size=8)
        results[f"fold_{fold}"] = {}
        for sname in STUDENT_BACKBONES:
            ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{fold}.pt"
            if not ckpt.exists():
                print(f"SKIP {ckpt} (missing)")
                continue
            fp = make_student(sname).cpu()
            fp.load_state_dict(torch.load(ckpt, map_location="cpu"))
            fp_iou = evaluate_cpu(fp, va)["iou"]
            try:
                q = quantize_int8(fp, tr)
                q_iou = evaluate_cpu(q, va)["iou"]
                results[f"fold_{fold}"][sname] = {
                    "fp32_iou": fp_iou,
                    "int8_iou": q_iou,
                    "delta_iou": q_iou - fp_iou,
                    "fp32_size_mb": model_size_mb(fp),
                    "int8_size_mb": model_size_mb(q),
                }
                print(f"fold {fold} {sname}: {fp_iou:.4f} -> {q_iou:.4f}")
            except Exception as e:
                results[f"fold_{fold}"][sname] = {"error": str(e), "fp32_iou": fp_iou}
                print(f"fold {fold} {sname}: QUANTIZATION FAILED: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
