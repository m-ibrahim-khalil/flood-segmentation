"""End-to-end driver: trains teacher + students + ablation + quantization on FSSD.

Usage (after `pip install -r requirements.txt`):
    python scripts/run_full_experiment.py --data_root /path/to/FSSD --fold 0

Set --folds 5 to loop over the full cross-validation.
"""
from __future__ import annotations
import argparse, json, time, copy, os
from pathlib import Path
import numpy as np
import torch

from floodlite.data import make_loaders, FSSD
from floodlite.models import make_teacher, make_student, count_params, estimate_flops, STUDENT_BACKBONES, make_baseline_mobilenetv2
from floodlite.train import train_teacher, train_student_kd
from floodlite.train import train_teacher as train_taskonly
from floodlite.quantize import quantize_int8, model_size_mb
from floodlite.benchmark import benchmark_latency
from floodlite.metrics import compute_metrics


def device_pick():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def evaluate(model, loader, device):
    model.eval(); ms = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ms.append(compute_metrics(model(x), y))
    n = len(ms)
    return {k: sum(getattr(m, k) for m in ms)/n for k in ("accuracy", "precision", "recall", "f1", "iou")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3, help="Number of folds to run starting from --fold")
    ap.add_argument("--epochs_teacher", type=int, default=35)
    ap.add_argument("--epochs_student", type=int, default=35)
    ap.add_argument("--include_baseline", action="store_true", default=False,
                    help="Train a MobileNetV2 baseline (task loss only) for comparison")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    device = device_pick()
    print(f"Device: {device}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for fold in range(args.fold, args.fold + args.folds):
        print(f"\n========== Fold {fold} ==========")
        tr, va = make_loaders(args.data_root, fold=fold, batch_size=args.batch_size)
        ckpt_teacher = Path(args.out_dir) / f"teacher_fold{fold}.pt"
        teacher = make_teacher().to(device)
        train_teacher(teacher, tr, va, epochs=args.epochs_teacher, device=device, ckpt_path=ckpt_teacher)
        teacher.load_state_dict(torch.load(ckpt_teacher, map_location=device))
        teacher_metrics = evaluate(teacher, va, device)
        print(f"Teacher final: {teacher_metrics}")

        baseline_metrics = None
        if args.include_baseline:
            ckpt_baseline = Path(args.out_dir) / f"baseline_mobilenetv2_fold{fold}.pt"
            baseline = make_baseline_mobilenetv2().to(device)
            train_taskonly(baseline, tr, va, epochs=args.epochs_student, device=device, ckpt_path=ckpt_baseline)
            baseline.load_state_dict(torch.load(ckpt_baseline, map_location=device))
            baseline_metrics = evaluate(baseline, va, device)
            print(f"Baseline (MobileNetV2): {baseline_metrics}")

        fold_results = {"teacher": teacher_metrics, "baseline_mobilenetv2": baseline_metrics, "students": {}}
        for sname in STUDENT_BACKBONES:
            for kd in ("none", "resp", "feat", "comb"):
                use_resp = kd in ("resp", "comb")
                use_feat = kd in ("feat", "comb")
                ckpt = Path(args.out_dir) / f"{sname}_{kd}_fold{fold}.pt"
                student = make_student(sname).to(device)
                train_student_kd(student, teacher, tr, va,
                                 epochs=args.epochs_student, use_response=use_resp, use_feature=use_feat,
                                 device=device, ckpt_path=ckpt)
                student.load_state_dict(torch.load(ckpt, map_location=device))
                m = evaluate(student, va, device)
                fold_results["students"][f"{sname}_{kd}"] = m
                print(f"[{sname} {kd}] {m}")

        # quantize the comb-config students; benchmark host latency
        for sname in STUDENT_BACKBONES:
            ckpt = Path(args.out_dir) / f"{sname}_comb_fold{fold}.pt"
            student_fp = make_student(sname).cpu()
            student_fp.load_state_dict(torch.load(ckpt, map_location="cpu"))
            try:
                student_q = quantize_int8(student_fp, tr)
                m_q = evaluate(student_q.cpu(), va, "cpu")
                fold_results.setdefault("quantized", {})[sname] = {
                    "iou": m_q["iou"],
                    "fp_size_mb": model_size_mb(student_fp),
                    "q_size_mb":  model_size_mb(student_q),
                }
            except Exception as e:
                print(f"Quantize {sname} failed: {e}")
            lat = benchmark_latency(make_student(sname), device=device)
            fold_results.setdefault("latency", {})[sname] = lat

        all_results[f"fold_{fold}"] = fold_results

    out_json = Path(args.out_dir) / "results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nDone. Results saved to {out_json}")


if __name__ == "__main__":
    main()
