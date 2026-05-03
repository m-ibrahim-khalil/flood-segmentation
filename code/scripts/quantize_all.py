"""Quantize all (student × fold × no-KD) checkpoints to INT8 and report IoU drop.

Uses ONNX dynamic quantization (the FloodLite-recommended path on Kaggle's
PyTorch — eager-mode static PTQ fails on timm Conv2dSame). Each checkpoint
becomes two artifacts under ``--exports_dir``: ``{sname}_fold{N}_fp32.onnx``
and ``{sname}_fold{N}_int8.onnx``. IoU is evaluated by running both ONNX
models through the FSSD val loader.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from floodlite.data import make_loaders
from floodlite.models import make_student, STUDENT_BACKBONES
from floodlite.quantize import (
    quantize_int8_onnx_static,
    evaluate_onnx,
    file_size_mb,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="Dir holding {sname}_none_fold{N}.pt")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--exports_dir", default="exports",
                    help="Output dir for FP32 and INT8 ONNX artifacts.")
    ap.add_argument("--out", default="runs/3fold/quantized.json")
    args = ap.parse_args()

    exports_dir = Path(args.exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for fold in args.folds:
        tr, va = make_loaders(args.data_root, fold=fold, batch_size=8)
        results[f"fold_{fold}"] = {}
        for sname in STUDENT_BACKBONES:
            ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{fold}.pt"
            if not ckpt.exists():
                print(f"SKIP {ckpt} (missing)")
                continue

            fp_model = make_student(sname).cpu()
            fp_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            fp_model.eval()

            fp_onnx = exports_dir / f"{sname}_fold{fold}_fp32.onnx"
            int8_onnx = exports_dir / f"{sname}_fold{fold}_int8.onnx"

            try:
                quantize_int8_onnx_static(
                    fp_model,
                    fp_onnx_path=fp_onnx,
                    int8_onnx_path=int8_onnx,
                    calib_loader=tr,
                    n_calib_batches=10,
                )
                fp_metrics = evaluate_onnx(fp_onnx, va)
                int8_metrics = evaluate_onnx(int8_onnx, va)
                results[f"fold_{fold}"][sname] = {
                    "fp32_iou": fp_metrics["iou"],
                    "int8_iou": int8_metrics["iou"],
                    "delta_iou": int8_metrics["iou"] - fp_metrics["iou"],
                    "fp32_size_mb": file_size_mb(fp_onnx),
                    "int8_size_mb": file_size_mb(int8_onnx),
                    "size_reduction_x": file_size_mb(fp_onnx) / max(file_size_mb(int8_onnx), 1e-6),
                    "fp32_metrics": fp_metrics,
                    "int8_metrics": int8_metrics,
                }
                print(f"fold {fold} {sname}: IoU {fp_metrics['iou']:.4f} -> {int8_metrics['iou']:.4f}  "
                      f"size {file_size_mb(fp_onnx):.1f} MB -> {file_size_mb(int8_onnx):.1f} MB")
            except Exception as e:
                results[f"fold_{fold}"][sname] = {"error": str(e)}
                print(f"fold {fold} {sname}: QUANTIZATION FAILED: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
