"""Latency on Apple M2 Pro: PyTorch CPU and ONNX Runtime CPU.

CoreML is intentionally omitted to save build complexity (see 3-day plan §1.5).

Usage
-----
python scripts/bench_m2.py \\
    --ckpt_dir runs/3fold \\
    --exports  exports \\
    --fold     0 \\
    --out      runs/3fold/lat_m2.json
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path

# Make the package importable when the script is run from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from floodlite.benchmark import benchmark_latency, benchmark_onnxruntime
from floodlite.models import (
    make_baseline_mobilenetv2,
    make_student,
    make_teacher,
    STUDENT_BACKBONES,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure PyTorch CPU and ONNX Runtime CPU latency on Apple M2 Pro."
    )
    ap.add_argument(
        "--ckpt_dir",
        required=True,
        help="Directory containing *.pt checkpoints (e.g. runs/3fold).",
    )
    ap.add_argument(
        "--exports",
        default="exports/",
        help="Directory containing *.onnx files produced by export_all.py. "
             "Default: exports/",
    )
    ap.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold's checkpoints to load. Default: 0",
    )
    ap.add_argument(
        "--out",
        default="runs/3fold/lat_m2.json",
        help="Output JSON path. Default: runs/3fold/lat_m2.json",
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    exports_dir = Path(args.exports)
    out: dict = {"host": platform.platform()}

    # ------------------------------------------------------------------
    # 1. PyTorch CPU — FP32 forward-pass latency
    # ------------------------------------------------------------------
    teacher_ckpt = ckpt_dir / f"teacher_fold{args.fold}.pt"
    if teacher_ckpt.exists():
        t = make_teacher()
        t.load_state_dict(torch.load(teacher_ckpt, map_location="cpu"))
        out["teacher_pt_cpu"] = benchmark_latency(t, device="cpu")
        print(f"teacher_pt_cpu: {out['teacher_pt_cpu']}")
    else:
        print(f"WARNING: checkpoint not found, skipping teacher PyTorch CPU: {teacher_ckpt}")

    for sname in STUDENT_BACKBONES:
        ckpt = ckpt_dir / f"{sname}_none_fold{args.fold}.pt"
        if ckpt.exists():
            s = make_student(sname)
            s.load_state_dict(torch.load(ckpt, map_location="cpu"))
            key = f"{sname}_pt_cpu_fp32"
            out[key] = benchmark_latency(s, device="cpu")
            print(f"{key}: {out[key]}")
        else:
            print(f"WARNING: checkpoint not found, skipping {sname} PyTorch CPU: {ckpt}")

    # ------------------------------------------------------------------
    # 2. ONNX Runtime CPU — FP32 and INT8 models
    # ------------------------------------------------------------------
    onnx_names = (
        ["teacher", "baseline_mobilenetv2"]
        + [f"{s}_fp32" for s in STUDENT_BACKBONES]
        + [f"{s}_int8" for s in STUDENT_BACKBONES]
    )
    for name in onnx_names:
        op = exports_dir / f"{name}.onnx"
        if not op.exists():
            print(f"INFO: {op} not found, skipping.")
            continue
        key = f"{name}_ort_cpu"
        out[key] = benchmark_onnxruntime(op)
        print(f"{key}: {out[key]}")

    # ------------------------------------------------------------------
    # 3. Write results
    # ------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
