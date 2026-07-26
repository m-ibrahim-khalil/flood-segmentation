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

import numpy as np
import onnxruntime as ort
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
    # Stamp the runtime versions so latency numbers are self-documenting and
    # reproducible (mirrors bench_teacher_ort.py). The ORT version in
    # particular determines whether the INT8 QDQ artifacts load at all
    # (they embed ai.onnx.ml opset 5 -> require ORT >= 1.18).
    out: dict = {
        "host": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

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
    #
    # File naming produced by quantize_all.py is per-fold:
    #   {sname}_fold{N}_fp32.onnx
    #   {sname}_fold{N}_int8.onnx
    # We resolve both that pattern and the flat plan-spec pattern
    #   {sname}_fp32.onnx
    # so the script tolerates either layout.
    # ------------------------------------------------------------------
    def _resolve(*candidates: Path) -> Path | None:
        for c in candidates:
            if c.exists():
                return c
        return None

    onnx_targets: list[tuple[str, Path | None]] = []
    onnx_targets.append((
        "teacher",
        _resolve(exports_dir / "teacher.onnx",
                 exports_dir / f"teacher_fold{args.fold}.onnx",
                 exports_dir / f"teacher_fold{args.fold}_fp32.onnx"),
    ))
    onnx_targets.append((
        "baseline_mobilenetv2",
        _resolve(exports_dir / "baseline_mobilenetv2.onnx",
                 exports_dir / f"baseline_mobilenetv2_fold{args.fold}.onnx"),
    ))
    for s in STUDENT_BACKBONES:
        onnx_targets.append((
            f"{s}_fp32",
            _resolve(exports_dir / f"{s}_fp32.onnx",
                     exports_dir / f"{s}_fold{args.fold}_fp32.onnx"),
        ))
        onnx_targets.append((
            f"{s}_int8",
            _resolve(exports_dir / f"{s}_int8.onnx",
                     exports_dir / f"{s}_fold{args.fold}_int8.onnx"),
        ))

    for name, op in onnx_targets:
        if op is None:
            print(f"INFO: ONNX for '{name}' not found in {exports_dir}, skipping.")
            continue
        key = f"{name}_ort_cpu"
        out[key] = benchmark_onnxruntime(op)
        print(f"{key} ({op.name}): {out[key]}")

    # ------------------------------------------------------------------
    # 3. Write results
    # ------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
