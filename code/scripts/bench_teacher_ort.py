"""Fair same-runtime teacher-vs-student latency re-benchmark (addresses review M1).

Motivation
----------
In ``runs/3fold/lat_m2.json`` the teacher latency was measured under **PyTorch
eager CPU** (key ``teacher_pt_cpu`` = 274 ms) while every student latency came
from **ONNX Runtime CPU** (keys ``*_fp32_ort_cpu``).  Table 5 of the manuscript
nonetheless labels *all* rows "ORT-CPU", so the headline teacher-relative
speedups conflate two effects:

  1. runtime (PyTorch eager  ->  ONNX Runtime graph optimisation), and
  2. architecture (EfficientNet-B0 teacher  ->  lightweight student).

The teacher was never exported to ONNX, so ``bench_m2.py`` silently skipped its
ORT-CPU row.  This script exports the teacher with the *same* ``export_onnx``
path used for the students and times teacher + students with the *same*
``benchmark_onnxruntime`` harness, so the comparison is apples-to-apples.

It writes to a SEPARATE json (default ``runs/3fold/lat_teacher_ort_check.json``)
and never overwrites the canonical ``lat_m2.json``.

IMPORTANT — version note for submission
---------------------------------------
The paper's canonical student numbers were measured on **onnxruntime 1.17**.
Re-run this on that same version before quoting numbers in the manuscript; a
different installed ORT version yields indicative (not submission-grade)
magnitudes.  The script prints the ORT version it actually used.

Usage
-----
    cd code/
    python scripts/bench_teacher_ort.py --ckpt_dir runs/3fold --fold 0
    # then inspect runs/3fold/lat_teacher_ort_check.json
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnxruntime as ort
import torch

from floodlite.benchmark import benchmark_latency, benchmark_onnxruntime
from floodlite.export import export_onnx
from floodlite.models import make_teacher, make_student, STUDENT_BACKBONES


def _speedup(ref_ms: float, val_ms: float) -> float:
    return ref_ms / val_ms if val_ms else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_dir", default="runs/3fold",
                    help="Directory with *.pt checkpoints (default: runs/3fold).")
    ap.add_argument("--exports", default="runs/3fold/exports",
                    help="Directory with student *.onnx files (default: runs/3fold/exports).")
    ap.add_argument("--fold", type=int, default=0, help="Fold index (default: 0).")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset for the teacher export; match the students "
                         "(export_all.py used 17). Default: 17.")
    ap.add_argument("--out", default="runs/3fold/lat_teacher_ort_check.json")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    exports_dir = Path(args.exports)
    out: dict = {
        "host": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "note": (
            "Same-runtime teacher-vs-student re-bench for review M1. Teacher and "
            "students both timed under ONNX Runtime CPU via benchmark_onnxruntime. "
            "Re-run on onnxruntime==1.17 for submission-grade numbers."
        ),
        "fold": args.fold,
    }

    # ------------------------------------------------------------------
    # 1. Export the teacher to ONNX (the missing artifact) and bench it.
    # ------------------------------------------------------------------
    teacher_ckpt = ckpt_dir / f"teacher_fold{args.fold}.pt"
    if not teacher_ckpt.exists():
        sys.exit(f"ERROR: teacher checkpoint not found: {teacher_ckpt}")

    teacher = make_teacher()
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu"))

    teacher_onnx = exports_dir / f"teacher_fold{args.fold}_fp32.onnx"
    export_onnx(teacher, teacher_onnx, opset=args.opset)
    print(f"[export] teacher -> {teacher_onnx}")

    out["teacher_pt_cpu"] = benchmark_latency(teacher, device="cpu")
    out["teacher_fp32_ort_cpu"] = benchmark_onnxruntime(teacher_onnx)
    print(f"[bench] teacher PyTorch-CPU : {out['teacher_pt_cpu']['p50_ms']:.1f} ms")
    print(f"[bench] teacher ORT-CPU     : {out['teacher_fp32_ort_cpu']['p50_ms']:.1f} ms")

    # ------------------------------------------------------------------
    # 2. Re-bench the student FP32 ONNX files on the SAME ORT version so
    #    the teacher/student comparison is internally consistent on this
    #    machine (decoupled from the 1.17-vs-installed version question).
    # ------------------------------------------------------------------
    for sname in STUDENT_BACKBONES:
        onnx = exports_dir / f"{sname}_fold{args.fold}_fp32.onnx"
        if not onnx.exists():
            onnx = exports_dir / f"{sname}_fp32.onnx"
        if not onnx.exists():
            print(f"INFO: student FP32 ONNX not found for {sname}, skipping.")
            continue
        out[f"{sname}_fp32_ort_cpu"] = benchmark_onnxruntime(onnx)
        print(f"[bench] {sname} FP32 ORT-CPU : {out[f'{sname}_fp32_ort_cpu']['p50_ms']:.1f} ms")

    # ------------------------------------------------------------------
    # 3. Summary: the honest, same-runtime speedups.
    # ------------------------------------------------------------------
    t_pt = out["teacher_pt_cpu"]["p50_ms"]
    t_ort = out["teacher_fp32_ort_cpu"]["p50_ms"]
    print("\n================  SAME-RUNTIME SUMMARY (ORT %s)  ================" % ort.__version__)
    print(f"Teacher   PyTorch-CPU : {t_pt:7.1f} ms")
    print(f"Teacher   ORT-CPU     : {t_ort:7.1f} ms   (runtime-only speedup {_speedup(t_pt, t_ort):.1f}x)")
    print("-" * 64)
    print(f"{'Student (FP32)':22s} {'ORT p50':>9s}  {'vs teacher-ORT':>15s}  {'vs teacher-PT':>14s}")
    for sname in STUDENT_BACKBONES:
        k = f"{sname}_fp32_ort_cpu"
        if k not in out:
            continue
        s_ort = out[k]["p50_ms"]
        print(f"{sname:22s} {s_ort:7.1f}ms  {_speedup(t_ort, s_ort):13.1f}x  {_speedup(t_pt, s_ort):12.1f}x")
    print("=" * 64)
    print("The 'vs teacher-PT' column is the mislabelled comparison currently in "
          "Table 5; the 'vs teacher-ORT' column is the honest same-runtime one.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
