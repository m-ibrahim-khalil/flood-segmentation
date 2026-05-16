"""Export all final models (FP32 + INT8 students) to ONNX, with parity check."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

# Allow running as: python3 scripts/export_all.py from inside code/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from floodlite.models import make_teacher, make_baseline_mobilenetv2, make_student, STUDENT_BACKBONES
from floodlite.export import export_onnx
from floodlite.quantize import quantize_int8_onnx_static
from floodlite.data import make_loaders


def parity(model, onnx_path, atol=1e-3):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y_pt = model.cpu().eval()(x).numpy()
    y_ort = sess.run(None, {"input": x.numpy().astype(np.float32)})[0]
    return float(np.max(np.abs(y_pt - y_ort)))


def main():
    ap = argparse.ArgumentParser(
        description="Export all final FloodLite models to ONNX FP32 and INT8 QDQ, "
                    "with numerical parity check (PyTorch vs ONNX Runtime)."
    )
    ap.add_argument("--ckpt_dir", required=True,
                    help="Directory containing .pt checkpoint files")
    ap.add_argument("--data_root", required=True,
                    help="FSSD root for INT8 calibration (train split is used)")
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold index to load checkpoints for (default: 0)")
    ap.add_argument("--out_dir", default="exports/",
                    help="Output directory for .onnx files and parity.json (default: exports/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parity_log = {}

    # Teacher + baseline
    for name, factory in [("teacher", make_teacher), ("baseline_mobilenetv2", make_baseline_mobilenetv2)]:
        ckpt = Path(args.ckpt_dir) / f"{name}_fold{args.fold}.pt"
        if not ckpt.exists():
            print(f"[SKIP] {ckpt} not found")
            continue
        m = factory()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        op = out_dir / f"{name}.onnx"
        export_onnx(m, op, opset=17)
        parity_log[name] = parity(m, op)
        print(f"[OK] {name}  max_abs_diff={parity_log[name]:.6f}")

    # Students FP32 (no-KD checkpoints)
    for sname in STUDENT_BACKBONES:
        ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{args.fold}.pt"
        if not ckpt.exists():
            print(f"[SKIP] {ckpt} not found")
            continue
        m = make_student(sname)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        op = out_dir / f"{sname}_fp32.onnx"
        export_onnx(m, op, opset=17)
        parity_log[f"{sname}_fp32"] = parity(m, op)
        print(f"[OK] {sname}_fp32  max_abs_diff={parity_log[f'{sname}_fp32']:.6f}")

    # Students INT8 via ONNX static quantization (QDQ path — avoids Conv2dSame issues)
    tr, _ = make_loaders(args.data_root, fold=args.fold, batch_size=8)
    for sname in STUDENT_BACKBONES:
        ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{args.fold}.pt"
        if not ckpt.exists():
            print(f"[SKIP INT8] {ckpt} not found")
            continue
        m = make_student(sname).cpu()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        fp_onnx_path = out_dir / f"{sname}_fp32.onnx"
        int8_onnx_path = out_dir / f"{sname}_int8.onnx"
        try:
            quantize_int8_onnx_static(
                m,
                fp_onnx_path=fp_onnx_path,
                int8_onnx_path=int8_onnx_path,
                calib_loader=tr,
                n_calib_batches=25,
            )
            parity_log[f"{sname}_int8"] = "exported (no parity check for INT8)"
            print(f"[OK] {sname}_int8  (QDQ static)")
        except Exception as e:
            parity_log[f"{sname}_int8"] = f"FAILED: {e}"
            print(f"[FAIL] {sname}_int8  {e}")

    parity_json = out_dir / "parity.json"
    parity_json.write_text(json.dumps(parity_log, indent=2))
    print("Parity log written to", parity_json)
    print("Parity:", parity_log)


if __name__ == "__main__":
    main()
