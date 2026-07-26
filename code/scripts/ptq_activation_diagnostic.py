"""Activation-range diagnostic for the PTQ-stability mechanism.

Tests the manuscript's claim (Sec. 5.3/6.3) that MobileNetV3-Small's INT8 PTQ
collapse is driven by the *unbounded positive tail* of its HardSwish activations,
whereas EfficientNet-Lite0's ReLU6 (bounded at 6) quantizes cleanly. We measure
per-layer post-activation ranges (max and 99.999-percentile of |activation|) on
the exact INT8 calibration images, across the encoder of both students.

A single per-tensor INT8 activation scale must cover the whole range; a heavy,
unbounded tail forces a large scale that destroys resolution on the bulk of
activations. This script quantifies that tail.

Usage:
    python scripts/ptq_activation_diagnostic.py --data_root <FSSD> \
        --ckpt_dir <runs/3fold> --fold 0 --out runs/3fold/ptq_activation_diag.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn

from floodlite.data import make_loaders
from floodlite.models import make_student, STUDENT_BACKBONES

ACT_TYPES = (nn.Hardswish, nn.ReLU6, nn.ReLU, nn.SiLU, nn.GELU)
PER_BATCH_SAMPLE = 20_000  # random |act| values kept per layer per batch


@torch.no_grad()
def layer_ranges(model, loader, n_batches=10, device="cpu"):
    samples: dict[str, list] = {}
    maxes: dict[str, float] = {}
    kinds: dict[str, str] = {}
    handles = []
    rng = np.random.default_rng(0)

    def mk(name, kind):
        def hook(_m, _i, o):
            v = o.detach().float().abs().flatten()
            maxes[name] = max(maxes.get(name, 0.0), float(v.max()))
            a = v.numpy()
            if a.size > PER_BATCH_SAMPLE:
                a = a[rng.integers(0, a.size, PER_BATCH_SAMPLE)]
            samples.setdefault(name, []).append(a)
            kinds[name] = kind
        return hook

    for name, m in model.named_modules():
        if isinstance(m, ACT_TYPES):
            handles.append(m.register_forward_hook(mk(name, type(m).__name__)))

    model.eval()
    seen = 0
    for x, _ in loader:
        model(x.to(device))
        seen += 1
        if seen >= n_batches:
            break
    for h in handles:
        h.remove()

    out = {}
    for name, chunks in samples.items():
        v = np.concatenate(chunks)
        out[name] = {"act": kinds[name], "max": maxes[name],
                     "p99_999": float(np.percentile(v, 99.999)),
                     "p99_9": float(np.percentile(v, 99.9))}
    return out


def summarize(per_layer):
    maxes = [d["max"] for d in per_layer.values()]
    p = [d["p99_999"] for d in per_layer.values()]
    return {"n_act_layers": len(per_layer),
            "act_types": sorted({d["act"] for d in per_layer.values()}),
            "layer_max_of_max": max(maxes), "layer_max_median": float(np.median(maxes)),
            "p99999_max": max(p), "p99999_median": float(np.median(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_calib_batches", type=int, default=10)
    ap.add_argument("--out", default="runs/3fold/ptq_activation_diag.json")
    args = ap.parse_args()

    tr, _ = make_loaders(args.data_root, fold=args.fold, batch_size=8, num_workers=0)
    result = {}
    for sname in ("mobilenetv3_small", "efficientnet_lite0"):
        m = make_student(sname)
        m.load_state_dict(torch.load(Path(args.ckpt_dir) / f"{sname}_none_fold{args.fold}.pt",
                                     map_location="cpu"))
        per_layer = layer_ranges(m, tr, n_batches=args.n_calib_batches)
        result[sname] = {"summary": summarize(per_layer), "per_layer": per_layer}
        s = result[sname]["summary"]
        print(f"{sname}: acts={s['act_types']} max-of-layer-max={s['layer_max_of_max']:.2f} "
              f"median-layer-max={s['layer_max_median']:.2f} p99.999-max={s['p99999_max']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
