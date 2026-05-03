"""Evaluate FSSD-trained models on Sen1Floods11 RGB chips (no fine-tuning)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from floodlite.data import make_sen1floods11_loader
from floodlite.models import make_teacher, make_baseline_mobilenetv2, make_student, STUDENT_BACKBONES
from floodlite.metrics import compute_metrics


def evaluate(model, loader, device):
    model.eval()
    ms = []
    with torch.no_grad():
        for x, y in loader:
            ms.append(compute_metrics(model(x.to(device)), y.to(device)))
    n = len(ms)
    return {k: sum(getattr(m, k) for m in ms) / n
            for k in ("accuracy", "precision", "recall", "f1", "iou")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--fold", type=int, default=0, help="Which fold's checkpoint to evaluate")
    ap.add_argument("--best_kd", default="none",
                    choices=["none", "resp", "feat", "comb"],
                    help="Which KD config to consider for each student")
    ap.add_argument("--out", default="runs/3fold/ood_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_sen1floods11_loader(args.ood_root, n_chips=250, batch_size=8)
    out = {}

    teacher = make_teacher().to(device)
    teacher.load_state_dict(torch.load(Path(args.ckpt_dir) / f"teacher_fold{args.fold}.pt", map_location=device))
    out["teacher"] = evaluate(teacher, loader, device)
    print(f"teacher: {out['teacher']}")

    bp = Path(args.ckpt_dir) / f"baseline_mobilenetv2_fold{args.fold}.pt"
    if bp.exists():
        baseline = make_baseline_mobilenetv2().to(device)
        baseline.load_state_dict(torch.load(bp, map_location=device))
        out["baseline_mobilenetv2"] = evaluate(baseline, loader, device)
        print(f"baseline: {out['baseline_mobilenetv2']}")

    for sname in STUDENT_BACKBONES:
        ckpt = Path(args.ckpt_dir) / f"{sname}_{args.best_kd}_fold{args.fold}.pt"
        if not ckpt.exists():
            print(f"SKIP {ckpt}")
            continue
        s = make_student(sname).to(device)
        s.load_state_dict(torch.load(ckpt, map_location=device))
        out[f"{sname}_{args.best_kd}"] = evaluate(s, loader, device)
        print(f"{sname}_{args.best_kd}: {out[f'{sname}_{args.best_kd}']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
