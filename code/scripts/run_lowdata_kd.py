"""Low-data KD ablation (review Mod4): does knowledge distillation reappear when
the benchmark is *not* saturated?

The paper's negative-KD finding is scoped to the saturated full-FSSD regime. Its
mechanistic claim --- KD does not help because strong ImageNet priors plus 663
labelled images leave the student no headroom --- predicts that KD *should* help
once labels are scarce. This script tests that boundary condition directly.

Design
------
* Teacher: the existing full-data teacher (``teacher_fold{f}.pt`` from the main
  run) is reused unchanged --- a strong teacher that has seen all the labels.
* Student: trained on a *subsample* of the fold's training set
  (``train_frac`` in {0.1, 0.25, 0.5, 1.0} by default), with and without the
  combined KD loss, and evaluated on the *full* validation fold (so IoU is
  comparable across fractions). Subsets are deterministic and nested
  (see ``make_loaders(train_frac=...)``), so the no-KD and KD arms see identical
  data at each fraction.
* Metric of interest: KD gain = IoU(comb) - IoU(none), as a function of
  ``train_frac``. Expected shape: gain ~ 0 at frac = 1.0 (reproduces the
  negative result) and gain > 0 as frac -> 0.1 (KD helps when data-starved).

Resumability
------------
Results are written to ``<out_dir>/lowdata_kd.json`` after every cell, and
``--skip_if_exists`` skips any (frac, fold, student, kd) cell already present ---
so a Kaggle session that times out can be resumed by re-running the same command.
Checkpoints are written to a temp path and deleted after evaluation unless
``--keep_checkpoints`` is set (default off, to bound disk).

Usage (Kaggle T4 / Colab)
-------------------------
    cd code/
    python scripts/run_lowdata_kd.py \
        --data_root /kaggle/input/flood-semantic-segmentation-dataset \
        --teacher_dir runs/3fold --out_dir runs/lowdata \
        --fracs 0.1,0.25,0.5,1.0 --fold 0 --folds 3 \
        --students mobilenetv3_small,efficientnet_lite0,mobilevit_xxs \
        --epochs 35 --skip_if_exists

Then: python scripts/make_lowdata_figure.py --lowdata runs/lowdata/lowdata_kd.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from floodlite.data import make_loaders
from floodlite.models import make_teacher, make_student, STUDENT_BACKBONES
from floodlite.train import train_student_kd
from floodlite.metrics import compute_metrics


def device_pick() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ms, n = [], 0
    for x, y in loader:
        m = compute_metrics(model(x.to(device)), y.to(device))
        ms.append(m); n += 1
    return {k: sum(getattr(m, k) for m in ms) / max(n, 1)
            for k in ("accuracy", "precision", "recall", "f1", "iou")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--teacher_dir", default="runs/3fold",
                    help="Directory with full-data teacher_fold{f}.pt checkpoints.")
    ap.add_argument("--out_dir", default="runs/lowdata")
    ap.add_argument("--fracs", default="0.1,0.25,0.5,1.0",
                    help="Comma-separated training-data fractions.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3, help="Number of folds from --fold.")
    ap.add_argument("--students", default=",".join(STUDENT_BACKBONES),
                    help="Comma-separated student backbones.")
    ap.add_argument("--kd", default="none,comb",
                    help="Comma-separated KD configs to compare (none,resp,feat,comb).")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers; use 0 on macOS to avoid worker crashes.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_if_exists", action="store_true", default=False)
    ap.add_argument("--keep_checkpoints", action="store_true", default=False)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    kd_configs = [k.strip() for k in args.kd.split(",") if k.strip()]
    device = device_pick()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpts"; ckpt_dir.mkdir(exist_ok=True)
    out_json = out_dir / "lowdata_kd.json"

    torch.manual_seed(args.seed)

    # Resume: load prior results if present.
    if out_json.exists():
        blob = json.loads(out_json.read_text())
        results = blob.get("results", {})
    else:
        results = {}

    def have(frac, fold, student, kd) -> bool:
        return (f"{frac}" in results and str(fold) in results[f"{frac}"]
                and student in results[f"{frac}"][str(fold)]
                and kd in results[f"{frac}"][str(fold)][student])

    def record(frac, fold, student, kd, metrics):
        results.setdefault(f"{frac}", {}).setdefault(str(fold), {}) \
               .setdefault(student, {})[kd] = metrics
        out_json.write_text(json.dumps({
            "meta": {"fracs": fracs, "folds": list(range(args.fold, args.fold + args.folds)),
                     "students": students, "kd": kd_configs, "epochs": args.epochs,
                     "seed": args.seed, "device": device,
                     "description": "Low-data KD ablation (Mod4): student trained on "
                                    "train_frac of FSSD, full-data teacher, evaluated on full val."},
            "results": results,
        }, indent=2))

    t0 = time.perf_counter()
    for frac in fracs:
        for fold in range(args.fold, args.fold + args.folds):
            teacher_ckpt = Path(args.teacher_dir) / f"teacher_fold{fold}.pt"
            if not teacher_ckpt.exists():
                print(f"[SKIP fold {fold}] missing teacher {teacher_ckpt}"); continue
            teacher = make_teacher().to(device)
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
            teacher.eval()

            # Subsampled train loader + full val loader for this (frac, fold).
            tr, va = make_loaders(args.data_root, fold=fold, batch_size=args.batch_size,
                                  num_workers=args.num_workers, seed=args.seed,
                                  train_frac=frac)
            n_train = len(tr.dataset)
            for student_name in students:
                for kd in kd_configs:
                    if args.skip_if_exists and have(frac, fold, student_name, kd):
                        print(f"[skip] frac={frac} fold={fold} {student_name}/{kd}")
                        continue
                    use_resp = kd in ("resp", "comb")
                    use_feat = kd in ("feat", "comb")
                    student = make_student(student_name).to(device)
                    ckpt = ckpt_dir / f"{student_name}_{kd}_frac{frac}_fold{fold}.pt"
                    train_student_kd(student, teacher, tr, va, epochs=args.epochs,
                                     use_response=use_resp, use_feature=use_feat,
                                     device=device, ckpt_path=ckpt)
                    student.load_state_dict(torch.load(ckpt, map_location=device))
                    m = evaluate(student, va, device)
                    m["n_train"] = n_train
                    record(frac, fold, student_name, kd, m)
                    print(f"[frac={frac} fold={fold} {student_name}/{kd}] "
                          f"IoU={m['iou']:.4f}  (n_train={n_train}, {time.perf_counter()-t0:.0f}s)")
                    if not args.keep_checkpoints:
                        ckpt.unlink(missing_ok=True)

    # ---- summary: KD gain vs fraction ----
    print("\n================  KD GAIN vs TRAINING-DATA FRACTION  ================")
    print("gain = IoU(comb) - IoU(none), averaged over folds & students\n")
    base_kd = "comb" if "comb" in kd_configs else kd_configs[-1]
    for frac in fracs:
        gains = []
        for fold in range(args.fold, args.fold + args.folds):
            for s in students:
                cell = results.get(f"{frac}", {}).get(str(fold), {}).get(s, {})
                if "none" in cell and base_kd in cell:
                    gains.append(cell[base_kd]["iou"] - cell["none"]["iou"])
        if gains:
            mean_gain = sum(gains) / len(gains)
            print(f"  frac={frac:<5}  {base_kd}-none gain = {mean_gain:+.4f}  "
                  f"(n={len(gains)} runs)")
    print(f"\nWrote {out_json}  (total {time.perf_counter()-t0:.0f}s)")
    print("Next: python scripts/make_lowdata_figure.py --lowdata", out_json)


if __name__ == "__main__":
    main()
