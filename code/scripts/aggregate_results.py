"""Aggregate per-fold results.json files into one summary."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_fold = []
    for p in args.inputs:
        per_fold.append(json.loads(Path(p).read_text()))

    # Collect all (config_key, metric_dict) pairs across folds
    keys = set()
    for f in per_fold:
        for fold_id, fold in f.items():
            keys.add(("teacher", fold_id))
            if fold.get("baseline_mobilenetv2"):
                keys.add(("baseline_mobilenetv2", fold_id))
            for s in fold.get("students", {}):
                keys.add((s, fold_id))

    # Mean ± std across folds
    summary = {}
    for f in per_fold:
        for fold_id, fold in f.items():
            for cfg in ["teacher", "baseline_mobilenetv2"] + list(fold.get("students", {}).keys()):
                if cfg == "baseline_mobilenetv2" and not fold.get(cfg):
                    continue
                m = fold[cfg] if cfg in ("teacher", "baseline_mobilenetv2") else fold["students"][cfg]
                if m is None:
                    continue
                summary.setdefault(cfg, []).append(m)

    final = {}
    for cfg, runs in summary.items():
        agg = {}
        for k in ("accuracy", "precision", "recall", "f1", "iou"):
            vals = [r[k] for r in runs]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        final[cfg] = agg

    Path(args.out).write_text(json.dumps(final, indent=2))
    print(f"Wrote {args.out} with {len(final)} configs.")


if __name__ == "__main__":
    main()
