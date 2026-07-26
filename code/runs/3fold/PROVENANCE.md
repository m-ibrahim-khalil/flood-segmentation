# `runs/3fold` provenance

Authoritative source file for each quantity reported in the manuscript
(matches the `main.tex` "Numbers source of truth" header):

| Reported quantity | Authoritative file | Producer |
|---|---|---|
| Segmentation metrics, Tables 2 & 3 | `summary.json` (mean±std over folds 0–2); `results.json` (per-fold raw) | `run_full_experiment.py` → `aggregate_results.py` |
| INT8 quantization, Table 4 | **`quantized.json`** | `quantize_all.py` (commit `cf9577e`, 2026-05-16) — Percentile-99.999% static QDQ recipe |
| Latency, Table 5 | `lat_m2.json` (M2 Pro ORT-CPU), `lat_wasm.json` (browser WASM SIMD) | `bench_m2.py`, `bench.mjs` |
| Fold-level Wilcoxon | `stats.json` | `stats.py` |

## Known discrepancy — do NOT use the INT8 block inside `results.json`

`results.json` (commit `d5557ff`, 2026-05-03) was produced by an earlier run of
`run_full_experiment.py` whose **inline** INT8 step used a superseded PTQ recipe.
Its per-fold INT8 IoU values differ materially from `quantized.json` — e.g. fold-0
MobileNetV3-Small INT8 IoU is 0.449 in the old block vs 0.772 in `quantized.json` —
and are **not** the numbers reported in the paper. The dedicated `quantize_all.py`
sweep (`quantized.json`) supersedes it.

To prevent accidental reuse, the stale per-fold block in `results.json` has been
renamed from `"quantized"` to `"quantized__SUPERSEDED__see_PROVENANCE"`. No code in
`code/` reads that block (`aggregate_results.py` and `stats.py` read only the
`teacher` / `baseline_mobilenetv2` / `students` entries).

## Statistical significance

The powered per-image paired bootstrap for the KD and teacher-vs-student contrasts is
**`stats_bootstrap.json`** (produced by `scripts/stats_bootstrap.py --data_root <FSSD>
--ckpt_dir runs/3fold`, using the 42 per-fold checkpoints; $n=399$ pooled held-out
images, 10,000 replicates, seed-fixed). It was independently re-run and reproduces
bit-for-bit. This file backs every confidence interval and p-value in the manuscript
(Sections 3.5, 5.2, 7).

The `"bootstrap_fold0": {"skipped_reason": ...}` entry inside `stats.json` is a
**different, earlier** analysis — a fold-0-only pixel bootstrap from `scripts/stats.py`
— which was superseded by the per-image `stats_bootstrap.json` and is not referenced by
the manuscript. `stats.json` remains authoritative only for the fold-level Wilcoxon.
