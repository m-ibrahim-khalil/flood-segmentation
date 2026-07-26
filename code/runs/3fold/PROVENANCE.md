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

`scripts/stats_bootstrap.py` implements a powered per-image paired bootstrap for the
KD and teacher-vs-student contrasts, but it was **not executed** for this version of
the manuscript: it requires a local FSSD copy, and `stats.json` records the run as
skipped (`"bootstrap_fold0": {"skipped_reason": ...}`). Accordingly, the manuscript
reports these contrasts as fold-level directional effects (paired Wilcoxon, minimum
two-sided p = 0.25) and makes **no** bootstrap confidence-interval or p-value claim.
Running the bootstrap against a local FSSD download is the top statistical follow-up.
