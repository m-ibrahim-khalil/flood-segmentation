# FloodLite 3-Day Minimum-Viable Submission Plan

**Date:** 2026-05-03
**Supersedes:** parts of `2026-05-02-floodlite-edge-benchmark-pivot.md` (the long-form plan stays as the canonical reference; this is the compressed sprint variant)
**Target:** IJDRR submission in 72 hours. IGARSS condensation deferred to week 2.

## Why this plan exists

The 3-fold sweep is in. The pivot narrative (KD doesn't help; students beat teacher; first multi-platform edge benchmark) is empirically locked. Remaining work is mostly mechanical: get 2-platform latency numbers, generate figures, rewrite manuscript, submit. The original plan budgeted 4-6 weeks; the user wants 3 days. This plan is the minimum surface area that still delivers a credible paper.

## The corrected paper narrative (unchanged from v1 spec)

> *FSSD's saturation at IoU ≈ 0.93 means modern ImageNet-pretrained lightweight backbones already match a UNet+EfficientNet-B0 teacher. We present the first systematic edge-deployment benchmark for flood segmentation on Apple M2 Pro CPU and AWS Graviton ARM CPU, with INT8 ONNX quantization, and an honest KD ablation showing distillation does not help on this saturated benchmark.*

### Three contributions (final)
1. First multi-platform edge-deployment benchmark for flood segmentation (FP32 + INT8 on M2 Pro CPU + AWS Graviton ARM CPU).
2. Negative empirical finding on knowledge distillation for FSSD-class flood segmentation.
3. Open-source release: code, FP32 + INT8 ONNX weights, reproducible Kaggle notebook.

### Demoted to "future work"
- OOD generalization on Sen1Floods11 RGB (data prep done; eval deferred to follow-up).
- Android phone + browser WASM measurements (ONNX format already in place; reproducible by downstream users).
- Per-pixel bootstrap statistical test (3-fold Wilcoxon reported with explicit n=3 caveat).

## Experimental matrix (no new training runs needed)

| Component | Status | Source |
|---|---|---|
| FP32 segmentation (3-fold mean ± std for 14 configs) | ✅ done | `code/runs/3fold/summary.json` |
| 3-fold paired Wilcoxon (no-KD vs comb-KD) | ✅ done | `code/runs/3fold/stats.json` |
| INT8 quantization (3 students × 3 folds) | ⏳ pending Percentile-cal re-run | `code/runs/3fold/quantized.json` (to be regenerated) |
| Latency on M2 Pro CPU | ⏳ Day 1 | run `scripts/bench_m2.py` locally |
| Latency on AWS Graviton t4g.small | ⏳ Day 1 | provision + `scripts/bench_graviton.sh` |
| Pareto figure (4-axis) | ⏳ Day 2 | `scripts/make_figures.py` from above |
| Qualitative panel (4 images) | ⏳ Day 2 | `scripts/make_qualitative.py`, no OOD rows |
| Manuscript rewrite | ⏳ Day 3 | `docs/FloodLite_Manuscript.docx` |

## Day-by-day plan

### Day 1 — Verify INT8 + measure 2 platforms (≈6 working hours)

| Step | Where | Time | Owner |
|---|---|---|---|
| 1.1 Re-quantize on Kaggle with Percentile calibration | Kaggle | 10 min | user |
| 1.2 Decision: 3-of-3 architectures quantize cleanly? | — | 1 min | user |
| 1.3a If yes: download `quantized.json` + INT8 ONNX files | Kaggle → local | 5 min | user |
| 1.3b If no for MNv3/MViT: keep EffLite0 INT8 only; document the other two as "PTQ unstable; QAT future work" | — | — | — |
| 1.4 ONNX export of all FP32 models on M2 Pro | local M2 | 15 min | user |
| 1.5 `bench_m2.py` (PyTorch CPU only — skip CoreML to save time) | local M2 | 30 min | user |
| 1.6 Provision AWS Graviton `t4g.small` (free tier, 12-month) | AWS console | 15 min | user |
| 1.7 `bench_graviton.sh` (scp exports, run, collect JSON) | AWS | 1 hr | user |
| 1.8 Commit all latency + quant JSONs | local | 5 min | me |
| **Decision gate** | If 1.6/1.7 blow up (AWS auth, billing, etc.), drop Graviton and use **browser WASM in Chrome on M2** as the 2nd platform — runs in 30 min with no external service | — | — |

### Day 2 — Figures + write 80% of manuscript (≈8 working hours)

| Step | Where | Time | Owner |
|---|---|---|---|
| 2.1 Generate Table 1 (footprints): params + GFLOPs for 5 architectures | local | 30 min | me — script |
| 2.2 Generate Table 2 (3-fold mean ± std) from summary.json | local | 15 min | me — script |
| 2.3 Generate Table 3 (KD ablation on best student) | local | 15 min | me — script |
| 2.4 Generate Table 4 (quantization FP32 vs INT8) | local | 15 min | me — script |
| 2.5 Generate Table 5 (per-platform latency) | local | 15 min | me — script |
| 2.6 Pareto figure (4-axis: IoU vs Params, FLOPs, M2 ms, Graviton ms) | local | 30 min | me — `make_figures.py` |
| 2.7 Qualitative panel (2 rows: easy, hard from FSSD val) — skip OOD rows | local | 30 min | me — `make_qualitative.py` |
| 2.8 Pipeline diagram F1 (PowerPoint/Keynote one-pager) | manual | 30 min | user |
| 2.9 Rewrite manuscript Abstract + Intro | Word | 1.5 hr | user + me drafts |
| 2.10 Rewrite Method (§3.1-§3.4; remove §3.5 OOD) | Word | 1 hr | user + me drafts |
| 2.11 Rewrite Experimental Setup | Word | 30 min | user + me drafts |
| 2.12 Rewrite Results (paste T1–T5 + F1–F3) | Word | 1.5 hr | user |
| 2.13 Title change to *FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation* | Word | 1 min | user |

### Day 3 — Finish manuscript + submit (≈6 working hours)

| Step | Where | Time | Owner |
|---|---|---|---|
| 3.1 Rewrite Discussion (lead with negative KD; per-platform recommendations) | Word | 2 hr | user + me drafts |
| 3.2 Add new Limitations section | Word | 30 min | user + me drafts |
| 3.3 Rewrite Conclusion | Word | 30 min | user + me drafts |
| 3.4 Delete Appendix A (placeholder map) | Word | 1 min | user |
| 3.5 Add single 1-page "Reproducibility" appendix (point to GitHub + Zenodo DOI) | Word | 30 min | user |
| 3.6 Final read-through, fix typos, check tables/figures | Word | 1 hr | user |
| 3.7 Tag GitHub `v1.0-ijdrr`, push to origin | git | 5 min | user |
| 3.8 Mint Zenodo DOI from GitHub release (web UI) | Zenodo | 15 min | user |
| 3.9 Add DOI to manuscript front-matter | Word | 1 min | user |
| 3.10 Draft IJDRR cover letter (3 short paragraphs) | text | 30 min | user + me drafts |
| 3.11 Export manuscript to PDF | Word | 5 min | user |
| 3.12 IJDRR Editorial Manager submission | web | 30 min | user |
| **End state: submitted** | | | |

## Hard kill switches

- **D1.2 INT8 still broken for MNv3/MViT after Percentile fix.** Action: keep only EfficientNet-Lite0 in Table 4; reframe the quantization claim as "INT8 PTQ for the most quantization-friendly student" rather than "for all students". Drop the others' INT8 cells with "PTQ unstable; QAT-based quantization left to future work."
- **D1.6 AWS Graviton account setup > 30 min.** Action: drop Graviton; substitute browser WASM (ONNX Runtime Web in Chrome on M2 Pro). The two platforms become "M2 Pro CPU" and "browser WASM". The Pareto x-axis becomes M2 ms + WASM ms.
- **D2.6 Pareto figure looks confusing with 2 platforms.** Action: collapse to a 2-axis Pareto (IoU vs Params + IoU vs FLOPs) and put the per-platform latency in a separate horizontal bar chart.
- **D3.12 IJDRR Editorial Manager rejects format.** Action: fall back to MDPI Remote Sensing (closer formatting, similar scope, ~$2.6k APC).

## Tables and figures (cut list)

Kept:
- T1 Model footprints
- T2 3-fold segmentation (mean ± std)
- T3 KD ablation isolated (Wilcoxon)
- T4 Quantization (FP32 vs INT8; possibly just EffLite0)
- T5 Per-platform latency (M2 + Graviton OR M2 + WASM)
- F1 Pipeline diagram
- F2 Pareto (2- or 4-axis depending on platform count)
- F3 Qualitative panel (2 rows: easy, hard; no OOD)

Cut (deferred to follow-up paper):
- T6 OOD on Sen1Floods11
- F4 Per-platform throughput bars (folded into T5)
- Appendix B Model Card (replaced with concise GitHub README)
- Appendix C Datasheet (replaced with citation of FSSD's own data card)
- Appendix D Full Reproducibility Checklist (replaced with 1-page repro appendix)

## What "submitted" means at the end of Day 3

1. PDF of revised manuscript uploaded to IJDRR Editorial Manager
2. GitHub repo tagged `v1.0-ijdrr` with `runs/3fold/results.json` + tagged release
3. Zenodo DOI minted from that tag and cited in manuscript
4. Cover letter explaining the contribution

Hugging Face Hub upload + IGARSS 4-page condensation happen in week 2, *after* submission. They're not blocking.

## What this plan is NOT

- It is not "a thorough study". It is the minimum credible submission.
- It does not maximize impact. A 6-week version with OOD + 4 platforms + statistical bootstrap would be stronger.
- It does not include IGARSS. That's a 1-day condensation post-IJDRR-submission.
- It does not include 5 random seeds. The 3-fold variance is what we report; spec §3 already documents this limit.

If reviewers reject for "missing OOD" or "missing Android/Pi", we either (a) add it in revision, or (b) cite the follow-up plan and resubmit. Better to ship a tight paper now and iterate than to spend 4 more weeks polishing.
