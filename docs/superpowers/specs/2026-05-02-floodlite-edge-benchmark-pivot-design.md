# FloodLite revision design — pivot to edge-deployment benchmark

**Date**: 2026-05-02
**Author**: Ibrahim Khalil (with Claude Code)
**Status**: Awaiting final user approval before transition to implementation plan
**Target venue**: Elsevier *International Journal of Disaster Risk Reduction* (IJDRR), primary
**Backup venue**: IGARSS 2026 (4-page extended abstract)
**Timeline**: 4–6 week sprint (Approach 2)

---

## 1. Why we are pivoting

The Fold-0 results contradict the manuscript-as-drafted:

```
                       none      resp      feat      comb
mobilenetv3_small    0.9203    0.9213    0.9310    0.9228
efficientnet_lite0   0.9348    0.9321    0.9329    0.9302   ← KD hurts
mobilevit_xxs        0.9450    0.9430    0.9467    0.9395   ← combined is worst
teacher (UNet+EffB0)        0.9340                          ← students > teacher
```

- Two students *exceed* the teacher without distillation.
- Combined response+feature KD is the worst of the four KD configs on EfficientNet-Lite0 and MobileViT-XXS.
- Quantization in `floodlite/quantize.py` failed (CUDA-tensor calibration; PyTorch INT8 PTQ requires CPU).
- Per-platform latency was measured on a Kaggle T4 GPU, not on M2 Pro / Android / WASM.
- Only Fold 0 was run; no statistical significance, no error bars.

Mechanistic interpretation: FSSD has 663 images and saturates at IoU ≈ 0.93. ImageNet-pretrained lightweight backbones already match a UNet+EfficientNet-B0 teacher. KD has no headroom to transfer.

**The manuscript's primary claim ("KD enables edge flood segmentation") is empirically false.** We pivot the contribution rather than fabricate results.

## 2. New paper narrative

> *FSSD's saturation at IoU ≈ 0.93 means modern ImageNet-pretrained lightweight backbones already match a UNet+EfficientNet-B0 teacher. The open question is therefore not "how do we close the gap with KD" but "do these candidate models actually deploy". We present the first systematic edge-deployment benchmark for flood segmentation across Apple M2 Pro, Android (Termux + TFLite), browser WASM (ONNX Runtime Web), and AWS Graviton ARM CPU, with INT8 post-training quantization, an OOD test on Sen1Floods11 RGB, and an honest KD ablation showing distillation does not help on a saturated benchmark with strong pretrained students.*

### Three contributions (final)

1. **First multi-platform edge-deployment benchmark for flood segmentation.** FP32 + INT8 latency on M2 Pro, Android phone, browser WASM, AWS Graviton ARM, with paired-Wilcoxon-tested IoU.
2. **Negative empirical finding on knowledge distillation for FSSD-class flood segmentation.** ImageNet-pretrained lightweight UNet students match the teacher without KD; combined response+feature KD does not improve over no-KD on this saturated benchmark.
3. **Single-domain OOD characterization on Sen1Floods11 RGB.** Quantifies the cost of training on FSSD only; sets a baseline for follow-up work.

### Demoted from contribution → ablation

Combined-KD recipe (response KL on softened logits + feature MSE through a 1×1 adapter on decoder block 2). Code remains; we simply do not claim it works.

### Niche gap (sharpened, kept)

Karcı et al. 2026 + 26 cited works → none reports parameters, FLOPs, per-platform latency, OOD performance, or quantized-model behavior. We close all four.

## 3. Experiment matrix

### Training set
FSSD (663 images), KFold(n_splits=5, shuffle=True, random_state=42), folds {0, 1, 2}. Image size 256×256. Existing `floodlite/data.py` already supports.

### Models (5 architectures × KD configs × folds)

| # | Encoder | Decoder | Pretrained | KD configs | Role |
|---|---|---|---|---|---|
| 1 | EfficientNet-B0 (smp) | UNet | ImageNet | — | Teacher (Karcı parity) |
| 2 | MobileNetV2 (smp) | UNet | ImageNet | — | **NEW** off-the-shelf lightweight baseline (no KD) |
| 3 | MobileNetV3-Small (timm) | UNet | ImageNet | none, resp, feat, comb | Student A |
| 4 | EfficientNet-Lite0 (timm) | UNet | ImageNet | none, resp, feat, comb | Student B |
| 5 | MobileViT-XXS (timm) | UNet | ImageNet | none, resp, feat, comb | Student C |

Total: 1 teacher + 1 baseline + 3 students × 4 KD = **14 configs × 3 folds = 42 training runs**.

### Compute budget
- Cap epochs at **35** (current Fold-0 plateaus by ep 45).
- T4 ≈ 10 min/epoch on this workload → 35 ep × 42 runs ≈ 245 T4-hr.
- Distributed across Kaggle T4×2 + Colab T4 → 3–4 calendar days.

### Quantization (INT8 PTQ)
- Apply to the **no-KD** versions of MobileNetV3-Small, EfficientNet-Lite0, MobileViT-XXS — these are the deployment candidates.
- Run on CPU (fix existing CUDA bug). Calibrate with 200 train images.
- Known risk: MobileViT-XXS may not quantize cleanly; fall back to FP16 if so.

### OOD set
~250 chips from Sen1Floods11 (Sentinel-2 RGB visualizations), CC-BY 4.0. Eval-only — no fine-tuning. Same transforms as FSSD val pipeline.

### Latency benchmarks (4 platforms)

| Platform | Runtime | Models | Iterations |
|---|---|---|---|
| Apple M2 Pro | PyTorch CPU + CoreML | teacher + 3 students FP32 + 3 students INT8 | 20 warm + 200 timed |
| Android phone | Termux + tflite-runtime | 3 students FP32 (+ INT8 if convertible) | 20 warm + 100 timed (mobile) |
| Browser | onnxruntime-web (WASM SIMD), Chrome on M2 Pro | 3 students FP32 | 20 warm + 200 timed |
| AWS Graviton t4g.small | onnxruntime CPU (ARM) | 3 students FP32 + INT8 | 20 warm + 200 timed |

All single-image batch, 256×256 input.

### Statistical tests
- 3-fold means ± std for every cell of T2.
- **"Best student"** is defined post-hoc as the (architecture, KD config) pair with the highest mean IoU across the 3 folds on FSSD val.
- Paired Wilcoxon signed-rank across folds for two headline comparisons:
  1. no-KD vs comb-KD (per student)
  2. teacher vs best student INT8
- **Caveat**: with only 3 paired folds, the Wilcoxon test has very low power (the smallest reportable two-sided p-value is 0.25 for n=3). We will (a) report exact p-values and acknowledge the limit, and (b) supplement with a per-image paired bootstrap (1000 resamples) on the held-out validation pixels of fold 0 for the same two comparisons — this gives the publication-grade statistical claim. The 3-fold Wilcoxon is reported only as a sanity check on cross-fold variance.

## 4. Tables and figures

### Tables (in-paper)
- **T1 — Model footprints.** Params (M), GFLOPs @256², FP32 disk (MB), INT8 disk (MB).
- **T2 — In-distribution segmentation.** Acc / Prec / Rec / F1 / IoU; mean ± std over 3 folds; for {teacher, MN-V2 baseline, 3 students × 4 KD configs}.
- **T3 — KD ablation isolated on best student.** MobileViT-XXS at {none, resp, feat, comb} with paired Wilcoxon vs no-KD.
- **T4 — Quantization.** FP32 IoU vs INT8 IoU; ΔIoU; size reduction × ; for the three deployable students.
- **T5 — Per-platform latency.** p50 / p95 ms + FPS for each model on each platform; INT8 vs FP32 speedup factor.
- **T6 — OOD (Sen1Floods11 RGB).** IoU/F1 for five models, FSSD-trained → S1F11-tested: teacher, MobileNetV2 baseline, and the best KD-config per student (MobileNetV3-S, EfficientNet-Lite0, MobileViT-XXS), all FP32.

### Figures
- **F1 — Pipeline diagram.** Teacher → KD students → quantization → 4-platform deployment.
- **F2 — 4-axis Pareto.** IoU vs {Params, FLOPs, M2 latency, Android latency}.
- **F3 — Qualitative panel.** 4 rows × 5 cols: image, GT, teacher pred, best student INT8 pred, error map. Two in-distribution (easy + hard), two OOD (success + failure).
- **F4 (supplementary) — Per-platform throughput bars** + INT8/FP32 speedup factor.

### Killed
Training-curve plots; placeholder Pareto from `code/README.md` §4.

## 5. Manuscript section-by-section rewrite

| Section | Action | Hours |
|---|---|---|
| Title | Drop "Knowledge-Distilled" → propose: *FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation* | 1 |
| Abstract | Full rewrite around three new contributions | 3 |
| 1. Introduction | Keep niche-gap framing; reframe KD from contribution → ablation | 4 |
| 2. Related work | Add §2.4 *Lightweight segmentation backbones*; cite BiSeNetV2, PIDNet, Fast-SCN | 4 |
| 3. Method | §3.1 teacher; §3.2 students + KD ablation; §3.3 PTQ INT8 (CPU constraint); §3.4 export pipeline (ONNX → CoreML / TFLite); §3.5 OOD protocol | 6 |
| 4. Experimental setup | Add baseline; 3-fold protocol; OOD; 4-platform latency rigging | 4 |
| 5. Results | Full rewrite around T1–T6 / F1–F4 | 8 |
| 6. Discussion | Lead with the negative-KD finding and mechanistic explanation (saturated benchmark, strong pretrains). Per-platform deployment recommendations. OOD interpretation. | 6 |
| **7. Limitations (NEW)** | Single-domain training; RGB-only; no SAR/multispectral; no energy measurements; phone-specific Android numbers | 2 |
| 8. Conclusion | Rewrite around benchmark contribution + open artifacts | 2 |
| Appendix A | **DELETE** (placeholder map) | 0.1 |
| **Appendix B (NEW)** | Model Card (HuggingFace format) | 3 |
| **Appendix C (NEW)** | Datasheet for the FSSD preprocessing pipeline | 2 |
| **Appendix D (NEW)** | ML Reproducibility Checklist | 1 |

Total writing: ~46 hours ≈ 6 working days.

## 6. Primary research deliverables (artifacts)

1. **GitHub repo** — already prepared; tag `v1.0-ijdrr`, MIT license.
2. **Hugging Face Hub** — 6 artifacts (3 students × {FP32, INT8}), Apache-2.0, with Model Card.
3. **Zenodo DOI** for tagged code release.
4. **`runs/results.json`** + per-fold checkpoints uploaded to HF.
5. **Datasheet** for the FSSD preprocessing pipeline (FSSD itself is third-party).
6. **ML Reproducibility Checklist** + Papers-With-Code submission.

## 7. Code changes required

| File | Change | Hours |
|---|---|---|
| `floodlite/quantize.py` | Force CPU device on model and calibration tensors before `torch.ao.quantization.prepare()`; document INT8 PTQ requires CPU | 1 |
| `floodlite/models.py` | Add `make_baseline_mobilenetv2()` returning `smp.Unet(encoder_name="mobilenet_v2")` | 0.5 |
| `floodlite/data.py` | Add `make_sen1floods11_loader(root, n_chips=250)` for OOD eval; same transforms as FSSD val | 2 |
| `floodlite/benchmark.py` | Add `benchmark_onnxruntime(onnx_path, providers=["CPUExecutionProvider"])` for ARM/WASM-equivalent runs | 2 |
| `scripts/run_full_experiment.py` | Default to `--epochs_teacher 35 --epochs_student 35 --folds 3`; add `--baseline mobilenetv2` flag | 1 |
| `scripts/run_edge_benchmarks.py` | NEW — drives M2 / Android / WASM / Graviton runs and writes a single CSV | 4 |
| `notebooks/floodlite_kaggle.ipynb` | Update final cells to print T1–T6 ready-to-paste; remove placeholder pattern | 2 |
| `floodlite/train.py` | Migrate `torch.cuda.amp.*` → `torch.amp.*` for forward-compat (low priority) | 0.5 |

Total code: ~13 hours ≈ 1.5 working days.

## 8. Per-week plan

### Week 1 (D1–D7) — fix the foundation
- D1: fix INT8 quant CPU bug; add MobileNetV2 baseline; add Sen1Floods11 loader.
- D2: smoke-test full pipeline on Fold 0 (1 student) end-to-end including INT8 + ONNX export.
- D3–D7: 3-fold sweep on Kaggle T4×2 (parallelize across two accounts if available): 14 configs × 3 folds = 42 training runs.

### Week 2 (D8–D14) — quantize, OOD, qualitative
- D8: INT8-quantize 3 student no-KD configs × 3 folds; eval on FSSD val.
- D9–D10: download Sen1Floods11 chips; preprocess; eval all final models OOD.
- D11–D12: ONNX export with opset 17; verify numerical parity FP32 PyTorch ↔ FP32 ONNX < 1e-3.
- D13: generate qualitative panel (4 examples, manual selection); generate Pareto plot.
- D14: write Method + Experimental Setup sections.

### Week 3 (D15–D21) — measure deployment
- D15: CoreML conversion + M2 Pro latency.
- D16: AWS Graviton t4g.small benchmark.
- D17–D18: Android Termux setup + TFLite conversion + latency on your phone.
- D19: WASM ONNX-RT-Web latency in Chrome on M2 Pro (mid-range proxy).
- D20–D21: write Results + Discussion + Limitations.

### Week 4 (D22–D28) — assemble + submit IJDRR
- D22: rewrite Abstract + Intro.
- D23: write Model Card + Datasheet + Reproducibility checklist.
- D24: full read-through, internal review, fix tables/figures.
- D25: Hugging Face upload + Zenodo DOI mint.
- D26: IJDRR cover letter; submit.
- D27–D28: buffer.

### Week 5 (D29–D35) — IGARSS submission + cleanup
- D29–D31: condense IJDRR draft to IGARSS 4 pages (no new experiments — just trim).
- D32: IGARSS submit.
- D33–D35: buffer / co-author review responses if any.

### Hard gates / kill switches
- If by D7 the 3-fold sweep is < 70% complete, drop to 2 folds.
- If Android Termux setup blocks > 1 day, fall back to AWS Graviton as the headline ARM number; frame Android as future work.
- If quantization breaks MobileViT-XXS (known INT8 ViT issue), report FP16 only for that model.

## 9. Out of scope (paper-2 / paper-3 territory)

- Calibration / temperature scaling / MC-Dropout / evidential head → Paper 2 (FloodCalibrate).
- SAR / Sentinel-1 / multispectral → Paper 3 (FloodEDL-Mamba).
- Mamba / state-space backbones → Paper 3.
- Energy / Joules-per-inference → future work (no USB power meter available).
- Jetson Nano / Orin → future work.
- QAT (quantization-aware training) → out of scope; PTQ only.
- Additional baselines beyond MobileNetV2 → resist scope creep.

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle session timeout mid-run | Med | Med | Save checkpoints every epoch; resume from latest |
| MobileViT-XXS INT8 PTQ fails | Med-High | Med | Report FP16 for that model; document |
| Sen1Floods11 chip preprocessing mismatch (e.g., bit depth) | Med | Med | Match FSSD normalization exactly; sanity-check on first 10 chips |
| Android Termux + TFLite setup time blowup | Med | Med | Fall back to AWS Graviton as headline ARM; keep Android as supplementary |
| 3-fold sweep slips past D7 | Med | Med | Drop to 2 folds; document |
| IJDRR rejects / long review | Med | Low | IGARSS submission as parallel hedge; submission-ready by D32 |
| Concurrent paper publishes similar idea | Low-Med | High | Differentiate via 4-platform multi-runtime benchmark — very few do this |

---

**Awaiting user review.** Once approved, I transition to the writing-plans skill to produce a code-level implementation plan for §7 + §3.
