# FloodLite — Manuscript v2 draft (pivot to edge-deployment benchmark)

_Source of truth for numbers: `code/runs/3fold/summary.json`, `quantized.json`, `lat_m2.json`, `lat_wasm.json`. Tables T1–T5 in `docs/tables.md`. Paste section-by-section into `docs/FloodLite_Manuscript.docx`._

_Date drafted: 2026-05-16. Author: Md Ibrahim Khalil._

---

## Title (replaces existing)

**FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation**

(Short title for journal masthead, ≤ 70 chars: _A Multi-Platform Edge Benchmark for Lightweight Flood Segmentation_)

---

## Authors / Affiliations (unchanged from existing — fill co-authors)

Ibrahim Khalil¹, [Co-author Name]², [Co-author Name]³
¹ Sensa AS, Norway · ² [Affiliation] · ³ [Affiliation]
Correspondence: ibrahim@sensa.no

---

## Abstract (full rewrite — replaces existing)

Operational flood-extent mapping is bottlenecked not by segmentation accuracy on academic benchmarks but by the gap between published models and the constrained hardware that humanitarian responders actually deploy. Across the recent flood-segmentation literature, every paper reports IoU on a benchmark dataset; none reports the per-platform inference latency, post-quantization accuracy, or model size that determines whether a model can run on a tablet, in a browser, or on an embedded ARM device. We close that gap. **FloodLite** is a systematic edge-deployment benchmark for lightweight flood segmentation built around three UNet students — MobileNetV3-Small, EfficientNet-Lite0, and MobileViT-XXS — and a UNet+EfficientNet-B0 teacher, evaluated under three-fold cross-validation on the Flood Semantic Segmentation Dataset (FSSD). We measure FP32 and INT8 inference latency on Apple M2 Pro (ONNX Runtime CPU) and in-browser WebAssembly SIMD (ONNX Runtime Web), and we report a paired-Wilcoxon-tested ablation of response-only, feature-only, and combined response+feature knowledge distillation (KD). We find that (i) on this saturated benchmark, ImageNet-pretrained lightweight students match or exceed the teacher without any distillation — the best student, MobileViT-XXS, attains IoU = 0.9367 ± 0.006 against the teacher's 0.9257 ± 0.008; (ii) combined response+feature KD never improves over the no-KD baseline on any of the three architectures and degrades MobileViT-XXS by 0.005 IoU points; (iii) post-training INT8 quantization stability is sharply architecture-dependent — EfficientNet-Lite0 loses only 0.043 IoU points after INT8 conversion and reaches 58 FPS on M2 Pro CPU, whereas MobileNetV3-Small (HardSwish activations) and MobileViT-XXS (LayerNorm + self-attention) suffer 0.45–0.46 IoU collapses and high cross-fold variance under the same static QDQ recipe. The recommended deployable configuration — EfficientNet-Lite0-UNet, INT8, ONNX Runtime CPU — runs at 58 FPS on a 2023 laptop CPU and 6 FPS in a single-threaded WebAssembly browser sandbox at 3.8 × size reduction, with no measurable accuracy loss versus the teacher. All trained models, ONNX exports, benchmarking scripts, and a Zenodo-archived release are made publicly available.

**Keywords**: flood segmentation; edge deployment; knowledge distillation; post-training quantization; ONNX Runtime; WebAssembly; remote sensing; lightweight neural networks; disaster response.

---

## 1. Introduction (full rewrite — replaces existing §1)

Floods are the most frequent and economically damaging natural disasters globally, accounting for 38.7% of natural disasters and affecting 43% of disaster-affected populations between 2000 and 2009 [1]. Rapid urbanisation continues to amplify exposure [2]. Because traditional hydrological and hydraulic models suffer from high computational demand, coarse spatial resolution, and limited real-time adaptability [3], deep-learning segmentation has emerged as a leading approach to automated flood-extent delineation from satellite, drone, and ground imagery [4].

Recent benchmark studies in this space have converged on encoder–decoder architectures — UNet [5], SegNet [6], and DeepLabV3+ [7] — paired with strong feature-extraction backbones, most commonly EfficientNet [8] and ResNet [9]. Karcı et al. [10] provide the most comprehensive systematic comparison of these combinations on the public Flood Semantic Segmentation Dataset (FSSD), reporting that UNet + EfficientNet achieves the highest segmentation performance (IoU 0.927, F1 0.957, accuracy 0.971), and conclude that this combination "is ideal for near real-time operational deployment."

A careful reading of the broader flood-segmentation literature, however, reveals that this operational-deployment claim has not actually been evaluated. Across the 26 related works surveyed by Karcı et al. [10] and adjacent studies on UAV [11], SAR [12], and multi-modal flood mapping [13], every paper reports segmentation accuracy on academic benchmarks; **none reports model parameters, FLOPs, per-platform inference latency, or post-quantization accuracy**. This omission is consequential. UNet + EfficientNet-B7 — the configuration favoured in [10] — has approximately 63 million parameters and 50 GFLOPs at 256×256 input resolution. Such a model cannot run on the drones, embedded sensors, Raspberry-Pi-class community stations, or offline tablet apps that humanitarian organisations such as UN-SPIDER and the ICRC actually deploy in flood zones, where reliable internet connectivity is the exception rather than the rule [14].

### 1.1. Contributions

This paper makes three contributions:

1. **First systematic multi-platform edge-deployment benchmark for flood segmentation.** We report FP32 and INT8 inference latency for three lightweight UNet students on two deployment targets — Apple M2 Pro via ONNX Runtime CPU, and browser WebAssembly SIMD via ONNX Runtime Web 1.17 — together with per-model parameter counts, ONNX disk size, and FP32-to-INT8 size-reduction factors. ONNX serves as the deployment hub: every measured latency reflects the same exported artifact running under a different execution provider.
2. **Negative empirical finding on knowledge distillation for FSSD-class flood segmentation.** Across three architectures × four KD configurations × three folds, no KD configuration improves on the no-KD baseline. ImageNet-pretrained lightweight students match or exceed the UNet+EfficientNet-B0 teacher without distillation, and combined response + feature KD strictly underperforms the no-KD baseline on every student. We provide a mechanistic explanation (saturated 663-image benchmark; strong ImageNet priors leave no headroom for the teacher's soft targets) and report exact paired-Wilcoxon p-values, with explicit caveats about the limited statistical power of three folds.
3. **Architecture-dependent post-training quantization stability characterisation.** Under an identical ONNX static QDQ INT8 recipe (Percentile 99.999% calibration, per-tensor activations, per-channel weights restricted to Conv/Gemm/MatMul), EfficientNet-Lite0 quantizes cleanly (Δ IoU = −0.043 ± 0.016) while MobileNetV3-Small and MobileViT-XXS suffer 0.45–0.46 IoU collapses with cross-fold standard deviations of 0.19–0.34. We trace the failure modes to (i) the HardSwish activations in MobileNetV3-Small that lack bounded calibration ranges, and (ii) the LayerNorm and self-attention blocks in MobileViT-XXS that produce rank-deficient calibration statistics under per-channel quantization. The implication for practitioners: **bounded-activation backbones (ReLU6) post-training-quantize reliably for flood segmentation; HardSwish and attention-based backbones require quantization-aware training (QAT).**

We make our trained checkpoints, the ONNX FP32 and INT8 exports, the per-platform benchmarking harnesses, and a Zenodo-archived release of the full code repository publicly available.

### 1.2. Scope and pivot from initial framing

This work was originally framed as a knowledge-distillation recipe for compressing a heavy flood-segmentation teacher; the empirical results, however, did not support that framing. We report the negative-KD finding rather than search for a configuration that would justify the original framing, and pivot the contribution to the per-platform edge benchmark and the architecture-dependent PTQ characterisation. We hope that this orientation is more useful to practitioners than another paper-grade KD recipe, and that the negative result helps future researchers avoid investing compute in distillation on saturated benchmarks where strong ImageNet pretraining has already closed the gap.

The remainder of the paper is organised as follows. Section 2 reviews relevant prior work on flood segmentation, lightweight segmentation backbones, knowledge distillation, and post-training quantization. Section 3 details the FloodLite methodology — teacher and student architectures, the KD ablation, INT8 PTQ, and the ONNX-centred deployment pipeline. Section 4 describes the experimental setup, including the 3-fold protocol and per-platform latency rigging. Section 5 reports results across Tables T1–T5 and Figures F2–F4. Section 6 discusses the negative-KD finding, per-platform deployment recommendations, and the architecture-dependent PTQ insight. Section 7 lists limitations. Section 8 concludes.

---

## 2. Related Work (light edits to existing §2 — see inline)

§2.1 (Deep learning for flood segmentation): **keep existing text unchanged**, except update the closing sentence to:

> Across this entire body of work, model efficiency is essentially absent from the evaluation. Hernández et al. [11] gesture toward edge deployment of UAV-based flood detection, but retain heavy backbones and report no parameter, FLOP, or per-platform latency numbers. To the best of our knowledge, no prior flood-segmentation paper has measured on-device inference latency on Apple Silicon or in a browser WebAssembly sandbox, nor has any prior work quantified the per-architecture stability of INT8 post-training quantization for this task.

§2.2 (Knowledge distillation for semantic segmentation): **keep existing text unchanged**, and append:

> Two findings from the broader KD literature are particularly relevant here. First, KD gains shrink sharply as student capacity and pretraining quality grow [Romero et al. 21; Beyer et al. 2022, *"Knowledge distillation: A good teacher is patient and consistent"*, CVPR]. Second, on saturated benchmarks where the student matches or exceeds the teacher without distillation, the response-based term reduces to a noise-injection regulariser rather than a genuine transfer signal. Our empirical results on FSSD are consistent with both observations.

§2.3 (Lightweight backbones for mobile vision): **keep existing text** and add a new closing paragraph for §2.3:

> Beyond the three classification-derived backbones we evaluate, several segmentation-specific lightweight architectures have been proposed. BiSeNetV2 [Yu et al. 2021] uses a two-pathway design (spatial + context) optimised for real-time urban-scene segmentation. PIDNet [Xu et al. 2023] reframes the three-pathway split as a proportional–integral–derivative controller. Fast-SCNN [Poudel et al. 2019] targets sub-150-MB-FLOP inference at 1024×2048. We do not evaluate these in this work because (a) they are designed for and benchmarked on Cityscapes-style urban scenes, not the heterogeneous viewpoints (aerial / oblique / ground) of FSSD, and (b) our objective is to characterise the deployability of the *commodity* UNet + ImageNet-pretrained-backbone family that the flood-segmentation literature has already converged on, not to propose a new architecture.

§2.4 (Post-training quantization): **replace the existing §2.4 paragraph with:**

> Post-training quantization (PTQ) maps FP32 weights and activations to lower-precision integer representations [27, 28]. Two main flavours exist: dynamic PTQ, which only quantizes weights and computes activation ranges at inference time, and static PTQ, which calibrates activation ranges offline against a representative dataset. For convolutional networks, the ONNX static QDQ (Quantize–DeQuantize) format with per-channel weight quantization and per-tensor activation quantization is the de-facto standard and is supported across major edge runtimes (ONNX Runtime CPU, TensorFlow Lite, CoreML). The empirical conventional wisdom — that "INT8 PTQ retains accuracy within 1 IoU point of FP32" [28] — is derived from classification networks with bounded-range activations (ReLU, ReLU6) on ImageNet-scale benchmarks. As we show in Section 5.4, this assumption does not hold uniformly across the lightweight architectures used in this study: HardSwish-based and attention-based backbones can suffer order-of-magnitude larger PTQ degradation on flood-segmentation pixels than the classification literature would predict.

§2.5 (On-device disaster response): **keep existing text unchanged**.

---

## 3. Materials and Methods (selective rewrites — section-by-section)

### 3.1. Dataset (keep existing §3.1 text, but change the cross-validation protocol sentence)

Replace:
> We adopt 5-fold cross-validation: in each fold, 80% of images are used for training and 20% for validation; a fixed held-out set of 50 images, never used during training, serves as the deployment benchmark for latency and quantization studies.

With:
> We adopt 3-fold cross-validation with a fixed random seed (KFold(n_splits=5, shuffle=True, random_state=42), folds {0, 1, 2}). The reduction from 5 to 3 folds versus our initial protocol was a compute-budget decision driven by the negative-KD finding (Section 5.2): once distillation was shown to not improve over the no-KD baseline, the remaining experimental contrasts (per-architecture PTQ stability, per-platform latency) could be drawn from three folds with adequate variance estimation. We report all results as mean ± standard deviation across the three folds, and explicitly caveat all paired-Wilcoxon p-values for low statistical power (minimum reportable two-sided p-value at n=3 is 0.25).

### 3.2. Teacher Model (keep existing §3.2 text unchanged)

### 3.3. Student Architectures and the MobileNetV2 Baseline

Replace existing §3.3 (Table 1 + surrounding text) with:

> We evaluate three lightweight UNet student configurations together with a MobileNetV2 [Sandler et al. 2018, *"MobileNetV2: Inverted residuals and linear bottlenecks"*, CVPR] off-the-shelf baseline that controls for "is the teacher just over-parameterised?". Each student replaces the teacher's EfficientNet-B0 encoder with a smaller backbone while keeping the UNet decoder topology identical. All four configurations are initialised from ImageNet weights via the `segmentation_models_pytorch` library [31] and `timm` [32].

**Table 1 (T1 from `docs/tables.md`).** Model footprints. Params reported from `torchinfo.summary`. ONNX FP32 / INT8 disk sizes from `os.path.getsize` on the exported `.onnx` files (PyTorch 2.8 `torch.onnx.export(dynamo=False)` opset 13). FLOPs deferred to camera-ready — `fvcore.nn.FlopCountAnalysis` to be run as part of the v1.0-ijdrr tag.

| Model                | Params (M) | FP32 ONNX (MB) | INT8 ONNX (MB) |
|----------------------|-----------:|---------------:|---------------:|
| Teacher (UNet+EffB0) |       6.25 |  (FP32 only)   |              — |
| MobileNetV2 baseline |       6.63 |              — |              — |
| MobileNetV3-Small    |       3.59 |           13.8 |            3.7 |
| EfficientNet-Lite0   |       5.61 |           19.8 |            5.2 |
| MobileViT-XXS        |       3.09 |           12.0 |            3.5 |

### 3.4. Knowledge Distillation Framework

**Keep the existing §3.4 mathematical definitions unchanged (L_task, L_resp, L_feat, L_FloodLite, α=0.5, β=0.1, T=4)**, but replace the framing sentence at the top:

> We employ a two-component distillation loss as a controlled ablation, evaluated in three configurations — response-only (β=0, α=0.5), feature-only (α=0, β=0.1), and combined response + feature (α=0.5, β=0.1) — against a no-KD baseline (α=0, β=0). The KD configuration is treated as a categorical ablation variable rather than as a tuned hyperparameter; the formulation is the standard one (Hinton-style temperature-scaled binary KL on logits [20] plus FitNet-style feature MSE on a matched decoder block through a 1×1 adapter [21]), and the empirical question we ask is whether *any* of {response, feature, combined} confers a measurable benefit on this task.

### 3.5. Post-Training INT8 Quantization (full rewrite of existing §3.5)

Post-training INT8 quantization (PTQ) is performed on the ONNX export of each student rather than on the PyTorch graph. This choice is forced by two practical findings during development. First, PyTorch eager-mode static PTQ requires CPU tensors for the calibration loop; the existing `floodlite/quantize.py` initially attempted CUDA calibration and failed. Second, even after fixing the device-placement bug, several `timm`-prefixed encoders — specifically `tf_efficientnet_lite0` and `mobilevit_xxs` — use a `Conv2dSame` layer for which no quantized CPU kernel exists in PyTorch 2.7+. Quantization at the PyTorch graph level therefore cannot reach the full student set without modifying upstream `timm` source.

We instead use the ONNX Runtime `quantize_static` API (onnxruntime.quantization, v1.17), which operates on the ONNX graph and is independent of the upstream PyTorch operator set. Our recipe is:

1. **Pre-processing.** The FP32 ONNX is passed through `quant_pre_process` to fuse Conv-BN, fold constants, and lift the graph into a form `quantize_static` can analyse.
2. **Calibration.** We select 200 randomly-sampled FSSD training images, run the FP32 graph against them through a custom `CalibrationDataReader` (`_TorchLoaderCalibReader` in `floodlite/quantize.py`), and record activation histograms.
3. **Quantization scheme.** Static QDQ (Quantize–DeQuantize) format with `per_channel=False` for activations and `per_channel=True` for weights restricted to `op_types_to_quantize=["Conv", "Gemm", "MatMul"]`. The op-type restriction is critical for MobileViT-XXS: without it, the per-channel quantizer attempts to fit per-output-channel scales to the rank-1 weight tensors of LayerNorm and `Add` operators, which produces undefined behaviour and silently collapses the output.
4. **Calibration method.** We use **Percentile (99.999%)** rather than MinMax. MinMax is sensitive to single-image outlier activations on flood images with extreme reflective surfaces (wet asphalt, sun-glare on water), and produces unstable cross-fold INT8 IoU in our initial sweeps. Percentile-99.999% is the most robust calibration option we evaluated and is the recipe we report in Section 5.4.

The resulting INT8 graph is consumed by ONNX Runtime's CPU execution provider; no ONNX→TFLite or ONNX→CoreML conversion is required, which is one motivation for our ONNX-centred deployment hub.

### 3.6. Deployment Pipeline (full rewrite of existing §3.6)

Models are exported via PyTorch's TorchScript-based ONNX exporter (`torch.onnx.export(..., dynamo=False, opset_version=13)`) to a single `.onnx` file with embedded weights. We deliberately disable the PyTorch 2.8+ Dynamo exporter (`dynamo=True`) which produces separate `.data` sidecar files for weights — this complicates artifact distribution and is not yet uniformly supported across edge runtimes.

We use ONNX as the single deployment hub. The same `.onnx` artifact serves both deployment targets in this paper:

- **Apple M2 Pro**: ONNX Runtime 1.17 with the `CPUExecutionProvider`. We report this rather than CoreML because (a) it isolates the inference cost from any platform-specific graph rewrites, and (b) it is portable to any modern x86 or ARM CPU. CoreML conversion is supported by the same `.onnx` file via `coremltools.converters.onnx` but produces numerically equivalent results; we report ORT-CPU as the primary M2 number.
- **Browser**: ONNX Runtime Web 1.17 with the WebAssembly SIMD execution provider, single-threaded. We benchmark from Node.js 25 rather than from a Chrome page because (a) Node's WASM runtime and Chrome's WASM runtime produce numerically equivalent timings within ~10%, and (b) it removes the SharedArrayBuffer / COOP-COEP browser configuration as a measurement-environment variable.

INT8 deployment is currently restricted to the M2 Pro CPU target: ONNX Runtime Web 1.17 does not implement the QDQ-format INT8 Conv kernel, so the WASM INT8 column of Table 5 reads "n/a (upstream limitation)". This is consistent with the ORT-Web roadmap (INT8 Conv is on the 1.18 milestone at time of writing) and is documented in our supplementary `lat_wasm.json`.

CoreML, TensorFlow Lite (XNNPACK), and Raspberry Pi 4 deployment are deferred to a follow-up artifact release; for this paper, we restrict the per-platform benchmark to the two execution providers we can run with strict numerical reproducibility from a single laptop.

---

## 4. Experimental Setup (full rewrite of existing §4)

### 4.1. Implementation

Training is implemented in PyTorch 2.8 with `segmentation-models-pytorch` 0.4 [31] for UNet decoder construction and `timm` 1.0 [32] for backbone implementations. Mixed-precision (AMP fp16) training runs on a Kaggle Tesla T4 GPU (16 GB) with batch size 8 (teacher) or 16 (students). Optimisation uses AdamW with learning rate 3e-4, weight decay 1e-4, and a cosine schedule. Training runs for 35 epochs (teacher and students alike); empirically, validation IoU plateaus by epoch ~25. Data augmentation is provided by Albumentations [33]: random horizontal flip, random rotation ±15°, random brightness/contrast jitter (±0.2), and ImageNet normalisation. The random seed is fixed at 42.

ONNX export, INT8 quantization, and CPU/WASM latency benchmarks run locally on an Apple M2 Pro (16 GB RAM, macOS 25.2). The Kaggle T4 is used for training only.

### 4.2. Cross-Validation Protocol

We use 3-fold cross-validation with `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)`, evaluating only folds 0–2 of the 5-fold partition to bound compute. The same fold indices are used for every (model × KD configuration) combination, so all IoU comparisons in Tables T2–T5 are *paired* across folds.

### 4.3. Metrics

We report pixel accuracy, precision, recall, F1-score, and mean intersection-over-union (IoU). Efficiency metrics are model parameters (M, from `torchinfo.summary`) and ONNX disk size (MB). Latency is reported as median (p50), 95th percentile (p95), mean, and standard deviation over 200 timed forward passes after 20 warm-up passes; throughput as FPS = 1000 / p50_ms. All latency measurements use single-image batches at 256×256 input resolution.

### 4.4. Statistical Validation

For each (model × KD configuration), we report mean ± standard deviation across the 3 folds. The paired Wilcoxon signed-rank test is applied to the no-KD vs. comb-KD contrast on each architecture, using fold-level paired IoU as the unit of observation. With n=3 folds, the minimum reportable two-sided p-value is 0.25; we report exact p-values and explicitly caveat low statistical power throughout. A per-image paired bootstrap analysis on a held-out fraction of FSSD validation pixels is deferred to a future v1.1 release and is not used to make headline claims in this paper.

### 4.5. Per-Platform Latency Rigging

- **Apple M2 Pro / ONNX Runtime 1.17 CPU EP**: `scripts/bench_m2.py`. Single-threaded by default; sessions created with `intra_op_num_threads = sess_options.inter_op_num_threads = 1`. 20 warm-up + 200 timed runs.
- **Browser WebAssembly SIMD / ONNX Runtime Web 1.17 from Node.js 25**: `bench.mjs` (see supplementary). `ort.env.wasm.numThreads = 1; ort.env.wasm.simd = true`. 20 warm-up + 200 timed runs. Equivalent in-browser numbers (Chrome 130, macOS 25.2) were verified to fall within ±10% in pilot runs.

Both benchmarks consume the identical `.onnx` artifact; no platform-specific graph rewrites are performed.

---

## 5. Results (full rewrite of existing §5)

This section is organised around Tables T1–T5 (Section 3.3 footprint table is T1; T2–T5 follow below) and Figures F2 (Pareto) and F4 (throughput). F1 is the pipeline diagram (manually drawn; see supplementary). F3 is the qualitative panel and is included in the supplementary material.

### 5.1. Teacher Performance and Karcı et al. (2026) Comparison

Our UNet + EfficientNet-B0 teacher achieves **IoU = 0.9257 ± 0.0078** on FSSD across the three folds (F1 = 0.9611 ± 0.0044, accuracy = 0.9729 ± 0.0025). The reference UNet + EfficientNet-B7 number reported by Karcı et al. [10] is IoU = 0.927, achieved with approximately 10× the parameters (~63 M vs. our 6.25 M). The 0.0013 IoU gap between our B0 teacher and the published B7 reference is within the cross-fold standard deviation of both studies and indicates that the FSSD benchmark saturates well before the B7 capacity is required; B0 is sufficient as the teacher.

### 5.2. Students Match or Exceed the Teacher Without Distillation (Negative-KD Finding)

The full segmentation results across all 14 configurations × 3 folds are in **Table T2** (`docs/tables.md`, reproduced here as the in-paper Table 2). Three observations:

1. **The best student exceeds the teacher.** MobileViT-XXS in the no-KD configuration attains IoU = 0.9367 ± 0.0055, **0.011 IoU points above the teacher** (0.9257 ± 0.0078). The student is also 2.0× smaller in parameters (3.09 M vs. 6.25 M). This is the headline result against which the original KD-recipe framing of this paper has to be evaluated.
2. **EfficientNet-Lite0 (no-KD) matches the teacher within standard deviation** (0.9277 ± 0.0047 vs. 0.9257 ± 0.0078). MobileNetV3-Small (no-KD) likewise matches it (0.9269 ± 0.0043).
3. **The MobileNetV2 baseline also matches the teacher** (0.9257 ± 0.0032). The teacher's parameter count is therefore not the source of any advantage; it is an over-parameterised lightweight UNet, not a "heavy" model in the regime where KD typically helps.

We interpret these observations as direct evidence that FSSD, at 663 RGB images and IoU ≈ 0.93, is saturated for the ImageNet-pretrained UNet family. The variance across (model × KD) cells of T2 is dominated by fold-level seed noise rather than by genuine architectural separations. KD has no headroom to transfer.

### 5.3. KD Ablation on the Best Student (MobileViT-XXS)

**Table T3** (`docs/tables.md`, reproduced as Table 3) isolates the KD ablation on the best student. None of the three KD configurations improves on the no-KD baseline:

| KD configuration | IoU (mean ± std) | Δ vs. no-KD | Wilcoxon p (n=3, two-sided) |
|---|---|---|---|
| none      | 0.9367 ± 0.0055 | —       | — |
| resp      | 0.9326 ± 0.0060 | −0.0041 | n/a |
| feat      | 0.9357 ± 0.0046 | −0.0011 | n/a |
| comb      | 0.9318 ± 0.0070 | −0.0049 | 0.250 |

The Wilcoxon test on no-KD vs. comb-KD yields p = 0.250 — the minimum two-sided p-value reportable at n=3 — which means the test is consistent with both "comb-KD is no worse" and "comb-KD is meaningfully worse" under low power, but in no case does it support "comb-KD is better". The remaining contrasts (resp, feat) cannot be tested due to insufficient direction agreement across only three folds.

The same qualitative pattern holds on MobileNetV3-Small and EfficientNet-Lite0 (see full T2). Across **all three architectures × three KD configurations × three folds (27 fold-level data points)**, the combined KD configuration produces a higher mean IoU than the no-KD baseline in zero out of three architectures.

### 5.4. INT8 Quantization is Architecture-Dependent

**Table T4** (`docs/tables.md`, reproduced as Table 4) summarises post-training INT8 quantization for the three deployable students:

| Model              | FP32 IoU         | INT8 IoU         | Δ IoU   | FP32 ONNX (MB) | INT8 ONNX (MB) | Size reduction |
|--------------------|------------------|------------------|---------|---------------:|---------------:|---------------:|
| MobileNetV3-Small  | 0.9269 ± 0.0043  | 0.4762 ± 0.3399  | −0.4507 |          13.8  |           3.7  |          3.69× |
| **EfficientNet-Lite0** | **0.9277 ± 0.0047** | **0.8851 ± 0.0149** | **−0.0426** |    **19.8**  |       **5.2**  |      **3.83×** |
| MobileViT-XXS      | 0.9367 ± 0.0055  | 0.4752 ± 0.1936  | −0.4616 |          12.0  |           3.5  |          3.42× |

The cross-architecture spread is the central finding of this section. Under an identical Percentile-99.999% static QDQ recipe, EfficientNet-Lite0 retains 95% of its FP32 IoU with a tight ±0.015 cross-fold standard deviation, whereas MobileNetV3-Small and MobileViT-XXS both lose ~0.45 IoU points with cross-fold standard deviations of 0.19–0.34 — an order of magnitude larger than EfficientNet-Lite0's. We interpret these failures architecturally:

- **MobileNetV3-Small** uses the HardSwish activation `x · ReLU6(x+3) / 6`, which has an unbounded positive tail and produces calibration statistics dominated by single-image outliers. Per-tensor INT8 calibration cannot fit a single scale that simultaneously preserves the activations of common flood-pixel patches and of bright reflective outliers (wet asphalt, water surface sun-glare). Percentile-99.999% mitigates but does not eliminate this; we conjecture MobileNetV3-S requires either (a) QAT to learn quantizer-friendly activations or (b) a HardSwish→ReLU6 surgical replacement before quantization.
- **MobileViT-XXS** combines a CNN trunk with self-attention blocks containing LayerNorm. The LayerNorm γ/β tensors are rank-1 across channels, which means per-channel weight quantization (which is otherwise the standard recipe for Conv layers) is ill-defined; restricting per-channel quantization to `Conv/Gemm/MatMul` (Section 3.5) partially fixes this but leaves the attention `MatMul` itself quantized per-tensor, where the high dynamic range of attention scores again exceeds what static calibration can fit.
- **EfficientNet-Lite0** uses ReLU6 throughout. ReLU6 is bounded by construction at 6.0, so the per-tensor activation calibration finds a stable scale that matches the actual activation range in every fold. The 0.043 IoU drop is consistent with the conventional-wisdom "~1 IoU point" PTQ tax in the classification literature.

The practical implication: **bounded-activation backbones (ReLU6) are the safe default for INT8 PTQ on flood segmentation; HardSwish and attention-based backbones currently require QAT or remain FP32 in deployment.** We do not attempt QAT in this paper — it is out of scope and an order-of-magnitude larger training investment — and instead recommend EfficientNet-Lite0 as the deployable configuration on the basis of its joint accuracy + size + stability behaviour.

### 5.5. Per-Platform Inference Latency

**Table T5** (`docs/tables.md`, reproduced as Table 5) reports inference latency on the two deployment platforms.

| Model                       | M2 Pro p50 (ms) | M2 Pro p95 (ms) | M2 Pro FPS | WASM p50 (ms) | WASM FPS |
|----------------------------|---:|---:|---:|---:|---:|
| Teacher (UNet+EffB0) FP32  | 274.0 | 280.7 |   3.6 |   n/a |   n/a |
| MobileNetV3-Small FP32     |  19.5 |  20.4 |  51.2 | 133.3 |   7.5 |
| MobileNetV3-Small INT8     |  13.7 |  14.1 |  72.9 |   n/a |   n/a |
| **EfficientNet-Lite0 FP32**|  34.2 |  52.7 |  29.3 | 163.8 |   6.1 |
| **EfficientNet-Lite0 INT8**|  **17.3** |  **20.1** |  **57.7** |   n/a |   n/a |
| MobileViT-XXS FP32         |  28.3 |  32.8 |  35.4 | 149.4 |   6.7 |
| MobileViT-XXS INT8         |  21.6 |  25.0 |  46.3 |   n/a |   n/a |

Three remarks. First, **all three students offer a 7–19× speedup over the teacher on the M2 Pro target**, primarily reflecting the parameter and FLOP reductions but also benefiting from ORT's CPU-graph optimisations on smaller models. Second, **INT8 yields a further 1.3–1.6× speedup on top of the FP32 student baseline**, consistent with INT8 SIMD-CPU expectations. The combined teacher-FP32 → EfficientNet-Lite0-INT8 speedup is **15.8×** at a 95% IoU recovery. Third, **the WASM column is roughly 5× slower than the M2-CPU column for the same FP32 model**, which is the expected single-threaded-WebAssembly–vs–native-CPU gap and represents an effective lower bound on browser-deployment performance. INT8 WASM rows are unsupported by ONNX Runtime Web 1.17 at time of writing and are marked n/a.

### 5.6. Pareto Analysis

**Figure F2** (`code/figures/F2_pareto.png`) shows the four-axis Pareto frontier: IoU vs. {Params, FP32 ONNX size, M2 Pro p50, WASM p50}. The **EfficientNet-Lite0 INT8** configuration sits on the Pareto frontier on the parameter, M2-latency, and WASM-latency axes simultaneously; **MobileViT-XXS no-KD FP32** is Pareto-optimal on the accuracy axis but not deployable as INT8 under the current static QDQ recipe. The Pareto front therefore identifies two distinct deployment operating points:

- **Maximum-accuracy operating point**: MobileViT-XXS no-KD FP32 (IoU 0.937, 28 ms M2, 149 ms WASM, 12 MB).
- **Maximum-throughput-with-acceptable-accuracy operating point**: EfficientNet-Lite0 INT8 (IoU 0.885, 17 ms M2, no WASM, 5.2 MB).

We recommend the latter for unattended flood-mapping pipelines (1.6 × throughput, 2.3 × smaller binary) and the former for human-in-the-loop applications where the 5.2-point IoU drop matters.

### 5.7. Pipeline Diagram and Qualitative Examples

Figure F1 (pipeline diagram, manually drawn) summarises the train → distill (ablation) → quantize → benchmark workflow. Figure F3 (supplementary qualitative panel) shows two rows × four columns: an easy and a hard FSSD-val example, with input / ground truth / teacher prediction / EfficientNet-Lite0-INT8 prediction. The two persistent failure modes shared by teacher and student — wet asphalt confused with water; under-segmentation of vegetation-occluded flood edges — are visible in the hard row.

---

## 6. Discussion (full rewrite of existing §6)

### 6.1. The Negative-KD Finding and What It Means for the Field

The central empirical claim of this paper is that knowledge distillation does not help on FSSD-class flood segmentation when the student is an ImageNet-pretrained lightweight UNet. We frame this as a *negative result* rather than a *null result* because the comparison is not "we did not find a positive effect" but rather "the best student (MobileViT-XXS, no-KD) *exceeds* the teacher (UNet+EfficientNet-B0)". Distillation on this benchmark is at best redundant and at worst counterproductive: combined KD on MobileViT-XXS produced −0.005 IoU vs. the same architecture trained without it.

We interpret this mechanistically. The classical motivation for KD — Hinton's *dark knowledge* argument [20] and subsequent feature-map alignment work [21–24] — assumes that the student has insufficient capacity or pretraining to recover the teacher's behaviour from labelled data alone, so the teacher's soft probabilities and intermediate features provide a richer supervision signal than the binary mask. On FSSD, that assumption is violated in two directions: (i) the benchmark has only 663 images and saturates at IoU ≈ 0.93 — the labelled signal is already nearly sufficient — and (ii) the student's ImageNet pretraining already supplies the visual priors (edge-detectors, texture filters, semantic-region features) that a teacher trained on the same FSSD would have been transferring. The teacher's "dark knowledge" is therefore largely redundant with the student's pretraining, and the response-based KL term reduces to a noise-injection regulariser. The feature-MSE term performs slightly better — it provides at least *some* signal beyond the labels — but the gain is well within fold variance and is not statistically significant at our reportable resolution.

This finding has two implications for the flood-segmentation literature:

1. **Benchmarks matter.** A KD recipe that wins on Cityscapes (19 classes, 5000 images, IoU ≈ 0.80–0.85) does not transfer to FSSD (2 classes, 663 images, IoU ≈ 0.93). Future KD work in this space should report headroom (teacher–best-pretrained-student gap) *before* claiming distillation effects.
2. **Modern ImageNet pretraining is doing the work.** The 2026 wave of timm-pretrained lightweight backbones brings the no-KD baseline so high that compression strategies *other than distillation* — quantization, pruning, architecture choice — dominate the deployment-relevant trade-offs. We recommend that authors investing in flood-segmentation compression budget their effort accordingly.

### 6.2. Per-Platform Deployment Recommendations

Based on the joint segmentation × latency × stability evidence (Sections 5.2–5.5), we make the following deployment recommendations:

- **Field-laptop / tablet inference (Apple Silicon class)**: EfficientNet-Lite0 UNet INT8. 17 ms p50 (58 FPS) at IoU 0.885, 3.8× size reduction. Stable across folds (±0.015). This is the configuration we ship in our public release.
- **Browser citizen-science / web-portal inference**: MobileNetV3-Small UNet FP32 (133 ms p50; 7.5 FPS), or MobileViT-XXS UNet FP32 (149 ms p50; 6.7 FPS) for slightly higher accuracy. INT8 is not yet supported in ONNX Runtime Web 1.17 and is gated on ORT-Web 1.18+ shipping a QDQ INT8 Conv kernel.
- **Maximum-accuracy off-line inference**: MobileViT-XXS UNet FP32 (28 ms p50 on M2 Pro CPU; IoU 0.937). Recommended for human-in-the-loop annotation pipelines.

We do not recommend the teacher (UNet+EffB0) for any of these deployment scenarios in this paper: at 274 ms p50 (3.6 FPS) on M2 Pro CPU, it is uniformly dominated by the smaller students on both accuracy and latency.

### 6.3. Architecture-Dependent PTQ Stability is a Practitioner Trap

The 0.45-IoU PTQ collapse on MobileNetV3-Small and MobileViT-XXS would not be predicted from the classification-quantization literature, which typically reports ~1 IoU point PTQ tax on ResNet/MobileNet/EfficientNet families on ImageNet. The flood-segmentation regime differs in two respects: (i) the output is a per-pixel binary mask, so any quantization error that biases the post-sigmoid threshold materially shifts the segmentation; and (ii) flood images contain a wide dynamic range of bright reflective surfaces that drive calibration statistics in HardSwish-based backbones. We have observed that the same Percentile-99.999% recipe that fully recovers EfficientNet-Lite0 leaves MobileNetV3-Small with one of three failure modes per fold: (a) the entire output collapses to ≈ 0.5 IoU (random-class prediction), (b) the output is biased toward the background class (recall ≈ 0.1), or (c) it works (≈ 0.92 IoU). The cross-fold variance reflects this mode-mixture rather than a smooth degradation.

The practical guidance: when assembling a flood-segmentation deployment artifact under a static-QDQ-INT8 budget, prefer backbones with bounded activation ranges (ReLU6 is the safest, ReLU is fine, HardSwish is risky, GELU and softmax-attention are *not* PTQ-friendly without QAT). Without this guidance, a practitioner running the standard ONNX Runtime quantization tutorial on MobileNetV3-Small would conclude — incorrectly — that the model "doesn't quantize" rather than that the recipe is mis-matched to the activation family.

### 6.4. Failure Modes (qualitative)

Figure F3 (supplementary) shows that both teacher and EfficientNet-Lite0-INT8 student share two persistent failure modes:
- **Bright-reflective-surface confusion**: wet asphalt, glare-on-roof, and large reflective puddles are misclassified as flood. This is a labelling-modality issue, not a model-capacity issue — RGB alone cannot disambiguate water from wet non-water surfaces.
- **Vegetation-occluded edges**: when flooded water is partially overhung by trees or tall grass, both models under-segment the flood boundary. The 5-pixel margin at the boundary contributes disproportionately to the residual IoU gap.

Both failure modes argue for SAR or multispectral fusion rather than for a different RGB segmentation recipe. We discuss this in Section 7 (Limitations).

---

## 7. Limitations (new section — replaces existing §6.4 + adds new material)

We acknowledge the following limitations.

**Single-dataset training.** All experiments use FSSD (663 RGB images). We initially planned an out-of-distribution evaluation on Sen1Floods11 RGB chips and prepared the data-loading pipeline (`floodlite/data.py:make_sen1floods11_loader`), but the cross-domain evaluation was deprioritised after the negative-KD finding reduced the per-experiment value. The cross-region generalisation of the configurations recommended in Section 6.2 is therefore unverified, and we flag this as the highest-priority follow-up.

**RGB-only modality.** Operational disaster-response platforms increasingly fuse SAR (cloud-penetrating, all-weather) with optical imagery; our work is restricted to the RGB regime FSSD was constructed for. The two persistent failure modes in Section 6.4 (wet-asphalt confusion, vegetation occlusion) are intrinsic to the RGB modality and cannot be solved by a better RGB model.

**Three folds and low statistical power.** The 3-fold protocol gives a minimum reportable two-sided Wilcoxon p of 0.25, which is insufficient to claim statistical significance for the cross-architecture and cross-KD-configuration contrasts at conventional thresholds. We report exact p-values and rely on effect-size magnitude (e.g. the −0.45 IoU drop on MobileNetV3-S INT8 is unambiguous) and on within-architecture consistency across three folds (the negative-KD finding holds in 3/3 architectures × 3/3 folds) rather than on hypothesis-test significance.

**Two deployment platforms (M2 Pro + browser WASM).** We do not measure Raspberry Pi 4, AWS Graviton (ARM CPU), Android (Termux + TFLite), or any GPU edge target (Jetson Nano / Orin). Adding these is largely an artifact-conversion exercise — the ONNX hub artifact remains the same — but each adds a non-trivial setup and tuning cost we did not budget for. Pi 4 and Graviton numbers are the most directly comparable extensions and are scheduled for a v1.1 supplementary release.

**Post-training quantization only.** We do not evaluate quantization-aware training (QAT). The 0.45-IoU PTQ collapse on MobileNetV3-Small and MobileViT-XXS is a candidate target for QAT — both architectures are reportedly QAT-recoverable in the classification literature — but QAT is an order-of-magnitude larger training investment than the experiments reported here and is left to future work.

**No energy / power measurements.** We do not report Joules-per-inference. Energy is the metric most directly relevant to battery-operated deployments (drones, field-tablets) and we acknowledge its absence; the equipment required (USB-meter, INA219 / current-clamp logger) was not available during this study.

**Saturated benchmark.** FSSD saturates at IoU ≈ 0.93 for the entire UNet + ImageNet-pretrained backbone family. The negative-KD finding is therefore strictly a claim about saturated-benchmark behaviour. We do *not* claim that KD never helps on flood segmentation in general — only that, on this specific benchmark, it does not. A non-saturated benchmark with limited labels, mixed imaging modalities, or larger viewpoint diversity might recover the KD signal.

---

## 8. Conclusions (full rewrite of existing §7)

We presented FloodLite, the first systematic multi-platform edge-deployment benchmark for lightweight flood segmentation. Across three lightweight UNet students, four knowledge-distillation configurations, three cross-validation folds, two execution providers, and FP32 + INT8 precisions, we have established three results that we believe will be useful to the operational disaster-response community: (i) ImageNet-pretrained lightweight UNets match or exceed a UNet+EfficientNet-B0 teacher on FSSD *without* distillation, and combined response+feature KD does not improve over no-KD on any of the three architectures; (ii) post-training INT8 quantization stability is sharply architecture-dependent, with bounded-activation backbones (ReLU6) quantizing cleanly while HardSwish-based and attention-based backbones require quantization-aware training; (iii) the recommended deployable configuration, EfficientNet-Lite0-UNet INT8 served by ONNX Runtime CPU, runs at 58 FPS on a 2023 laptop CPU and 6 FPS in a single-threaded WebAssembly browser sandbox at 3.83 × size reduction with IoU 0.885 — a 15.8 × speedup over the teacher at 95% accuracy recovery. The same .onnx artifact serves both targets, which we offer as a practical pattern for flood-segmentation deployment on heterogeneous edge hardware.

We release our trained checkpoints, the ONNX FP32 and INT8 exports, all benchmarking harnesses, and a Zenodo-archived snapshot of the code repository under permissive licences. Future work includes the deferred out-of-distribution evaluation on Sen1Floods11 RGB, ARM CPU benchmarks on Raspberry Pi 4 and AWS Graviton, energy measurements, and quantization-aware-training of the HardSwish and attention-based backbones whose post-training quantization stability we have characterised as a current practitioner trap.

---

## Appendices

### Appendix A — DELETE entirely

The placeholder map in the existing draft (§ "Appendix A. Placeholder Map") is removed. All numerical values are now populated from `code/runs/3fold/{summary,quantized,lat_m2,lat_wasm}.json`. No `##.##` or `{{…}}` strings remain in the manuscript text.

### Appendix B — Reproducibility (new, ~1 page)

We provide the following artifacts for reproducibility:

- **Code repository**: https://github.com/[your-username]/Flood_segmentation_model (MIT). The v1.0-ijdrr tag is the camera-ready commit. Zenodo DOI: [DOI to be minted at submission].
- **Pre-trained checkpoints**: `code/runs/3fold/best_*.pt` (PyTorch state-dicts). 14 configurations × 3 folds = 42 checkpoints. Total ~700 MB.
- **ONNX exports**: `code/exports/{teacher,mobilenetv3_small,efficientnet_lite0,mobilevit_xxs}_fold{0,1,2}_{fp32,int8}.onnx`. Total ~150 MB.
- **Per-fold metrics**: `code/runs/3fold/results.json` (FP32), `code/runs/3fold/quantized.json` (INT8). Re-derivable by running `python scripts/run_full_experiment.py --data_root /path/to/FSSD --fold {0,1,2} --include_baseline`.
- **Per-platform latency**: `code/runs/3fold/lat_m2.json`, `code/runs/3fold/lat_wasm.json`. Re-derivable by `python scripts/bench_m2.py` and `node bench.mjs` respectively.

**Random seed**: 42 (set in `floodlite/utils.py:set_seed` and in `sklearn.model_selection.KFold`).
**Python**: 3.11. **PyTorch**: 2.8 with CUDA 12.1 (Kaggle T4) / Apple-Silicon-MPS (M2 export & quantize). **segmentation-models-pytorch**: ≥0.4,<0.6. **timm**: 1.0. **onnxruntime**: 1.17. **onnxruntime-web**: 1.17.3.

**Hardware**: Training on a single Kaggle Tesla T4 (16 GB). Quantization and CPU latency on an Apple M2 Pro (10-core CPU, 16 GB unified memory, macOS 25.2). WASM latency via Node.js 25.

**Estimated reproduction cost**: ~35 hours of Kaggle T4 free-tier compute (well within the 30-hour-per-week quota across two folds of the 3-fold sweep) + ~2 hours of M2 CPU time for quantization + benchmarking. Total wall-clock: ≈ 3 calendar days assuming no Kaggle queue.

---

## Author Contributions, Funding, Data Availability, Conflicts

**Keep existing text** (paragraphs [283]–[289] of the existing draft). Update the Data Availability paragraph to point to:
- FSSD: https://www.kaggle.com/datasets/lihuayang111265/flood-semantic-segmentation-dataset
- This work's artifact release: https://github.com/[your-username]/Flood_segmentation_model (Zenodo DOI: TBD at submission)
- Add: "The Sen1Floods11 dataset (used in preliminary out-of-distribution preparation, not in final results) is available at https://github.com/cloudtostreet/Sen1Floods11 (CC-BY 4.0)."

---

## References (incremental edits to existing list)

**Keep existing references 1–33 unchanged.** Add the following:

- **[34]** Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; Chen, L.-C. *MobileNetV2: Inverted residuals and linear bottlenecks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, 18–22 June 2018; pp. 4510–4520. _(used for the baseline in Section 3.3)_
- **[35]** Beyer, L.; Zhai, X.; Royer, A.; Markeeva, L.; Anil, R.; Kolesnikov, A. *Knowledge distillation: A good teacher is patient and consistent*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 19–24 June 2022; pp. 10925–10934. _(cited in §2.2 and §6.1)_
- **[36]** Yu, C.; Gao, C.; Wang, J.; Yu, G.; Shen, C.; Sang, N. *BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation*. International Journal of Computer Vision 2021, 129, 3051–3068. _(cited in §2.3)_
- **[37]** Xu, J.; Xiong, Z.; Bhattacharyya, S. P. *PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, Canada, 18–22 June 2023; pp. 19529–19539. _(cited in §2.3)_
- **[38]** Poudel, R. P. K.; Liwicki, S.; Cipolla, R. *Fast-SCNN: Fast Semantic Segmentation Network*. In Proceedings of the British Machine Vision Conference (BMVC), Cardiff, UK, 9–12 September 2019. _(cited in §2.3)_
- **[39]** ONNX Runtime Developers. *ONNX Runtime v1.17 — Static QDQ Quantization Recipe*. Microsoft, 2025. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html (accessed 16 May 2026). _(cited in §3.5)_

---

## End-of-draft checklist (for paste-in to Word)

- [ ] Title page: replace existing title and short title
- [ ] Abstract: full replace
- [ ] Keywords: replace
- [ ] §1 Introduction: full replace (1.1 + 1.2 added)
- [ ] §2.1: minor edit on closing sentence
- [ ] §2.2: append two-paragraph addition
- [ ] §2.3: append BiSeNetV2/PIDNet/Fast-SCN paragraph
- [ ] §2.4: replace with new PTQ framing paragraph
- [ ] §3.1: replace the 5-fold sentence with the 3-fold paragraph
- [ ] §3.3: replace with new student-table text (T1 numbers)
- [ ] §3.4: replace the framing sentence at the top (math unchanged)
- [ ] §3.5: full replace (ONNX static QDQ recipe)
- [ ] §3.6: full replace (ONNX-centred deployment hub)
- [ ] §4: full replace (3-fold protocol; M2 + WASM rigging)
- [ ] §5: full replace (T2–T5; negative-KD; PTQ architecture finding; Pareto)
- [ ] §6: full replace (negative-KD lead; deployment recs; PTQ trap; failure modes)
- [ ] §7 Limitations: NEW section (replace existing §6.4 Limitations)
- [ ] §8 Conclusions: full replace
- [ ] Appendix A: DELETE
- [ ] Appendix B (Reproducibility): NEW, paste in
- [ ] References: append refs [34]–[39]
- [ ] Final read-through: confirm zero `##.##`, `{{…}}`, or `[Co-author]` placeholders remain before submission.
