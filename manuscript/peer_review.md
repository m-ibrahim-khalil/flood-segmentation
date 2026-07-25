# Peer Review — *FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation*

**Target venue:** *International Journal of Disaster Risk Reduction* (Elsevier, IJDRR)
**Manuscript type:** Original research (methods / benchmark)
**Review basis:** Full LaTeX source (`manuscript/`) cross-checked line-by-line against the source-of-truth data in `code/runs/3fold/{summary,quantized,stats,lat_m2,lat_wasm}.json`. Where a claim could be re-measured, it was (see M1).
**Reviewer note:** This is an internal pre-submission review generated to stress-test the manuscript before IJDRR submission. All numbers below were verified against the repository data on 2026-07-26.

---

## Summary statement

FloodLite reframes flood segmentation around the question the literature has ignored — *can these models actually run on the hardware responders deploy?* — and delivers three useful results: a multi-platform latency/size benchmark, an honestly-reported **negative** knowledge-distillation (KD) finding, and an architecture-dependent INT8 post-training-quantization (PTQ) characterization. The framing is timely, the writing clear, the pivot honest, and the reproducibility posture strong.

**Recommendation: Major Revision.** The paper's central *deployment* claim rests on a latency comparison that is confounded by runtime — the teacher is timed in PyTorch eager mode while every student is timed in ONNX Runtime, yet Table 5 labels all rows "ORT-CPU." I re-benchmarked under a common runtime: the true FP32 architecture speedup is **1.1–1.8×, not the claimed 7–19×**. Separately, the Abstract and Conclusions contain **two factual self-contradictions** refuted by the paper's own tables, and one mechanistic narrative (§6.3) does not match the reported per-fold numbers. None of these require new modeling — they need a re-benchmark (script provided), three wording fixes, and one reconciliation — but they are load-bearing, so they block acceptance as-is.

**Key strengths**
- A real gap, correctly identified: no prior FSSD-class flood-seg paper reports latency/size/PTQ; this one does.
- The negative-KD finding is scoped carefully to "saturated benchmark" and reported against the incentive to bury it.
- Reproducibility: fixed seeds, per-fold JSON, ONNX artifacts, Zenodo release, single-artifact deployment story.

**Key weaknesses**
- **M1:** teacher-vs-student latency is an apples-to-oranges runtime comparison; the headline speedups are largely a PyTorch-vs-ORT artifact (measured below).
- **M2 / M3:** the Abstract and Conclusions claim (a) the INT8 config runs in-browser at 6 FPS and (b) "no measurable accuracy loss versus the teacher" — both contradicted by the paper's own data.
- **M4:** the §6.3 per-fold PTQ failure-mode story doesn't match the actual fold values.

---

## Major comments

### M1. The teacher latency is measured under a different runtime than the students; Table 5 mislabels it; the central speedup claim is confounded. *(Confirmed by re-measurement.)*

`lat_m2.json` contains the teacher only as `teacher_pt_cpu` (**PyTorch eager CPU**, 274 ms). Every student latency in Table 5 comes from the `_ort_cpu` keys (**ONNX Runtime CPU**, 17–34 ms). Table 5's column header reads "Apple M2 Pro (ORT-CPU)" for **all** rows including the teacher — factually wrong for the teacher row.

The runtime change alone is worth ~6–8× on these models. Because the teacher was never exported to ONNX, `bench_m2.py` silently skipped its ORT-CPU row. I exported the teacher with the same `export_onnx` path used for the students and timed teacher + students with the same `benchmark_onnxruntime` harness (`scripts/bench_teacher_ort.py`; output `runs/3fold/lat_teacher_ort_check.json`):

| | PyTorch-CPU p50 | ORT-CPU p50 | runtime-only speedup |
|---|---|---|---|
| **Teacher (UNet+EffB0)** | 293 ms | **37.8 ms** | **7.8×** (no architecture change) |

Honest **same-runtime** FP32 speedups (vs teacher-ORT 37.8 ms):

| Student (FP32) | ORT-CPU p50 | vs teacher-ORT (honest) | vs teacher-PT (as printed in Table 5) |
|---|---|---|---|
| MobileNetV3-Small | 20.5 ms | **1.8×** | 14.3× |
| EfficientNet-Lite0 | 34.7 ms | **1.1×** | 8.5× |
| MobileViT-XXS | 23.4 ms | **1.6×** | 12.5× |

So "all three students offer a 7–19× speedup over the teacher" (§5.4) and "15.8× speedup over the teacher" (Abstract, §8) are mostly the PyTorch→ORT runtime effect, not "the parameter and FLOP reductions" (§5.4). Under the same runtime, the recommended EfficientNet-Lite0 FP32 (34.7 ms) is only **1.1×** faster than the teacher (37.8 ms). My re-benched student numbers (20.5/34.7/23.4 ms) closely track the paper's ORT-1.17 values (19.5/34.2/28.3 ms), so the teacher-ORT ≈ 38 ms estimate will hold on ORT 1.17.

**Required fix:** re-bench the teacher on the canonical **onnxruntime 1.17** (script provided), add a teacher ORT-CPU row, recompute every teacher-relative speedup (Abstract, §5.4, §6.2, §8), and regenerate F2 (Pareto latency axes) and F4. Keep an optional PyTorch-CPU-vs-PyTorch-CPU column if you also want the architecture-only story separable from runtime. The INT8 story still holds (EfficientNet-Lite0 INT8 17.3 ms is ~2.2× vs teacher-ORT), but must be stated on a same-runtime basis.

**[RESOLVED — 2026-07-26, ORT 1.17.]** Teacher re-benched under ONNX Runtime 1.17 (`runs/3fold/lat_m2_ort117.json`): teacher ORT-CPU = **37.6 ms p50 / 26.6 FPS** (PyTorch eager was 274 ms → 7.3× runtime-only). Same-runtime FP32 student speedups: MobileNetV3-Small 1.72×, EfficientNet-Lite0 1.04×, MobileViT-XXS 1.56× (i.e. **1.0–1.7×**, not 7–19×); EfficientNet-Lite0 INT8 vs teacher-ORT = **2.2×**. Applied to Table 5 (teacher row + footnote), §5.4, §6.2, §8, and the `\teacherMTwoFPS` macro (3.6→26.6). **Reproducibility note discovered:** the INT8 QDQ ONNX files fail to load under ORT 1.17 (`ai.onnx.ml` opset 5 > 1.17's supported opset 4) — they require a newer ORT; the paper should state the minimum ORT version for the INT8 artifacts. **F2 regenerated** (`code/scripts/make_figures.py`): teacher latency point moved 274→37.6 ms; F4 needed no change (students-only, teacher not plotted). While regenerating I also fixed two **pre-existing** F2 defects: the Params axis used placeholder counts (MNV3-S 1.5M, EffLite0 4.4M) contradicting Table 1 — now the real 3.59/5.61/3.09/6.25 M; and the second panel showed fabricated FLOPs while its caption promised "FP32 ONNX size" — now shows the real ONNX sizes (12.0/13.8/19.8/22.3 MB) matching the caption. MobileViT-XXS is now correctly Pareto-optimal on params, size, and accuracy.

### M2. The Abstract and Conclusions state the INT8 configuration runs in the browser at 6 FPS — it does not run in the browser at all.

`lat_wasm.json`: `efficientnet_lite0_int8: {"error": ...}` — INT8 **fails at session-load** under ORT-Web 1.17 (no QDQ INT8 Conv), exactly as you state in §3.6 and Table 5 ("n/a"). The "6 FPS" figure is the **FP32** model (`efficientnet_lite0_fp32`: 163.8 ms → 6.10 FPS). Yet the Abstract and §8 say *"EfficientNet-Lite0-UNet INT8 … runs at 58 FPS on a 2023 laptop CPU and 6 FPS in a single-threaded WebAssembly browser sandbox"* — fusing an M2-INT8 result with a WASM-**FP32** result and attributing both to one INT8 config on both platforms. This contradicts §3.6, Table 5, and the data.

**Required fix:** state that the deployable INT8 config is **M2/CPU-only**; the browser path is the **FP32** model (6.1 FPS). The macro `\deployableWASMFPS = 6` is attached to the wrong precision in the prose.

### M3. "No measurable accuracy loss versus the teacher" (Abstract) is contradicted by the paper's own numbers.

EfficientNet-Lite0 **INT8** IoU = 0.885 (verified per-fold [0.903, 0.885, 0.867], mean 0.8851 ± 0.0149) vs teacher 0.9257 → **−0.041 IoU, a 4.4% relative drop.** §8 calls this "95% accuracy recovery" and §5.6 a "5.2-point IoU drop [that] matters." The Abstract contradicts both. ("No measurable loss" is true only for the *FP32* student vs teacher — not the INT8 config the sentence describes.)

**Required fix:** replace with the honest framing (≈4 IoU points / 95% recovery).

### M4. The §6.3 failure-mode narrative for MobileNetV3-Small INT8 does not match the data.

§6.3 says each fold lands in "(a) collapse to ≈0.5 IoU, (b) background-biased recall ≈0.1, or (c) it works (≈0.92 IoU)." The actual per-fold INT8 IoU (`quantized.json`) is **[0.772, 0.0001, 0.657]** — no ≈0.92 "works" fold and no ≈0.5 fold; one near-total collapse (0.0001) and two partial degradations. The narrative reads as carried over from an earlier calibration sweep. Relatedly, reporting **mean ± std = 0.476 ± 0.340** (Table 4) for a variable taking {0.77, 0.0001, 0.66} is misleading — the mean describes no fold.

**Required fix:** rewrite §6.3 to match the true per-fold values, and for the collapsing configs report per-fold IoU (or min/median/max) rather than mean ± std. The headline PTQ conclusion (bounded-activation backbones quantize; HardSwish/attention don't) survives.

---

## Moderate comments

- **Mod1 — Statistical power + fold non-independence. [RESOLVED — bootstrap run 2026-07-26.]** n = 3, folds {0,1,2} of a 5-fold split → training sets overlap ~50% pairwise → fold-IoUs positively correlated, cross-fold std understates variance, paired-Wilcoxon floor p = 0.25. The per-image paired bootstrap (previously skipped in `stats.json`) has now been run over the pooled held-out set (`scripts/stats_bootstrap.py` → `runs/3fold/stats_bootstrap.json`; n = 399 images, 10,000 replicates). Result, folded into §4.4/§5.2/§1/§7: across all **9** (architecture × KD) contrasts **none** significantly improves over no-KD; in **3/9** KD is significantly *worse* (response-KD on MobileNetV3-Small Δ=+0.0047 [+0.0008,+0.0091] and MobileViT-XXS Δ=+0.0065 [+0.0029,+0.0103]; combined-KD on MobileViT-XXS Δ=+0.0076 [+0.0035,+0.0118]); the other 6 are indistinguishable. The "best student beats teacher" headline is now significant (MobileViT-XXS no-KD − teacher = +0.015 per image, 95% CI [+0.010,+0.020], p<0.001). This replaces the p≥0.25 floor as the basis for the significance claims. Residual (unaddressed): fold-level generalization — the bootstrap is over images, not independent dataset draws; the 5→3 fold reduction is still circular; consider restoring all 5 folds.
- **Mod2 — "1.3–1.6× INT8 speedup" (§5.4) is wrong.** EfficientNet-Lite0 FP32 34.2 → INT8 17.3 ms = **1.98×**, outside the stated range; true range 1.3–2.0×.
- **Mod3 — "7–19× over the teacher" is loose and contingent on M1.** Recompute after M1.
- **Mod4 — KD mechanism is inferred, not demonstrated. [SETUP DONE — 2026-07-26; ready to run.]** Show KD *reappearing* when headroom exists: retrain at 10–25% of FSSD train (same pipeline) and/or run the deferred Sen1Floods11 eval. Converts "null on one saturated set" into "here is the boundary condition," which is far more citable. Built and smoke-tested: `code/floodlite/data.py` (`make_loaders(train_frac=…)`, nested seeded subsample), `code/scripts/run_lowdata_kd.py` (full-data teacher + data-starved students × {none,comb} × fracs {0.1,0.25,0.5,1.0} × 3 folds, resumable), `code/scripts/make_lowdata_figure.py` (F5). Plan + run command + cost (~15–20 T4-h, resumable) in `manuscript/mod4_lowdata_kd_plan.md`. A 2-epoch smoke run already showed comb-KD +0.07 IoU over no-KD at 10% data (promising, not a result). **Pending:** the full Kaggle run (needs GPU-hours), then draft the new results subsection + F5.

---

## Minor comments

- **Min1 — param-count confusion.** §2.3 gives backbone-only counts (2.5/4.7/1.3 M); Table 1 gives full-UNet counts (3.59/5.61/3.09 M). Say which is which.
- **Min2 — teacher ≈ MobileNetV2.** Verified distinct runs (IoU 0.925708 vs 0.925715) that coincidentally round to 0.9257 / 0.9611 / 0.9729. Not an error, but reads like a copy-paste; add a footnote or one extra decimal in Table 2.
- **Min3 — figure labeling.** §5 calls F1/F3 "supplementary," but F1 is a `figure*` in the §3 body. Reconcile; regenerate F2/F4 after M1.
- **Min4 — crisper KD statement.** `stats.json` mean-deltas (none−comb) are +0.0027/+0.0013/+0.0049 → no-KD beats comb-KD in 3/3 architectures by 0.001–0.005 IoU. State it directly.
- **Min5 — version strings.** Appendix "PyTorch 2.8 / CUDA 12.1"; §3.5 "PyTorch 2.7+"; §3.6 "2.8+." Pin one. (Also: `export_all.py` uses opset 17 while §3.6 and `floodlite/export.py`'s default say opset 13 — reconcile.)
- **Min6 — Zenodo DOI.** Back matter says "to be minted"; git history already has a Zenodo/`v1.0-ijdrr` release commit. Mint and insert the real DOI before submission.
- **Min7 — abstract length.** ~350 words in one block; Elsevier/IJDRR generally wants ≤ 250. Tighten (M2/M3 fixes help).

---

## Venue calibration (IJDRR-specific)

IJDRR weights **operational realism** heavily. Two things will draw fire:
1. **Neither benchmarked platform is a genuinely constrained device.** A 2023 MacBook M2 Pro and Node.js-WASM are developer machines. The Raspberry-Pi-4 / Graviton / Android targets you defer (§2.5, §7) are *precisely* the evidence an IJDRR reviewer wants for "resource-constrained disaster zones." Landing even the Pi-4 number would convert the framing from aspirational to demonstrated.
2. **No field/operational validation** and single-dataset (FSSD, 663 RGB images). You flag both honestly, but for this venue the OOD/Sen1Floods11 evaluation is close to necessary.

Combined with M1, these are the revisions most likely to move an IJDRR decision. If the Pi-4/OOD work is out of scope this cycle, IGARSS (your stated backup) tolerates the current scope better.

---

## Questions for the authors
1. On ORT-CPU (script provided), the teacher is ~38 ms and the FP32 architecture speedup is 1.1–1.8×. Does the deployment narrative survive re-framing around INT8 (~2.2× vs teacher-ORT) rather than FP32 architecture speedup?
2. §6.3's three failure modes vs the actual folds [0.772, 0.0001, 0.657] — which is correct? Were these from a different calibration recipe?
3. Was the per-image bootstrap ever run (skipped in `stats.json`)? Can you provide it for the KD and PTQ contrasts?
4. Does a low-label subsample of FSSD recover any KD gain?

---

**Recommendation: Major Revision.** The contributions are real and the honesty is a genuine asset; the blocking issues are a confounded comparison (M1, re-measured above), two self-contradictory headline claims (M2, M3), and one unsupported narrative (M4) — all fixable without new architectures.
