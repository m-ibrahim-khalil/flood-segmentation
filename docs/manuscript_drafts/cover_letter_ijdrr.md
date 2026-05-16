# Cover Letter — Elsevier International Journal of Disaster Risk Reduction (IJDRR)

_Draft, 2026-05-16. To be moved into Editorial Manager's cover-letter field at submission. Replace [Editor's name] with the handling-editor name from the IJDRR masthead, or use "Dear Editor" if the handling editor is not yet assigned._

---

**Submission**: _FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation_
**Article type**: Research Article
**Corresponding author**: Md Ibrahim Khalil, Sensa AS, Norway · ibrahim@sensa.no

---

Dear Editor,

We are pleased to submit our manuscript, *FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation*, for consideration as a Research Article in the *International Journal of Disaster Risk Reduction*.

Operational flood-extent mapping is bottlenecked not by segmentation accuracy on academic benchmarks — that problem is essentially solved at IoU ≈ 0.93 on the public Flood Semantic Segmentation Dataset (FSSD) by current encoder–decoder architectures — but by the gap between published models and the constrained hardware that humanitarian responders actually deploy. Across the recent flood-segmentation literature, every paper reports IoU on a benchmark; **none reports per-platform inference latency, post-quantization accuracy, or model size**. Our manuscript closes that gap with the first systematic edge-deployment benchmark for flood segmentation, evaluated across two execution providers (Apple Silicon CPU and browser WebAssembly), three lightweight UNet students, and FP32 + INT8 precisions.

Two empirical findings make this work distinctive:

1. **A negative result on knowledge distillation.** Across three architectures × four distillation configurations × three folds, no configuration of response-, feature-, or combined-KD improves over the no-KD baseline; the best student (MobileViT-XXS-UNet, no-KD, FP32) *exceeds* the UNet+EfficientNet-B0 teacher by 0.011 IoU points. We trace this to the saturation of FSSD (663 images, IoU ≈ 0.93) combined with strong ImageNet pretraining, which leaves no headroom for the teacher's soft targets to transfer. We report this honestly rather than search for a configuration that would have justified a more conventional positive-result framing, because we believe the field is better served by an explicit benchmark-saturation caveat than by yet another marginal-gain KD recipe.

2. **Architecture-dependent post-training quantization stability.** Under an identical ONNX static QDQ INT8 recipe, EfficientNet-Lite0 quantizes cleanly (Δ IoU = −0.043 ± 0.016) while MobileNetV3-Small and MobileViT-XXS suffer 0.45-IoU collapses with high cross-fold variance. We trace the failures to the HardSwish activation family and to LayerNorm-plus-attention blocks, and we make a concrete recommendation: ReLU6-bounded backbones (such as EfficientNet-Lite0) are the safe default for INT8 PTQ on flood-segmentation pixels, and HardSwish- or attention-based backbones require quantization-aware training. To our knowledge this is the first systematic characterisation of this failure mode in the flood-segmentation literature, and it directly informs deployment-architecture choice for practitioners.

Our recommended deployable configuration — EfficientNet-Lite0-UNet INT8 served by ONNX Runtime CPU — runs at **58 FPS on a 2023 laptop CPU** and **6 FPS in a single-threaded WebAssembly browser sandbox** at **3.83× size reduction** and **95% IoU recovery vs. the teacher**. The same .onnx artifact serves both deployment targets, which we offer as a practical pattern for heterogeneous flood-response edge hardware.

The work fits IJDRR's scope because the contribution is not a new neural-network design but a *deployment-readiness benchmark for disaster-response practice*: the metrics we report (per-platform latency, INT8 size, INT8 IoU stability) are the metrics a humanitarian-data-science team needs to make a deployment decision, and they have been systematically absent from the prior flood-segmentation literature. We hope this orientation is useful to IJDRR's interdisciplinary readership of disaster-management practitioners and applied researchers.

**Reproducibility and openness.** All code, ONNX exports, per-fold checkpoints, and benchmarking harnesses are released under MIT licence at https://github.com/[your-username]/Flood_segmentation_model. A Zenodo-archived snapshot (DOI to be minted at acceptance) will be linked in the camera-ready. The FSSD dataset is publicly available on Kaggle.

**Conflicts of interest.** None to declare. **Funding.** The work received no external funding. **Co-author and reviewer suggestions.** Suggested reviewers and their affiliations are listed in Editorial Manager. We confirm that the manuscript has not been published previously, is not under consideration elsewhere, and has been read and approved by all listed authors.

We would be grateful for your consideration of this submission and welcome any feedback from the editorial team and reviewers.

Sincerely,

**Md Ibrahim Khalil**
Sensa AS, Norway
ibrahim@sensa.no
On behalf of all authors

---

### Suggested reviewers (to enter into Editorial Manager separately)

_Pick 4–6 from the list below; avoid anyone you've co-authored with in the last 4 years. Pull current affiliation and ORCID from each reviewer's most recent paper before pasting into the submission portal. Update entries as needed for IJDRR's actual reviewer-suggestion form._

1. **Şeyma Karcı** — Hacettepe University, Türkiye. Lead author of [10] (Karcı et al., 2026, *Nat. Hazards*), the direct flood-segmentation benchmark we extend. Strongest candidate for accuracy-pipeline review.
2. **Saurabh Garg** — Helmholtz Centre Potsdam, GFZ German Research Centre for Geosciences. Author of [18] (Ghosh, Garg, Motagh et al., 2024). Strong on SAR-flood pipelines and benchmark protocols.
3. **Dimitrios Hernández** — Universidad Católica San Antonio de Murcia, Spain. Author of [11] (Hernández, Cecilia, Cano, Calafate, 2022, *Remote Sens.*) — the closest prior UAV-edge-flood-segmentation work.
4. **Mariana M. Kuglitsch** — independent / ZAMG, Austria. Co-author of [29] (Kuglitsch et al., 2022, *Nat. Commun.*) on AI for disaster management. Strong on operational-deployment framing.
5. **Hairo Munawar** — University of New South Wales, Australia. Author of [14] (Munawar et al., 2021, *Autom. Constr.*) on flood ML for operational practice.
6. **Sachin Mehta** — Apple ML Research / University of Washington. Co-author of [26] (MobileViT). Strong on the lightweight-architecture and on-device-inference side.

_Avoid (recent co-authors / collaborators)_: [none currently, but verify per Sensa's external-collaboration register before submission].

---

### Submission checklist before clicking Submit

- [ ] PDF compiled from the final .docx with all figures embedded
- [ ] Highlights file (3–5 bullets, ≤ 85 chars each — see below)
- [ ] Graphical abstract (optional but recommended — the F1 pipeline diagram, 600×400 px)
- [ ] Author CRediT taxonomy completed (already in §Author Contributions of the manuscript)
- [ ] Zenodo DOI: pending mint at https://zenodo.org/uploads/new (task #30)
- [ ] ORCID linked for corresponding author
- [ ] Cover letter pasted into the cover-letter field
- [ ] Suggested reviewers entered

### Highlights (Elsevier format, ≤ 85 chars each)

1. First multi-platform edge-deployment benchmark for flood segmentation.
2. Negative-KD result: ImageNet-pretrained students match teacher without distillation.
3. INT8 PTQ stability is architecture-dependent; ReLU6 backbones are safe defaults.
4. EfficientNet-Lite0 INT8 ONNX: 58 FPS on M2 CPU, 6 FPS in WASM, 95% IoU.
5. All code, ONNX exports, and Zenodo-archived release publicly available.
