# AI Usage Disclosure

**Manuscript.** FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation.
**Lead author.** Md Ibrahim Khalil.
**Target venue.** Elsevier International Journal of Disaster Risk Reduction (IJDRR), primary.
**Statement prepared.** 2026-05-16. To be re-confirmed at submission.

---

In preparing this manuscript and its accompanying code, the authors used Anthropic's Claude (model versions noted in the project log) as a writing-and-engineering assistant. Specifically, Claude was used for:

1. **Code scaffolding and debugging.** Skeleton implementations of `floodlite/quantize.py` (ONNX static QDQ flow), `scripts/bench_m2.py` (M2 ORT-CPU latency rigging), `bench.mjs` (Node.js / onnxruntime-web WASM benchmark), `scripts/make_qualitative.py` (F3 panel rendering), and `scripts/make_tables.py` / `scripts/make_figures.py` (T1–T5 and F2/F4 generation) were drafted with Claude assistance and then reviewed, executed, and corrected by the lead author. All numerical results in Tables T1–T5 and Figures F2 and F4 were produced by author-executed runs against `code/runs/3fold/{summary,quantized,lat_m2,lat_wasm}.json`.
2. **Manuscript scaffolding.** The LaTeX skeleton (`manuscript/main.tex`, per-section `.tex` files, `refs.bib` skeleton) was drafted with Claude assistance. Section prose was iterated between author intent and assistant drafts; the lead author retains editorial responsibility for every sentence in the camera-ready.
3. **Editorial polish.** Concision, register, British-English consistency, and citation-formatting checks.

Claude was **not** used to fabricate experimental results, to generate numerical values that do not appear in `code/runs/`, to author the negative-KD finding's mechanistic interpretation independent of author judgement, or to manufacture citations or DOIs. All methodological choices (3-fold protocol, Percentile-99.999% PTQ calibration, ONNX-hub deployment design, the decision to report rather than rescue the negative-KD finding) are the authors' own. The authors have read and approved the final manuscript and accept full responsibility for its content.
