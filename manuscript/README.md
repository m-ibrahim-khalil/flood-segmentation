# FloodLite — Manuscript

LaTeX source for the FloodLite paper, targeting Elsevier *International Journal of Disaster Risk Reduction* (IJDRR).

```
manuscript/
├── main.tex                         # entry point (elsarticle, preprint+review)
├── refs.bib                         # bibliography
├── cover_letter.tex                 # standalone cover letter for Editorial Manager
├── highlights.tex                   # Elsevier highlights (5 bullets ≤ 85 char)
├── outline.md                       # section-by-section status + numerical update workflow
├── AI_disclosure.md                 # supplementary AI-usage statement
├── README.md                        # this file
├── sections/                        # per-section TeX
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methods.tex
│   ├── 04_experimental_setup.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   ├── 07_limitations.tex
│   ├── 08_conclusions.tex
│   ├── 09_appendix_reproducibility.tex
│   └── 99_back_matter.tex
├── tables/
│   ├── T1_footprints.tex
│   ├── T2_segmentation.tex
│   ├── T3_kd_ablation.tex
│   ├── T4_quantization.tex
│   └── T5_latency.tex
└── figures/
    ├── F2_pareto.pdf
    └── F4_throughput.png
```

## Build

```bash
cd manuscript
latexmk -pdf main.tex                 # full build with BibTeX
latexmk -pdf -pvc main.tex            # continuous preview
latexmk -pdf cover_letter.tex         # cover letter standalone PDF
latexmk -pdf highlights.tex           # highlights standalone PDF
latexmk -C                            # clean intermediates
```

## Numerical source of truth

Every numerical claim in the manuscript traces to one of:
- `code/runs/3fold/summary.json` — FP32 segmentation metrics (3-fold mean ± std)
- `code/runs/3fold/quantized.json` — INT8 quantized metrics
- `code/runs/3fold/lat_m2.json` — M2 Pro ORT-CPU latency
- `code/runs/3fold/lat_wasm.json` — Browser WASM SIMD latency
- `code/runs/3fold/stats.json` — Wilcoxon p-values

Headline numbers are centralised as macros at the top of `main.tex`
(`\teacherIoU`, `\bestStudentIoU`, `\deployableMTwoFPS`, etc.). Update
the macro once and the value propagates to abstract / results /
discussion / conclusion.

## Remaining placeholders before submission

Search `main.pdf` and confirm zero matches for:
- `[your-username]` — GitHub username substitution
- `[DOI to be minted at submission]` — Zenodo DOI (task #30)
- `Figure placeholder` (F1 pipeline + F3 qualitative not yet finalised)
- `[Co-author Name]` or affiliations marked TBD

The corresponding files to edit are: `main.tex` (author block), `sections/09_appendix_reproducibility.tex` (DOI + GitHub URL), `sections/99_back_matter.tex` (data availability), and `sections/05_results.tex` (F1/F3 placeholders).
