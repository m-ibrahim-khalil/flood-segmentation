# Manuscript Outline — FloodLite (IJDRR)

**Owner:** Md Ibrahim Khalil (lead) + co-authors TBD.
**Venue:** Elsevier International Journal of Disaster Risk Reduction (IJDRR), primary; IGARSS 2026 (4-page extended abstract) backup.
**Target word budget:** 9,000–11,000 words main text (IJDRR is generous on length).
**Numerical source of truth:** `code/runs/3fold/{summary,quantized,lat_m2,lat_wasm}.json`. Do not paste numbers that aren't traceable back to these JSONs.
**Status (2026-05-16):** LaTeX scaffold complete and compiling. All five tables T1–T5 populated. F2 + F4 figures present; F1 (pipeline) and F3 (qualitative) are placeholders pending manual diagram and a local FSSD-disk run.

---

## Section-by-section status

| Section | File | Status | Notes |
|---|---|---|---|
| Abstract | `sections/00_abstract.tex` | Drafted | Macro-driven numbers (`\teacherIoU`, `\bestStudentIoU`, etc.) |
| 1. Introduction | `sections/01_introduction.tex` | Drafted | Contributions + scope-pivot subsections |
| 2. Related Work | `sections/02_related_work.tex` | Drafted | 5 subsections including new lightweight-backbone subsection |
| 3. Methods | `sections/03_methods.tex` | Drafted | Includes `tables/T1_footprints.tex` |
| 4. Experimental Setup | `sections/04_experimental_setup.tex` | Drafted | 3-fold protocol; M2 + WASM rigging |
| 5. Results | `sections/05_results.tex` | Drafted | T2–T5 + F2 + F4; F1/F3 are framed placeholders |
| 6. Discussion | `sections/06_discussion.tex` | Drafted | Leads with negative-KD; PTQ-trap subsection |
| 7. Limitations | `sections/07_limitations.tex` | Drafted | OOD, RGB-only, 3-fold power, PTQ-only, no energy |
| 8. Conclusions | `sections/08_conclusions.tex` | Drafted | — |
| Back matter | `sections/99_back_matter.tex` | Drafted | CRediT, funding, data availability |
| Appendix (Reproducibility) | `sections/09_appendix_reproducibility.tex` | Drafted | — |

## Tables T1–T5 (all populated, source-of-truth-traceable)

| Table | File | Source |
|---|---|---|
| T1 footprints | `tables/T1_footprints.tex` | Params from `torchinfo.summary`; ONNX sizes from `code/exports/` |
| T2 segmentation | `tables/T2_segmentation.tex` | `code/runs/3fold/summary.json` |
| T3 KD ablation | `tables/T3_kd_ablation.tex` | `code/runs/3fold/summary.json` + `stats.json` |
| T4 quantization | `tables/T4_quantization.tex` | `code/runs/3fold/quantized.json` |
| T5 latency | `tables/T5_latency.tex` | `code/runs/3fold/{lat_m2,lat_wasm}.json` |

## Figures status

| Figure | Path | Status | Action |
|---|---|---|---|
| F1 pipeline | `figures/F1_pipeline.pdf` | Placeholder (framed box in §5) | Manual: PowerPoint / draw.io, ~30 min |
| F2 Pareto | `figures/F2_pareto.pdf` | Present | — |
| F3 qualitative | `figures/F3_qualitative.pdf` | Placeholder | Run `code/scripts/make_qualitative.py` when FSSD is locally reachable |
| F4 throughput | `figures/F4_throughput.png` | Present | — |

## Compile

```bash
cd manuscript
latexmk -pdf main.tex                  # full build, includes BibTeX
latexmk -pdf -pvc main.tex             # continuous preview
latexmk -C                             # clean
```

## Numerical-update workflow

If experiments are re-run and a JSON changes:
1. Re-run `python code/scripts/make_tables.py` → updates `docs/tables.md`.
2. Manually port the affected cells into `manuscript/tables/Tn_*.tex`.
3. Update headline-number macros at the top of `manuscript/main.tex` (`\teacherIoU`, `\bestStudentIoU`, `\deployableINTIoU`, `\deployableMTwoFPS`, `\deployableWASMFPS`).
4. `latexmk -pdf main.tex` and visually diff `main.pdf`.

The macros propagate to abstract, results, discussion, and conclusion — one edit point per headline number.

## Remaining before submission

- [ ] Co-author final list locked.
- [ ] F1 pipeline diagram exported as PDF and dropped into `manuscript/figures/`.
- [ ] F3 qualitative rendered locally and dropped into `manuscript/figures/`.
- [ ] Zenodo DOI minted; placeholder `[DOI to be minted at submission]` replaced in `09_appendix_reproducibility.tex` and `99_back_matter.tex`.
- [x] GitHub username substituted into URLs (`m-ibrahim-khalil/flood-segmentation`).
- [ ] Cover letter (`cover_letter.tex`) finalised with handling-editor name.
- [ ] Suggested-reviewer list finalised (entered separately in Editorial Manager).
- [ ] Final read-through: confirm zero `[your-username]` or `\fbox{...placeholder...}` strings remain (only the Zenodo DOI placeholder is acceptable until the release is minted via `scripts/release_v1_ijdrr.sh`).
