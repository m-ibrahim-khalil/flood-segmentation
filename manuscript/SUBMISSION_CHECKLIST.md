# IJDRR Submission Checklist — FloodLite

Keep this open next to https://www.editorialmanager.com/jidr/ while submitting. Every field below has the exact paste-ready value.

---

## 0. Before you start (one-time, do once)

- [ ] Zenodo DOI minted → string in form `10.5281/zenodo.NNNNNNN`
- [ ] DOI substituted into `manuscript/sections/09_appendix_reproducibility.tex`, `manuscript/cover_letter.tex`, `CITATION.cff` (`doi:` field added)
- [ ] `main.pdf` and `cover_letter.pdf` rebuilt after DOI substitution (`latexmk -pdf main.tex && latexmk -pdf cover_letter.tex`)
- [ ] Final grep — should return zero matches:
      ```bash
      grep -rn "your-username\|\[DOI\|to be minted" manuscript/sections/ manuscript/cover_letter.tex
      ```
- [ ] Co-author list locked in `manuscript/main.tex` `\author{}` block; affiliations correct
- [ ] ORCID added to corresponding-author block if you have one

---

## 1. Article type

| Field | Value |
|---|---|
| **Article type** | Research Paper (the default; not "Review", not "Short Communication") |

---

## 2. Title

```
FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation
```

Short title (if asked, ≤ 70 chars):

```
A Multi-Platform Edge Benchmark for Lightweight Flood Segmentation
```

---

## 3. Authors

For each author, EM will ask for first name, last name, email, affiliation, ORCID. Paste in order. Mark Md Ibrahim Khalil as corresponding author.

| # | First name | Last name | Email | Affiliation | ORCID | Role |
|---|---|---|---|---|---|---|
| 1 | Md Ibrahim | Khalil | ibrahim@sensa.no | Sensa AS, Oslo, Norway | _(fill if available)_ | Corresponding |
| 2 | _(co-author)_ | _(if applicable)_ | | | | |

---

## 4. Keywords (paste comma-separated)

```
flood segmentation; edge deployment; knowledge distillation; post-training quantization; ONNX Runtime; WebAssembly; remote sensing; lightweight neural networks; disaster response
```

---

## 5. Abstract

Open `manuscript/sections/00_abstract.tex`, remove the `\begin{abstract}...\end{abstract}` wrapper and any LaTeX math/markup, paste the resulting plain text into EM's abstract field. (EM accepts ~1500–2500 chars; the FloodLite abstract is ~2300 chars — fits.)

Quick plain-text version is in `docs/manuscript_drafts/cover_letter_ijdrr.md` and in `highlights.tex` for the bulletised version.

---

## 6. Highlights (Elsevier requires 3–5 bullets ≤ 85 chars each)

Paste each bullet on its own line:

```
First multi-platform edge-deployment benchmark for flood segmentation.
Negative-KD: ImageNet-pretrained students match teacher without distillation.
INT8 PTQ stability is architecture-dependent; ReLU6 backbones are safe defaults.
EfficientNet-Lite0 INT8 ONNX: 58 FPS on M2 CPU, 6 FPS in WASM, 95% IoU.
All code, ONNX exports, and a Zenodo-archived release are publicly available.
```

---

## 7. Suggested reviewers (enter 4–6)

| # | Name | Affiliation | Email (look up via DOI of cited paper) | Why |
|---|---|---|---|---|
| 1 | Şeyma Karcı | Hacettepe University, Türkiye | _(from [10] author block)_ | Direct FSSD benchmark precedent |
| 2 | Saurabh Garg | Helmholtz / GFZ Potsdam, Germany | _(from [18])_ | SAR-flood pipelines |
| 3 | Daniel Hernández | UCAM, Spain | _(from [11])_ | Closest prior UAV-edge work |
| 4 | Monique M. Kuglitsch | ZAMG, Austria | _(from [29])_ | AI for operational disaster mgmt |
| 5 | Hafiz S. Munawar | UNSW, Australia | _(from [14])_ | Flood ML for ops practice |
| 6 | Sachin Mehta | Apple ML / UW, USA | _(from [26])_ | MobileViT + on-device inference |

Opposed reviewers: leave blank unless you have a documented conflict.

---

## 8. Cover letter

Upload `manuscript/cover_letter.pdf` in the "Cover Letter" slot OR paste the body text (without the `\opening`/`\closing` wrappers) into the cover-letter text field if EM provides one.

---

## 9. Funding statement (paste into the funding field)

```
This research received no external funding.
```

## 10. Conflict of interest (paste)

```
The authors declare no conflict of interest.
```

## 11. Data Availability (paste)

```
The Flood Semantic Segmentation Dataset (FSSD) is publicly available at https://www.kaggle.com/datasets/lihuayang111265/flood-semantic-segmentation-dataset. All trained models, training code, and deployment scripts produced for this study are available at https://github.com/m-ibrahim-khalil/flood-segmentation (MIT licence; Zenodo DOI 10.5281/zenodo.NNNNNNN). The Sen1Floods11 dataset, used in preliminary out-of-distribution preparation, is available at https://github.com/cloudtostreet/Sen1Floods11 (CC-BY 4.0).
```

(Substitute the real Zenodo DOI.)

---

## 12. CRediT taxonomy

Open the CRediT picker in EM. For Md Ibrahim Khalil, tick:

- Conceptualization
- Methodology
- Software
- Validation
- Formal analysis
- Investigation
- Data curation
- Writing – original draft
- Writing – review & editing
- Visualization
- Project administration

(Spread roles across co-authors if you have any.)

---

## 13. File upload — exact list and order

EM asks for "Manuscript" first, then "Figures", "Highlights", "Cover letter", "Supplementary", in that conventional order. The two-column elsarticle PDF embeds the figures already, so the figure-upload step can be **a) skipped** (Elsevier accepts inline figures in the PDF for first submission) **or b)** you upload the four figure files separately if EM insists.

| Order | Type | File | Notes |
|---|---|---|---|
| 1 | Manuscript | `manuscript/main.pdf` | 16 pages, 2.9 MB |
| 2 | Highlights | `manuscript/highlights.pdf` | 1 page, 40 KB |
| 3 | Cover letter | `manuscript/cover_letter.pdf` | 2 pages, 146 KB |
| 4 | Figure (optional separate) | `manuscript/figures/F1_pipeline.pdf` | Pipeline schematic |
| 5 | Figure (optional separate) | `manuscript/figures/F2_pareto.pdf` | 4-axis Pareto |
| 6 | Figure (optional separate) | `manuscript/figures/F3_qualitative.pdf` | Easy / hard val examples |
| 7 | Figure (optional separate) | `manuscript/figures/F4_throughput.png` | Per-platform FPS bars |
| 8 | Supplementary | (none for first submission) | OOD + Pi 4 data deferred to revision |

If EM requires LaTeX source rather than (or alongside) the PDF, zip the entire `manuscript/` directory:

```bash
cd /Users/ibrahim/Desktop/personal/Flood_segmentation_model
zip -r floodlite_latex_source.zip manuscript -x 'manuscript/*.aux' 'manuscript/*.log' 'manuscript/*.fls' 'manuscript/*.fdb_latexmk' 'manuscript/*.bbl' 'manuscript/*.blg' 'manuscript/*.synctex.gz' 'manuscript/*.out' 'manuscript/*.spl'
```

Upload `floodlite_latex_source.zip` in the "LaTeX source" slot.

---

## 14. Final pre-submit checks

- [ ] Open `main.pdf` end-to-end one last time. Check pages 1, 5 (F1), 8 (T2), 10 (F2+F4), 12 (F3), 15–16 (refs + appendix).
- [ ] Confirm zero `\fbox{...placeholder...}` rectangles in `main.pdf`.
- [ ] Confirm DOI substitution: `grep "10.5281/zenodo" manuscript/sections/09_appendix_reproducibility.tex` returns a line.
- [ ] Confirm GitHub repo is public (or at least readable by the reviewer auth route) — flip from private to public via GitHub Settings → Danger Zone if needed.
- [ ] Confirm Zenodo record is public, not "restricted".
- [ ] All authors have read the final PDF and signed off via email (paper trail).
- [ ] Save a snapshot of EM's final preview PDF on disk before clicking Submit.

---

## 15. After submission

- [ ] Note the EM manuscript number (e.g. `IJDR-D-26-NNNNN`) — save in `docs/` somewhere.
- [ ] EM emails an acknowledgement within ~1 hour. Keep that email.
- [ ] First editorial decision typically lands in 3–8 weeks for IJDRR. Watch the EM "Submissions Being Processed" queue.
- [ ] If a "Revisions Required" decision comes back, parse the reviewer comments into a revision roadmap (`docs/manuscript_drafts/` or a new `manuscript/revision_v2/` directory) — don't paste responses inline in EM before you have a tracked-changes plan locally.
- [ ] If "Reject", pivot to IGARSS 2026 (4-page extended abstract) from the same source — `manuscript/main.tex` compresses to IGARSS format by switching documentclass and dropping the limitations + reproducibility appendix.

---

_Last updated: 2026-05-17 (v1.0-ijdrr commit pending Zenodo mint)._
