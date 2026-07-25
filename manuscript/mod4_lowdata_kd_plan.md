# Mod4 — Low-data KD boundary-condition experiment (setup)

**Status:** setup complete and smoke-tested; **ready to run on Kaggle/Colab.** Not yet run at scale (needs GPU-hours the review machine doesn't have).

## Why

The paper's negative-KD finding is scoped to the *saturated* full-FSSD regime, and its mechanism — strong ImageNet priors + 663 labelled images leave the student no headroom — **predicts** that KD should reappear when labels are scarce. Right now §7 only *speculates* this ("A non-saturated benchmark … might recover the KD signal"). This experiment turns that speculation into evidence and converts the negative result into a positive characterization of *when* KD helps — a far more citable contribution.

## What was built

| Component | File | Role |
|---|---|---|
| Train-subsampling | `code/floodlite/data.py` → `make_loaders(train_frac=…)` | Deterministic, **nested** subsample of the train split (10% ⊂ 25% ⊂ 50% ⊂ 100%); val kept whole so IoU is comparable across fractions. |
| Driver | `code/scripts/run_lowdata_kd.py` | Reuses the **full-data teacher** (`teacher_fold{f}.pt`); trains data-starved students with/without KD; evaluates on full val; records metrics **incrementally** (resumable). |
| Figure | `code/scripts/make_lowdata_figure.py` | F5: KD gain `IoU(comb) − IoU(none)` vs training-data fraction. |

Design choices that make the result clean: (1) the teacher always sees all labels, so at low student-fractions it genuinely knows more than the student's data — the exact regime where KD is supposed to help; (2) no-KD and KD arms see **identical** subsampled data at each fraction (nested seeded subset); (3) evaluation is on the full val fold, so the x-axis is the only thing that changes.

## How to run (Kaggle T4 / Colab)

```bash
cd code/
python scripts/run_lowdata_kd.py \
    --data_root /kaggle/input/flood-semantic-segmentation-dataset \
    --teacher_dir runs/3fold \
    --out_dir runs/lowdata \
    --fracs 0.1,0.25,0.5,1.0 --fold 0 --folds 3 \
    --students mobilenetv3_small,efficientnet_lite0,mobilevit_xxs \
    --kd none,comb --epochs 35 --skip_if_exists
python scripts/make_lowdata_figure.py --lowdata runs/lowdata/lowdata_kd.json
```

- **Prereq:** the full-data teacher checkpoints (`teacher_fold{0,1,2}.pt`) must be in `--teacher_dir`. They already exist in `runs/3fold/`.
- **Resumability:** `--skip_if_exists` skips any (frac, fold, student, kd) cell already in `lowdata_kd.json`, so a timed-out Kaggle session resumes by re-running the same command. Checkpoints are deleted after eval unless `--keep_checkpoints`.
- **macOS:** add `--num_workers 0` (DataLoader workers crash on macOS otherwise).

## Cost

Default grid = 4 fracs × 3 folds × 3 students × 2 KD configs = 72 trainings, but low fractions run proportionally fewer iterations (Σ fracs = 1.85), so ≈ 33 full-data-equivalent student trainings ≈ **15–20 T4-hours**, i.e. ~2 Kaggle sessions with resume.

**Fast first look** (≈2–3 T4-hours): `--students efficientnet_lite0,mobilevit_xxs --folds 1`. Enough to see the trend before committing the full grid.

## Expected outcome

KD gain ≈ 0 at frac = 1.0 (reproduces the paper's negative result) rising above 0 as frac → 0.1. A throwaway **2-epoch** smoke run already showed comb-KD beating no-KD by **+0.07 IoU at 10% data** on MobileViT-XXS — promising, but **not a result** (n=1, 2 epochs). The real 35-epoch × 3-fold run is what the manuscript should cite.

## Manuscript integration (after the run)

1. New results subsection (e.g. §5.3.x "When does KD help? A low-data boundary") + **Figure F5**.
2. Rewrite §7's "Saturated benchmark" paragraph: replace the speculative "might recover the KD signal" with the measured boundary — KD gain becomes significant below ~X% of the data.
3. Strengthen §6.1: the saturation mechanism is now demonstrated, not inferred.
4. One line in the abstract's contribution (iii) / intro contribution 2: "…and we locate the data-fraction boundary below which KD does help."

I can draft these sections once `lowdata_kd.json` exists — with real numbers, not placeholders.
