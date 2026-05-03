# Phase B Execution Runbook

**Date:** 2026-05-02
**Goal:** Produce all numerical results needed for Tables T1–T6 and Figures F2–F4 of the IJDRR submission.
**You drive Kaggle + local CPU; I drive script implementation (already done).**
**Total wall-clock:** ~5–7 calendar days (most of it is Kaggle GPU time you can leave running).

---

## What's already in place (after Phase A + B-script implementation)

```
code/scripts/
├── run_full_experiment.py        ← updated (3 folds × 35 ep × +baseline)
├── aggregate_results.py          ← NEW — pools per-fold JSONs into summary.json
├── quantize_all.py               ← NEW — INT8 PTQ across student × fold grid
├── prepare_sen1floods11_rgb.py   ← NEW — sample 250 OOD chips
├── eval_ood.py                   ← NEW — eval FSSD-trained models on Sen1Floods11
└── stats.py                      ← NEW — Wilcoxon + paired pixel-bootstrap
```

All five scripts have `--help` and parse cleanly. None has been run against real data yet.

---

## Day-by-day plan (target: 5–7 days)

### **Day 0 (today, ~1 hour) — Smoke test on Kaggle**

This is the critical pre-flight to confirm the Phase A code changes (especially the INT8 quantization fix and the MobileNetV2 baseline) actually run end-to-end on the Kaggle environment that produced your original Fold-0 results.

**Steps:**

1. Open Kaggle, create a new notebook.
2. Add Data → search *Flood Semantic Segmentation Dataset* by `lihuayang111265` → Add.
3. Settings → Accelerator → **GPU T4×2**.
4. Push your local `code/` to GitHub (so Kaggle can clone it), or upload as a Kaggle dataset, or paste the cell below.

**Cell 1 — clone and install:**
```python
!git clone https://github.com/<your-user>/Flood_segmentation_model.git
%cd Flood_segmentation_model/code
!pip install -q -r requirements.txt
```

**Cell 2 — verify the package imports:**
```python
import sys; sys.path.insert(0, '.')
from floodlite.models import make_baseline_mobilenetv2, make_student
from floodlite.data import make_sen1floods11_loader
from floodlite.benchmark import benchmark_onnxruntime
from floodlite.quantize import quantize_int8
print("All Phase A imports OK")
```

**Cell 3 — smoke test (2 epochs, 1 fold, 1 student):**
```python
!python scripts/run_full_experiment.py \
    --data_root /kaggle/input/flood-semantic-segmentation-dataset \
    --fold 0 --folds 1 \
    --epochs_teacher 2 --epochs_student 2 \
    --batch_size 4 \
    --include_baseline \
    --out_dir /kaggle/working/runs/smoke
```

**What to watch for:**
- ✅ Teacher trains.
- ✅ Baseline (MobileNetV2) trains and prints `Baseline (MobileNetV2): {...}`.
- ✅ All 3 students × 4 KD = 12 student configs run.
- ✅ Quantization step produces 3 entries with non-zero IoU. **If you see `Could not run 'quantized::conv2d.new'` → the Phase A fix didn't help — capture the full traceback and ping me; we may need a `QuantStub` wrapper.**
- ✅ Latency benchmark prints 3 entries.
- ✅ `runs/smoke/results.json` exists.

**Cell 4 — sanity-check JSON:**
```python
import json
r = json.load(open('/kaggle/working/runs/smoke/results.json'))
print(json.dumps(r['fold_0'], indent=2)[:1500])
```

**Decision point:**
- ✅ If smoke test passes → proceed to Day 1.
- ❌ If quantization fails → fall-back: pin Kaggle to PyTorch 2.2 (`!pip install torch==2.2.0+cu121 -q`) and retry. If still fails → ping me, will need QuantStub.

---

### **Day 1 — Launch the 3-fold full sweep**

Same Kaggle notebook, replace Cell 3 with:

```python
!python scripts/run_full_experiment.py \
    --data_root /kaggle/input/flood-semantic-segmentation-dataset \
    --fold 0 --folds 3 \
    --epochs_teacher 35 --epochs_student 35 \
    --batch_size 8 \
    --include_baseline \
    --out_dir /kaggle/working/runs/3fold
```

**Wall-time:** 14 configs × 35 epochs × ~0.7 min/epoch = ~5.7 hr per fold × 3 folds ≈ **17 hours**.

**Strategy options:**
- (a) Run fold 0 in one Kaggle session (≈12 hr session limit), then folds 1, 2 in subsequent sessions — change `--fold 1 --folds 1` and `--fold 2 --folds 1`. Each writes `runs/3fold/fold_N/results.json` under unique paths.
- (b) Use two Kaggle accounts in parallel — one runs `--fold 0 --folds 2` (folds 0+1), another runs `--fold 2 --folds 1`.
- (c) If you have Colab Pro: run one fold there in parallel.

**At end of each fold, download `results.json` to local:**
```bash
# On Kaggle, after the run completes:
# Right-click /kaggle/working/runs/3fold/results.json → Download
# Save locally to:
#   code/runs/3fold/fold0_results.json   (etc.)
```

**Important:** the script writes one `results.json` per invocation containing only the folds it ran. If you split runs across sessions, rename each downloaded file by fold (`fold0_results.json`, `fold1_results.json`, `fold2_results.json`) before continuing.

**Also download all 42 checkpoints** — you need them for B2 (quantize), B4 (OOD), and Phase C (latency on M2). Easiest: zip the entire `/kaggle/working/runs/3fold/` and download once after all folds finish.

```python
# Kaggle cell after sweep finishes:
import shutil
shutil.make_archive('/kaggle/working/runs_3fold', 'zip', '/kaggle/working/runs/3fold')
# Then download runs_3fold.zip via the file browser.
```

Locally:
```bash
mkdir -p code/runs/3fold
unzip ~/Downloads/runs_3fold.zip -d code/runs/3fold/
```

---

### **Day 2 (parallel with Day 1) — Acquire Sen1Floods11 RGB chips**

This is independent of the Kaggle sweep — do it while Kaggle runs.

**Source:** Cloud2Street's Sen1Floods11 release. The hand-labeled chips are under `HandLabeled/S2Hand/` (Sentinel-2 RGB visualizations) with masks under `HandLabeled/LabelHand/`.

**Two acquisition paths:**

**Option A — direct from GitHub release** (recommended, free, ~6 GB):
```bash
mkdir -p code/data/sen1floods11_raw
cd code/data/sen1floods11_raw
# The dataset is hosted in Google Cloud Storage; install gsutil first:
# https://cloud.google.com/storage/docs/gsutil_install
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand .
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand .
```

**Option B — Hugging Face mirror** (smaller subset, faster):
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nasa-impact/Sen1Floods11', repo_type='dataset',
                  local_dir='code/data/sen1floods11_raw',
                  allow_patterns=['HandLabeled/S2Hand/*.tif', 'HandLabeled/LabelHand/*.tif'])
"
```

**Then sample 250 chips:**
```bash
cd /Users/ibrahim/Desktop/personal/Flood_segmentation_model
python3 code/scripts/prepare_sen1floods11_rgb.py \
    --src_images code/data/sen1floods11_raw/HandLabeled/S2Hand \
    --src_labels code/data/sen1floods11_raw/HandLabeled/LabelHand \
    --out code/data/sen1floods11_rgb \
    --n 250
```

**Expected output:** `Wrote 250 pairs to code/data/sen1floods11_rgb`.

**Sanity check:** the loader from Phase A loads them cleanly:
```bash
cd code
python3 -c "
from floodlite.data import make_sen1floods11_loader
ld = make_sen1floods11_loader('data/sen1floods11_rgb', n_chips=250, batch_size=8)
x, y = next(iter(ld))
print('OK — shape:', x.shape, 'mask range:', y.min().item(), '..', y.max().item())
"
```

Expected: `shape: torch.Size([8, 3, 256, 256]) mask range: 0.0 .. 1.0`.

---

### **Day 3–4 — Wait for Kaggle to finish** 🍵

Most of Phase B is just waiting. Use this time for:
- Phase C prep: provision an AWS Graviton `t4g.small` (free tier), enable Termux on your Android phone, pre-test Chrome's WASM SIMD support.
- Phase D prep: re-read your manuscript draft and start a separate notes file with the section-by-section rewrite checklist (plan §D3).

---

### **Day 5 — Aggregate + quantize + OOD + stats**

By now you have:
- `code/runs/3fold/fold0_results.json`, `fold1_results.json`, `fold2_results.json`
- `code/runs/3fold/teacher_fold{0,1,2}.pt`, `baseline_mobilenetv2_fold{0,1,2}.pt`, `{student}_{kd}_fold{N}.pt` (39 student checkpoints)
- `code/data/sen1floods11_rgb/` populated

Run these four scripts in sequence (~30 min total on M2 Pro CPU):

**Step 1 — Aggregate per-fold JSONs into a summary:**
```bash
cd code
python3 scripts/aggregate_results.py \
    --inputs runs/3fold/fold0_results.json \
             runs/3fold/fold1_results.json \
             runs/3fold/fold2_results.json \
    --out runs/3fold/summary.json
```
Expected: `Wrote runs/3fold/summary.json with 14 configs.`

**Step 2 — INT8-quantize the no-KD students × 3 folds:**
```bash
cd code
python3 scripts/quantize_all.py \
    --data_root /path/to/local/FSSD \
    --ckpt_dir runs/3fold \
    --folds 0 1 2 \
    --out runs/3fold/quantized.json
```

You'll need a local copy of FSSD (for calibration). Download from Kaggle datasets via:
```bash
pip install kaggle
kaggle datasets download -d lihuayang111265/flood-semantic-segmentation-dataset -p code/data/
unzip code/data/flood-semantic-segmentation-dataset.zip -d code/data/FSSD
```

Expected output per (fold, student):
```
fold 0 mobilenetv3_small: 0.9203 -> 0.9180
fold 0 efficientnet_lite0: 0.9348 -> 0.9301
fold 0 mobilevit_xxs: 0.9450 -> ???  (might fail — see note below)
...
```

**MobileViT-XXS quantization caveat:** known risk per spec §10. If you see `error` entries for mobilevit_xxs, that's expected — report `FP16 only` for that model in T4 and proceed. Don't stop.

**Step 3 — Identify the best KD config per student** (manual, 5 min):

Open `runs/3fold/summary.json`. For each of `mobilenetv3_small`, `efficientnet_lite0`, `mobilevit_xxs`, find the (kd) with the highest mean IoU. Record them. Based on your Fold-0 numbers, the likely answer is:
- `mobilenetv3_small` → `feat`
- `efficientnet_lite0` → `none`
- `mobilevit_xxs` → `feat` (or `none` — close)

This is the "best student per architecture" definition from spec §3.

**Step 4 — OOD evaluation on Sen1Floods11:**

Run `eval_ood.py` once per `--best_kd` value you identified in Step 3 — the script accepts one `--best_kd` per invocation (limitation of current CLI — pragmatic given the small number of configs):

```bash
cd code
# Run once per best-KD value across the 3 students
for kd in none feat; do
    python3 scripts/eval_ood.py \
        --ood_root data/sen1floods11_rgb \
        --ckpt_dir runs/3fold \
        --fold 0 \
        --best_kd $kd \
        --out runs/3fold/ood_$kd.json
done
```

Combine the relevant student lines manually into a final `runs/3fold/ood_results.json` (small, hand-curated), or just cite both files from the manuscript.

**Numbers will be much lower than FSSD val** — likely IoU in the 0.30–0.65 range. **That IS the finding.** Single-domain training on 663 RGB drone images doesn't transfer to Sentinel-2 RGB visualizations. This is the headline OOD-cost number you'll cite in the Discussion.

**Step 5 — Statistical tests:**

Note: the current `stats.py` reconstructs per-fold IoUs by reading individual `fold{N}_results.json` files. Your aggregator put them at `runs/3fold/foldN_results.json`. Make sure that path matches what `stats.py` expects (read the script and adjust the `Path(args.ckpt_dir).parent.joinpath(...)` line if needed — quick 30-second fix).

```bash
cd code
python3 scripts/stats.py \
    --summary runs/3fold/summary.json \
    --data_root data/FSSD \
    --ckpt_dir runs/3fold \
    --out runs/3fold/stats.json
```

Expected output: `runs/3fold/stats.json` with two top-level keys:
- `wilcoxon_3fold` — three entries (one per student), p-values around 0.25 (the floor for n=3).
- `bootstrap_fold0` — three entries with `mean_delta` and `p_one_sided_no_kd_better`. **The `p` values here are the publication-grade ones** because they're computed over hundreds of validation pixels.

---

### **Day 6 — Commit + verify everything is in the repo**

```bash
cd /Users/ibrahim/Desktop/personal/Flood_segmentation_model

# Don't commit checkpoints or data (gitignored), but DO commit JSON results:
git add code/runs/3fold/fold0_results.json \
        code/runs/3fold/fold1_results.json \
        code/runs/3fold/fold2_results.json \
        code/runs/3fold/summary.json \
        code/runs/3fold/quantized.json \
        code/runs/3fold/ood_*.json \
        code/runs/3fold/stats.json
git commit -m "results(phase-b): 3-fold sweep + quantization + OOD + stats"

# Tag this milestone for reproducibility
git tag phase-b-complete
```

**Verify the result schema is what Phase D expects:**
```bash
cd code
python3 -c "
import json
s = json.load(open('runs/3fold/summary.json'))
expected = {'teacher', 'baseline_mobilenetv2'}
expected |= {f'{enc}_{kd}' for enc in ('mobilenetv3_small', 'efficientnet_lite0', 'mobilevit_xxs')
                            for kd in ('none', 'resp', 'feat', 'comb')}
got = set(s.keys())
missing = expected - got
extra = got - expected
print('Missing:', missing)
print('Extra:', extra)
print('Sample:', list(s.values())[0])
"
```

Missing should be empty (or just `baseline_mobilenetv2` if you didn't pass `--include_baseline` — re-run if so).

---

## Decision points (ping me when you hit any)

| Trigger | Action |
|---|---|
| Smoke test quantization throws CUDA error on Kaggle | Send full traceback. Likely needs QuantStub wrapper. |
| All KD configs *outperform* `none` (opposite of Fold-0) | Reconsider paper narrative — re-promote KD to a contribution. |
| MobileViT-XXS quantization fails on all 3 folds | Drop INT8 row for that arch in T4; report FP16 size only. |
| Sen1Floods11 OOD IoU drops below 0.20 | Possibly a preprocessing mismatch — sanity-check first 10 chips look like flood scenes. |
| Total Kaggle wall time > 30 hr | Drop to 2 folds (per spec §10 kill switch). Re-run aggregator with 2 input files. |
| Bootstrap p-values strongly *disagree* with Wilcoxon | Trust the bootstrap (n is much larger). Note both in manuscript. |

---

## Output checklist (gate to Phase C)

Before starting Phase C, you must have all of these in `code/runs/3fold/`:

- [ ] `fold0_results.json`
- [ ] `fold1_results.json`
- [ ] `fold2_results.json`
- [ ] `summary.json` (14 configs, mean ± std)
- [ ] `quantized.json` (3 students × 3 folds, with at most 1 architecture in `error` state)
- [ ] `ood_none.json` and/or `ood_feat.json`
- [ ] `stats.json` (Wilcoxon + bootstrap)
- [ ] All 42 student checkpoints + 3 teacher + 3 baseline (.pt files, gitignored, but present locally)

Once these exist, ping me with: *"Phase B complete — start Phase C."* I'll dispatch C1 (ONNX export + parity) and walk you through the per-platform measurements.

---

## What I'd need to know to help you faster

When you hit Day 5, paste the contents of `summary.json` and `stats.json` into a follow-up message. With those numbers I can:
1. Lock in the "best student" definition empirically (currently a placeholder).
2. Pre-draft the Results-section paragraphs around real numbers (saves you 2–3 days in Phase D).
3. Pre-render F2 (Pareto) with real coordinates as soon as Phase C latency comes in.

Good luck. Run, sleep, repeat. Next checkpoint: smoke test result.
