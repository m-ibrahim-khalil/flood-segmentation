# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Reproducible code for the manuscript **"FloodLite: Knowledge-Distilled Lightweight Segmentation for On-Device Flood Mapping in Resource-Constrained Disaster Zones"** (Khalil et al., under review at *Remote Sensing*, 2026). The paper draft and roadmap live under `docs/` (`FloodLite_Manuscript.docx`, `FloodLite_Roadmap_and_Plan.docx`, `FloodEDL_Mamba_Research_Plan.docx`); all code lives under `code/`.

Target compute is free-tier: Kaggle T4×2, Colab T4, or a MacBook M2 Pro. The notebook is the primary reproduction path; the script is for headless multi-fold runs.

## Common commands

All commands assume `cd code/`.

```bash
# Setup (Python 3.11)
pip install -r requirements.txt

# Single fold, full pipeline (teacher → 4 KD configs × 3 students → quantize → benchmark)
python scripts/run_full_experiment.py --data_root /path/to/FSSD --fold 0

# Full 5-fold cross-validation
python scripts/run_full_experiment.py --data_root /path/to/FSSD --fold 0 --folds 5

# Faster smoke test
python scripts/run_full_experiment.py --data_root /path/to/FSSD --fold 0 \
    --epochs_teacher 2 --epochs_student 2 --batch_size 4
```

There is no test suite, linter config, or build step. The notebook `notebooks/floodlite_kaggle.ipynb` is the recommended end-to-end entry point on Kaggle (Add Data → *Flood Semantic Segmentation Dataset* by `lihuayang111265`; mounts at `/kaggle/input/flood-semantic-segmentation-dataset/`).

## Architecture

### Pipeline shape

`run_full_experiment.py` (and the notebook) execute, per fold:

1. **Teacher train** — UNet + EfficientNet-B0 (`make_teacher`), task loss only (BCE+Dice).
2. **Student train × 3 backbones × 4 KD configs** — for each of `mobilenetv3_small`, `efficientnet_lite0`, `mobilevit_xxs`, run an ablation grid over `kd ∈ {none, resp, feat, comb}` (toggles `use_response` / `use_feature` in `train_student_kd`).
3. **Quantize** — INT8 PTQ via `torch.ao.quantization` on the `comb`-KD students, calibrated against the train loader (`floodlite/quantize.py`).
4. **Benchmark** — `benchmark_latency` reports p50/p95/mean/std/fps.
5. Results dumped to `runs/results.json`.

### Loss composition (`floodlite/losses.py`)

Combined student loss is `(1 - α)·L_task + α·L_resp + β·L_feat` with defaults `α=0.5, β=0.1, T=4.0`.
- `L_task` = `0.5·BCE + 0.5·Dice` (binary segmentation).
- `L_resp` = temperature-scaled binary KL between teacher/student logits (T² rescaled).
- `L_feat` = MSE between student decoder features (passed through a learned 1×1 adapter `FeatureKDLoss.adapter`) and detached teacher features. Hooks tap **decoder block index 2** on both nets via `attach_decoder_hook` — channel dimensions are discovered with one peek-forward, then the adapter is sized and added to the optimizer's parameter list.

### Dataset wiring (`floodlite/data.py`)

- Expected layout: `<root>/dataset/{train,val}/{images,labels}/`. `find_split_dirs` is forgiving about casing (`Image`, `IMAGES`, `Mask`, `masks`, …) so older FSSD packagings also work.
- **Default behaviour**: train+val are pooled and split via `KFold(n_splits=5, shuffle=True, random_state=42)`. Pass `use_predefined_split=True` to `make_loaders` to use the dataset's own train/val folders instead (matches Karcı et al. 2026's protocol when pooling).
- Masks are binarized at `>127`. Default image size 256×256, batch 8, ImageNet normalization.

### Models (`floodlite/models.py`)

All models are `segmentation_models_pytorch.Unet` with different encoders. Teacher uses smp's native `efficientnet-b0`; students use `timm-`-prefixed encoders (`timm-mobilenetv3_small_100`, `timm-tf_efficientnet_lite0`, `timm-mobilevit_xxs`). All output raw logits — apply `sigmoid` externally.

### Reproducibility invariants

- `SEED=42` is fixed in the notebook and KFold split.
- Three random seeds are re-run for the final reported configuration (re-invoke with different seeds; not parameterized via CLI).
- Models trained with mixed precision when `device == "cuda"`; quantization runs on CPU (deployment-target conditions).

## Manuscript ↔ results coupling

The manuscript contains placeholders that are populated **after** running the code:

| Pattern | Replacement source |
|---|---|
| `##.##` | 4-decimal floats from `compute_metrics` (printed by the notebook's final cell) |
| `##` (latency cols.) | Integer ms from `benchmark_latency` |
| `{{IOU_RECOVERY_PCT}}`, `{{PARAM_PCT}}`, `{{FLOPS_PCT}}` | Ratios of student vs. teacher metrics (see `code/README.md` §3) |
| `{{FPS_M2}}`, `{{FPS_PI}}`, `{{FPS_WASM}}` | `1000 / p50_ms`, rounded |

When asked to update manuscript numbers, the source of truth is `runs/results.json` (or notebook output), not memorized values. **Appendix A** of the manuscript is the placeholder map and must be deleted before submission.

## Deployment exports (`floodlite/export.py`)

ONNX is the hub format. `export_onnx` produces a dynamic-batch ONNX with opset 13. `export_coreml` requires `coremltools` (macOS); `export_tflite` requires `onnx-tf` + `tensorflow`. ONNX Runtime Web consumes the ONNX directly. Pi 4 latency is measured off-host with `tflite-runtime` (snippet in `code/README.md` §5).

## Conventions worth knowing

- Code uses `from __future__ import annotations` everywhere — type hints are forward-compatible strings.
- `train_*` functions take `ckpt_path` and save the **best-IoU** checkpoint, not the last; downstream steps assume best.
- `train.py` still uses `torch.cuda.amp.autocast` / `GradScaler` (deprecated in newer PyTorch). Don't migrate to `torch.amp.*` casually — verify on the target Kaggle/Colab Torch version first.
- Licenses: code is MIT (`code/LICENSE`); model weights, when published to HF Hub, are Apache-2.0.
