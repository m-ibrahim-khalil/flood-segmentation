# FloodLite Edge-Benchmark Pivot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot the FloodLite manuscript from a knowledge-distillation contribution to a multi-platform edge-deployment benchmark, with corrected experiments, real per-platform latency, an OOD test, and an honest negative-KD ablation, ready for IJDRR submission within 4–6 weeks and IGARSS 2026 as parallel backup.

**Architecture:** Repo stays as-is (`code/floodlite/` package + `scripts/` + `notebooks/`). Five surgical code changes (~13 hr): fix INT8 PTQ CPU bug; add MobileNetV2 baseline; add Sen1Floods11 OOD loader; add ONNX Runtime CPU benchmark; update training script defaults. Then a 3-fold sweep of 14 configs, 4-platform latency measurement (M2 / Android Termux / WASM / AWS Graviton), and a manuscript rewrite around the corrected story.

**Tech Stack:** Python 3.11, PyTorch 2.2+, segmentation-models-pytorch 0.3.4, timm 1.0.11, Albumentations 1.4.18, onnx + onnxruntime, coremltools (M2 only), tflite-runtime (Android), onnxruntime-web (browser).

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-02-floodlite-edge-benchmark-pivot-design.md`
- Project conventions: `CLAUDE.md`
- Manuscript: `docs/FloodLite_Manuscript.docx`

---

## Phase A — Code foundation (~1.5 working days)

### Task A0: Initialize git + minimal test scaffolding

**Files:**
- Create: `code/.gitignore`
- Create: `code/tests/__init__.py`
- Create: `code/tests/conftest.py`

**Why:** The repo is currently not under git per the environment report; reproducibility for both IJDRR and Zenodo requires a tagged release. There is also no test suite — we add a minimal one for the new code.

- [ ] **Step 1: Initialize git in the project root**

```bash
cd /Users/ibrahim/Desktop/personal/Flood_segmentation_model
git init
git add CLAUDE.md docs/ code/
git commit -m "chore: initial snapshot before edge-benchmark pivot"
```

- [ ] **Step 2: Create `code/.gitignore`**

```
__pycache__/
*.pyc
*.pt
runs/
exports/
data/
.ipynb_checkpoints/
.DS_Store
.venv/
.env
*.onnx
*.tflite
*.mlmodel
.pytest_cache/
```

- [ ] **Step 3: Create `code/tests/__init__.py` (empty)**

```python
```

- [ ] **Step 4: Create `code/tests/conftest.py`**

```python
"""Shared pytest fixtures for floodlite tests."""
from __future__ import annotations
import torch
import pytest


@pytest.fixture
def tiny_batch():
    """A single 3-channel 256x256 image and binary mask, on CPU."""
    x = torch.randn(1, 3, 256, 256)
    y = (torch.rand(1, 1, 256, 256) > 0.5).float()
    return x, y


@pytest.fixture
def cpu_device():
    return torch.device("cpu")
```

- [ ] **Step 5: Add pytest to requirements**

Modify `code/requirements.txt`, add at the bottom (after `onnxruntime`):

```
pytest>=7.0
```

- [ ] **Step 6: Verify pytest discovers the fixtures**

```bash
cd code
pip install pytest
python -m pytest tests/ --collect-only
```

Expected output: `0 tests collected` (no tests yet, but no errors).

- [ ] **Step 7: Commit**

```bash
git add code/.gitignore code/tests/ code/requirements.txt
git commit -m "chore: add git, .gitignore, pytest scaffolding"
```

---

### Task A1: Fix INT8 PTQ CPU bug

**Files:**
- Modify: `code/floodlite/quantize.py`
- Test: `code/tests/test_quantize.py` (NEW)

**Why:** Existing call path leaves model on CUDA when `prepare()` runs, producing the runtime error "Could not run 'quantized::conv2d.new' with arguments from the 'CUDA' backend". Fix is to force-move both the model and the calibration inputs to CPU before `prepare()`, and to unwrap dataloader tensors that may live on GPU.

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_quantize.py`:

```python
"""Tests for INT8 post-training quantization."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from floodlite.models import make_student
from floodlite.quantize import quantize_int8, model_size_mb


@pytest.fixture
def cuda_loader():
    """A loader whose tensors start on CUDA (when available) — simulates the failing path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 256, 256, device=device)
    y = (torch.rand(2, 1, 256, 256, device=device) > 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_quantize_int8_runs_on_cpu_even_when_loader_is_cuda(cuda_loader):
    """Repro for the Fold-0 bug: calibration loader on CUDA must not crash quantization."""
    model = make_student("mobilenetv3_small")
    qmodel = quantize_int8(model, cuda_loader, n_calib_batches=1, device="cpu")
    # Forward pass on CPU must succeed
    with torch.no_grad():
        out = qmodel(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 256, 256)


def test_quantize_int8_reduces_size():
    """INT8 model should be smaller than FP32 baseline."""
    fp32 = make_student("mobilenetv3_small")
    x = torch.randn(2, 3, 256, 256)
    y = torch.zeros(2, 1, 256, 256)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    int8 = quantize_int8(fp32, loader, n_calib_batches=1)
    assert model_size_mb(int8) < model_size_mb(fp32) * 0.6  # at least 40% smaller
```

- [ ] **Step 2: Run test and confirm it fails**

```bash
cd code
python -m pytest tests/test_quantize.py -v
```

Expected: FAIL with the same CUDA-backend error message from your Fold-0 run.

- [ ] **Step 3: Fix `floodlite/quantize.py`**

Replace the entire file content with:

```python
"""Post-training INT8 quantization (PyTorch static quantization).

INT8 PTQ in torch.ao.quantization requires the model and calibration inputs
to live on CPU; running with CUDA tensors raises:

    NotImplementedError: Could not run 'quantized::conv2d.new' with
    arguments from the 'CUDA' backend.

This module forces both the model and the calibration tensors to CPU before
`prepare()`, regardless of where the calibration loader yields them.
"""
from __future__ import annotations
from pathlib import Path
import copy
import io

import torch
import torch.nn as nn


def quantize_int8(model: nn.Module, calib_loader, *, n_calib_batches: int = 25,
                  device: str = "cpu") -> nn.Module:
    """Apply post-training static INT8 quantization with calibration.

    The calibration set should be drawn from the training distribution.
    Both ``model`` and every batch yielded by ``calib_loader`` are moved to
    CPU before quantization runs — this is mandatory for torch.ao.quantization.
    """
    if device != "cpu":
        # Quietly enforce CPU; INT8 PTQ does not support CUDA.
        device = "cpu"

    model = copy.deepcopy(model).to(device).eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()  # force CPU + FP32 regardless of loader
            model(x)
            if i + 1 >= n_calib_batches:
                break

    torch.ao.quantization.convert(model, inplace=True)
    return model


def save_quantized(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def model_size_mb(model: nn.Module) -> float:
    """Estimate on-disk model size in MB by serialising to a buffer."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd code
python -m pytest tests/test_quantize.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/floodlite/quantize.py code/tests/test_quantize.py
git commit -m "fix(quantize): force CPU device for INT8 PTQ regardless of loader device"
```

---

### Task A2: Add MobileNetV2-UNet baseline

**Files:**
- Modify: `code/floodlite/models.py`
- Test: `code/tests/test_models.py` (NEW)

**Why:** Reviewers will ask "why didn't you compare to a published lightweight backbone besides the teacher?" MobileNetV2 is the canonical answer; smp ships it natively.

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_models.py`:

```python
"""Tests for model factory functions."""
from __future__ import annotations
import torch
from floodlite.models import (
    make_teacher, make_student, make_baseline_mobilenetv2,
    count_params, STUDENT_BACKBONES,
)


def test_baseline_mobilenetv2_returns_unet_with_correct_output_shape():
    model = make_baseline_mobilenetv2()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 256, 256)


def test_baseline_param_count_smaller_than_teacher():
    teacher = make_teacher()
    baseline = make_baseline_mobilenetv2()
    assert count_params(baseline) < count_params(teacher)


def test_all_three_students_build_and_forward():
    for name in STUDENT_BACKBONES:
        model = make_student(name)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256))
        assert out.shape == (1, 1, 256, 256), f"shape mismatch for {name}"
```

- [ ] **Step 2: Run test and confirm `make_baseline_mobilenetv2` fails to import**

```bash
cd code
python -m pytest tests/test_models.py -v
```

Expected: FAIL with `ImportError: cannot import name 'make_baseline_mobilenetv2'`.

- [ ] **Step 3: Add the factory in `floodlite/models.py`**

After the existing `make_student` function (around line 44), add:

```python
def make_baseline_mobilenetv2(num_classes: int = 1) -> nn.Module:
    """UNet + MobileNetV2 baseline.

    A published lightweight encoder trained from ImageNet pretraining;
    used as an off-the-shelf reference point against the KD students.
    No KD is applied — task loss only.
    """
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd code
python -m pytest tests/test_models.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/floodlite/models.py code/tests/test_models.py
git commit -m "feat(models): add MobileNetV2-UNet lightweight baseline"
```

---

### Task A3: Add Sen1Floods11 RGB OOD loader

**Files:**
- Modify: `code/floodlite/data.py`
- Test: `code/tests/test_data_ood.py` (NEW)

**Why:** Spec §3 OOD requires ~250 chips of Sen1Floods11 RGB visualizations evaluated against FSSD-trained models. The existing `make_loaders` is FSSD-specific; we need a separate eval-only loader that re-uses the same val transforms.

- [ ] **Step 1: Inspect the existing transforms to reuse them**

```bash
grep -n "Normalize\|Resize" code/floodlite/data.py
```

Expected: see `A.Resize(img_size, img_size)` and `A.Normalize(mean=...)` already inside `get_transforms`.

- [ ] **Step 2: Write the failing test**

Create `code/tests/test_data_ood.py`:

```python
"""Tests for the Sen1Floods11 RGB OOD loader."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import pytest

from floodlite.data import make_sen1floods11_loader


@pytest.fixture
def fake_s1f11_root(tmp_path: Path) -> Path:
    """Build a fake Sen1Floods11-style folder with 4 RGB chips and binary masks."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "labels"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        msk = (rng.integers(0, 2, (512, 512), dtype=np.uint8) * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"chip_{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_dir / f"chip_{i:03d}.png"), msk)
    return tmp_path


def test_make_sen1floods11_loader_yields_correct_tensor_shape(fake_s1f11_root):
    loader = make_sen1floods11_loader(fake_s1f11_root, n_chips=4, batch_size=2)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 256, 256)
    assert y.shape == (2, 1, 256, 256)
    # Mask should be 0/1 binary
    unique_vals = set(y.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_make_sen1floods11_loader_caps_at_n_chips(fake_s1f11_root):
    loader = make_sen1floods11_loader(fake_s1f11_root, n_chips=2, batch_size=1)
    n = sum(1 for _ in loader)
    assert n == 2
```

- [ ] **Step 3: Run test and confirm it fails**

```bash
cd code
python -m pytest tests/test_data_ood.py -v
```

Expected: FAIL with `ImportError: cannot import name 'make_sen1floods11_loader'`.

- [ ] **Step 4: Implement `make_sen1floods11_loader` in `floodlite/data.py`**

At the bottom of `floodlite/data.py`, append:

```python
def make_sen1floods11_loader(root: str | Path, *, n_chips: int = 250,
                              batch_size: int = 8, img_size: int = 256,
                              num_workers: int = 2):
    """Build a DataLoader for the Sen1Floods11 RGB OOD evaluation set.

    Expects ``<root>/images/*.png`` and ``<root>/labels/*.png`` with matching
    stems. Same val-style transforms as FSSD (no augmentation, ImageNet
    normalization, 256x256). Eval-only — no shuffle.
    """
    root = Path(root)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    pairs = gather_pairs(img_dir, lbl_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No Sen1Floods11 image/label pairs found under {root!s}. "
            "Expected images/ and labels/ subdirs with matching stems."
        )
    pairs = pairs[:n_chips]
    ds = FSSD(pairs, img_size=img_size, train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd code
python -m pytest tests/test_data_ood.py -v
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add code/floodlite/data.py code/tests/test_data_ood.py
git commit -m "feat(data): add Sen1Floods11 RGB OOD loader"
```

---

### Task A4: Add ONNX Runtime CPU benchmark function

**Files:**
- Modify: `code/floodlite/benchmark.py`
- Test: `code/tests/test_benchmark.py` (NEW)

**Why:** AWS Graviton + WASM both run via onnxruntime; we need a single function whose output schema matches `benchmark_latency` so downstream tables aggregate cleanly.

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_benchmark.py`:

```python
"""Tests for benchmark utilities."""
from __future__ import annotations
from pathlib import Path
import torch
import pytest

from floodlite.models import make_student
from floodlite.export import export_onnx
from floodlite.benchmark import benchmark_latency, benchmark_onnxruntime


def test_benchmark_latency_returns_expected_keys():
    model = make_student("mobilenetv3_small")
    out = benchmark_latency(model, n_warm=2, n_iter=4, device="cpu")
    assert set(out.keys()) == {"p50_ms", "p95_ms", "mean_ms", "std_ms", "fps"}
    assert out["p50_ms"] > 0


def test_benchmark_onnxruntime_returns_same_schema(tmp_path: Path):
    model = make_student("mobilenetv3_small")
    onnx_path = tmp_path / "m.onnx"
    export_onnx(model, onnx_path, opset=17)
    out = benchmark_onnxruntime(onnx_path, n_warm=2, n_iter=4)
    assert set(out.keys()) == {"p50_ms", "p95_ms", "mean_ms", "std_ms", "fps"}
    assert out["p50_ms"] > 0
```

- [ ] **Step 2: Run test, confirm `benchmark_onnxruntime` is missing**

```bash
cd code
python -m pytest tests/test_benchmark.py -v
```

Expected: 1 PASS (existing), 1 FAIL with `ImportError: cannot import name 'benchmark_onnxruntime'`.

- [ ] **Step 3: Add the function to `floodlite/benchmark.py`**

Append to the end of the file:

```python
def benchmark_onnxruntime(onnx_path, *, img_size: int = 256, n_warm: int = 20,
                          n_iter: int = 200,
                          providers: list[str] | None = None) -> dict[str, float]:
    """Measure latency of an ONNX model under onnxruntime.

    Used for AWS Graviton (ARM CPU) and WASM targets. Output schema matches
    benchmark_latency so downstream code can pool both.
    """
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=providers or ["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {in_name: x})
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "std_ms":  float(arr.std()),
        "fps":     float(1000.0 / np.percentile(arr, 50)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd code
python -m pytest tests/test_benchmark.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/floodlite/benchmark.py code/tests/test_benchmark.py
git commit -m "feat(benchmark): add benchmark_onnxruntime for ARM/WASM targets"
```

---

### Task A5: Update training script defaults and add baseline flag

**Files:**
- Modify: `code/scripts/run_full_experiment.py`

**Why:** Defaults need to align with the spec (35 epochs, 3 folds, MobileNetV2 baseline included). No new test — this is a CLI entrypoint, exercised end-to-end in Phase B.

- [ ] **Step 1: Update default args**

In `code/scripts/run_full_experiment.py`, locate:

```python
    ap.add_argument("--folds", type=int, default=1, help="Number of folds to run starting from --fold")
    ap.add_argument("--epochs_teacher", type=int, default=50)
    ap.add_argument("--epochs_student", type=int, default=50)
```

Replace with:

```python
    ap.add_argument("--folds", type=int, default=3, help="Number of folds to run starting from --fold")
    ap.add_argument("--epochs_teacher", type=int, default=35)
    ap.add_argument("--epochs_student", type=int, default=35)
    ap.add_argument("--include_baseline", action="store_true",
                    help="Also train an off-the-shelf MobileNetV2-UNet baseline (no KD).")
```

- [ ] **Step 2: Add the baseline import and training call**

At the top of `run_full_experiment.py`, change:

```python
from floodlite.models import make_teacher, make_student, count_params, estimate_flops, STUDENT_BACKBONES
```

to:

```python
from floodlite.models import (
    make_teacher, make_student, make_baseline_mobilenetv2,
    count_params, estimate_flops, STUDENT_BACKBONES,
)
```

Then, inside the `for fold in range(...)` loop, immediately after the `train_teacher(...)` call and the `teacher_metrics = evaluate(...)` line, add:

```python
        # Optional off-the-shelf lightweight baseline (no KD, task loss only)
        baseline_metrics = None
        if args.include_baseline:
            ckpt_b = Path(args.out_dir) / f"baseline_mobilenetv2_fold{fold}.pt"
            from floodlite.train import train_teacher as train_taskonly
            baseline = make_baseline_mobilenetv2().to(device)
            train_taskonly(baseline, tr, va, epochs=args.epochs_student,
                           device=device, ckpt_path=ckpt_b)
            baseline.load_state_dict(torch.load(ckpt_b, map_location=device))
            baseline_metrics = evaluate(baseline, va, device)
            print(f"Baseline (MobileNetV2): {baseline_metrics}")
        fold_results = {"teacher": teacher_metrics, "baseline_mobilenetv2": baseline_metrics, "students": {}}
```

(Replace the existing `fold_results = {"teacher": teacher_metrics, "students": {}}` line with the new dict-with-baseline above.)

- [ ] **Step 3: Smoke-test the script with help**

```bash
cd code
python scripts/run_full_experiment.py --help
```

Expected: argparse help output that includes `--include_baseline`, `--folds 3`, and `--epochs_teacher 35` defaults.

- [ ] **Step 4: Commit**

```bash
git add code/scripts/run_full_experiment.py
git commit -m "feat(scripts): add baseline flag, default to 3 folds × 35 epochs"
```

---

### Task A6: Migrate `torch.cuda.amp` → `torch.amp` in train.py

**Files:**
- Modify: `code/floodlite/train.py`

**Why:** `CLAUDE.md` flags this as "deprecated in newer PyTorch — verify on the target Kaggle/Colab Torch version first." Kaggle now ships Torch 2.4+; the deprecation warning clutters logs. Low risk; identical numerics.

- [ ] **Step 1: Replace deprecated calls**

In `code/floodlite/train.py`, locate every occurrence of:

```python
torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
```

Replace with:

```python
torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
```

And every occurrence of:

```python
with torch.cuda.amp.autocast(enabled=(device == "cuda")):
```

Replace with:

```python
with torch.amp.autocast("cuda", enabled=(device == "cuda")):
```

There are 2 occurrences of each (one in `train_teacher`, one in `train_student_kd`).

- [ ] **Step 2: Smoke-test by importing the module**

```bash
cd code
python -c "from floodlite.train import train_teacher, train_student_kd; print('ok')"
```

Expected: prints `ok` with no DeprecationWarning.

- [ ] **Step 3: Commit**

```bash
git add code/floodlite/train.py
git commit -m "chore(train): migrate torch.cuda.amp -> torch.amp"
```

---

### Task A7: Smoke test — full single-config end-to-end on Fold 0

**Files:** none (executes existing code paths)

**Why:** Before launching the 3-fold sweep, verify all five code changes compose correctly. ~25 minutes on a Kaggle T4.

- [ ] **Step 1: Run the smoke pipeline**

```bash
cd code
python scripts/run_full_experiment.py \
    --data_root /kaggle/input/flood-semantic-segmentation-dataset \
    --fold 0 --folds 1 \
    --epochs_teacher 2 --epochs_student 2 \
    --batch_size 4 \
    --include_baseline \
    --out_dir runs/smoke
```

Expected: console shows teacher → baseline → 3 students × 4 KD configs → quantization (no CUDA error this time) → latency. No exceptions.

- [ ] **Step 2: Sanity-check `runs/smoke/results.json`**

```bash
cd code
python -c "import json; r = json.load(open('runs/smoke/results.json')); print(json.dumps(r, indent=2))" | head -80
```

Expected: valid JSON with `fold_0` containing keys `teacher`, `baseline_mobilenetv2`, `students` (12 entries), `quantized` (3 entries with non-zero IoU), `latency` (3 entries).

- [ ] **Step 3: If quantization still fails, debug**

If you see "quantized::conv2d.new" again, run:

```bash
cd code
python -m pytest tests/test_quantize.py -v
```

If the unit test passes but the smoke run fails, the bug is in `run_full_experiment.py` — confirm the line `student_fp = make_student(sname).cpu()` is unchanged (it explicitly moves to CPU; do not modify).

- [ ] **Step 4: Commit if any drift was needed**

```bash
git add -u code/
git diff --staged --stat
git commit -m "chore: smoke-test fixes" --allow-empty
```

---

## Phase B — Experiments (3–4 calendar days, mostly waiting)

### Task B1: 3-fold full sweep on Kaggle T4×2

**Files:** none (executes existing code)

**Why:** Produces the headline numbers for T1–T4.

- [ ] **Step 1: Spin up Kaggle notebook with T4×2 accelerator**

In the Kaggle UI, open `code/notebooks/floodlite_kaggle.ipynb`, attach the FSSD dataset, set Accelerator → GPU T4×2, set persistent runtime if you have Pro.

- [ ] **Step 2: Add a parameter cell at the top**

```python
DATA_ROOT = "/kaggle/input/flood-semantic-segmentation-dataset"
FOLDS = 3
EPOCHS = 35
BATCH = 8
OUT_DIR = "/kaggle/working/runs"
SEED = 42
```

- [ ] **Step 3: Replace the "Run All" cells with a CLI driver call**

```python
import subprocess, sys
subprocess.run([
    sys.executable, "scripts/run_full_experiment.py",
    "--data_root", DATA_ROOT,
    "--fold", "0", "--folds", str(FOLDS),
    "--epochs_teacher", str(EPOCHS),
    "--epochs_student", str(EPOCHS),
    "--batch_size", str(BATCH),
    "--include_baseline",
    "--out_dir", OUT_DIR,
], check=True)
```

- [ ] **Step 4: Run the sweep**

Wall-time estimate: 3 folds × (1 teacher + 1 baseline + 3 students × 4 KD) = 14 runs × 35 ep × 0.7 min/ep ≈ 343 min ≈ 5.7 hr per fold ≈ 17 hr total. Spread across 2 Kaggle accounts (free tier = 30 hr/week each) or 2 separate sessions; 4 calendar days of evenings.

- [ ] **Step 5: Download `runs/results.json` to local repo**

```bash
mkdir -p code/runs/3fold
# After each fold completes: download from Kaggle UI to:
#   code/runs/3fold/fold0_results.json
#   code/runs/3fold/fold1_results.json
#   code/runs/3fold/fold2_results.json
```

- [ ] **Step 6: Aggregate**

Create `code/scripts/aggregate_results.py`:

```python
"""Aggregate per-fold results.json files into one summary."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_fold = []
    for p in args.inputs:
        per_fold.append(json.loads(Path(p).read_text()))

    # Collect all (config_key, metric_dict) pairs across folds
    keys = set()
    for f in per_fold:
        for fold_id, fold in f.items():
            keys.add(("teacher", fold_id))
            if fold.get("baseline_mobilenetv2"):
                keys.add(("baseline_mobilenetv2", fold_id))
            for s in fold.get("students", {}):
                keys.add((s, fold_id))

    # Mean ± std across folds
    summary = {}
    for f in per_fold:
        for fold_id, fold in f.items():
            for cfg in ["teacher", "baseline_mobilenetv2"] + list(fold.get("students", {}).keys()):
                if cfg == "baseline_mobilenetv2" and not fold.get(cfg):
                    continue
                m = fold[cfg] if cfg in ("teacher", "baseline_mobilenetv2") else fold["students"][cfg]
                if m is None:
                    continue
                summary.setdefault(cfg, []).append(m)

    final = {}
    for cfg, runs in summary.items():
        agg = {}
        for k in ("accuracy", "precision", "recall", "f1", "iou"):
            vals = [r[k] for r in runs]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        final[cfg] = agg

    Path(args.out).write_text(json.dumps(final, indent=2))
    print(f"Wrote {args.out} with {len(final)} configs.")


if __name__ == "__main__":
    main()
```

Then run:

```bash
cd code
python scripts/aggregate_results.py \
    --inputs runs/3fold/fold0_results.json runs/3fold/fold1_results.json runs/3fold/fold2_results.json \
    --out runs/3fold/summary.json
```

Expected: prints `Wrote runs/3fold/summary.json with 14 configs.`

- [ ] **Step 7: Commit aggregation script + results**

```bash
cd /Users/ibrahim/Desktop/personal/Flood_segmentation_model
git add code/scripts/aggregate_results.py code/runs/3fold/
git commit -m "feat(experiments): 3-fold sweep results + aggregation"
```

---

### Task B2: Quantize the three deployable students × 3 folds

**Files:** Create `code/scripts/quantize_all.py`

- [ ] **Step 1: Write the script**

Create `code/scripts/quantize_all.py`:

```python
"""Quantize all (student × fold × no-KD) checkpoints to INT8 and report IoU drop."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from floodlite.data import make_loaders
from floodlite.models import make_student, STUDENT_BACKBONES
from floodlite.quantize import quantize_int8, model_size_mb
from floodlite.metrics import compute_metrics


def evaluate_cpu(model, loader):
    model.eval()
    ms = []
    with torch.no_grad():
        for x, y in loader:
            ms.append(compute_metrics(model(x), y))
    n = len(ms)
    return {k: sum(getattr(m, k) for m in ms) / n
            for k in ("accuracy", "precision", "recall", "f1", "iou")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="Dir holding {sname}_none_fold{N}.pt")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs/3fold/quantized.json")
    args = ap.parse_args()

    results = {}
    for fold in args.folds:
        tr, va = make_loaders(args.data_root, fold=fold, batch_size=8)
        results[f"fold_{fold}"] = {}
        for sname in STUDENT_BACKBONES:
            ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{fold}.pt"
            if not ckpt.exists():
                print(f"SKIP {ckpt} (missing)")
                continue
            fp = make_student(sname).cpu()
            fp.load_state_dict(torch.load(ckpt, map_location="cpu"))
            fp_iou = evaluate_cpu(fp, va)["iou"]
            try:
                q = quantize_int8(fp, tr)
                q_iou = evaluate_cpu(q, va)["iou"]
                results[f"fold_{fold}"][sname] = {
                    "fp32_iou": fp_iou,
                    "int8_iou": q_iou,
                    "delta_iou": q_iou - fp_iou,
                    "fp32_size_mb": model_size_mb(fp),
                    "int8_size_mb": model_size_mb(q),
                }
                print(f"fold {fold} {sname}: {fp_iou:.4f} -> {q_iou:.4f}")
            except Exception as e:
                results[f"fold_{fold}"][sname] = {"error": str(e), "fp32_iou": fp_iou}
                print(f"fold {fold} {sname}: QUANTIZATION FAILED: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on Kaggle/local with downloaded checkpoints**

```bash
cd code
python scripts/quantize_all.py \
    --data_root /path/to/FSSD \
    --ckpt_dir runs/3fold \
    --out runs/3fold/quantized.json
```

Expected: 3 folds × 3 students = 9 entries; each prints `fold N name: 0.93xx -> 0.92xx`. If MobileViT-XXS errors, that's expected — flag in T4.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/quantize_all.py code/runs/3fold/quantized.json
git commit -m "feat(experiments): INT8 PTQ across all 3 students × 3 folds"
```

---

### Task B3: Download and prep Sen1Floods11 RGB OOD chips

**Files:** Create `code/scripts/prepare_sen1floods11_rgb.py`

- [ ] **Step 1: Acquire chips**

```bash
# Via Hugging Face / public Sen1Floods11 release. The RGB visualizations are
# under HandLabeled/S2Hand/ and corresponding masks under HandLabeled/LabelHand/.
# Pick a single subdirectory and 250 chips (random sample, seeded).
mkdir -p code/data/sen1floods11_rgb
# Manual download from https://github.com/cloudtostreet/Sen1Floods11
# OR: pip install sen1floods11; from the package's data utility.
```

- [ ] **Step 2: Write the prep script**

Create `code/scripts/prepare_sen1floods11_rgb.py`:

```python
"""Sample 250 chips from a Sen1Floods11 release into images/ + labels/ folders."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import random
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_images", required=True, help="Dir of *.tif or *.png S2 RGB chips")
    ap.add_argument("--src_labels", required=True, help="Dir of *.tif or *.png label masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src_imgs = sorted(Path(args.src_images).iterdir())
    paired = []
    for ip in src_imgs:
        if ip.suffix.lower() not in (".png", ".jpg", ".tif", ".tiff"):
            continue
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            mp = Path(args.src_labels) / (ip.stem + ext)
            if mp.exists():
                paired.append((ip, mp))
                break
    print(f"Found {len(paired)} paired chips.")
    sampled = random.sample(paired, min(args.n, len(paired)))

    out_imgs = Path(args.out) / "images"
    out_lbls = Path(args.out) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    for ip, mp in sampled:
        # Convert to PNG, resize/normalize-on-load handled by FSSD transforms.
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(out_imgs / (ip.stem + ".png")), img)
        msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        msk = (msk > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_lbls / (mp.stem + ".png")), msk)
    print(f"Wrote {len(sampled)} pairs to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the prep**

```bash
cd code
python scripts/prepare_sen1floods11_rgb.py \
    --src_images /path/to/Sen1Floods11/HandLabeled/S2Hand \
    --src_labels /path/to/Sen1Floods11/HandLabeled/LabelHand \
    --out data/sen1floods11_rgb --n 250
```

Expected: prints `Wrote 250 pairs to data/sen1floods11_rgb`.

- [ ] **Step 4: Sanity-check via the loader**

```bash
cd code
python -c "
from floodlite.data import make_sen1floods11_loader
loader = make_sen1floods11_loader('data/sen1floods11_rgb', n_chips=250)
x, y = next(iter(loader))
print('shape:', x.shape, 'mask range:', y.min().item(), y.max().item())
"
```

Expected: `shape: torch.Size([8, 3, 256, 256]) mask range: 0.0 1.0`

- [ ] **Step 5: Commit (script + small README, NOT the data)**

```bash
git add code/scripts/prepare_sen1floods11_rgb.py
git commit -m "feat(experiments): Sen1Floods11 RGB OOD prep script"
```

---

### Task B4: Evaluate all final models on Sen1Floods11 OOD

**Files:** Create `code/scripts/eval_ood.py`

- [ ] **Step 1: Write the script**

Create `code/scripts/eval_ood.py`:

```python
"""Evaluate FSSD-trained models on Sen1Floods11 RGB chips (no fine-tuning)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from floodlite.data import make_sen1floods11_loader
from floodlite.models import make_teacher, make_baseline_mobilenetv2, make_student, STUDENT_BACKBONES
from floodlite.metrics import compute_metrics


def evaluate(model, loader, device):
    model.eval()
    ms = []
    with torch.no_grad():
        for x, y in loader:
            ms.append(compute_metrics(model(x.to(device)), y.to(device)))
    n = len(ms)
    return {k: sum(getattr(m, k) for m in ms) / n
            for k in ("accuracy", "precision", "recall", "f1", "iou")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--fold", type=int, default=0, help="Which fold's checkpoint to evaluate")
    ap.add_argument("--best_kd", default="none",
                    choices=["none", "resp", "feat", "comb"],
                    help="Which KD config to consider for each student")
    ap.add_argument("--out", default="runs/3fold/ood_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_sen1floods11_loader(args.ood_root, n_chips=250, batch_size=8)
    out = {}

    teacher = make_teacher().to(device)
    teacher.load_state_dict(torch.load(Path(args.ckpt_dir) / f"teacher_fold{args.fold}.pt", map_location=device))
    out["teacher"] = evaluate(teacher, loader, device)
    print(f"teacher: {out['teacher']}")

    bp = Path(args.ckpt_dir) / f"baseline_mobilenetv2_fold{args.fold}.pt"
    if bp.exists():
        baseline = make_baseline_mobilenetv2().to(device)
        baseline.load_state_dict(torch.load(bp, map_location=device))
        out["baseline_mobilenetv2"] = evaluate(baseline, loader, device)
        print(f"baseline: {out['baseline_mobilenetv2']}")

    for sname in STUDENT_BACKBONES:
        ckpt = Path(args.ckpt_dir) / f"{sname}_{args.best_kd}_fold{args.fold}.pt"
        if not ckpt.exists():
            print(f"SKIP {ckpt}")
            continue
        s = make_student(sname).to(device)
        s.load_state_dict(torch.load(ckpt, map_location=device))
        out[f"{sname}_{args.best_kd}"] = evaluate(s, loader, device)
        print(f"{sname}_{args.best_kd}: {out[f'{sname}_{args.best_kd}']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run for the per-student best KD config**

For each student, identify its best-mean-IoU KD config from `runs/3fold/summary.json` and run:

```bash
cd code
# Adjust --best_kd per student or run multiple times
python scripts/eval_ood.py \
    --ood_root data/sen1floods11_rgb \
    --ckpt_dir runs/3fold \
    --fold 0 \
    --best_kd none \
    --out runs/3fold/ood_results_none.json
```

Expected: prints IoU for teacher, baseline, and each student. Numbers will be lower than FSSD val (probably 0.40–0.70) — that IS the finding to report.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/eval_ood.py code/runs/3fold/ood_results*.json
git commit -m "feat(experiments): OOD evaluation on Sen1Floods11 RGB"
```

---

### Task B5: Statistical tests (Wilcoxon + paired pixel-bootstrap)

**Files:** Create `code/scripts/stats.py`

- [ ] **Step 1: Write the script**

Create `code/scripts/stats.py`:

```python
"""Statistical comparisons across folds and within fold-0 pixels.

Outputs:
  - 3-fold paired Wilcoxon for: no-KD vs comb-KD (per student); teacher vs best.
  - Fold-0 paired pixel-bootstrap (n=1000 resamples) for the same comparisons.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon

from floodlite.data import make_loaders
from floodlite.models import make_teacher, make_student, STUDENT_BACKBONES
from floodlite.metrics import compute_metrics


def per_image_iou(model, loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            for i in range(logits.size(0)):
                m = compute_metrics(logits[i:i+1], y[i:i+1])
                ious.append(m.iou)
    return np.array(ious)


def bootstrap_p(deltas, n_boot=1000, seed=42):
    """Paired bootstrap p-value: P(mean(delta_resample) <= 0)."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.array([rng.choice(deltas, n, replace=True).mean() for _ in range(n_boot)])
    return float((means <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="runs/3fold/summary.json")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", default="runs/3fold/stats.json")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())

    # ----- 3-fold Wilcoxon -----
    wilc = {}
    for sname in STUDENT_BACKBONES:
        none_iou = []
        comb_iou = []
        # Reconstruct per-fold IoU from per-fold result files (mean=single fold)
        for fold in (0, 1, 2):
            f = json.loads(Path(args.ckpt_dir).parent.joinpath(
                f"3fold/fold{fold}_results.json").read_text())
            none_iou.append(f[f"fold_{fold}"]["students"][f"{sname}_none"]["iou"])
            comb_iou.append(f[f"fold_{fold}"]["students"][f"{sname}_comb"]["iou"])
        try:
            stat, p = wilcoxon(none_iou, comb_iou)
            wilc[f"{sname}_none_vs_comb"] = {
                "n": 3, "p_value": float(p), "stat": float(stat),
                "mean_delta": float(np.mean(np.array(none_iou) - np.array(comb_iou))),
                "note": "n=3 -> minimum two-sided p ≈ 0.25 (low power)",
            }
        except ValueError as e:
            wilc[f"{sname}_none_vs_comb"] = {"n": 3, "error": str(e)}

    # ----- Fold-0 paired pixel-bootstrap -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va = make_loaders(args.data_root, fold=0, batch_size=8)
    boot = {}
    for sname in STUDENT_BACKBONES:
        s_none = make_student(sname).to(device)
        s_none.load_state_dict(torch.load(
            Path(args.ckpt_dir) / f"{sname}_none_fold0.pt", map_location=device))
        s_comb = make_student(sname).to(device)
        s_comb.load_state_dict(torch.load(
            Path(args.ckpt_dir) / f"{sname}_comb_fold0.pt", map_location=device))
        none_per = per_image_iou(s_none, va, device)
        comb_per = per_image_iou(s_comb, va, device)
        deltas = none_per - comb_per
        boot[f"{sname}_none_vs_comb"] = {
            "n": len(deltas),
            "mean_delta": float(deltas.mean()),
            "p_one_sided_no_kd_better": bootstrap_p(deltas),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"wilcoxon_3fold": wilc, "bootstrap_fold0": boot}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
cd code
python scripts/stats.py \
    --summary runs/3fold/summary.json \
    --data_root /path/to/FSSD \
    --ckpt_dir runs/3fold \
    --out runs/3fold/stats.json
```

Expected: writes JSON with three Wilcoxon entries (one per student) and three bootstrap entries.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/stats.py code/runs/3fold/stats.json
git commit -m "feat(experiments): Wilcoxon + paired pixel-bootstrap stats"
```

---

## Phase C — Per-platform latency (1 week)

### Task C1: ONNX export + parity check

**Files:** Create `code/scripts/export_all.py`

- [ ] **Step 1: Write the export script**

```python
"""Export all final models (FP32 + INT8 students) to ONNX, with parity check."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from floodlite.models import make_teacher, make_baseline_mobilenetv2, make_student, STUDENT_BACKBONES
from floodlite.export import export_onnx
from floodlite.quantize import quantize_int8
from floodlite.data import make_loaders


def parity(model, onnx_path, atol=1e-3):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y_pt = model.cpu().eval()(x).numpy()
    y_ort = sess.run(None, {"input": x.numpy().astype(np.float32)})[0]
    return float(np.max(np.abs(y_pt - y_ort)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_root", required=True, help="FSSD root for INT8 calibration")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out_dir", default="exports/")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    parity_log = {}

    # Teacher + baseline
    for name, factory in [("teacher", make_teacher), ("baseline_mobilenetv2", make_baseline_mobilenetv2)]:
        ckpt = Path(args.ckpt_dir) / f"{name}_fold{args.fold}.pt"
        if not ckpt.exists():
            continue
        m = factory()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        op = Path(args.out_dir) / f"{name}.onnx"
        export_onnx(m, op, opset=17)
        parity_log[name] = parity(m, op)

    # Students FP32 (no-KD checkpoints)
    for sname in STUDENT_BACKBONES:
        ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{args.fold}.pt"
        if not ckpt.exists():
            continue
        m = make_student(sname)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        op = Path(args.out_dir) / f"{sname}_fp32.onnx"
        export_onnx(m, op, opset=17)
        parity_log[f"{sname}_fp32"] = parity(m, op)

    # Students INT8 (export the *quantized* graph)
    tr, _ = make_loaders(args.data_root, fold=args.fold, batch_size=8)
    for sname in STUDENT_BACKBONES:
        ckpt = Path(args.ckpt_dir) / f"{sname}_none_fold{args.fold}.pt"
        if not ckpt.exists():
            continue
        m = make_student(sname).cpu()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        try:
            q = quantize_int8(m, tr)
            op = Path(args.out_dir) / f"{sname}_int8.onnx"
            # PyTorch quantized models export via the QAT path; if it fails,
            # use ONNX dynamic quantization as a fallback.
            try:
                export_onnx(q, op, opset=17)
            except Exception:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                fp32_path = Path(args.out_dir) / f"{sname}_fp32.onnx"
                quantize_dynamic(str(fp32_path), str(op), weight_type=QuantType.QInt8)
            parity_log[f"{sname}_int8"] = "exported (no parity check for INT8)"
        except Exception as e:
            parity_log[f"{sname}_int8"] = f"FAILED: {e}"

    Path(args.out_dir + "/parity.json").write_text(json.dumps(parity_log, indent=2))
    print("Parity:", parity_log)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on M2 Pro (CPU)**

```bash
cd code
python scripts/export_all.py \
    --ckpt_dir runs/3fold \
    --data_root data/FSSD \
    --fold 0 --out_dir exports/
```

Expected: `exports/{teacher,baseline_mobilenetv2,3 students}.onnx` + INT8 variants. Parity values < 1e-3 for FP32 entries.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/export_all.py code/exports/parity.json
# Do NOT commit the .onnx files (.gitignore handles it)
git commit -m "feat(export): ONNX export pipeline with parity check"
```

---

### Task C2: M2 Pro latency (PyTorch CPU + CoreML)

**Files:** Create `code/scripts/bench_m2.py`

- [ ] **Step 1: Write the benchmark**

```python
"""Latency on Apple M2 Pro: PyTorch CPU and CoreML."""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import torch

from floodlite.models import make_teacher, make_student, STUDENT_BACKBONES
from floodlite.benchmark import benchmark_latency, benchmark_onnxruntime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--exports", default="exports/")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="runs/3fold/lat_m2.json")
    args = ap.parse_args()

    out = {"host": platform.platform()}

    # PyTorch CPU
    t = make_teacher()
    t.load_state_dict(torch.load(Path(args.ckpt_dir) / f"teacher_fold{args.fold}.pt",
                                  map_location="cpu"))
    out["teacher_pt_cpu"] = benchmark_latency(t, device="cpu")
    for sname in STUDENT_BACKBONES:
        s = make_student(sname)
        s.load_state_dict(torch.load(
            Path(args.ckpt_dir) / f"{sname}_none_fold{args.fold}.pt", map_location="cpu"))
        out[f"{sname}_pt_cpu_fp32"] = benchmark_latency(s, device="cpu")

    # ONNX Runtime CPU (M2 native)
    for name in ["teacher", "baseline_mobilenetv2"] + [f"{s}_fp32" for s in STUDENT_BACKBONES] + [f"{s}_int8" for s in STUDENT_BACKBONES]:
        op = Path(args.exports) / f"{name}.onnx"
        if not op.exists():
            continue
        out[f"{name}_ort_cpu"] = benchmark_onnxruntime(op)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on M2 Pro**

```bash
cd code
python scripts/bench_m2.py \
    --ckpt_dir runs/3fold \
    --exports exports \
    --fold 0 \
    --out runs/3fold/lat_m2.json
```

Expected: prints `Wrote runs/3fold/lat_m2.json` with ~10 entries.

- [ ] **Step 3 (optional): CoreML conversion + benchmark**

```bash
cd code
pip install coremltools
python -c "
from floodlite.export import export_coreml
for s in ('mobilenetv3_small', 'efficientnet_lite0', 'mobilevit_xxs'):
    try:
        export_coreml(f'exports/{s}_fp32.onnx', f'exports/{s}.mlmodel')
        print(f'{s}: OK')
    except Exception as e:
        print(f'{s}: FAIL: {e}')
"
```

If conversion succeeds, write a small CoreML latency probe (skipped here; coremltools can hit reliability issues with timm-prefixed encoders — if it fails, report PyTorch CPU only on M2).

- [ ] **Step 4: Commit**

```bash
git add code/scripts/bench_m2.py code/runs/3fold/lat_m2.json
git commit -m "feat(benchmark): M2 Pro PyTorch CPU + ONNX Runtime CPU latency"
```

---

### Task C3: AWS Graviton t4g.small ARM benchmark

**Files:** Create `code/scripts/bench_graviton.sh`

- [ ] **Step 1: Provision the instance**

In AWS console: launch `t4g.small` (ARM64), Ubuntu 22.04, free-tier eligible. SSH in.

- [ ] **Step 2: Write the bootstrap script**

```bash
cat > code/scripts/bench_graviton.sh <<'BASH'
#!/usr/bin/env bash
# Run on AWS Graviton t4g.small after `git clone` and `scp exports/`
set -e
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy
python - <<'PY'
import json, time
import numpy as np
import onnxruntime as ort
results = {}
for name in ("mobilenetv3_small_fp32", "efficientnet_lite0_fp32", "mobilevit_xxs_fp32",
             "mobilenetv3_small_int8", "efficientnet_lite0_int8", "mobilevit_xxs_int8"):
    path = f"exports/{name}.onnx"
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, 256, 256).astype(np.float32)
        for _ in range(20): sess.run(None, {"input": x})
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {"input": x})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        results[name] = {"p50_ms": float(np.percentile(ts, 50)),
                         "p95_ms": float(np.percentile(ts, 95)),
                         "fps": float(1000.0/np.percentile(ts, 50))}
        print(name, results[name])
    except Exception as e:
        results[name] = {"error": str(e)}
print(json.dumps(results, indent=2))
PY
BASH
chmod +x code/scripts/bench_graviton.sh
```

- [ ] **Step 3: Run remotely**

```bash
# Locally
scp -r code/exports ubuntu@<graviton-ip>:~/exports
scp code/scripts/bench_graviton.sh ubuntu@<graviton-ip>:~/
# Remote
ssh ubuntu@<graviton-ip> 'bash ~/bench_graviton.sh' | tee runs/3fold/lat_graviton.json
```

Expected: per-model p50/p95/fps. INT8 should be ~2× faster than FP32 on ARM CPU.

- [ ] **Step 4: Commit**

```bash
git add code/scripts/bench_graviton.sh code/runs/3fold/lat_graviton.json
git commit -m "feat(benchmark): AWS Graviton ARM CPU latency"
```

---

### Task C4: Android Termux + TFLite latency

**Files:** Create `code/scripts/bench_android_termux.sh`

- [ ] **Step 1: TFLite conversion on host**

```bash
cd code
pip install onnx-tf tensorflow
python -c "
from floodlite.export import export_tflite
for s in ('mobilenetv3_small', 'efficientnet_lite0', 'mobilevit_xxs'):
    try:
        export_tflite(f'exports/{s}_fp32.onnx', f'exports/{s}.tflite')
        print(f'{s}: OK')
    except Exception as e:
        print(f'{s}: FAIL: {e}')
"
```

If onnx-tf chokes on a model (common for `mobilevit_xxs`), record it as "TFLite conversion unsupported" in T5; do not block on this.

- [ ] **Step 2: Write the on-phone benchmark**

```bash
cat > code/scripts/bench_android_termux.sh <<'BASH'
#!/data/data/com.termux/files/usr/bin/bash
# Run inside Termux on the Android phone after `pkg install python` and
# transferring exports/*.tflite via `scp` from the laptop.
set -e
pip install --upgrade pip
pip install tflite-runtime numpy
python <<'PY'
import json, time
import numpy as np
import tflite_runtime.interpreter as tflite

results = {}
for name in ("mobilenetv3_small", "efficientnet_lite0", "mobilevit_xxs"):
    path = f"{name}.tflite"
    try:
        i = tflite.Interpreter(path)
        i.allocate_tensors()
        in_d = i.get_input_details()[0]
        x = np.random.randn(*in_d["shape"]).astype(np.float32)
        for _ in range(20):
            i.set_tensor(in_d["index"], x); i.invoke()
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            i.set_tensor(in_d["index"], x); i.invoke()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        results[name] = {"p50_ms": float(np.percentile(ts, 50)),
                         "p95_ms": float(np.percentile(ts, 95)),
                         "fps": float(1000.0/np.percentile(ts, 50))}
    except Exception as e:
        results[name] = {"error": str(e)}
print(json.dumps(results, indent=2))
PY
BASH
chmod +x code/scripts/bench_android_termux.sh
```

- [ ] **Step 3: Transfer + run**

```bash
# Use Termux SSHd or scp via USB/Wi-Fi:
scp code/exports/*.tflite code/scripts/bench_android_termux.sh u@phone:~/
ssh u@phone 'bash ~/bench_android_termux.sh' | tee code/runs/3fold/lat_android.json
```

Note the phone model + Android version + chipset in the resulting JSON manually before committing (e.g., add a top-level `"device": "Pixel 7, Tensor G2, Android 14"` key).

- [ ] **Step 4: Commit**

```bash
git add code/scripts/bench_android_termux.sh code/runs/3fold/lat_android.json
git commit -m "feat(benchmark): Android Termux + TFLite latency"
```

---

### Task C5: Browser WASM (ONNX Runtime Web) latency

**Files:** Create `code/scripts/bench_wasm.html`

- [ ] **Step 1: Write the HTML harness**

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>FloodLite WASM bench</title></head>
<body><pre id="out">running…</pre>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js"></script>
<script>
const out = document.getElementById('out');
const log = (s) => { out.textContent += s + "\n"; };
async function bench(name) {
  const sess = await ort.InferenceSession.create(name, {executionProviders: ['wasm']});
  const x = new ort.Tensor('float32', new Float32Array(3*256*256), [1,3,256,256]);
  const feeds = {input: x};
  for (let i=0; i<20; i++) await sess.run(feeds);
  const ts = [];
  for (let i=0; i<200; i++) {
    const t0 = performance.now();
    await sess.run(feeds);
    ts.push(performance.now() - t0);
  }
  ts.sort((a,b)=>a-b);
  const p50 = ts[Math.floor(ts.length*0.5)];
  const p95 = ts[Math.floor(ts.length*0.95)];
  log(`${name}: p50=${p50.toFixed(2)}ms  p95=${p95.toFixed(2)}ms  fps=${(1000/p50).toFixed(1)}`);
}
(async () => {
  for (const m of ["mobilenetv3_small_fp32.onnx",
                   "efficientnet_lite0_fp32.onnx",
                   "mobilevit_xxs_fp32.onnx"]) {
    try { await bench(m); } catch(e) { log(`${m}: FAIL: ${e}`); }
  }
})();
</script></body></html>
```

Save as `code/scripts/bench_wasm.html` and copy the three FP32 ONNX files alongside it under `code/exports/wasm/`.

- [ ] **Step 2: Serve over HTTP and run in Chrome**

```bash
cd code/exports/wasm
cp ../mobilenetv3_small_fp32.onnx .
cp ../efficientnet_lite0_fp32.onnx .
cp ../mobilevit_xxs_fp32.onnx .
cp ../../scripts/bench_wasm.html index.html
python -m http.server 8080
# Open http://localhost:8080/index.html in Chrome on M2 Pro
```

Manually copy the printed lines into `code/runs/3fold/lat_wasm.json`:

```json
{
  "device": "Chrome 124 on M2 Pro, WASM SIMD",
  "mobilenetv3_small_fp32": {"p50_ms": ..., "p95_ms": ..., "fps": ...},
  "efficientnet_lite0_fp32": {"p50_ms": ..., "p95_ms": ..., "fps": ...},
  "mobilevit_xxs_fp32": {"p50_ms": ..., "p95_ms": ..., "fps": ...}
}
```

- [ ] **Step 3: Commit**

```bash
git add code/scripts/bench_wasm.html code/runs/3fold/lat_wasm.json
git commit -m "feat(benchmark): browser WASM ONNX Runtime Web latency"
```

---

## Phase D — Figures + manuscript writing (~6 working days)

### Task D1: Generate Pareto figure (F2)

**Files:** Create `code/scripts/make_figures.py`

- [ ] **Step 1: Write the figure script**

```python
"""Generate F2 (Pareto) and F4 (throughput) for the manuscript."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    summary = json.loads(Path("runs/3fold/summary.json").read_text())
    lat_m2 = json.loads(Path("runs/3fold/lat_m2.json").read_text())
    lat_android = json.loads(Path("runs/3fold/lat_android.json").read_text())

    # Static map of (config, params_M, gflops) — fill from your T1 table
    arch = {
        "teacher":              {"p": 6.6,  "g": 1.4},
        "baseline_mobilenetv2": {"p": 6.6,  "g": 0.9},
        "mobilenetv3_small_none": {"p": 1.5, "g": 0.4},
        "efficientnet_lite0_none": {"p": 4.4, "g": 1.0},
        "mobilevit_xxs_none":   {"p": 1.3, "g": 0.6},
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Params (M)", "FLOPs (G)", "M2 latency (ms)", "Android latency (ms)"]
    for ax, key, title in zip(axes, ["p", "g", "m2", "android"], titles):
        for cfg, a in arch.items():
            iou = summary[cfg]["iou"]["mean"]
            if key == "m2":
                x = lat_m2.get(f"{cfg}_pt_cpu", lat_m2.get(f"{cfg}_pt_cpu_fp32", {})).get("p50_ms")
            elif key == "android":
                short = cfg.replace("_none", "")
                x = lat_android.get(short, {}).get("p50_ms")
            else:
                x = a[key]
            if x is None:
                continue
            ax.scatter(x, iou)
            ax.annotate(cfg, (x, iou), fontsize=8)
        ax.set_xlabel(title); ax.set_ylabel("IoU"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/F2_pareto.png", dpi=300)
    plt.savefig("figures/F2_pareto.pdf")
    print("Wrote figures/F2_pareto.{png,pdf}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
cd code
python scripts/make_figures.py
```

Expected: produces `code/figures/F2_pareto.{png,pdf}`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/make_figures.py code/figures/F2_pareto.*
git commit -m "feat(figures): Pareto F2"
```

---

### Task D2: Qualitative panel (F3)

**Files:** Create `code/scripts/make_qualitative.py`

- [ ] **Step 1: Write the panel script**

```python
"""F3 — qualitative panel: image, GT, teacher pred, best student INT8 pred, error map.

Picks 4 examples manually by index (override after first run)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from floodlite.data import make_loaders, make_sen1floods11_loader
from floodlite.models import make_teacher, make_student
from floodlite.quantize import quantize_int8


def overlay(img, mask, color=(0, 255, 0), alpha=0.5):
    out = img.copy()
    out[mask > 0.5] = (out[mask > 0.5] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ood_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--id_indices", nargs=2, type=int, default=[0, 1], help="In-dist (easy, hard)")
    ap.add_argument("--ood_indices", nargs=2, type=int, default=[0, 1], help="OOD (success, fail)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = make_teacher().to(device)
    teacher.load_state_dict(torch.load(Path(args.ckpt_dir)/"teacher_fold0.pt", map_location=device))
    student = make_student("mobilevit_xxs").cpu()
    student.load_state_dict(torch.load(Path(args.ckpt_dir)/"mobilevit_xxs_none_fold0.pt", map_location="cpu"))
    tr, va = make_loaders(args.data_root, fold=0, batch_size=1)
    sq = quantize_int8(student, tr)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    rows = [("ID-easy", va, args.id_indices[0]), ("ID-hard", va, args.id_indices[1])]
    ood = make_sen1floods11_loader(args.ood_root, n_chips=250, batch_size=1)
    rows += [("OOD-success", ood, args.ood_indices[0]), ("OOD-fail", ood, args.ood_indices[1])]

    for r, (label, loader, idx) in enumerate(rows):
        for k, (x, y) in enumerate(loader):
            if k != idx: continue
            with torch.no_grad():
                t_pred = (torch.sigmoid(teacher(x.to(device))).cpu() > 0.5).float().numpy()[0, 0]
                s_pred = (torch.sigmoid(sq(x.cpu())) > 0.5).float().numpy()[0, 0]
            img_np = (x[0].permute(1,2,0).numpy() * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            gt_np = y[0,0].numpy()
            err = np.abs(s_pred - gt_np)
            for c, (im, ti) in enumerate(zip(
                [img_np, overlay(img_np, gt_np), overlay(img_np, t_pred), overlay(img_np, s_pred), (err*255).astype(np.uint8)],
                ["image", f"GT ({label})", "teacher", "MViT-XXS INT8", "error"])):
                ax = axes[r, c]
                ax.imshow(im, cmap="gray" if c==4 else None); ax.set_title(ti, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            break
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/F3_qualitative.png", dpi=300)
    plt.savefig("figures/F3_qualitative.pdf")
    print("Wrote figures/F3_qualitative.{png,pdf}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run with default indices, then iterate**

```bash
cd code
python scripts/make_qualitative.py \
    --data_root data/FSSD --ood_root data/sen1floods11_rgb \
    --ckpt_dir runs/3fold
# Inspect figures/F3_qualitative.png. If the chosen "hard" example is too easy
# or the "OOD-fail" looks fine, override the indices and rerun:
# --id_indices 17 42 --ood_indices 9 88
```

- [ ] **Step 3: Commit final selection**

```bash
git add code/scripts/make_qualitative.py code/figures/F3_qualitative.*
git commit -m "feat(figures): qualitative panel F3"
```

---

### Task D3: Manuscript rewrite (sections, tables, appendices)

**Files:** Modify `docs/FloodLite_Manuscript.docx` (in Word)

This is the largest writing task. Use the spec §5 mapping. **Do not** re-export the docx via Python — Word's track-changes + reviewer comments are easier to maintain by editing the existing file in Word/Apple Pages directly.

- [ ] **Step 1: Title + Abstract**

Replace title with:
> **FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight Flood Segmentation**

Rewrite the abstract around the three contributions in spec §2.

- [ ] **Step 2: Introduction — soften KD claim**

In Section 1, find every place that frames combined KD as a contribution and reframe as ablation. Keep the niche-gap narrative.

- [ ] **Step 3: Add §2.4 Lightweight segmentation backbones**

Cite BiSeNetV2 (Yu et al. 2021), PIDNet (Xu et al. 2023), Fast-SCNN (Poudel et al. 2019).

- [ ] **Step 4: Method — split §3.2 into students-and-baseline + KD-as-ablation**

Add a paragraph in §3.3 Quantization explicitly stating: "INT8 PTQ requires CPU; calibration tensors and model are moved to CPU before `prepare()`." This is the single sentence reviewers will grep for.

- [ ] **Step 5: Method §3.5 OOD protocol**

State: "Sen1Floods11 RGB (~250 chips) is used eval-only. No fine-tuning; same val-time transforms as FSSD. This characterizes single-domain training generalization."

- [ ] **Step 6: Results — replace Tables 2–6 with T1–T6**

Paste mean ± std values from `runs/3fold/summary.json`. Paste Wilcoxon and bootstrap p-values from `runs/3fold/stats.json`. Paste latency from `runs/3fold/lat_*.json`.

- [ ] **Step 7: Discussion — lead with the negative finding**

Three paragraphs:
1. **Why KD did not help.** FSSD saturates at 0.93; ImageNet-pretrained students already match teacher. Mechanistic interpretation.
2. **Per-platform deployment recommendations.** M2 Pro for triage; Android for field; WASM for browser-based dashboards; Graviton as cloud-edge fallback.
3. **OOD interpretation.** Drop from FSSD-val IoU to Sen1Floods11 IoU quantifies the cost of single-domain training.

- [ ] **Step 8: Add §7 Limitations**

Bullet-style: 663-image single-domain training, RGB-only, no SAR/multispectral, no energy measurements, single-phone Android, INT8 ViT may not quantize on all toolchains.

- [ ] **Step 9: Conclusion**

Rewrite around benchmark + open artifacts. No "≥95% recovery" claims.

- [ ] **Step 10: Delete Appendix A (placeholder map)**

- [ ] **Step 11: Add Appendix B — Model Card**

Use the HuggingFace template. One model card covering all 3 deployable students (FP32 + INT8). Sections: Model Details, Intended Use, Limitations, Training Data, Evaluation, Quantitative Analyses, Ethical Considerations, Caveats.

- [ ] **Step 12: Add Appendix C — Datasheet**

Gebru et al. format applied to the FSSD preprocessing pipeline (NOT to FSSD itself, which is third-party).

- [ ] **Step 13: Add Appendix D — Reproducibility checklist**

Use the ML Reproducibility Checklist (Pineau et al.). Tick boxes against your repo: code release, dataset release, hyperparameters, seeds, hardware specs, training time, evaluation.

- [ ] **Step 14: Final read-through pass**

Search the document for `##.##`, `##`, `{{`, `[Co-author Name]`, `[Affiliation]`. Replace each. Confirm no placeholders remain.

- [ ] **Step 15: Commit**

```bash
git add docs/FloodLite_Manuscript.docx
git commit -m "docs(manuscript): full rewrite for IJDRR submission"
```

---

## Phase E — Artifacts + submission (~2 working days)

### Task E1: Hugging Face Hub upload

- [ ] **Step 1: Create model repos**

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create floodlite-mobilevit-xxs --type model
huggingface-cli repo create floodlite-mobilenetv3-small --type model
huggingface-cli repo create floodlite-efficientnet-lite0 --type model
```

- [ ] **Step 2: Push artifacts + Model Card**

For each student: upload `*_fp32.pt`, `*_int8.pt`, `*_fp32.onnx`, `*_int8.onnx`, plus a `README.md` Model Card.

```bash
for s in mobilevit_xxs mobilenetv3_small efficientnet_lite0; do
  cd /tmp && rm -rf $s && huggingface-cli download <user>/floodlite-$s --local-dir $s || mkdir -p $s
  cp /Users/.../runs/3fold/${s}_none_fold0.pt $s/${s}_fp32.pt
  cp /Users/.../exports/${s}_fp32.onnx $s/
  cp /Users/.../exports/${s}_int8.onnx $s/ 2>/dev/null || true
  # Write README.md (Model Card) into $s/
  cd $s && git init && git lfs install && git lfs track "*.pt" "*.onnx"
  git remote add origin https://huggingface.co/<user>/floodlite-$s
  git add . && git commit -m "v1.0-ijdrr"
  git push origin main
done
```

- [ ] **Step 3: Verify URLs are public and 6 artifacts are accessible**

---

### Task E2: Zenodo DOI

- [ ] **Step 1: Tag the GitHub release**

```bash
cd /Users/ibrahim/Desktop/personal/Flood_segmentation_model
git tag v1.0-ijdrr
git push origin --tags
```

- [ ] **Step 2: Mint Zenodo DOI**

In Zenodo UI: link your GitHub repo, enable releases sync, copy the resulting DOI. Add the DOI to manuscript front matter and to `code/README.md` §7 Citation.

---

### Task E3: IJDRR cover letter + submit

- [ ] **Step 1: Draft cover letter**

3 short paragraphs:
1. Why IJDRR (humanitarian disaster framing).
2. Three contributions (paste from spec §2).
3. Reproducibility statement: code on GitHub, weights on HF, DOI on Zenodo.

- [ ] **Step 2: Submit via IJDRR Editorial Manager**

Upload manuscript PDF + supplementary (Model Card, Datasheet, Reproducibility checklist as separate files).

---

### Task E4: IGARSS 4-page condensation + submit

- [ ] **Step 1: Trim to 4 pages**

Drop appendices, keep T1, T2 (compressed), T5, T6, F1, F2, F3 (single column). Cite the IJDRR submission as preprint (Zenodo DOI).

- [ ] **Step 2: Submit via IGARSS portal**

Confirm you have the submission deadline correct (typical IGARSS abstract deadline = January, but check the 2026 site).

---

## Self-review (executed against the spec)

**Spec coverage check (§1–10 of the spec):**
- §1 motivation → covered in Phase D Task D3 Step 7 (Discussion).
- §2 narrative + 3 contributions → covered in D3 Steps 1–3, 7.
- §3 experiment matrix → Phase A (code) + B (runs).
  - 14 configs × 3 folds → B1.
  - INT8 on 3 deployable students × 3 folds → B2.
  - OOD on 5 models → B3 + B4.
  - 4-platform latency → C1–C5.
  - Statistical tests (Wilcoxon + paired pixel-bootstrap) → B5.
  - "Best student" definition → applied in eval scripts (B5, C1, C2).
- §4 tables/figures → Phase D (T1–T6 in D3 Step 6, F1 hand-drawn diagram in D3, F2 + F4 in D1, F3 in D2).
- §5 manuscript rewrite → D3 Steps 1–14.
- §6 deliverables → Phase E (HF + Zenodo + repro checklist).
- §7 code changes → Phase A Tasks A1–A6 (one task per file, matches the table exactly).
- §8 per-week plan → Phase A=W1, B=W1–W2, C=W3, D=W4, E=W4–W5.
- §9 out-of-scope → not implemented (correct).
- §10 risk register → reflected in kill switches inside C4 (Android fallback) and C1 (INT8 ViT fallback).

**Placeholder scan:** I see explicit `<user>` and `<graviton-ip>` and `u@phone` — these are user-substituted, not plan-author placeholders. Acceptable. No "TBD/TODO/etc."

**Type consistency:**
- `make_baseline_mobilenetv2()` defined in A2, used in A5 import, B4, C1.
- `make_sen1floods11_loader(root, n_chips=...)` defined in A3, used in B4, D2.
- `benchmark_onnxruntime(onnx_path, ...)` defined in A4, used in C2.
- All consistent.

**One spec gap I patched while writing:** F1 (pipeline diagram) is hand-drawn; I did not script it, since it's a one-off illustration. Adding a step in D3 to create it manually would clutter; assume the user produces F1 in PowerPoint/Keynote/draw.io alongside the manuscript edits.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-floodlite-edge-benchmark-pivot.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for Phase A (code changes are isolated and testable).
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Best when you want to watch each step.

**Which approach?**
