# FloodLite — Reproducible Code

Code accompanying the manuscript **"FloodLite: Knowledge-Distilled Lightweight Segmentation for On-Device Flood Mapping in Resource-Constrained Disaster Zones."**

This repository reproduces every numerical result in the paper, end-to-end, on free-tier compute (Kaggle T4, Google Colab T4, or a MacBook M2 Pro).

---

## 1. Repository layout

```
code/
├── README.md                              ← you are here
├── requirements.txt                       ← pip install -r requirements.txt
├── floodlite/                             ← core Python package
│   ├── __init__.py
│   ├── data.py                            ← FSSD dataset + transforms + 5-fold splits
│   ├── models.py                          ← teacher (UNet+EffNet-B0) + 3 students
│   ├── losses.py                          ← BCE+Dice + response-KD + feature-KD
│   ├── metrics.py                         ← accuracy/precision/recall/F1/IoU
│   ├── train.py                           ← teacher and KD-student training loops
│   ├── quantize.py                        ← post-training INT8 quantization
│   ├── benchmark.py                       ← latency benchmarking
│   └── export.py                          ← ONNX → CoreML / TFLite
├── notebooks/
│   └── floodlite_kaggle.ipynb             ← single end-to-end notebook (recommended)
└── scripts/
    └── run_full_experiment.py             ← CLI driver for full 5-fold sweep
```

---

## 2. Quickstart

### Option A — Kaggle (recommended)

1. Open <https://kaggle.com> → Create new Notebook.
2. Upload `notebooks/floodlite_kaggle.ipynb`.
3. In the right-hand panel, **Add Data** → search for *Flood Semantic Segmentation Dataset* (lihuayang111265) → Add.
4. Settings → Accelerator → **GPU T4 x2**.
5. Run All. Wall-time per fold: ~2-3 hours.

The Kaggle dataset mounts at `/kaggle/input/flood-semantic-segmentation-dataset/` with the layout:

```
dataset/
├── train/{images, labels}
└── val/{images, labels}
```

By default the notebook **combines train + val and runs 5-fold cross-validation** over the combined pool (matches Karcı et al. 2026's methodology). To use the dataset's predefined train/val split instead, set `USE_PREDEFINED_SPLIT = True` in the data-loading cell.

### Option B — Google Colab

1. Open the notebook in Colab.
2. Runtime → Change runtime type → GPU.
3. Mount Google Drive (optional) for checkpoint persistence.
4. Install the Kaggle API and download the dataset, OR upload the dataset to your Drive.
5. Run All.

### Option C — Local M2 Pro

```bash
conda create -n floodlite python=3.11
conda activate floodlite
pip install -r requirements.txt
# Download FSSD from Kaggle and unzip into ./data/FSSD/
python scripts/run_full_experiment.py --data_root ./data/FSSD --fold 0 --folds 5
```

PyTorch's MPS backend on M2 Pro is roughly 0.6× the speed of a Kaggle T4 for this workload — usable but slow. INT8 quantization needs CPU; latency benchmarks on M2 Pro should be run with `device='cpu'` to match deployment-target conditions.

---

## 3. Filling in the manuscript placeholders

The manuscript (`FloodLite_Manuscript.docx`) contains three kinds of placeholders that need to be replaced after running the code:

| Placeholder pattern     | Where it appears             | What replaces it                                      |
|-------------------------|------------------------------|-------------------------------------------------------|
| `##.##`                 | Inside Tables 2-6            | A 4-decimal float printed by the notebook (IoU, F1…)  |
| `##` (in latency cols.) | Table 6                      | An integer ms-latency printed by `benchmark_latency`  |
| `{{NAME}}`              | Abstract & Conclusions       | A computed percentage / ratio (see table below)       |

### Step-by-step

1. **Run `floodlite_kaggle.ipynb`** end-to-end.
2. The last cell ("Print a results table ready to paste into the manuscript") prints all numbers in a tidy text format. Copy each printed value into the corresponding cell in `FloodLite_Manuscript.docx`.
3. Compute the abstract / conclusion `{{...}}` placeholders with the formulas below:

| Token                  | Formula                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `{{IOU_RECOVERY_PCT}}` | (`best_student_IoU` / `teacher_IoU`) × 100, rounded to one decimal      |
| `{{PARAM_PCT}}`        | (`student_params` / `teacher_params`) × 100, rounded to one decimal     |
| `{{FLOPS_PCT}}`        | (`student_FLOPs` / `teacher_FLOPs`) × 100, rounded to one decimal       |
| `{{FPS_M2}}`           | 1000 / `M2_p50_ms`, rounded to nearest integer                          |
| `{{FPS_PI}}`           | 1000 / `Pi_p50_ms`,  rounded to nearest integer                         |
| `{{FPS_WASM}}`         | 1000 / `WASM_p50_ms`, rounded to nearest integer                        |

4. Open Word's **Find & Replace** (Cmd-Shift-H on Mac), search for `##.##` to spot remaining placeholders. Search for `{{` to find any unfilled bracket placeholders.
5. Delete **Appendix A** (the placeholder map) before submission — it is intended for you, the author, not the reviewer.

---

## 4. Generating the four-axis Pareto figure

The notebook produces all data needed for Figure 3 (Pareto). After the notebook finishes, run this cell to produce the figure:

```python
import matplotlib.pyplot as plt
# manually fill in the dict below from your results
data = [
    ("Teacher",     {"iou": 0.92, "params": 6.3, "flops": 1.4, "lat_m2": 50, "lat_pi": 800}),
    ("MobileNetV3", {"iou": 0.85, "params": 1.5, "flops": 0.42, "lat_m2": 12, "lat_pi": 70}),
    ("EffNet-Lite", {"iou": 0.88, "params": 3.5, "flops": 0.94, "lat_m2": 18, "lat_pi": 110}),
    ("MobileViT",   {"iou": 0.89, "params": 1.3, "flops": 0.62, "lat_m2": 15, "lat_pi": 95}),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, key, xlabel in zip(axes, ["params","flops","lat_m2","lat_pi"], ["Params (M)","FLOPs (G)","M2 latency (ms)","Pi latency (ms)"]):
    for name, d in data:
        ax.scatter(d[key], d["iou"]); ax.annotate(name, (d[key], d["iou"]))
    ax.set_xlabel(xlabel); ax.set_ylabel("IoU"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("figure3_pareto.png", dpi=300)
```

---

## 5. Deployment artifacts

After the notebook runs, `exports/*.onnx` contains ONNX models. Convert as follows:

```bash
# CoreML (Apple Silicon, requires macOS)
pip install coremltools
python -c "from floodlite.export import export_coreml; export_coreml('exports/mobilevit_xxs.onnx', 'exports/mobilevit_xxs.mlmodel')"

# TensorFlow Lite (Raspberry Pi, Android)
pip install onnx-tf tensorflow
python -c "from floodlite.export import export_tflite; export_tflite('exports/mobilevit_xxs.onnx', 'exports/mobilevit_xxs.tflite')"

# Browser (ONNX Runtime Web)
# The ONNX file in exports/ already runs in onnxruntime-web 1.17+ via:
#   import * as ort from 'onnxruntime-web';
#   const sess = await ort.InferenceSession.create('mobilevit_xxs.onnx');
```

For Pi 4 latency, copy `mobilevit_xxs.tflite` to a Pi 4 running Raspberry Pi OS and run:

```bash
pip install tflite-runtime
python -c "
import time, tflite_runtime.interpreter as tflite, numpy as np
i = tflite.Interpreter('mobilevit_xxs.tflite'); i.allocate_tensors()
in_d, out_d = i.get_input_details()[0], i.get_output_details()[0]
x = np.random.randn(1,3,256,256).astype(np.float32)
for _ in range(20): i.set_tensor(in_d['index'], x); i.invoke()  # warm
t0=time.perf_counter()
for _ in range(100): i.set_tensor(in_d['index'], x); i.invoke()
print('Pi 4 mean ms:', (time.perf_counter()-t0)/100*1000)
"
```

---

## 6. Reproducibility checklist

- [x] All seeds fixed (`SEED=42` in notebook + scripts)
- [x] Versioned dependencies in `requirements.txt`
- [x] Public dataset (Kaggle FSSD)
- [x] All metrics computed by the same `compute_metrics()` function
- [x] 5-fold cross-validation
- [x] Three random seeds for the final reported configuration (set via `SEED` and re-run)
- [x] Open-source: MIT license for code, Apache-2.0 for model checkpoints

---

## 7. Citation

If you use this code, please cite:

```bibtex
@article{khalil2026floodlite,
  title   = {FloodLite: Knowledge-Distilled Lightweight Segmentation for On-Device Flood Mapping in Resource-Constrained Disaster Zones},
  author  = {Khalil, Ibrahim and others},
  journal = {Remote Sensing},
  year    = {2026},
  note    = {(under review)}
}
```

---

## 8. License

- **Code**: MIT License (see `LICENSE`)
- **Model weights** (uploaded separately to Hugging Face Hub): Apache-2.0 License
