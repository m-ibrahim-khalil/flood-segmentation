"""make_tables.py — emit manuscript tables T1-T5 as Markdown and plain-text.

Usage:
    python3 scripts/make_tables.py --runs runs/3fold --out_md docs/tables.md --out_txt docs/tables.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

# Allow `import floodlite` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    """Load JSON; return None and warn on any error."""
    if not path.exists():
        warnings.warn(f"[make_tables] {path} not found — affected table cells will show n/a")
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        warnings.warn(f"[make_tables] Cannot parse {path}: {exc}")
        return None


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def _ms(v: float | None, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def _mean_std(d: dict, key: str) -> str:
    """Extract 'key.mean ± key.std' from a summary-json model entry."""
    sub = d.get(key, {})
    m = sub.get("mean")
    s = sub.get("std")
    if m is None:
        return "n/a"
    return f"{m:.4f} ± {s:.4f}"


# ---------------------------------------------------------------------------
# Markdown table builders
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    header_line = "|" + "|".join(f" {h:<{widths[i]}} " for i, h in enumerate(headers)) + "|"
    lines = [header_line, sep]
    for row in rows:
        lines.append("|" + "|".join(f" {str(cell):<{widths[i]}} " for i, cell in enumerate(row)) + "|")
    return "\n".join(lines)


def _txt_table(headers: list[str], rows: list[list[str]]) -> str:
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt_header = "  ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(cols))
    lines = [fmt_header, sep]
    for row in rows:
        lines.append("  ".join(f"{str(cell):<{widths[i]}}" for i, cell in enumerate(row)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T1 — Model footprints
# ---------------------------------------------------------------------------

# Canonical model registry: (display_name, factory_fn_name, student_name_or_None)
_MODEL_REGISTRY = [
    ("Teacher (UNet+EffB0)",    "teacher",              None),
    ("MobileNetV2 baseline",    "baseline_mobilenetv2", None),
    ("MobileNetV3-Small",       "mobilenetv3_small",    "mobilenetv3_small"),
    ("EfficientNet-Lite0",      "efficientnet_lite0",   "efficientnet_lite0"),
    ("MobileViT-XXS",           "mobilevit_xxs",        "mobilevit_xxs"),
]


def build_t1(quantized: dict | None) -> tuple[str, str]:
    """T1: Model footprints."""
    try:
        from floodlite.models import (
            make_teacher, make_student, make_baseline_mobilenetv2,
            count_params, estimate_flops,
        )
        models_available = True
    except Exception as exc:
        warnings.warn(f"[T1] Cannot import floodlite.models: {exc}. Params/GFLOPs will show n/a.")
        models_available = False

    headers = ["Model", "Params (M)", "GFLOPs @256²", "FP32 ONNX (MB)", "INT8 ONNX (MB)"]
    rows = []

    for display, key, student_arg in _MODEL_REGISTRY:
        params_s = "n/a"
        gflops_s = "n/a"

        if models_available:
            try:
                import torch
                if key == "teacher":
                    m = make_teacher()
                elif key == "baseline_mobilenetv2":
                    m = make_baseline_mobilenetv2()
                else:
                    m = make_student(student_arg)

                params_s = _fmt(count_params(m), 2)
                gf = estimate_flops(m)
                gflops_s = _fmt(gf, 2)
            except Exception as exc:
                warnings.warn(f"[T1] Cannot build model '{key}': {exc}")

        # ONNX sizes from quantized.json
        fp32_mb = "n/a"
        int8_mb = "n/a"
        if quantized is not None:
            entry = quantized.get(key) or quantized.get(f"{key}_none") or {}
            fp32_mb = _fmt(entry.get("fp32_onnx_mb"), 1) if entry.get("fp32_onnx_mb") is not None else "n/a"
            int8_mb = _fmt(entry.get("int8_onnx_mb"), 1) if entry.get("int8_onnx_mb") is not None else "n/a"

        rows.append([display, params_s, gflops_s, fp32_mb, int8_mb])

    md = "## T1 — Model Footprints\n\n" + _md_table(headers, rows)
    txt = "T1 — Model Footprints\n" + "=" * 60 + "\n" + _txt_table(headers, rows)
    return md, txt


# ---------------------------------------------------------------------------
# T2 — In-distribution segmentation
# ---------------------------------------------------------------------------

_T2_ROW_ORDER = [
    ("Teacher (UNet+EffB0)",   "teacher"),
    ("MobileNetV2 baseline",   "baseline_mobilenetv2"),
    ("MobileNetV3-S / none",   "mobilenetv3_small_none"),
    ("MobileNetV3-S / resp",   "mobilenetv3_small_resp"),
    ("MobileNetV3-S / feat",   "mobilenetv3_small_feat"),
    ("MobileNetV3-S / comb",   "mobilenetv3_small_comb"),
    ("EfficientNet-Lite0 / none", "efficientnet_lite0_none"),
    ("EfficientNet-Lite0 / resp", "efficientnet_lite0_resp"),
    ("EfficientNet-Lite0 / feat", "efficientnet_lite0_feat"),
    ("EfficientNet-Lite0 / comb", "efficientnet_lite0_comb"),
    ("MobileViT-XXS / none",   "mobilevit_xxs_none"),
    ("MobileViT-XXS / resp",   "mobilevit_xxs_resp"),
    ("MobileViT-XXS / feat",   "mobilevit_xxs_feat"),
    ("MobileViT-XXS / comb",   "mobilevit_xxs_comb"),
]


def build_t2(summary: dict | None) -> tuple[str, str]:
    """T2: In-distribution segmentation across all model/KD combos."""
    headers = ["Model / KD", "Acc", "Prec", "Rec", "F1", "IoU"]
    rows = []

    for display, key in _T2_ROW_ORDER:
        if summary is None or key not in summary:
            rows.append([display, "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        entry = summary[key]
        rows.append([
            display,
            _mean_std(entry, "accuracy"),
            _mean_std(entry, "precision"),
            _mean_std(entry, "recall"),
            _mean_std(entry, "f1"),
            _mean_std(entry, "iou"),
        ])

    note = "_Values are mean ± std over 3 folds (FSSD val, 256×256)._"
    md = "## T2 — In-Distribution Segmentation\n\n" + _md_table(headers, rows) + "\n\n" + note
    txt = "T2 — In-Distribution Segmentation\n" + "=" * 60 + "\n" + _txt_table(headers, rows) + "\n\n" + note
    return md, txt


# ---------------------------------------------------------------------------
# T3 — KD ablation on best student (MobileViT-XXS)
# ---------------------------------------------------------------------------

_T3_KD_KEYS = [
    ("none", "mobilevit_xxs_none"),
    ("resp", "mobilevit_xxs_resp"),
    ("feat", "mobilevit_xxs_feat"),
    ("comb", "mobilevit_xxs_comb"),
]


def build_t3(summary: dict | None, stats: dict | None) -> tuple[str, str]:
    """T3: KD ablation on MobileViT-XXS (best student)."""
    headers = ["KD config", "IoU (mean ± std)", "Δ IoU vs none", "Wilcoxon p (3 folds)"]
    rows = []

    # baseline IoU for none config
    none_iou = None
    if summary and "mobilevit_xxs_none" in summary:
        none_iou = summary["mobilevit_xxs_none"].get("iou", {}).get("mean")

    wilcoxon = {}
    if stats:
        wil = stats.get("wilcoxon_3fold", {})
        wilcoxon = {
            "none": None,
            "resp": wil.get("mobilevit_xxs_none_vs_resp"),  # may not exist
            "feat": wil.get("mobilevit_xxs_none_vs_feat"),  # may not exist
            "comb": wil.get("mobilevit_xxs_none_vs_comb"),
        }

    for kd_label, key in _T3_KD_KEYS:
        iou_str = "n/a"
        delta_str = "n/a" if kd_label != "none" else "—"
        p_str = "—" if kd_label == "none" else "n/a"

        if summary and key in summary:
            entry = summary[key]
            iou_str = _mean_std(entry, "iou")
            m = entry.get("iou", {}).get("mean")
            if kd_label != "none" and none_iou is not None and m is not None:
                delta_str = f"{m - none_iou:+.4f}"

        # Wilcoxon p-value from stats.json (only none_vs_comb is computed)
        if kd_label == "comb":
            wkey = "mobilevit_xxs_none_vs_comb"
            w = stats.get("wilcoxon_3fold", {}).get(wkey) if stats else None
            if w:
                p_str = f"{w['p_value']:.3f} (n=3, low power)"

        rows.append([kd_label, iou_str, delta_str, p_str])

    note = ("_Best student = MobileViT-XXS (highest mean IoU = 0.9367 for 'none' config). "
            "Wilcoxon two-sided; n=3 folds → minimum reportable p ≈ 0.25._")
    md = "## T3 — KD Ablation on Best Student (MobileViT-XXS)\n\n" + _md_table(headers, rows) + "\n\n" + note
    txt = "T3 — KD Ablation on Best Student (MobileViT-XXS)\n" + "=" * 60 + "\n" + _txt_table(headers, rows) + "\n\n" + note
    return md, txt


# ---------------------------------------------------------------------------
# T4 — Quantization
# ---------------------------------------------------------------------------

_T4_STUDENTS = [
    ("MobileNetV3-Small",  "mobilenetv3_small"),
    ("EfficientNet-Lite0", "efficientnet_lite0"),
    ("MobileViT-XXS",      "mobilevit_xxs"),
]


def build_t4(summary: dict | None, quantized: dict | None) -> tuple[str, str]:
    """T4: Quantization impact (FP32 vs INT8 IoU + size reduction)."""
    if quantized is None:
        note = "_(awaiting Phase C quantized.json output)_"
        md = "## T4 — Quantization\n\n" + note
        txt = "T4 — Quantization\n" + "=" * 60 + "\n" + note.replace("_", "")
        return md, txt

    headers = [
        "Model", "FP32 IoU (mean ± std)", "INT8 IoU (mean ± std)",
        "Δ IoU", "FP32 ONNX (MB)", "INT8 ONNX (MB)", "Size reduction",
    ]
    rows = []

    for display, key in _T4_STUDENTS:
        none_key = f"{key}_none"

        # FP32 IoU from summary.json (no-KD config = deployment candidate)
        fp32_iou_str = "n/a"
        fp32_iou_val = None
        if summary and none_key in summary:
            fp32_iou_str = _mean_std(summary[none_key], "iou")
            fp32_iou_val = summary[none_key].get("iou", {}).get("mean")

        # INT8 IoU from quantized.json
        qentry = quantized.get(key) or quantized.get(none_key) or {}
        int8_iou_val = qentry.get("int8_iou_mean")
        int8_iou_std = qentry.get("int8_iou_std")
        if int8_iou_val is not None:
            int8_iou_str = f"{int8_iou_val:.4f}" + (f" ± {int8_iou_std:.4f}" if int8_iou_std is not None else "")
        else:
            int8_iou_str = "n/a"

        delta_str = "n/a"
        if fp32_iou_val is not None and int8_iou_val is not None:
            delta_str = f"{int8_iou_val - fp32_iou_val:+.4f}"

        fp32_mb = _fmt(qentry.get("fp32_onnx_mb"), 1) if qentry.get("fp32_onnx_mb") is not None else "n/a"
        int8_mb = _fmt(qentry.get("int8_onnx_mb"), 1) if qentry.get("int8_onnx_mb") is not None else "n/a"

        size_red = "n/a"
        if qentry.get("fp32_onnx_mb") and qentry.get("int8_onnx_mb"):
            ratio = qentry["fp32_onnx_mb"] / qentry["int8_onnx_mb"]
            size_red = f"{ratio:.2f}×"

        rows.append([display, fp32_iou_str, int8_iou_str, delta_str, fp32_mb, int8_mb, size_red])

    note = "_INT8 PTQ calibrated on 200 train images, CPU inference._"
    md = "## T4 — Quantization\n\n" + _md_table(headers, rows) + "\n\n" + note
    txt = "T4 — Quantization\n" + "=" * 60 + "\n" + _txt_table(headers, rows) + "\n\n" + note
    return md, txt


# ---------------------------------------------------------------------------
# T5 — Per-platform latency
# ---------------------------------------------------------------------------

_T5_MODELS = [
    "teacher_fp32",
    "mobilenetv3_small_fp32",
    "mobilenetv3_small_int8",
    "efficientnet_lite0_fp32",
    "efficientnet_lite0_int8",
    "mobilevit_xxs_fp32",
    "mobilevit_xxs_int8",
]

_PLATFORM_FILES = [
    ("M2 Pro",           "lat_m2.json"),
    ("Graviton (ARM)",   "lat_graviton.json"),
    ("Browser (WASM)",   "lat_wasm.json"),
]


def build_t5(runs_dir: Path) -> tuple[str, str]:
    """T5: Per-platform latency."""
    lat_data: dict[str, dict | None] = {}
    for platform_name, fname in _PLATFORM_FILES:
        lat_data[platform_name] = _load_json(runs_dir / fname)

    if all(v is None for v in lat_data.values()):
        note = "_(awaiting Phase C output — lat_m2.json / lat_graviton.json / lat_wasm.json not yet generated)_"
        md = "## T5 — Per-Platform Latency\n\n" + note
        txt = "T5 — Per-Platform Latency\n" + "=" * 60 + "\n" + note.replace("_", "")
        return md, txt

    # Build column list dynamically from available platforms
    platform_cols: list[tuple[str, str]] = []
    for platform_name, _ in _PLATFORM_FILES:
        if lat_data[platform_name] is not None:
            platform_cols.append((platform_name, platform_name))

    sub_headers = ["p50 (ms)", "p95 (ms)", "FPS"]
    headers = ["Model"] + [f"{p} — {s}" for p, _ in platform_cols for s in sub_headers]
    rows = []

    def _lookup(pdata: dict, model_key: str) -> dict | None:
        """Find a latency entry under multiple legal key shapes.

        bench_m2.py writes:
          teacher_pt_cpu                       <- PyTorch CPU (no ORT for teacher)
          {sname}_pt_cpu_fp32                  <- PyTorch CPU FP32
          {sname}_{fp32|int8}_ort_cpu          <- ONNX Runtime CPU

        bench_graviton.sh writes:
          {sname}_{fp32|int8}                  <- flat names

        bench_wasm.html writes:
          {sname}_{fp32|int8}                  <- flat names

        We prefer ONNX-RT over PyTorch CPU when both exist (it's the
        deployable path) and fall back through the recognised shapes.
        """
        if pdata is None:
            return None
        if model_key == "teacher_fp32":
            return pdata.get("teacher_ort_cpu") or pdata.get("teacher_pt_cpu") or pdata.get("teacher")
        candidates = [
            model_key,                         # graviton/wasm flat names
            f"{model_key}_ort_cpu",            # bench_m2 ORT keys
            f"{model_key.replace('_fp32', '_pt_cpu_fp32')}",  # bench_m2 PyTorch keys
        ]
        for c in candidates:
            if c in pdata:
                return pdata[c]
        return None

    for model_key in _T5_MODELS:
        row = [model_key.replace("_", " ")]
        for platform_name, _ in platform_cols:
            entry = _lookup(lat_data[platform_name], model_key)
            if entry:
                p50 = entry.get("p50_ms")
                p95 = entry.get("p95_ms")
                fps = (1000.0 / p50) if p50 else None
                row.extend([_ms(p50), _ms(p95), _ms(fps, 1)])
            else:
                row.extend(["n/a", "n/a", "n/a"])
        rows.append(row)

    note = "_Single-image inference, batch=1, 256×256 input. FPS = 1000/p50._"
    md = "## T5 — Per-Platform Latency\n\n" + _md_table(headers, rows) + "\n\n" + note
    txt = "T5 — Per-Platform Latency\n" + "=" * 60 + "\n" + _txt_table(headers, rows) + "\n\n" + note
    return md, txt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit manuscript tables T1-T5 as Markdown and plain-text."
    )
    parser.add_argument("--runs", type=Path, default=Path("runs/3fold"),
                        help="Directory containing summary.json, stats.json, quantized.json, lat_*.json")
    parser.add_argument("--out_md", type=Path, default=Path("docs/tables.md"),
                        help="Output Markdown file")
    parser.add_argument("--out_txt", type=Path, default=Path("docs/tables.txt"),
                        help="Output plain-text file")
    args = parser.parse_args()

    runs_dir: Path = args.runs
    summary  = _load_json(runs_dir / "summary.json")
    stats    = _load_json(runs_dir / "stats.json")
    quantized_raw = _load_json(runs_dir / "quantized.json")

    # Normalize quantized.json from nested per-fold {fold_N: {arch: {fp32_iou, int8_iou, ...}}}
    # to the flat {arch_none: {int8_iou_mean, int8_iou_std, fp32_onnx_mb, int8_onnx_mb}}
    # schema the T1/T4 builders expect.
    quantized: dict | None = None
    if quantized_raw is not None:
        per_arch: dict[str, dict[str, list]] = {}
        for fold_key, fold in quantized_raw.items():
            if not fold_key.startswith("fold_"):
                continue
            for arch, entry in fold.items():
                if "error" in entry:
                    continue
                bucket = per_arch.setdefault(arch, {"fp32_iou": [], "int8_iou": [],
                                                    "fp32_mb": [], "int8_mb": []})
                if "fp32_iou" in entry: bucket["fp32_iou"].append(entry["fp32_iou"])
                if "int8_iou" in entry: bucket["int8_iou"].append(entry["int8_iou"])
                if "fp32_size_mb" in entry: bucket["fp32_mb"].append(entry["fp32_size_mb"])
                if "int8_size_mb" in entry: bucket["int8_mb"].append(entry["int8_size_mb"])
        import statistics as _stat
        quantized = {}
        for arch, b in per_arch.items():
            row = {}
            if b["int8_iou"]:
                row["int8_iou_mean"] = float(_stat.fmean(b["int8_iou"]))
                row["int8_iou_std"]  = float(_stat.pstdev(b["int8_iou"])) if len(b["int8_iou"]) > 1 else 0.0
            if b["fp32_iou"]:
                row["fp32_iou_mean"] = float(_stat.fmean(b["fp32_iou"]))
                row["fp32_iou_std"]  = float(_stat.pstdev(b["fp32_iou"])) if len(b["fp32_iou"]) > 1 else 0.0
            if b["fp32_mb"]:
                row["fp32_onnx_mb"] = float(_stat.fmean(b["fp32_mb"]))
            if b["int8_mb"]:
                row["int8_onnx_mb"] = float(_stat.fmean(b["int8_mb"]))
            # Register under multiple key forms so both T1 (looks up by `key`) and
            # T4 (looks up by `key` or `{key}_none`) find it.
            quantized[arch] = row
            quantized[f"{arch}_none"] = row

    sections_md: list[str] = []
    sections_txt: list[str] = []

    header_md = (
        "# FloodLite — Manuscript Tables T1–T5\n\n"
        "_Auto-generated by `scripts/make_tables.py`. "
        "Source: `runs/3fold/summary.json` + supplementary JSONs._\n"
    )
    sections_md.append(header_md)
    sections_txt.append("FloodLite — Manuscript Tables T1-T5\n" + "=" * 70 + "\n")

    for label, fn in [
        ("T1", lambda: build_t1(quantized)),
        ("T2", lambda: build_t2(summary)),
        ("T3", lambda: build_t3(summary, stats)),
        ("T4", lambda: build_t4(summary, quantized)),
        ("T5", lambda: build_t5(runs_dir)),
    ]:
        try:
            md, txt = fn()
        except Exception as exc:
            warnings.warn(f"[make_tables] {label} failed: {exc}")
            md = f"## {label}\n\n_(generation error: {exc})_"
            txt = f"{label}\n_(generation error: {exc})_"
        sections_md.append(md)
        sections_txt.append(txt)

    out_md_str = "\n\n---\n\n".join(sections_md) + "\n"
    out_txt_str = "\n\n" + ("-" * 70) + "\n\n".join(sections_txt) + "\n"

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)

    args.out_md.write_text(out_md_str, encoding="utf-8")
    args.out_txt.write_text(out_txt_str, encoding="utf-8")

    print(f"[make_tables] Markdown → {args.out_md}")
    print(f"[make_tables] Plain-text → {args.out_txt}")


if __name__ == "__main__":
    main()
