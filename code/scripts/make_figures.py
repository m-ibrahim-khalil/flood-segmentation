"""make_figures.py — emit F2 (Pareto scatter) and F4 (throughput bars) for the FloodLite manuscript.

Usage:
    python3 scripts/make_figures.py --runs runs/3fold --out_dir figures

Outputs:
    figures/F2_pareto.png   (300 DPI)
    figures/F2_pareto.pdf
    figures/F4_throughput.png  (only if at least one platform lat JSON exists)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe (CI / Kaggle / Linux)

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Architecture metadata. Params (M) are the full-UNet counts from Table 1
# (torchinfo.summary), matching tables/T1_footprints.tex. FLOPs are deferred to
# the camera-ready release (Table 1), so the FLOPs panel renders as a placeholder
# rather than fabricated numbers.
# ---------------------------------------------------------------------------
ARCH = {
    "teacher":                 {"params": 6.25, "size_mb": 22.3},
    "baseline_mobilenetv2":    {"params": 6.63, "size_mb": None},  # ONNX not exported (Table 1: ---)
    "mobilenetv3_small_none":  {"params": 3.59, "size_mb": 13.8},
    "efficientnet_lite0_none": {"params": 5.61, "size_mb": 19.8},
    "mobilevit_xxs_none":      {"params": 3.09, "size_mb": 12.0},
}

# Short display labels per config
SHORT_LABEL = {
    "teacher":                 "Teacher\n(EffB0)",
    "baseline_mobilenetv2":    "MobileNetV2\nbaseline",
    "mobilenetv3_small_none":  "MNV3-S",
    "efficientnet_lite0_none": "EffLite0",
    "mobilevit_xxs_none":      "MobileViT\nXXS",
}

# Distinct colors — one per config, consistent across all subplots
COLORS = {
    "teacher":                 "#e41a1c",
    "baseline_mobilenetv2":    "#377eb8",
    "mobilenetv3_small_none":  "#4daf4a",
    "efficientnet_lite0_none": "#ff7f00",
    "mobilevit_xxs_none":      "#984ea3",
}

CONFIGS = list(ARCH.keys())

# Students for F4 (the three lightweight deployment candidates)
STUDENTS_F4 = [
    "mobilenetv3_small_none",
    "efficientnet_lite0_none",
    "mobilevit_xxs_none",
]
STUDENTS_SHORT = {
    "mobilenetv3_small_none":  "MNV3-S",
    "efficientnet_lite0_none": "EffLite0",
    "mobilevit_xxs_none":      "MViT-XXS",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """Return parsed JSON or None if the file is missing/invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_iou(summary: dict, key: str) -> float | None:
    """Extract mean IoU for a config from summary.json."""
    entry = summary.get(key)
    if entry is None:
        return None
    return entry.get("iou", {}).get("mean")


def get_p50_ms(lat: dict, key: str) -> float | None:
    """Extract p50 latency (ms) for a model key from a lat JSON.

    Tries several legal entry shapes:
      1. { "<key>": { "p50_ms": <float> } }              ← bench_m2/graviton/wasm
      2. { "<key>": { "p50":    <float> } }              ← legacy / WASM HTML
      3. { "<key>": { "fp32":   { "p50_ms": <float> } }} ← nested variant
    Returns the first match or None.
    """
    entry = lat.get(key)
    if entry is None:
        return None
    if "p50_ms" in entry:
        return float(entry["p50_ms"])
    if "p50" in entry:
        return float(entry["p50"])
    fp32 = entry.get("fp32", {})
    if "p50_ms" in fp32:
        return float(fp32["p50_ms"])
    return None


def _candidate_lat_keys(config: str) -> list[str]:
    """Map ARCH config to all key shapes used by the various benchmark scripts.

    Order matters: prefer the *deployable* path (ONNX-RT) over PyTorch CPU,
    and the *student-only no-KD* path over flat names.
    """
    if config == "teacher":
        # No teacher.onnx is exported by quantize_all.py, so we usually only have
        # the PyTorch CPU number on M2.
        return ["teacher_ort_cpu", "teacher_pt_cpu", "teacher", "teacher_fp32"]
    # config looks like "mobilenetv3_small_none" — strip the KD suffix for the
    # latency lookup since the deployable model is the no-KD variant.
    arch = config.replace("_none", "").replace("_resp", "") \
                 .replace("_feat", "").replace("_comb", "")
    return [
        f"{arch}_fp32_ort_cpu",     # bench_m2 ORT FP32 (preferred)
        f"{arch}_pt_cpu_fp32",      # bench_m2 PyTorch CPU FP32 fallback
        f"{arch}_fp32",             # graviton/wasm flat names
        arch,                       # raw architecture name as last resort
    ]


def _lat_lookup_any(lat: dict | None, config: str) -> float | None:
    """Try every candidate key and return the first p50_ms hit."""
    if lat is None:
        return None
    for key in _candidate_lat_keys(config):
        v = get_p50_ms(lat, key)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# F2: Pareto scatter (1 × 4 subplots)
# ---------------------------------------------------------------------------

def build_f2(
    summary: dict,
    lat_m2: dict | None,
    lat_graviton: dict | None,
    lat_wasm: dict | None,
    out_dir: Path,
) -> None:
    """Build and save F2_pareto.png / .pdf."""

    # Collect IoU for the 5 ARCH configs
    ious: dict[str, float | None] = {k: get_iou(summary, k) for k in CONFIGS}

    # Decide x-axis data for the 4 subplots
    # Subplot 3: M2 latency, or placeholder
    # Subplot 4: Graviton latency → fallback to WASM → placeholder
    use_wasm_as_4th = (lat_graviton is None and lat_wasm is not None)

    subplot_specs = [
        {
            "xlabel": "Params (M)",
            "data": {k: ARCH[k]["params"] for k in CONFIGS},
            "placeholder": None,
        },
        {
            "xlabel": "FP32 ONNX size (MB)",
            "data": {k: ARCH[k]["size_mb"] for k in CONFIGS},
            "placeholder": None,
        },
        {
            "xlabel": "Apple M2 p50 (ms)",
            "data": (
                {k: _lat_lookup_any(lat_m2, k) for k in CONFIGS}
                if lat_m2 is not None else None
            ),
            "placeholder": (
                None if lat_m2 is not None
                else "Awaiting Phase C output:\nlat_m2.json missing"
            ),
        },
        {
            "xlabel": (
                "Browser WASM p50 (ms)" if use_wasm_as_4th
                else "AWS Graviton p50 (ms)"
            ),
            "data": (
                {k: _lat_lookup_any(lat_graviton, k) for k in CONFIGS}
                if lat_graviton is not None
                else (
                    {k: _lat_lookup_any(lat_wasm, k) for k in CONFIGS}
                    if use_wasm_as_4th else None
                )
            ),
            "placeholder": (
                None if (lat_graviton is not None or use_wasm_as_4th)
                else "Awaiting Phase C output:\nlat_graviton.json missing"
            ),
        },
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(
        "F2 — FloodLite Pareto: IoU vs. efficiency metrics",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, spec in zip(axes, subplot_specs):
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylabel("IoU (3-fold mean)", fontsize=10)
        ax.set_xlabel(spec["xlabel"], fontsize=10)

        if spec["data"] is None or spec["placeholder"] is not None:
            # Draw placeholder
            ax.set_facecolor("#f8f8f8")
            ax.text(
                0.5, 0.5,
                spec["placeholder"],
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10, color="#888888",
                style="italic",
                wrap=True,
            )
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            continue

        # Plot each config
        for cfg in CONFIGS:
            x_val = spec["data"].get(cfg)
            y_val = ious.get(cfg)
            if x_val is None or y_val is None:
                continue
            ax.scatter(
                x_val, y_val,
                color=COLORS[cfg],
                s=90, zorder=3, edgecolors="white", linewidths=0.5,
            )
            ax.annotate(
                SHORT_LABEL[cfg],
                xy=(x_val, y_val),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=7.5,
                color=COLORS[cfg],
            )

        # Auto-scale y with a small margin
        valid_y = [v for v in ious.values() if v is not None]
        if valid_y:
            ymin, ymax = min(valid_y), max(valid_y)
            margin = max((ymax - ymin) * 0.4, 0.002)
            ax.set_ylim(ymin - margin, ymax + margin)

    # Legend (shared across all subplots)
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=COLORS[cfg],
            markeredgecolor="white",
            markersize=8,
            label=SHORT_LABEL[cfg].replace("\n", " "),
        )
        for cfg in CONFIGS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(CONFIGS),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = out_dir / f"F2_pareto.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F4: Throughput bar chart (supplementary)
# ---------------------------------------------------------------------------

def _fps(p50_ms: float | None) -> float | None:
    if p50_ms is None or p50_ms <= 0:
        return None
    return 1000.0 / p50_ms


def build_f4(
    lat_m2: dict | None,
    lat_graviton: dict | None,
    lat_wasm: dict | None,
    out_dir: Path,
) -> None:
    """Build and save F4_throughput.png if at least one platform JSON exists."""

    # Gather per-platform data
    platforms: list[dict] = []

    def _collect_platform(lat: dict | None, label: str) -> None:
        """Build FP32/INT8 FPS lists for the 3 students under multiple key shapes.

        bench_m2.py:   {arch}_fp32_ort_cpu / {arch}_int8_ort_cpu (preferred)
                       {arch}_pt_cpu_fp32 (PyTorch CPU fallback)
        bench_graviton/wasm: {arch}_fp32 / {arch}_int8 (flat)
        """
        if lat is None:
            return
        fp32_fps: list[float | None] = []
        int8_fps: list[float | None] = []
        for cfg in STUDENTS_F4:
            arch = cfg.replace("_none", "")
            fp32_keys = [f"{arch}_fp32_ort_cpu", f"{arch}_pt_cpu_fp32", f"{arch}_fp32"]
            int8_keys = [f"{arch}_int8_ort_cpu", f"{arch}_int8"]
            fp32_ms = next((get_p50_ms(lat, k) for k in fp32_keys
                            if get_p50_ms(lat, k) is not None), None)
            int8_ms = next((get_p50_ms(lat, k) for k in int8_keys
                            if get_p50_ms(lat, k) is not None), None)
            fp32_fps.append(_fps(fp32_ms))
            int8_fps.append(_fps(int8_ms))
        if any(v is not None for v in fp32_fps + int8_fps):
            platforms.append({"label": label, "fp32": fp32_fps, "int8": int8_fps})

    _collect_platform(lat_m2, "Apple M2 Pro")
    _collect_platform(lat_graviton, "AWS Graviton")
    _collect_platform(lat_wasm, "Browser WASM")

    if not platforms:
        print("  F4: no platform lat JSONs found — skipping F4_throughput.png")
        return

    n_platforms = len(platforms)
    fig, axes = plt.subplots(1, n_platforms, figsize=(5.5 * n_platforms, 4.5), sharey=False)
    if n_platforms == 1:
        axes = [axes]

    fig.suptitle(
        "F4 — FloodLite throughput: FP32 vs INT8 per platform",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(STUDENTS_F4))
    bar_width = 0.35
    student_labels = [STUDENTS_SHORT[s] for s in STUDENTS_F4]

    for ax, plat in zip(axes, platforms):
        fp32_vals = [v if v is not None else 0.0 for v in plat["fp32"]]
        int8_vals = [v if v is not None else 0.0 for v in plat["int8"]]

        bars_fp32 = ax.bar(
            x - bar_width / 2, fp32_vals, bar_width,
            label="FP32", color="#4e79a7", edgecolor="white",
        )
        bars_int8 = ax.bar(
            x + bar_width / 2, int8_vals, bar_width,
            label="INT8", color="#f28e2b", edgecolor="white",
        )

        # Value labels above bars
        for bar in bars_fp32:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8,
                )
        for bar in bars_int8:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_title(plat["label"], fontsize=11, fontweight="bold")
        ax.set_ylabel("FPS (= 1000 / p50 ms)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(student_labels, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "F4_throughput.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Emit F2 (Pareto scatter) and F4 (throughput bars) figures "
            "for the FloodLite manuscript."
        )
    )
    p.add_argument(
        "--runs",
        default="runs/3fold",
        help="Directory containing summary.json and lat_*.json files (default: runs/3fold)",
    )
    p.add_argument(
        "--out_dir",
        default="figures",
        help="Output directory for figures (default: figures)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs)
    out_dir = Path(args.out_dir)

    print(f"[make_figures] runs dir : {runs_dir.resolve()}")
    print(f"[make_figures] output   : {out_dir.resolve()}")

    # Load inputs
    summary = load_json(runs_dir / "summary.json")
    if summary is None:
        print(
            f"  WARNING: {runs_dir / 'summary.json'} not found — "
            "F2 IoU axis will be empty."
        )
        summary = {}

    lat_m2 = load_json(runs_dir / "lat_m2.json")
    lat_graviton = load_json(runs_dir / "lat_graviton.json")
    lat_wasm = load_json(runs_dir / "lat_wasm.json")

    for name, data in [("lat_m2", lat_m2), ("lat_graviton", lat_graviton), ("lat_wasm", lat_wasm)]:
        status = "loaded" if data is not None else "not found (placeholder will be shown)"
        print(f"  {name}.json: {status}")

    print("\n-- F2 Pareto --")
    build_f2(summary, lat_m2, lat_graviton, lat_wasm, out_dir)

    print("\n-- F4 Throughput --")
    build_f4(lat_m2, lat_graviton, lat_wasm, out_dir)

    print("\n[make_figures] Done.")


if __name__ == "__main__":
    main()
