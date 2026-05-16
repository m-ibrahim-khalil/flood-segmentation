"""make_qualitative.py — emit F3 qualitative panel for the FloodLite manuscript.

Layout: 2 rows x 5 cols (easy + hard from FSSD fold-0 val).
Columns: image | ground truth | teacher FP32 pred | EffLite0 INT8 pred | error map

"Easy" = highest per-image IoU under the EffLite0-INT8 model on fold-0 val.
"Hard" = lowest per-image IoU under the same model (excluding all-background masks).

Usage:
    python3 scripts/make_qualitative.py \\
        --data_root /path/to/FSSD \\
        --teacher_onnx exports/teacher.onnx \\
        --student_onnx exports/efficientnet_lite0_fold0_int8.onnx \\
        --out figures/F3_qualitative.png

Notes:
    Both ONNX graphs are run via ONNX Runtime CPU EP. We use ONNX (not PyTorch
    checkpoints) so the rendered predictions exactly match the artifacts whose
    latency is reported in T5 -- nothing in the qualitative panel is hidden
    behind a PyTorch eager-mode forward pass.

    The teacher ONNX is the "teacher.onnx" file produced by scripts/export_all.py
    (no fold suffix on teacher; it is exported from teacher_fold0.pt only).
    If it does not yet exist, export it once:
        python3 scripts/export_all.py \\
            --ckpt_dir runs/3fold --data_root /path/to/FSSD --fold 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floodlite.data import collect_pairs  # noqa: E402

IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image_mask(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (image_rgb_uint8, image_chw_normalised_fp32, mask_uint8) at IMG_SIZE."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)

    img_f = img_rgb.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img_f, (2, 0, 1))[None].astype(np.float32)  # (1, 3, H, W)
    return img_rgb, img_chw, mask


def _run_session(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Run a single forward pass and return a (H, W) binary prediction at 0.5."""
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: x})[0]
    # Models output raw logits at (1, 1, H, W); take sigmoid then threshold.
    probs = 1.0 / (1.0 + np.exp(-out[0, 0]))
    return (probs > 0.5).astype(np.uint8)


def _iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / (union + eps)


def _make_session(onnx_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def _build_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Render an RGB error map: green=TP, red=FP, blue=FN, black=TN."""
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = (46, 204, 113)
    out[fp] = (231, 76, 60)
    out[fn] = (52, 152, 219)
    return out


def _overlay_mask(rgb_uint8: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int] = (0, 153, 255), alpha: float = 0.45) -> np.ndarray:
    overlay = rgb_uint8.copy().astype(np.float32)
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[mask == 1] = np.array(color, dtype=np.float32)
    blended = np.where(mask[..., None] == 1,
                       overlay * (1 - alpha) + mask_rgb * alpha,
                       overlay)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _kfold_val_pairs(data_root: Path, fold: int, n_folds: int, seed: int):
    from sklearn.model_selection import KFold
    train_pairs, val_pairs = collect_pairs(data_root)
    all_pairs = train_pairs + val_pairs
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(len(all_pairs))))
    _, val_idx = splits[fold]
    return [all_pairs[i] for i in val_idx.tolist()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=Path,
                    help="FSSD root, containing dataset/{train,val}/{images,labels}")
    ap.add_argument("--teacher_onnx", required=True, type=Path)
    ap.add_argument("--student_onnx", required=True, type=Path,
                    help="EfficientNet-Lite0 INT8 ONNX (or any deployable student ONNX)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5,
                    help="Match the KFold(n_splits=...) used during training. "
                         "Default 5 mirrors floodlite/data.py defaults so fold-0 val matches "
                         "the segmentation eval split.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_flood_frac", type=float, default=0.02,
                    help="Discard images whose GT has <2%% flood pixels when picking the hard example "
                         "-- a near-empty GT is trivially easy to under-segment and not informative.")
    args = ap.parse_args()

    if not args.teacher_onnx.exists():
        print(f"ERROR: teacher ONNX not found: {args.teacher_onnx}", file=sys.stderr)
        print("       export it once with: python3 scripts/export_all.py --include_teacher", file=sys.stderr)
        return 2
    if not args.student_onnx.exists():
        print(f"ERROR: student ONNX not found: {args.student_onnx}", file=sys.stderr)
        return 2

    val_pairs = _kfold_val_pairs(args.data_root, args.fold, args.n_folds, args.seed)
    print(f"[F3] fold-{args.fold} val set: {len(val_pairs)} pairs")

    student_sess = _make_session(args.student_onnx)
    teacher_sess = _make_session(args.teacher_onnx)

    # Score every val image under the deployable (student INT8) model -- this is what
    # we want the easy/hard split to reflect, not the teacher.
    per_image_iou: list[tuple[int, float, float]] = []  # (index, student_iou, gt_flood_frac)
    for i, (img_p, lbl_p) in enumerate(val_pairs):
        _, x, gt = _load_image_mask(img_p, lbl_p)
        pred = _run_session(student_sess, x)
        iou = _iou(pred, gt)
        flood_frac = float(gt.mean())
        per_image_iou.append((i, iou, flood_frac))
    per_image_iou.sort(key=lambda t: t[1])

    candidate_hard = [t for t in per_image_iou if t[2] >= args.min_flood_frac]
    if not candidate_hard:
        print(f"WARNING: no val image has flood_frac >= {args.min_flood_frac}; "
              f"falling back to the lowest-IoU image overall.", file=sys.stderr)
        candidate_hard = per_image_iou

    hard_idx = candidate_hard[0][0]
    easy_idx = per_image_iou[-1][0]
    print(f"[F3] easy idx={easy_idx} student IoU={per_image_iou[-1][1]:.4f} "
          f"flood_frac={per_image_iou[-1][2]:.3f}")
    print(f"[F3] hard idx={hard_idx} student IoU={candidate_hard[0][1]:.4f} "
          f"flood_frac={candidate_hard[0][2]:.3f}")

    # Render: 2 rows x 5 cols.
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    column_titles = ["Input", "Ground truth",
                     "Teacher (UNet+EffB0, FP32)",
                     "EffLite0-UNet (INT8)",
                     "Error map (TP/FP/FN)"]
    row_labels = [("Easy", easy_idx), ("Hard", hard_idx)]

    for r, (row_lbl, idx) in enumerate(row_labels):
        img_p, lbl_p = val_pairs[idx]
        rgb, x, gt = _load_image_mask(img_p, lbl_p)
        teacher_pred = _run_session(teacher_sess, x)
        student_pred = _run_session(student_sess, x)
        teacher_iou = _iou(teacher_pred, gt)
        student_iou = _iou(student_pred, gt)

        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(_overlay_mask(rgb, gt, color=(0, 153, 255), alpha=0.50))
        axes[r, 2].imshow(_overlay_mask(rgb, teacher_pred, color=(255, 200, 0), alpha=0.50))
        axes[r, 3].imshow(_overlay_mask(rgb, student_pred, color=(255, 200, 0), alpha=0.50))
        axes[r, 4].imshow(_build_error_map(student_pred, gt))

        # IoU annotations beneath the prediction cells.
        axes[r, 2].set_xlabel(f"IoU = {teacher_iou:.3f}", fontsize=10)
        axes[r, 3].set_xlabel(f"IoU = {student_iou:.3f}", fontsize=10)

        # Row label on the left side.
        axes[r, 0].set_ylabel(f"{row_lbl}\n{img_p.name}", fontsize=11, rotation=90,
                              labelpad=8, va="center")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for c, title in enumerate(column_titles):
        axes[0, c].set_title(title, fontsize=11, pad=6)

    # Legend strip for the error map.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=(46 / 255, 204 / 255, 113 / 255), label="TP"),
        plt.Rectangle((0, 0), 1, 1, fc=(231 / 255, 76 / 255, 60 / 255), label="FP"),
        plt.Rectangle((0, 0), 1, 1, fc=(52 / 255, 152 / 255, 219 / 255), label="FN"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.82, -0.01), fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    pdf_out = args.out.with_suffix(".pdf")
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F3] wrote {args.out}")
    print(f"[F3] wrote {pdf_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
