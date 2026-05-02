"""Segmentation metrics."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class SegMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / (b + eps)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> SegMetrics:
    """Compute pixel accuracy, precision, recall, F1, IoU for binary masks."""
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    tgt = (target >= 0.5).float()

    tp = (pred * tgt).sum()
    fp = (pred * (1 - tgt)).sum()
    fn = ((1 - pred) * tgt).sum()
    tn = ((1 - pred) * (1 - tgt)).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    iou = _safe_div(tp, tp + fp + fn)

    return SegMetrics(acc.item(), prec.item(), rec.item(), f1.item(), iou.item())
