"""Task and knowledge-distillation losses for FloodLite."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation (logits → sigmoid → Dice)."""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(1)
    union = probs.sum(1) + target.sum(1)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def task_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * BCE + 0.5 * Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * dice_loss(logits, target)


def response_kd_loss(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor,
                     T: float = 4.0) -> torch.Tensor:
    """Temperature-scaled binary KL divergence (response-based KD).

    For binary segmentation we approximate the KL between Bernoulli
    distributions via element-wise BCE between teacher's soft labels and
    student's softened logits — equivalent up to an additive constant.
    """
    p_t = torch.sigmoid(teacher_logits / T).detach()
    log_q_s = F.logsigmoid(student_logits / T)
    log_q_s_neg = F.logsigmoid(-student_logits / T)
    loss = -(p_t * log_q_s + (1 - p_t) * log_q_s_neg).mean()
    return loss * (T * T)


class FeatureKDLoss(nn.Module):
    """Feature-based KD with a learned 1x1 channel adapter."""

    def __init__(self, in_channels_student: int, in_channels_teacher: int):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels_student, in_channels_teacher, kernel_size=1, bias=False)

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Match spatial sizes if needed
        if f_s.shape[-2:] != f_t.shape[-2:]:
            f_s = F.interpolate(f_s, size=f_t.shape[-2:], mode="bilinear", align_corners=False)
        f_s = self.adapter(f_s)
        return F.mse_loss(f_s, f_t.detach())
