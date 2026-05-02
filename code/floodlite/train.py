"""Training loops for teacher and FloodLite-distilled students."""
from __future__ import annotations
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .losses import task_loss, response_kd_loss, FeatureKDLoss
from .models import attach_decoder_hook
from .metrics import compute_metrics


def train_teacher(model, train_loader, val_loader, *, epochs: int = 50, lr: float = 1e-4,
                  device: str = "cuda", ckpt_path: str | Path | None = None):
    """Train the teacher model with task loss only."""
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in tqdm(train_loader, desc=f"Teacher ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = task_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        # validation
        model.eval()
        ms = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                ms.append(compute_metrics(logits, y).iou)
        miou = sum(ms) / len(ms)
        print(f"  ep {epoch+1:02d}  train_loss={loss.item():.4f}  val_IoU={miou:.4f}  time={time.time()-t0:.1f}s")

        if ckpt_path and miou > best_iou:
            best_iou = miou
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"   ✔ saved {ckpt_path} (IoU {best_iou:.4f})")
    return best_iou


def train_student_kd(student, teacher, train_loader, val_loader, *,
                     epochs: int = 50, lr: float = 5e-4,
                     alpha: float = 0.5, beta: float = 0.1, T: float = 4.0,
                     use_response: bool = True, use_feature: bool = True,
                     device: str = "cuda", ckpt_path: str | Path | None = None):
    """Train a student with the FloodLite combined KD loss.

    Args:
        alpha: weight on response-KD term (combined with task loss)
        beta:  weight on feature-KD term
        use_response, use_feature: toggle ablation modes
    """
    student = student.to(device)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # set up feature hooks at decoder block 2 (mid-resolution)
    s_hook = attach_decoder_hook(student, block_idx=2) if use_feature else None
    t_hook = attach_decoder_hook(teacher, block_idx=2) if use_feature else None

    feat_loss_fn = None
    opt_params = list(student.parameters())
    if use_feature:
        # peek at feature dims with one forward
        with torch.no_grad():
            x_peek = next(iter(train_loader))[0][:1].to(device)
            _ = student(x_peek); _ = teacher(x_peek)
            cs = s_hook.features.shape[1]
            ct = t_hook.features.shape[1]
        feat_loss_fn = FeatureKDLoss(cs, ct).to(device)
        opt_params += list(feat_loss_fn.parameters())

    opt = AdamW(opt_params, lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_iou = 0.0

    for epoch in range(epochs):
        student.train()
        t0 = time.time()
        for x, y in tqdm(train_loader, desc=f"Student ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                with torch.no_grad():
                    t_logits = teacher(x)
                    f_t = t_hook.features if use_feature else None
                s_logits = student(x)
                f_s = s_hook.features if use_feature else None

                L_task = task_loss(s_logits, y)
                L_resp = response_kd_loss(s_logits, t_logits, T=T) if use_response else torch.tensor(0.0, device=device)
                L_feat = feat_loss_fn(f_s, f_t) if use_feature else torch.tensor(0.0, device=device)
                L = (1 - alpha) * L_task + alpha * L_resp + beta * L_feat

            scaler.scale(L).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        student.eval()
        ms = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = student(x)
                ms.append(compute_metrics(logits, y).iou)
        miou = sum(ms) / len(ms)
        print(f"  ep {epoch+1:02d}  loss={L.item():.4f}  val_IoU={miou:.4f}  time={time.time()-t0:.1f}s")

        if ckpt_path and miou > best_iou:
            best_iou = miou
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), ckpt_path)
            print(f"   ✔ saved {ckpt_path} (IoU {best_iou:.4f})")

    if s_hook: s_hook.close()
    if t_hook: t_hook.close()
    return best_iou
