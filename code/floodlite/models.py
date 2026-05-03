"""Teacher and student segmentation models for FloodLite.

Uses segmentation-models-pytorch (smp) for UNet decoder + timm-backed
encoders. Three students are exposed:
    - mobilenetv3_small_100  (CPU-friendly)
    - tf_efficientnet_lite0  (quantization-friendly)
    - mobilevit_xxs          (hybrid CNN+ViT)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


TEACHER_BACKBONE = "efficientnet-b0"  # smp encoder name


def _mobilevit_encoder_name() -> str:
    """Pick a MobileViT-XXS encoder name that works on the installed smp version.

    smp 0.3.x ships ``mobilevit_xxs`` as a static encoder addressable via the
    ``timm-`` prefix. smp 0.4+ removed the static entry; the encoder is only
    reachable through the timm-universal (``tu-``) dynamic prefix instead.
    Kaggle's base image has ranged across both over time, so we detect at
    import and pick the form that resolves.
    """
    smp_version = tuple(int(p) for p in smp.__version__.split(".")[:2])
    return "tu-mobilevit_xxs" if smp_version >= (0, 4) else "timm-mobilevit_xxs"


STUDENT_BACKBONES = {
    "mobilenetv3_small": "timm-mobilenetv3_small_100",
    "efficientnet_lite0": "timm-tf_efficientnet_lite0",
    "mobilevit_xxs":      _mobilevit_encoder_name(),
}


def make_teacher(num_classes: int = 1) -> nn.Module:
    """UNet + EfficientNet-B0 teacher."""
    return smp.Unet(
        encoder_name=TEACHER_BACKBONE,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,  # logits (apply sigmoid externally)
    )


def make_student(name: str, num_classes: int = 1) -> nn.Module:
    """Build one of the three lightweight UNet students."""
    if name not in STUDENT_BACKBONES:
        raise ValueError(f"Unknown student '{name}'. Choose from {list(STUDENT_BACKBONES)}")
    return smp.Unet(
        encoder_name=STUDENT_BACKBONES[name],
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


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


class FeatureHook:
    """Tap a single decoder block's output for feature-based KD.

    Usage:
        h = FeatureHook(model.decoder.blocks[2])
        out = model(x)
        feat = h.features  # cleared by next forward
    """

    def __init__(self, module: nn.Module):
        self.features: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        self.features = out

    def close(self):
        self.handle.remove()


def attach_decoder_hook(model: nn.Module, block_idx: int = 2) -> FeatureHook:
    """Hook the decoder block at ``block_idx`` (counted from low res to high res)."""
    return FeatureHook(model.decoder.blocks[block_idx])


def count_params(model: nn.Module) -> float:
    """Return million-parameter count."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops(model: nn.Module, img_size: int = 256, device: str = "cpu") -> float:
    """Estimate GFLOPs at given input resolution using torchprofile.

    Falls back to ``thop`` or returns NaN if neither is installed.
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    try:
        from torchprofile import profile_macs
        macs = profile_macs(model.to(device), x)
        return macs * 2 / 1e9  # FLOPs ≈ 2× MACs
    except Exception:
        try:
            from thop import profile
            macs, _ = profile(model.to(device), inputs=(x,), verbose=False)
            return macs * 2 / 1e9
        except Exception:
            return float("nan")
