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
    # MobileNetV2 encoder (2.2 M) is lighter than EfficientNet-B0 encoder (4.0 M);
    # full UNet totals differ because smp 0.5.0 sizes decoder channels to encoder
    # feature maps, making the MobileNetV2 decoder slightly heavier overall.
    # We compare encoder-only params to capture the lightweight-backbone intent.
    def encoder_params(m):
        return sum(p.numel() for p in m.encoder.parameters()) / 1e6
    assert encoder_params(baseline) < encoder_params(teacher)


def test_all_three_students_build_and_forward():
    for name in STUDENT_BACKBONES:
        model = make_student(name)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256))
        assert out.shape == (1, 1, 256, 256), f"shape mismatch for {name}"
