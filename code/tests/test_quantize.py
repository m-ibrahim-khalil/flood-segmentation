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
