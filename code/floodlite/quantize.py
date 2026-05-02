"""Post-training INT8 quantization (PyTorch static quantization).

INT8 PTQ in torch.ao.quantization requires the model and calibration inputs
to live on CPU; running with CUDA tensors raises:

    NotImplementedError: Could not run 'quantized::conv2d.new' with
    arguments from the 'CUDA' backend.

This module forces both the model and the calibration tensors to CPU before
`prepare()`, regardless of where the calibration loader yields them.
"""
from __future__ import annotations
from pathlib import Path
import copy
import io

import torch
import torch.nn as nn


def _ensure_quantized_engine() -> str:
    """Ensure a quantization engine is active and return the qconfig string.

    On x86/Linux (Kaggle, Colab) torch typically auto-selects 'x86' or
    'onednn'.  On ARM/macOS (M-series) the default engine is 'none', so we
    explicitly set 'qnnpack' which works on both ARM and x86.
    """
    engine = torch.backends.quantized.engine
    if engine in ("x86", "onednn", "fbgemm"):
        return "x86"
    # Fall back to qnnpack (ARM macOS, Raspberry Pi, or engine == 'none')
    torch.backends.quantized.engine = "qnnpack"
    return "qnnpack"


def quantize_int8(model: nn.Module, calib_loader, *, n_calib_batches: int = 25,
                  device: str = "cpu") -> nn.Module:
    """Apply post-training static INT8 quantization with calibration.

    The calibration set should be drawn from the training distribution.
    Both ``model`` and every batch yielded by ``calib_loader`` are moved to
    CPU before quantization runs — this is mandatory for torch.ao.quantization.
    """
    if device != "cpu":
        # Quietly enforce CPU; INT8 PTQ does not support CUDA.
        device = "cpu"

    qconfig_str = _ensure_quantized_engine()
    model = copy.deepcopy(model).to(device).eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig(qconfig_str)
    torch.ao.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()  # force CPU + FP32 regardless of loader
            model(x)
            if i + 1 >= n_calib_batches:
                break

    torch.ao.quantization.convert(model, inplace=True)
    return model


def save_quantized(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def model_size_mb(model: nn.Module) -> float:
    """Estimate on-disk model size in MB by serialising to a buffer."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
