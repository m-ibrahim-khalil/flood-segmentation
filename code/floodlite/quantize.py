"""Post-training INT8 quantization (PyTorch static quantization)."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import copy


def quantize_int8(model: nn.Module, calib_loader, *, n_calib_batches: int = 25,
                  device: str = "cpu") -> nn.Module:
    """Apply post-training static INT8 quantization with calibration.

    The calibration set should be drawn from the training distribution.
    """
    model = copy.deepcopy(model).to(device).eval()

    # set qconfig (per-channel weights, per-tensor activations, symmetric)
    model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(model, inplace=True)

    # calibrate
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            model(x.to(device))
            if i + 1 >= n_calib_batches:
                break

    torch.ao.quantization.convert(model, inplace=True)
    return model


def save_quantized(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def model_size_mb(model: nn.Module) -> float:
    """Estimate on-disk model size in MB by serialising to a buffer."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
