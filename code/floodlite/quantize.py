"""Post-training INT8 quantization (PyTorch static quantization).

PyTorch eager-mode static PTQ has two hard requirements that SMP UNet does
not satisfy out of the box:

1. **CPU-only.** Calibration and conversion must run on CPU.

       NotImplementedError: Could not run 'quantized::conv2d.new' with
       arguments from the 'CUDA' backend.

2. **Explicit quant boundaries.** The model must wrap its input in a
   ``QuantStub`` (FP32 -> quint8) and its output in a ``DeQuantStub``
   (quint8 -> FP32). Without these, ``convert()`` swaps in quantized ops
   (e.g. ``quantized::batch_norm2d``) that then receive plain FP32 inputs
   and raise:

       Could not run 'quantized::batch_norm2d' with arguments from the
       'CPU' backend.

This module addresses both: forces CPU device for the model and calibration
tensors, and wraps the model in ``QuantWrapper`` to provide the missing
quant/dequant boundaries before ``prepare()`` runs.
"""
from __future__ import annotations
from pathlib import Path
import copy
import io

import torch
import torch.nn as nn
import torch.ao.quantization as tq


class QuantWrapper(nn.Module):
    """Adds QuantStub/DeQuantStub boundaries around a float model.

    PyTorch eager-mode static PTQ only quantizes the activations between a
    ``QuantStub`` (input) and a ``DeQuantStub`` (output). SMP UNet has
    neither, so without this wrapper the quantized graph has float inputs
    feeding into quantized ops and crashes at the first quantized op.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def _ensure_quantized_engine() -> str:
    """Ensure a quantization engine is active and return the qconfig string.

    On x86/Linux (Kaggle, Colab) torch typically auto-selects 'x86' or
    'onednn'.  On ARM/macOS (M-series) the default engine is 'none', so we
    explicitly set 'qnnpack' which works on both ARM and x86.
    """
    engine = torch.backends.quantized.engine
    if engine in ("x86", "onednn", "fbgemm"):
        return "x86"
    torch.backends.quantized.engine = "qnnpack"
    return "qnnpack"


def quantize_int8(model: nn.Module, calib_loader, *, n_calib_batches: int = 25,
                  device: str = "cpu") -> nn.Module:
    """Apply post-training static INT8 quantization with calibration.

    Returns a ``QuantWrapper(model)`` whose internal model has been converted
    to INT8. Forward calls accept FP32 input and return FP32 output, so
    downstream code that does ``compute_metrics(logits, target)`` or
    ``benchmark_latency(model)`` works unchanged.
    """
    if device != "cpu":
        # INT8 PTQ does not support CUDA — silently enforce CPU.
        device = "cpu"

    qconfig_str = _ensure_quantized_engine()

    # Wrap with quant/dequant boundaries BEFORE prepare()
    fp_model = copy.deepcopy(model).to(device).eval()
    wrapped = QuantWrapper(fp_model).to(device).eval()
    wrapped.qconfig = tq.get_default_qconfig(qconfig_str)

    tq.prepare(wrapped, inplace=True)

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            wrapped(x)
            if i + 1 >= n_calib_batches:
                break

    tq.convert(wrapped, inplace=True)
    return wrapped


def save_quantized(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def model_size_mb(model: nn.Module) -> float:
    """Estimate on-disk model size in MB by serialising to a buffer."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
