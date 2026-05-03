"""Post-training INT8 quantization for FloodLite.

Two independent paths are exposed:

1. ``quantize_int8`` — PyTorch eager-mode static PTQ. Wraps the model with
   QuantStub/DeQuantStub and runs prepare/calibrate/convert on CPU.
   Works for "vanilla" CNNs but fails on timm encoders that use
   ``Conv2dSame`` (TF-style asymmetric padding) because the underlying
   ``aten::_slow_conv2d_forward`` op has no quantized CPU kernel:

       Could not run 'aten::_slow_conv2d_forward' with arguments from
       the 'QuantizedCPU' backend.

   This affects MobileNetV3-Small, MobileNetV2 (depending on smp version),
   and other timm-backed encoders. **Use ``quantize_int8_onnx_dynamic``
   for these — it's the recommended path for FloodLite on Kaggle.**

2. ``quantize_int8_onnx_dynamic`` — exports the model to ONNX FP32, then
   uses ``onnxruntime.quantization.quantize_dynamic`` to produce an INT8
   ONNX file. Weights are quantized to INT8 at conversion time;
   activations are quantized at inference time by onnxruntime. Works on
   any model exportable to ONNX, including all three FloodLite students.
   This is the workflow used for the IJDRR submission's INT8 numbers.
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


def file_size_mb(path) -> float:
    """Return the on-disk size of a file in MB."""
    return Path(path).stat().st_size / (1024 ** 2)


def quantize_int8_onnx_dynamic(model: nn.Module, *, fp_onnx_path,
                               int8_onnx_path, img_size: int = 256,
                               opset: int = 17):
    """Export ``model`` to FP32 ONNX, then run ONNX dynamic INT8 quantization.

    Reliable alternative to PyTorch eager-mode static PTQ for models
    containing timm Conv2dSame, MobileViT attention, or other ops without
    quantized CPU kernels in PyTorch 2.4+. Only weights are stored as INT8;
    activations are quantized at runtime by onnxruntime.

    NOTE: produces ``ConvInteger`` ops which lack CPU EP support in some
    onnxruntime versions. Prefer ``quantize_int8_onnx_static`` for Conv-heavy
    models — that path uses QDQ format with standard Conv ops that run
    everywhere.

    Returns the path to the INT8 ONNX file.
    """
    from .export import export_onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    fp_onnx_path = Path(fp_onnx_path)
    int8_onnx_path = Path(int8_onnx_path)
    fp_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_onnx(model, fp_onnx_path, img_size=img_size, opset=opset)
    quantize_dynamic(
        str(fp_onnx_path),
        str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    return int8_onnx_path


class _TorchLoaderCalibReader:
    """Adapt a PyTorch DataLoader to onnxruntime.quantization.CalibrationDataReader.

    Yields up to ``n_batches`` calibration batches, each as a single-item dict
    keyed by the ONNX model's input tensor name. Returns ``None`` from
    ``get_next`` when exhausted, which is the contract onnxruntime expects.
    """

    def __init__(self, loader, input_name: str, n_batches: int = 10):
        self._gen = self._iterate(loader, input_name, n_batches)

    @staticmethod
    def _iterate(loader, input_name: str, n_batches: int):
        import numpy as np
        seen = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            yield {input_name: np.asarray(x, dtype="float32")}
            seen += 1
            if seen >= n_batches:
                return

    def get_next(self):
        try:
            return next(self._gen)
        except StopIteration:
            return None


def quantize_int8_onnx_static(model: nn.Module, *, fp_onnx_path, int8_onnx_path,
                              calib_loader, n_calib_batches: int = 25,
                              img_size: int = 256, opset: int = 17):
    """Static INT8 ONNX quantization in QDQ format with calibration.

    Workflow:
        1. Export ``model`` -> FP32 ONNX.
        2. Pre-process the FP32 ONNX (symbolic shape inference + BN-into-Conv
           fusion + other graph optimisations). Without this step the
           quantizer prints
           "Please consider to run pre-processing before quantization"
           and produces noticeably worse INT8 accuracy because BN weights
           are quantized separately rather than absorbed into Conv weights.
        3. Calibrate using up to ``n_calib_batches`` batches from
           ``calib_loader`` (drawn from the training distribution); 25 batches
           × batch_size 8 = 200 calibration images per the spec.
        4. Emit an INT8 ONNX in QDQ format — QuantizeLinear/DequantizeLinear
           are placed around standard Conv/MatMul ops, so the resulting
           graph uses only ops with broad onnxruntime CPU support.

    Both weights and activations are quantized to INT8. This is the
    recommended quantization path for FloodLite on Kaggle.

    Returns the path to the INT8 ONNX file.
    """
    from .export import export_onnx
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp_onnx_path = Path(fp_onnx_path)
    int8_onnx_path = Path(int8_onnx_path)
    fp_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_onnx(model, fp_onnx_path, img_size=img_size, opset=opset)

    # Pre-process: symbolic shape inference + Conv-BN fusion + cleanup.
    # Symbolic inference is brittle on some SMP UNet variants ('Incomplete
    # symbolic shape inference'); fall back to skipping that step alone if it
    # fails. ONNX shape inference + graph optimisation still run, which is
    # enough to get the BN-into-Conv fusion that drives the IoU-preservation
    # benefit.
    preproc_path = fp_onnx_path.with_name(fp_onnx_path.stem + "_preproc.onnx")
    try:
        quant_pre_process(
            input_model_path=str(fp_onnx_path),
            output_model_path=str(preproc_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
        )
    except Exception:
        try:
            quant_pre_process(
                input_model_path=str(fp_onnx_path),
                output_model_path=str(preproc_path),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=True,
            )
        except Exception:
            # Ultimate fallback: skip preprocessing entirely
            import shutil
            shutil.copy(str(fp_onnx_path), str(preproc_path))

    # Discover the ONNX input name from the pre-processed model
    import onnxruntime as ort
    sess = ort.InferenceSession(str(preproc_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    reader = _TorchLoaderCalibReader(calib_loader, input_name=input_name,
                                     n_batches=n_calib_batches)
    # Calibration method: Percentile (clip extremes at 99.999%) is much more
    # robust to activation outliers than MinMax. The 3-fold Kaggle sweep with
    # MinMax produced wildly inconsistent INT8 IoU (mobilenetv3_small dropped
    # to 0.0002 in fold-1 vs 0.59 in fold-2). MobileNetV3's HardSwish and
    # MobileViT's attention both produce occasional large activations that
    # MinMax interprets as the full quantization range, dropping resolution
    # for the bulk of the activation distribution. Percentile is the
    # standard fix.
    #
    # per_channel=False is required for ViT-style models (MobileViT).
    # LayerNorm weights are rank-1; the per-channel quantizer assumes axis-1
    # exists and silently produces broken weights for those layers.
    from onnxruntime.quantization.calibrate import CalibrationMethod
    quantize_static(
        str(preproc_path),
        str(int8_onnx_path),
        reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            "CalibPercentile": 99.999,
            "CalibMaxIntermediateOutputs": 50,
        },
    )
    return int8_onnx_path


def evaluate_onnx(onnx_path, loader, *, threshold: float = 0.5) -> dict:
    """Evaluate an ONNX model on a data loader, returning the same metrics
    schema as ``floodlite.metrics.compute_metrics`` (mean across batches)."""
    import onnxruntime as ort
    import numpy as np
    from .metrics import compute_metrics

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    keys = ("accuracy", "precision", "recall", "f1", "iou")
    accum = {k: 0.0 for k in keys}
    n = 0
    for x, y in loader:
        x_np = x.detach().cpu().numpy().astype("float32") if hasattr(x, "detach") else np.asarray(x, dtype="float32")
        logits_np = sess.run(None, {in_name: x_np})[0]
        logits = torch.from_numpy(logits_np)
        m = compute_metrics(logits, y)
        for k in keys:
            accum[k] += getattr(m, k)
        n += 1
    return {k: accum[k] / max(n, 1) for k in keys}
