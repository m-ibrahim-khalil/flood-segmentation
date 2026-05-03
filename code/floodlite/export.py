"""Convert trained checkpoint to ONNX → CoreML / TFLite / WASM."""
from __future__ import annotations
from pathlib import Path
import torch


def export_onnx(model: torch.nn.Module, out_path: str | Path,
                img_size: int = 256, opset: int = 13) -> Path:
    """Export a model to a single self-contained ONNX file.

    Forces ``dynamo=False`` to use the legacy TorchScript-based exporter:
      * Produces ONE .onnx file with weights embedded (no external .data
        sidecars), so file_size_mb() reports the true model size.
      * Honours ``opset_version`` exactly (the new dynamo exporter silently
        upgrades to opset 18 on PyTorch 2.8+ and then fails to down-convert
        ops like Pad/Resize back to 17).
      * Avoids the 'dynamic_axes is not recommended when dynamo=True' warning.

    The legacy exporter is deprecated in PyTorch 2.9+ but still functional;
    we will revisit when it is removed.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    return Path(out_path)


def export_coreml(onnx_path: str | Path, out_path: str | Path) -> Path:
    """Convert ONNX → CoreML using coremltools.

    Requires: pip install coremltools
    """
    import coremltools as ct
    mlmodel = ct.converters.onnx.convert(model=str(onnx_path), minimum_ios_deployment_target="13")
    mlmodel.save(str(out_path))
    return Path(out_path)


def export_tflite(onnx_path: str | Path, out_path: str | Path) -> Path:
    """Convert ONNX → TFLite via onnx-tf and TF Lite Converter.

    Requires: pip install onnx-tf tensorflow
    """
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_dir = str(Path(out_path).with_suffix(".tf"))
    tf_rep.export_graph(tf_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    return Path(out_path)
