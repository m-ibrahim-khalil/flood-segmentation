"""Tests for benchmark utilities."""
from __future__ import annotations
from pathlib import Path
import torch
import pytest

from floodlite.models import make_student
from floodlite.export import export_onnx
from floodlite.benchmark import benchmark_latency, benchmark_onnxruntime


def test_benchmark_latency_returns_expected_keys():
    model = make_student("mobilenetv3_small")
    out = benchmark_latency(model, n_warm=2, n_iter=4, device="cpu")
    assert set(out.keys()) == {"p50_ms", "p95_ms", "mean_ms", "std_ms", "fps"}
    assert out["p50_ms"] > 0


def test_benchmark_onnxruntime_returns_same_schema(tmp_path: Path):
    model = make_student("mobilenetv3_small")
    onnx_path = tmp_path / "m.onnx"
    export_onnx(model, onnx_path, opset=17)
    out = benchmark_onnxruntime(onnx_path, n_warm=2, n_iter=4)
    assert set(out.keys()) == {"p50_ms", "p95_ms", "mean_ms", "std_ms", "fps"}
    assert out["p50_ms"] > 0
