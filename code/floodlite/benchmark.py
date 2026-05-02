"""Latency benchmarking utilities (CPU / Apple MPS / CUDA)."""
from __future__ import annotations
import time
import torch
import numpy as np


@torch.no_grad()
def benchmark_latency(model: torch.nn.Module, *, img_size: int = 256, n_warm: int = 20,
                      n_iter: int = 200, device: str = "cpu") -> dict[str, float]:
    """Measure latency in ms over many forward passes.

    Returns p50, p95, mean, std in milliseconds.
    """
    model = model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    # warm-up
    for _ in range(n_warm):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "std_ms":  float(arr.std()),
        "fps":     float(1000.0 / np.percentile(arr, 50)),
    }
