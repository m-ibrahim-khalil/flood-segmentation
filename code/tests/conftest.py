"""Shared pytest fixtures for floodlite tests."""
from __future__ import annotations
import torch
import pytest


@pytest.fixture
def tiny_batch():
    """A single 3-channel 256x256 image and binary mask, on CPU."""
    x = torch.randn(1, 3, 256, 256)
    y = (torch.rand(1, 1, 256, 256) > 0.5).float()
    return x, y


@pytest.fixture
def cpu_device():
    return torch.device("cpu")
