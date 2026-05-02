"""Tests for the Sen1Floods11 RGB OOD loader."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import pytest

from floodlite.data import make_sen1floods11_loader


@pytest.fixture
def fake_s1f11_root(tmp_path: Path) -> Path:
    """Build a fake Sen1Floods11-style folder with 4 RGB chips and binary masks."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "labels"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        msk = (rng.integers(0, 2, (512, 512), dtype=np.uint8) * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"chip_{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_dir / f"chip_{i:03d}.png"), msk)
    return tmp_path


def test_make_sen1floods11_loader_yields_correct_tensor_shape(fake_s1f11_root):
    loader = make_sen1floods11_loader(fake_s1f11_root, n_chips=4, batch_size=2)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 256, 256)
    assert y.shape == (2, 1, 256, 256)
    # Mask should be 0/1 binary
    unique_vals = set(y.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_make_sen1floods11_loader_caps_at_n_chips(fake_s1f11_root):
    loader = make_sen1floods11_loader(fake_s1f11_root, n_chips=2, batch_size=1)
    n = sum(1 for _ in loader)
    assert n == 2
