import os

import numpy as np
import pytest
import torch
from PIL import Image

# Tiny tensors + many OpenMP threads on a shared node = seconds per op. Keep the
# suite deterministic and fast regardless of the host's core count.
torch.set_num_threads(min(4, os.cpu_count() or 1))


def _synthetic(size: int, seed: int = 0, smooth: bool = True) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    r = 127 + 100 * np.sin(2 * np.pi * (x * 1.3 + 0.1))
    g = 127 + 100 * np.cos(2 * np.pi * (y * 0.7 + x * 0.4))
    b = 127 + 90 * np.sin(2 * np.pi * (x * y * 2.0))
    img = np.stack([r, g, b], axis=-1)
    if not smooth:
        img += rng.normal(0, 25, img.shape)
        img[size // 3: 2 * size // 3, size // 2:] = 240  # a sharp edge
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


@pytest.fixture
def smooth64() -> Image.Image:
    return _synthetic(64, smooth=True)


@pytest.fixture
def noisy64() -> Image.Image:
    return _synthetic(64, seed=1, smooth=False)


@pytest.fixture(params=[8, 16])
def block_size(request) -> int:
    return request.param
