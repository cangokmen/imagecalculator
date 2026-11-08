import math

import numpy as np
import pytest
import torch

from imagecalculator.transform import (
    block_dct, block_idct, blockify, chroma_subsample, dct_matrices, deblockify, extract_diag_band,
    reduced_block_idct, rgb_to_ycrcb, ycrcb_to_rgb,
)
from imagecalculator.bands import diag_support_index


def _blockify_loop(im, size):
    bs, ch, h, w = im.shape
    out = torch.zeros(bs, ch, (h // size) * (w // size), size, size)
    for b in range(bs):
        for c in range(ch):
            L = 0
            for i in range(0, h, size):
                for j in range(0, w, size):
                    out[b, c, L] = im[b, c, i:i + size, j:j + size]
                    L += 1
    return out


def test_blockify_matches_loop_and_roundtrips():
    im = torch.randn(2, 3, 32, 48)
    b = blockify(im, 8)
    assert torch.equal(b, _blockify_loop(im, 8))
    assert torch.equal(deblockify(b, (32, 48)), im)


def test_dct_roundtrip_and_orthonormality(block_size):
    n, h = dct_matrices(block_size)
    x = torch.rand(1, 3, 4, block_size, block_size) * 255
    c = block_dct(x, n, h)
    assert torch.allclose(block_idct(c, n, h), x, atol=1e-3)
    assert torch.allclose((x ** 2).sum(dim=(-2, -1)), (c ** 2).sum(dim=(-2, -1)), rtol=1e-4)


def test_reduced_idct_constant_block_pins_scale():
    N = 16
    n, h = dct_matrices(N)
    v = 173.0
    c = block_dct(torch.full((1, 1, 1, N, N), v), n, h)
    assert abs(c[0, 0, 0, 0, 0].item() - N * v) < 1e-2
    for m in range(1, N + 1):
        tile = reduced_block_idct(c, m)
        assert tile.shape[-2:] == (m, m)
        assert torch.allclose(tile, torch.full_like(tile, v), atol=1e-3), m


def test_reduced_idct_full_size_is_block_idct():
    N = 8
    n, h = dct_matrices(N)
    c = torch.randn(1, 3, 5, N, N) * 50
    assert torch.equal(reduced_block_idct(c, N), block_idct(c, n, h))
    with pytest.raises(ValueError):
        reduced_block_idct(c, 0)
    with pytest.raises(ValueError):
        reduced_block_idct(c, N + 1)


def test_reduced_idct_is_close_to_area_downscale(smooth64):
    # m-point decode of a k-diagonal truncation vs area-downscale of the full image
    N, m = 8, 4
    x = torch.from_numpy(np.array(smooth64, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    n, h = dct_matrices(N)
    c = block_dct(blockify(x, N), n, h)
    small = deblockify(reduced_block_idct(c, m), (32, 32))
    ref = torch.nn.functional.avg_pool2d(x, 2)
    psnr = 10 * math.log10(255 ** 2 / ((small - ref) ** 2).mean().item())
    assert psnr > 35, psnr


def test_diag_gather_matches_mask_fold_oracle(block_size):
    S = 64
    x = torch.rand(1, 3, S, S) * 255
    n, h = dct_matrices(block_size)
    blocks = block_dct(blockify(x, block_size), n, h)
    plane = deblockify(blocks, (S, S)).reshape(3, -1)
    idx = diag_support_index(S, block_size)
    for d in range(2 * block_size - 1):
        oracle = extract_diag_band(blocks, d, d + 1, S).reshape(3, -1)
        assert torch.equal(oracle[:, idx[d]], plane[:, idx[d]])
        other = np.setdiff1d(np.arange(S * S), idx[d])
        assert (oracle[:, other] == 0).all()


def test_ycrcb_roundtrip_and_subsampling():
    x = torch.rand(1, 3, 16, 16) * 255
    y = rgb_to_ycrcb(x)
    assert torch.allclose(ycrcb_to_rgb(y), x, atol=0.6)
    assert torch.equal(chroma_subsample(y, "1x1"), y)
    s = chroma_subsample(y, "4x4")
    assert torch.equal(s[:, 0], y[:, 0]) and s.shape == y.shape
    assert torch.equal(s[:, 1, 0, 0], s[:, 1, 1, 1])  # 2x2 cells share a value
    with pytest.raises(ValueError):
        chroma_subsample(y, "3x3")
