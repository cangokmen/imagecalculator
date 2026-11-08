"""Pure-torch primitives: block DCT, blockify/deblockify, reduced-size inverse
DCT, YCrCb conversion and chroma subsampling.

Expression order is kept identical to the research code (``freq/dct.py``,
``freq/_block.py``, ``freq/ycrcb.py``) so results are bit-exact on the same
device.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# DCT matrices
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _dct_matrices_cpu(N: int) -> tuple[Tensor, Tensor]:
    n_mat = torch.ones((N, 1))
    n_mat[0, 0] = 1.0 / math.sqrt(2)
    n_mat = n_mat @ n_mat.t()
    sp = torch.arange(float(N)).reshape((N, 1))
    fr = torch.arange(float(N)).reshape((1, N))
    h_mat = torch.cos((2 * sp + 1) @ (fr * math.pi / (2 * N)))
    return n_mat, h_mat


def dct_matrices(N: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """``(n, h)``: the orthonormal scale matrix and the cosine harmonics for an ``N``-point DCT.

    ``h[x, i] = cos((2x+1) i pi / 2N)`` (spatial rows, spectral columns);
    ``n`` is 1 except ``1/sqrt(2)`` on row/column 0.
    """
    dev = torch.device(device)
    if dev.type == "cpu":
        return _dct_matrices_cpu(N)
    # Build on the device exactly as the research builder does (`_build_n_h`).
    n_mat = torch.ones((N, 1), device=dev)
    n_mat[0, 0] = 1.0 / math.sqrt(2)
    n_mat = n_mat @ n_mat.t()
    sp = torch.arange(float(N), device=dev).reshape((N, 1))
    fr = torch.arange(float(N), device=dev).reshape((1, N))
    h_mat = torch.cos((2 * sp + 1) @ (fr * math.pi / (2 * N)))
    return n_mat, h_mat


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def blockify(im: Tensor, size: int) -> Tensor:
    """``(B, C, H, W)`` -> non-overlapping blocks ``(B, C, L, size, size)`` (row-major)."""
    bs, ch, h, w = im.shape
    im = im.reshape(bs * ch, 1, h, w)
    im = F.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    return im.reshape(bs, ch, -1, size, size)


def deblockify(blocks: Tensor, size: tuple[int, int]) -> Tensor:
    """Inverse of :func:`blockify` for an output of ``size = (H, W)``."""
    bs, ch = blocks.shape[0], blocks.shape[1]
    block_size = blocks.shape[3]
    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = F.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    return blocks.reshape(bs, ch, size[0], size[1])


# ---------------------------------------------------------------------------
# Block DCT
# ---------------------------------------------------------------------------

def block_dct(blocks: Tensor, n: Tensor, h: Tensor) -> Tensor:
    """Orthonormal 2-D DCT-II of each ``(..., N, N)`` block."""
    N = blocks.shape[-1]
    return (2 / N) * n * (h.t() @ blocks @ h)


def block_idct(coeff: Tensor, n: Tensor, h: Tensor) -> Tensor:
    """Inverse of :func:`block_dct` (same expression as the research reader)."""
    N = coeff.shape[-1]
    return (2 / N) * (h @ (n * coeff) @ h.t())


def reduced_block_idct(coeff: Tensor, m: int, device: torch.device | str | None = None) -> Tensor:
    """``m``-point inverse DCT of the top-left ``m x m`` corner of each ``N x N`` block.

    Coefficients on anti-diagonals ``< k`` lie inside the ``k x k`` corner, so a
    block truncated to ``k`` diagonals decodes losslessly (w.r.t. what was kept)
    into an ``m x m`` pixel tile for any ``k <= m <= N``. The orthonormal 2-D DC
    of a constant block of value ``v`` is ``N * v``; the ``m``-point inverse
    returns ``N * v / m``, hence the ``m / N`` rescale. For ``m == N`` this is
    exactly :func:`block_idct`.
    """
    N = coeff.shape[-1]
    if not (1 <= m <= N):
        raise ValueError(f"m must be in 1..{N}; got {m}")
    dev = coeff.device if device is None else torch.device(device)
    n_m, h_m = dct_matrices(m, dev)
    if m == N:
        return block_idct(coeff, n_m, h_m)
    corner = coeff[..., :m, :m]
    return block_idct(corner, n_m, h_m) * (m / N)


# ---------------------------------------------------------------------------
# Anti-diagonal bands
# ---------------------------------------------------------------------------

def diag_mask(block_size: int, from_diag: int, to_diag: int, device: torch.device | str = "cpu") -> Tensor:
    i = torch.arange(block_size, device=device).unsqueeze(1)
    j = torch.arange(block_size, device=device).unsqueeze(0)
    d = i + j
    return (d >= from_diag) & (d < to_diag)


def extract_diag_band(blocks: Tensor, from_diag: int, to_diag: int, image_size: int) -> Tensor:
    """Dense ``(B, C, S, S)`` plane holding only diagonals ``[from_diag, to_diag)`` of every block.

    Port of the research ``deblockify_extract_diag_band``; kept as the test
    oracle for the gather-based encoder.
    """
    block_size = blocks.shape[-1]
    mask = diag_mask(block_size, from_diag, to_diag, blocks.device)
    mask_nd = mask.view(*((1,) * (blocks.ndim - 2)), block_size, block_size)
    out = blocks * mask_nd
    return deblockify(out, (image_size, image_size))


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------

def rgb_to_ycrcb(images: Tensor) -> Tensor:
    """``(B, 3, H, W)`` RGB in [0, 255] -> YCrCb (chroma shifted by +128)."""
    out = torch.empty_like(images)
    out[:, 0, :, :] = images[:, 0, :, :] * 0.299 + images[:, 1, :, :] * 0.587 + images[:, 2, :, :] * 0.114
    out[:, 1, :, :] = images[:, 0, :, :] * 0.5 + images[:, 1, :, :] * -0.4187 + images[:, 2, :, :] * -0.0813
    out[:, 2, :, :] = images[:, 0, :, :] * -0.1687 + images[:, 1, :, :] * -0.3313 + images[:, 2, :, :] * 0.5
    out[:, [1, 2], :, :] += 128
    return out


def ycrcb_to_rgb(ycrcb: Tensor) -> Tensor:
    """Inverse of :func:`rgb_to_ycrcb` (unclamped)."""
    out = torch.empty_like(ycrcb)
    y = ycrcb[:, 0, :, :]
    cr = ycrcb[:, 1, :, :] - 128
    cb = ycrcb[:, 2, :, :] - 128
    out[:, 0, :, :] = y + 1.402 * cr
    out[:, 1, :, :] = y - 0.714136 * cr - 0.344136 * cb
    out[:, 2, :, :] = y + 1.772 * cb
    return out


def chroma_subsample(img: Tensor, strategy: str) -> Tensor:
    """Average-pool the two chroma planes and nearest-upsample them back (``"1x1"`` = no-op)."""
    if strategy == "1x1":
        return img
    if strategy == "4x4":
        kernel = (2, 2)
    elif strategy == "2x2-r":
        kernel = (2, 1)
    elif strategy == "2x2-c":
        kernel = (1, 2)
    else:
        raise ValueError(f"Unknown subsampling strategy: {strategy}")
    out = img.clone()
    h, w = out.shape[-2:]
    for c in (1, 2):
        ch = out[:, c:c + 1, :, :]
        ch_down = F.avg_pool2d(ch, kernel_size=kernel, stride=kernel)
        out[:, c:c + 1, :, :] = F.interpolate(ch_down, size=(h, w), mode="nearest")
    return out


# ---------------------------------------------------------------------------
# Whole-plane transforms
# ---------------------------------------------------------------------------

def forward_plane(img: Tensor, block_size: int, use_ycrcb: bool, subsampling: str) -> Tensor:
    """``(1, 3, S, S)`` RGB float -> un-quantized DCT blocks ``(1, 3, L, N, N)``."""
    if use_ycrcb:
        img = rgb_to_ycrcb(img)
        img = chroma_subsample(img, subsampling)
    n, h = dct_matrices(block_size, img.device)
    return block_dct(blockify(img, block_size), n, h)


def inverse_plane(plane: Tensor, block_size: int, use_ycrcb: bool, m: int | None = None) -> Tensor:
    """Dense dequantized coefficient plane ``(1, 3, S, S)`` -> RGB ``(1, 3, S*m/N, S*m/N)`` clamped to [0, 255].

    ``m`` is the output block size (``None`` or ``N`` = full size). Chroma was
    stored on the full grid (subsampling is applied before the DCT at encode
    time), so no chroma upsampling is needed here.
    """
    S = plane.shape[-1]
    N = block_size
    m = N if m is None else int(m)
    blocks = blockify(plane, N)
    tiles = reduced_block_idct(blocks, m)
    out_side = S * m // N
    img = deblockify(tiles, (out_side, out_side))
    if use_ycrcb:
        img = ycrcb_to_rgb(img)
    return img.clamp(0, 255)
