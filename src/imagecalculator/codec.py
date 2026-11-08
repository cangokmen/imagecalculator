"""Encoder / EncodedImage / decode: the compress-for-the-model path.

Pipeline (bit-exact with the research dir-store builder when no ``transform``
is used): load -> resize -> [YCrCb -> chroma subsample] -> block DCT ->
``round(coeff / q[d])`` -> keep the first ``k`` anti-diagonals -> one compact
blosc payload per diagonal. Decoding runs an ``m``-point inverse DCT on each
block's low corner, so ``k`` diagonals come back as an ``S*k/N`` px image.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from . import tokens as _tokens
from .bands import decode_payload, diag_support_index, encode_payload
from .config import CodecConfig, quantization_bands
from .image import ImageLike, load_image, to_pil
from .policy import DiagonalPolicy, ResolvedPolicy
from .transform import deblockify, forward_plane, inverse_plane

MAGIC = b"ICF1"
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# EncodedImage
# ---------------------------------------------------------------------------

@dataclass
class EncodedImage:
    """One image's kept diagonals as compressed payloads.

    ``payloads[d]`` is byte-identical to ``img_<i>/diag_<d>.col`` of a research
    dir store built with the same config (``transform=None``).
    """

    config: CodecConfig
    payloads: list[bytes]
    transform_label: str | None = None
    _support: tuple[int, ...] = field(default_factory=tuple, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.payloads:
            raise ValueError("an EncodedImage needs at least one diagonal")
        if len(self.payloads) > self.config.max_diag:
            raise ValueError(f"{len(self.payloads)} payloads but max_diag is {self.config.max_diag}")
        if not self._support:
            self._support = tuple(int(ix.size) for ix in diag_support_index(self.config.image_size, self.config.block_size))

    # -- geometry ------------------------------------------------------------
    @property
    def keep_k(self) -> int:
        return len(self.payloads)

    @property
    def is_truncated(self) -> bool:
        return self.keep_k < self.config.max_diag

    @property
    def native_m(self) -> int:
        """Output block size of the native decode: the smallest corner holding the kept triangle."""
        return min(self.keep_k, self.config.block_size)

    @property
    def native_side(self) -> int:
        return self.config.image_size * self.native_m // self.config.block_size

    @property
    def pixels(self) -> tuple[int, int]:
        """``(width, height)`` of the native-size decode (tier 1: what the model sees)."""
        return (self.native_side, self.native_side)

    def valid_sizes(self) -> list[int]:
        S, N = self.config.image_size, self.config.block_size
        return [S * m // N for m in range(self.native_m, N + 1)]

    # -- bytes ---------------------------------------------------------------
    @property
    def payload_bytes(self) -> int:
        return sum(len(p) for p in self.payloads)

    @property
    def nbytes(self) -> int:
        """Bytes over the air: the ``.icf`` container (header + kept payloads) (tier 2)."""
        return len(self.to_bytes())

    def cumulative_bytes(self) -> np.ndarray:
        """int64 ``(keep_k,)``: payload bytes through each kept diagonal."""
        return np.cumsum([len(p) for p in self.payloads], dtype=np.int64)

    # -- tokens --------------------------------------------------------------
    def token_estimate(self, provider: str = "anthropic", detail: str | None = None) -> _tokens.TokenEstimate:
        """``provider`` is ``"anthropic"``, ``"openai[:model]"`` or ``"google[:model]"``."""
        w, h = self.pixels
        return _tokens.estimate(w, h, provider, detail)

    def savings(self, original: str | Path | int | None = None, provider: str = "anthropic",
                detail: str | None = None) -> dict:
        """Tier 1 and tier 2 numbers next to their full-image baselines."""
        S = self.config.image_size
        full = _tokens.estimate(S, S, provider, detail)
        red = self.token_estimate(provider, detail)
        if isinstance(original, (str, Path)):
            orig_bytes: int | None = os.path.getsize(original)
        elif isinstance(original, int):
            orig_bytes = original
        else:
            orig_bytes = None
        out = {
            "tokens": {"reduced": red.tokens, "full_image": full.tokens, "provider": provider},
            "pixels": {"reduced": self.pixels, "full_image": (S, S)},
            "bytes": {"over_the_air": self.nbytes, "original_file": orig_bytes},
            "keep": self.keep_k,
            "max_diag": self.config.max_diag,
        }
        if orig_bytes:
            out["bytes"]["ratio"] = orig_bytes / self.nbytes
        out["tokens"]["ratio"] = full.tokens / red.tokens
        return out

    # -- coefficients --------------------------------------------------------
    def coefficients(self, d: int) -> np.ndarray:
        """Quantized integer coefficients of diagonal ``d`` as ``(3, n_d)`` int64."""
        if not (0 <= d < self.keep_k):
            raise IndexError(f"diagonal {d} not kept (keep_k={self.keep_k})")
        return decode_payload(self.payloads[d], self._support[d])

    def dense_plane(self) -> np.ndarray:
        """Dequantized coefficients scattered into a dense ``(3, S, S)`` float32 plane (research reader arithmetic)."""
        S = self.config.image_size
        idx = diag_support_index(S, self.config.block_size)
        q = self.config.q_vector
        acc = np.zeros((3, S * S), dtype=np.float32)
        for d in range(self.keep_k):
            vals = self.coefficients(d)
            acc[:, idx[d]] += vals.astype(np.float32) * float(q[d])
        return acc.reshape(3, S, S)

    def truncate(self, k: int) -> "EncodedImage":
        if not (1 <= k <= self.keep_k):
            raise ValueError(f"k must be in 1..{self.keep_k}; got {k}")
        return EncodedImage(self.config, list(self.payloads[:k]), self.transform_label, self._support)

    # -- container -----------------------------------------------------------
    def header(self) -> dict:
        h = self.config.to_dict()
        h.update({
            "format_version": FORMAT_VERSION,
            "keep": self.keep_k,
            "lengths": [len(p) for p in self.payloads],
        })
        if self.transform_label is not None:
            h["transform"] = self.transform_label
        return h

    def to_bytes(self) -> bytes:
        hdr = json.dumps(self.header(), separators=(",", ":")).encode("utf-8")
        return b"".join([MAGIC, struct.pack("<I", len(hdr)), hdr, *self.payloads])

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncodedImage":
        if data[:4] != MAGIC:
            raise ValueError("not an .icf blob (bad magic)")
        (hlen,) = struct.unpack("<I", data[4:8])
        hdr = json.loads(data[8:8 + hlen].decode("utf-8"))
        config = CodecConfig.from_dict(hdr)
        lengths = [int(v) for v in hdr["lengths"]]
        if len(lengths) != int(hdr["keep"]):
            raise ValueError("corrupt header: keep != len(lengths)")
        pos = 8 + hlen
        payloads = []
        for n in lengths:
            payloads.append(data[pos:pos + n])
            pos += n
        if pos != len(data):
            raise ValueError(f"container length mismatch: expected {pos} bytes, got {len(data)}")
        return cls(config, payloads, hdr.get("transform"))

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: str | Path) -> "EncodedImage":
        return cls.from_bytes(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    """Encode images under one :class:`CodecConfig` on one device."""

    def __init__(self, config: CodecConfig, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        S, N = config.image_size, config.block_size
        self._support = diag_support_index(S, N)
        # per-position diagonal index of the dense S x S plane
        within = np.arange(S) % N
        self._diag_of_pos = torch.from_numpy((within[:, None] + within[None, :]).ravel()).to(self.device)

    def _plane(self, image: ImageLike) -> Tensor:
        """Un-quantized dense coefficient plane ``(3, S*S)`` float32 on the device."""
        cfg = self.config
        img = load_image(image, cfg.image_size, self.device)
        blocks = forward_plane(img, cfg.block_size, cfg.use_ycrcb, cfg.subsampling)
        return deblockify(blocks, (cfg.image_size, cfg.image_size)).reshape(3, -1)

    def _quantize(self, plane: Tensor, policy: ResolvedPolicy) -> np.ndarray:
        q_pos = torch.from_numpy(policy.q_vector).to(self.device)[self._diag_of_pos]
        if policy.transform is not None:
            plane = plane.clone()
            for d in range(policy.keep_k):
                ix = torch.from_numpy(self._support[d]).to(self.device)
                plane[:, ix] = policy.transform(d, plane[:, ix])
        return torch.round(plane / q_pos).to(torch.int64).cpu().numpy()

    def encode(self, image: ImageLike, policy: DiagonalPolicy | None = None, transform_label: str | None = None) -> EncodedImage:
        cfg = self.config
        pol = (policy or DiagonalPolicy()).resolve(cfg.max_diag, cfg.q_vector, cfg.clevel)
        plane_q = self._quantize(self._plane(image), pol)
        payloads = [
            encode_payload(np.ascontiguousarray(plane_q[:, self._support[d]]), pol.clevels[d])
            for d in range(pol.keep_k)
        ]
        eff_cfg = cfg
        if not np.array_equal(pol.q_vector, cfg.q_vector):
            qv = [int(v) for v in pol.q_vector]
            quant = qv[0] if len(set(qv)) == 1 else tuple(qv)
            eff_cfg = CodecConfig(cfg.block_size, cfg.image_size, cfg.use_ycrcb, cfg.subsampling, quant, cfg.clevel)
        label = transform_label if transform_label is not None else ("custom" if pol.transform is not None else None)
        return EncodedImage(eff_cfg, payloads, label, tuple(int(ix.size) for ix in self._support))

    def coefficient_plane(self, image: ImageLike) -> Tensor:
        """Quantized integer coefficient plane ``(1, 3, S, S)`` (config quantization, nothing truncated)."""
        cfg = self.config
        pol = DiagonalPolicy().resolve(cfg.max_diag, cfg.q_vector, cfg.clevel)
        plane_q = self._quantize(self._plane(image), pol)
        return torch.from_numpy(plane_q.reshape(3, cfg.image_size, cfg.image_size)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _resolve_m(enc: EncodedImage, size: str | int) -> int:
    S, N = enc.config.image_size, enc.config.block_size
    if size == "native":
        return enc.native_m
    if size == "full":
        return N
    side = int(size)
    if side * N % S != 0:
        raise ValueError(f"size {side} is not S*m/N for an integer m; valid sizes: {enc.valid_sizes()}")
    m = side * N // S
    if not (enc.native_m <= m <= N):
        raise ValueError(f"size {side} is outside the valid range; valid sizes: {enc.valid_sizes()}")
    return m


def decode(encoded: EncodedImage, size: str | int = "native", device: torch.device | str = "cpu") -> Tensor:
    """RGB float32 ``(1, 3, h, w)`` in [0, 255] at ``size`` = ``"native"`` (``S*k/N``), ``"full"`` or an int side."""
    m = _resolve_m(encoded, size)
    plane = torch.from_numpy(encoded.dense_plane()).unsqueeze(0).to(device)
    return inverse_plane(plane, encoded.config.block_size, encoded.config.use_ycrcb, m)


def decompress(encoded: EncodedImage | bytes | str | Path, size: str | int = "native", device: torch.device | str = "cpu") -> Image.Image:
    """Decode to a PIL image (uint8 truncation, as the research reader)."""
    if isinstance(encoded, (bytes, bytearray)):
        encoded = EncodedImage.from_bytes(bytes(encoded))
    elif isinstance(encoded, (str, Path)):
        encoded = EncodedImage.load(encoded)
    return to_pil(decode(encoded, size, device))


# ---------------------------------------------------------------------------
# One-call API
# ---------------------------------------------------------------------------

def _as_quantization(quantization, block_size: int):
    if quantization is None:
        return 1
    if isinstance(quantization, (int, np.integer)):
        return int(quantization)
    seq = list(quantization)
    if seq and isinstance(seq[0], (list, tuple)):
        return quantization_bands(seq, block_size)
    return tuple(int(v) for v in seq)


def compress(
    image: ImageLike,
    *,
    keep: int | float | str = "all",
    quantization: int | Sequence[int] | Sequence[Sequence[int]] | None = None,
    block_size: int = 32,
    image_size: int = 448,
    use_ycrcb: bool = True,
    subsampling: str = "1x1",
    clevel: int = 9,
    policy: DiagonalPolicy | None = None,
    device: torch.device | str = "cpu",
) -> EncodedImage:
    """Truncate to ``keep`` diagonals, quantize per diagonal, and pack the compact payloads.

    ``keep`` sets the pixel/token budget (tier 1). ``quantization``, ``use_ycrcb``,
    ``subsampling`` and ``clevel`` set bytes per pixel (tier 2). ``policy`` is
    the general form; when given, its ``keep``/``quantization``/``clevel`` win
    over the keyword arguments they override (``None`` fields fall back).
    """
    cfg = CodecConfig(block_size, image_size, use_ycrcb, subsampling, _as_quantization(quantization, block_size), clevel)
    if policy is None:
        policy = DiagonalPolicy(keep=keep)
    elif policy.keep == "all" and keep != "all":
        policy = DiagonalPolicy(keep, policy.quantization, policy.transform, policy.clevel)
    return Encoder(cfg, device).encode(image, policy)


def load(path: str | Path) -> EncodedImage:
    return EncodedImage.load(path)
