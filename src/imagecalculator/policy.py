"""Per-diagonal policy: how many diagonals to keep and what to do to each.

``keep`` is the truncation knob (tier 1: fewer pixels, fewer model tokens).
``quantization``, ``clevel`` and ``transform`` act per diagonal on the bytes
that leave the device (tier 2: fewer bytes over the air).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from torch import Tensor

from .config import DEFAULT_CLEVEL

DiagonalTransform = Callable[[int, Tensor], Tensor]


@dataclass(frozen=True)
class ResolvedPolicy:
    keep_k: int
    q_vector: np.ndarray        # float32 (max_diag,)
    clevels: tuple[int, ...]    # one per diagonal
    transform: DiagonalTransform | None = None


def resolve_keep(keep: int | float | str, max_diag: int) -> int:
    """Number of anti-diagonals to keep: an int count, a fraction in (0, 1], or "all"."""
    if isinstance(keep, str):
        if keep.lower() == "all":
            return max_diag
        raise ValueError(f"keep must be an int, a fraction in (0, 1], or 'all'; got {keep!r}")
    if isinstance(keep, bool):
        raise ValueError("keep must not be a bool")
    if isinstance(keep, float):
        if not (0.0 < keep <= 1.0):
            raise ValueError(f"a fractional keep must be in (0, 1]; got {keep}")
        return max(1, int(math.ceil(keep * max_diag)))
    k = int(keep)
    if not (1 <= k <= max_diag):
        raise ValueError(f"keep must be in 1..{max_diag}; got {k}")
    return k


@dataclass(frozen=True)
class DiagonalPolicy:
    """What happens to each anti-diagonal before it is stored.

    - ``keep``: diagonals ``>= keep`` are not stored at all (truncation).
    - ``quantization``: scalar step or per-diagonal vector (overrides the
      config's quantization when given explicitly to ``Encoder.encode``).
    - ``transform``: optional hook ``(d, coeffs) -> coeffs`` applied to the
      un-rounded, un-quantized ``(3, n_d)`` float coefficients of diagonal ``d``
      before quantization (dead-zone thresholding, per-diagonal gain, ...).
      With ``transform=None`` the stored integers are bit-exact with the
      research builder.
    - ``clevel``: blosc/zstd level, scalar or per diagonal.
    """

    keep: int | float | str = "all"
    quantization: int | Sequence[int] | None = None
    transform: DiagonalTransform | None = None
    clevel: int | Sequence[int] | None = None

    def resolve(self, max_diag: int, default_q: np.ndarray, default_clevel: int = DEFAULT_CLEVEL) -> ResolvedPolicy:
        keep_k = resolve_keep(self.keep, max_diag)
        if self.quantization is None:
            q = np.asarray(default_q, dtype=np.float32)
        elif isinstance(self.quantization, (int, np.integer)):
            if int(self.quantization) < 1:
                raise ValueError("quantization must be >= 1")
            q = np.full(max_diag, float(self.quantization), dtype=np.float32)
        else:
            q = np.asarray([int(v) for v in self.quantization], dtype=np.float32)
            if q.shape != (max_diag,):
                raise ValueError(f"quantization vector length {q.shape[0]} != max_diag {max_diag}")
            if (q < 1).any():
                raise ValueError("every quantization step must be >= 1")
        if self.clevel is None:
            cl = (int(default_clevel),) * max_diag
        elif isinstance(self.clevel, (int, np.integer)):
            cl = (int(self.clevel),) * max_diag
        else:
            cl = tuple(int(v) for v in self.clevel)
            if len(cl) != max_diag:
                raise ValueError(f"clevel vector length {len(cl)} != max_diag {max_diag}")
        if any(not (0 <= c <= 9) for c in cl):
            raise ValueError("clevel values must be in 0..9")
        return ResolvedPolicy(keep_k=keep_k, q_vector=q, clevels=cl, transform=self.transform)
