"""Codec configuration: block geometry, color handling, quantization.

Mirrors the ``metadata.json`` of the research Frequency-Store so that a
``CodecConfig`` round-trips through ``to_dict`` / ``from_dict`` without loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# The Image Calculator paper's design-space domains. They are documentation,
# not constraints: any block size dividing the image size and any integer
# quantization step >= 1 is accepted.
IC_BLOCK_SIZES = (8, 14, 16, 28, 32, 56, 64, 112, 128, 224, 256, 448)
IC_QUANTIZATION_FACTORS = (1, 2, 5, 20, 50, 100)
SUBSAMPLING_STRATEGIES = ("1x1", "2x2-r", "2x2-c", "4x4")
STORE_CODEC = "blosc-zstd"
DEFAULT_CLEVEL = 9


def quantization_bands(bands: Sequence[Sequence[int]], block_size: int) -> tuple[int, ...]:
    """Expand ``[[from, to, q], ...]`` band triples to a per-diagonal vector.

    ``[from, to)`` is a half-open range of anti-diagonals and ``q`` the integer
    step applied to each of them. The bands must tile ``[0, max_diag)`` exactly.
    Port of the research builder's ``expand_quantization_bands``.
    """
    max_diag = 2 * block_size - 1
    sched: list[int | None] = [None] * max_diag
    for entry in bands:
        if len(entry) != 3:
            raise ValueError(f"each band must be [from, to, q]; got {entry!r}")
        frm, to, q = int(entry[0]), int(entry[1]), int(entry[2])
        if q < 1:
            raise ValueError(f"quantization step must be >= 1; got {q} in {entry!r}")
        if not (0 <= frm < to <= max_diag):
            raise ValueError(
                f"band {entry!r} out of range for max_diag={max_diag} "
                f"(need 0 <= from < to <= {max_diag})"
            )
        for d in range(frm, to):
            if sched[d] is not None:
                raise ValueError(f"diagonal {d} covered by more than one band")
            sched[d] = q
    missing = [d for d, s in enumerate(sched) if s is None]
    if missing:
        raise ValueError(
            f"bands must cover every diagonal in [0, {max_diag}); "
            f"uncovered: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    return tuple(int(s) for s in sched)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CodecConfig:
    """Geometry and quantization of one encoding.

    ``quantization`` is a scalar step or a length-``max_diag`` vector (one step
    per anti-diagonal). Coefficients on diagonal ``d`` are stored as
    ``round(coeff / q[d])`` and dequantized by ``* q[d]`` on decode.
    """

    block_size: int
    image_size: int
    use_ycrcb: bool = True
    subsampling: str = "1x1"
    quantization: int | tuple[int, ...] = 1
    clevel: int = DEFAULT_CLEVEL

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.image_size < 1:
            raise ValueError("block_size and image_size must be positive")
        if self.image_size % self.block_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by block_size ({self.block_size})"
            )
        if self.subsampling not in SUBSAMPLING_STRATEGIES:
            raise ValueError(f"subsampling must be one of {SUBSAMPLING_STRATEGIES}; got {self.subsampling!r}")
        if self.subsampling != "1x1" and not self.use_ycrcb:
            raise ValueError("chroma subsampling other than '1x1' requires use_ycrcb=True")
        if not (0 <= self.clevel <= 9):
            raise ValueError(f"clevel must be in 0..9; got {self.clevel}")
        q = self.quantization
        if isinstance(q, (int, np.integer)) and not isinstance(q, bool):
            if int(q) < 1:
                raise ValueError(f"quantization must be >= 1; got {q}")
            object.__setattr__(self, "quantization", int(q))
        else:
            vec = tuple(int(v) for v in q)  # type: ignore[union-attr]
            if len(vec) != self.max_diag:
                raise ValueError(
                    f"quantization vector length {len(vec)} != max_diag {self.max_diag} "
                    f"(block_size {self.block_size})"
                )
            if any(v < 1 for v in vec):
                raise ValueError("every quantization step must be >= 1")
            object.__setattr__(self, "quantization", vec)

    # -- derived geometry -------------------------------------------------
    @property
    def max_diag(self) -> int:
        return 2 * self.block_size - 1

    @property
    def blocks_per_side(self) -> int:
        return self.image_size // self.block_size

    @property
    def is_uniform(self) -> bool:
        return isinstance(self.quantization, int)

    @property
    def q_vector(self) -> np.ndarray:
        """Per-diagonal dequantization steps as float32 ``(max_diag,)``."""
        if self.is_uniform:
            return np.full(self.max_diag, float(self.quantization), dtype=np.float32)
        return np.asarray(self.quantization, dtype=np.float32)

    @property
    def quantization_factor(self) -> int:
        """The scalar the research metadata reports (min of a schedule)."""
        if self.is_uniform:
            return int(self.quantization)  # type: ignore[arg-type]
        return int(min(self.quantization))  # type: ignore[arg-type]

    # -- metadata round-trip ---------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Keys and values as the research ``metadata.json`` writes them."""
        d: dict[str, Any] = {
            "codec": STORE_CODEC,
            "clevel": self.clevel,
            "block_size": self.block_size,
            "use_ycrcb": self.use_ycrcb,
            "subsampling_strategy": self.subsampling,
            "quantization_factor": self.quantization_factor,
            "image_size": self.image_size,
            "max_diag": self.max_diag,
        }
        if not self.is_uniform:
            d["quantization_schedule"] = list(self.quantization)  # type: ignore[arg-type]
        return d

    @classmethod
    def from_dict(cls, meta: dict[str, Any]) -> "CodecConfig":
        codec = meta.get("codec", "jxl")
        if codec != STORE_CODEC:
            raise ValueError(f"unsupported codec {codec!r} (only {STORE_CODEC!r}; legacy JXL stores are not supported)")
        sched = meta.get("quantization_schedule")
        quant: int | tuple[int, ...]
        if sched is not None:
            quant = tuple(int(v) for v in sched)
        else:
            quant = int(meta.get("quantization_factor", 1))
        return cls(
            block_size=int(meta["block_size"]),
            image_size=int(meta["image_size"]),
            use_ycrcb=bool(meta.get("use_ycrcb", True)),
            subsampling=str(meta.get("subsampling_strategy", "1x1")),
            quantization=quant,
            clevel=int(meta.get("clevel", DEFAULT_CLEVEL)),
        )
