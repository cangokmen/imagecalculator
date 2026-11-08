"""Per-diagonal band payloads: support indices and the blosc record format.

A payload is ``[1-byte dtype code][blosc(zstd, BITSHUFFLE) of the compact
(3, n_d) integer array]`` -- byte-identical to one ``img_<i>/diag_<d>.col``
file of the research dir store.
"""
from __future__ import annotations

from functools import lru_cache

import blosc
import numpy as np

DTYPE_CODES = {
    "int8": 0, "int16": 1, "int32": 2, "int64": 3,
    "uint8": 4, "uint16": 5, "uint32": 6, "uint64": 7,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_CODES.items()}


@lru_cache(maxsize=64)
def diag_support_index(image_size: int, block_size: int) -> tuple[np.ndarray, ...]:
    """Flat (C-raster) positions of each anti-diagonal's support in an ``S x S`` plane.

    Position ``(i, j)`` belongs to anti-diagonal ``(i % N) + (j % N)``. Entry
    ``d`` of the result gathers and scatters diagonal ``d`` with the same index,
    so a compact band round-trips to its dense positions exactly.
    """
    S, N = image_size, block_size
    within = np.arange(S) % N
    antidiag = (within[:, None] + within[None, :]).ravel()
    return tuple(np.flatnonzero(antidiag == d) for d in range(2 * N - 1))


def diag_support_sizes(image_size: int, block_size: int) -> tuple[int, ...]:
    return tuple(int(ix.size) for ix in diag_support_index(image_size, block_size))


def choose_int_dtype(min_val: int, max_val: int) -> np.dtype:
    if min_val >= 0:
        for t in (np.uint8, np.uint16, np.uint32):
            if max_val <= np.iinfo(t).max:
                return np.dtype(t)
        return np.dtype(np.uint64)
    for t in (np.int8, np.int16, np.int32):
        if min_val >= np.iinfo(t).min and max_val <= np.iinfo(t).max:
            return np.dtype(t)
    return np.dtype(np.int64)


def encode_blob(arr: np.ndarray, clevel: int) -> tuple[bytes, np.dtype]:
    """Narrow an int64 array to the smallest dtype holding its range and blosc-compress it."""
    arr = arr if arr.dtype == np.int64 else arr.astype(np.int64)
    dtype = choose_int_dtype(int(arr.min()), int(arr.max()))
    arr = np.ascontiguousarray(arr.astype(dtype, copy=False))
    blob = blosc.compress(
        arr.tobytes(),
        typesize=arr.itemsize,
        cname="zstd",
        clevel=int(clevel),
        shuffle=blosc.BITSHUFFLE,
    )
    return blob, dtype


def decode_blob(blob: bytes, dtype: str | np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    raw = blosc.decompress(blob)
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).astype(np.int64)


def encode_payload(compact: np.ndarray, clevel: int) -> bytes:
    """``(3, n_d)`` int64 -> ``[dtype code][blob]`` (the dir-store file content)."""
    blob, dtype = encode_blob(compact, clevel)
    return bytes((DTYPE_CODES[dtype.name],)) + blob


def decode_payload(payload: bytes, n_d: int) -> np.ndarray:
    """``[dtype code][blob]`` -> ``(3, n_d)`` int64."""
    if len(payload) < 2:
        raise ValueError("payload too short")
    code = payload[0]
    if code not in CODE_TO_DTYPE:
        raise ValueError(f"unknown dtype code {code}")
    return decode_blob(payload[1:], CODE_TO_DTYPE[code], (3, n_d))


def compact_to_dense(compact: np.ndarray, index: np.ndarray, image_size: int) -> np.ndarray:
    dense = np.zeros((3, image_size * image_size), dtype=compact.dtype)
    dense[:, index] = compact
    return dense.reshape(3, image_size, image_size)


def dense_to_compact(dense: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(dense.reshape(3, -1)[:, index])
