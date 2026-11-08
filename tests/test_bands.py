import numpy as np
import pytest

from imagecalculator.bands import (
    DTYPE_CODES, choose_int_dtype, compact_to_dense, decode_blob, decode_payload, dense_to_compact,
    diag_support_index, diag_support_sizes, encode_blob, encode_payload,
)


def test_support_indices_partition_the_plane():
    S, N = 64, 8
    idx = diag_support_index(S, N)
    assert len(idx) == 2 * N - 1
    allpos = np.concatenate(idx)
    assert allpos.size == S * S and np.array_equal(np.sort(allpos), np.arange(S * S))
    sizes = diag_support_sizes(S, N)
    assert sizes[0] == (S // N) ** 2 and sizes[N - 1] == N * (S // N) ** 2 and sizes[-1] == (S // N) ** 2


@pytest.mark.parametrize("lo,hi,name", [(0, 200, "uint8"), (0, 300, "uint16"), (-5, 100, "int8"), (-200, 100, "int16"), (-40000, 1, "int32"), (0, 2**40, "uint64"), (-(2**40), 0, "int64")])
def test_dtype_narrowing(lo, hi, name):
    assert choose_int_dtype(lo, hi).name == name


@pytest.mark.parametrize("name", list(DTYPE_CODES))
def test_blob_roundtrip_every_dtype(name):
    info = np.iinfo(np.dtype(name))
    rng = np.random.default_rng(0)
    lo, hi = max(info.min, -(2**40)), min(info.max, 2**40)
    arr = rng.integers(lo, hi, size=(3, 37), dtype=np.int64)
    arr[0, 0], arr[0, 1] = lo, hi  # force this dtype
    blob, dtype = encode_blob(arr, 9)
    assert dtype.name == name
    assert np.array_equal(decode_blob(blob, dtype, arr.shape), arr)
    payload = encode_payload(arr, 9)
    assert payload[0] == DTYPE_CODES[name]
    assert np.array_equal(decode_payload(payload, 37), arr)


def test_compact_dense_roundtrip():
    S, N = 16, 4
    idx = diag_support_index(S, N)
    dense = np.arange(3 * S * S, dtype=np.int64).reshape(3, S, S)
    for d, ix in enumerate(idx):
        comp = dense_to_compact(dense, ix)
        back = compact_to_dense(comp, ix, S)
        assert np.array_equal(back.reshape(3, -1)[:, ix], comp)
        assert (back.reshape(3, -1)[:, np.setdiff1d(np.arange(S * S), ix)] == 0).all()
