import numpy as np
import pytest

from imagecalculator import DiagonalPolicy
from imagecalculator.policy import resolve_keep


def test_resolve_keep():
    assert resolve_keep("all", 15) == 15
    assert resolve_keep(4, 15) == 4
    assert resolve_keep(0.25, 15) == 4   # ceil(3.75)
    assert resolve_keep(1.0, 15) == 15
    assert resolve_keep(0.01, 15) == 1
    for bad in (0, 16, 1.5, 0.0, "half", True):
        with pytest.raises(ValueError):
            resolve_keep(bad, 15)


def test_policy_resolve_defaults_and_overrides():
    default_q = np.full(15, 7, dtype=np.float32)
    r = DiagonalPolicy().resolve(15, default_q, 9)
    assert r.keep_k == 15 and (r.q_vector == 7).all() and r.clevels == (9,) * 15 and r.transform is None
    r = DiagonalPolicy(keep=3, quantization=2, clevel=[1] * 15).resolve(15, default_q, 9)
    assert r.keep_k == 3 and (r.q_vector == 2).all() and r.clevels == (1,) * 15
    with pytest.raises(ValueError):
        DiagonalPolicy(quantization=[1] * 14).resolve(15, default_q)
    with pytest.raises(ValueError):
        DiagonalPolicy(clevel=11).resolve(15, default_q)
