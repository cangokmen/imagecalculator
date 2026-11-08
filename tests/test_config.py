import numpy as np
import pytest

from imagecalculator import CodecConfig, quantization_bands


def test_geometry_and_uniform_q():
    c = CodecConfig(block_size=8, image_size=64, quantization=5)
    assert c.max_diag == 15 and c.blocks_per_side == 8 and c.is_uniform
    assert c.q_vector.shape == (15,) and c.q_vector.dtype == np.float32 and (c.q_vector == 5).all()
    assert c.quantization_factor == 5


def test_validation():
    with pytest.raises(ValueError):
        CodecConfig(block_size=8, image_size=60)
    with pytest.raises(ValueError):
        CodecConfig(block_size=8, image_size=64, quantization=0)
    with pytest.raises(ValueError):
        CodecConfig(block_size=8, image_size=64, quantization=[1] * 14)
    with pytest.raises(ValueError):
        CodecConfig(block_size=8, image_size=64, subsampling="4x4", use_ycrcb=False)
    with pytest.raises(ValueError):
        CodecConfig(block_size=8, image_size=64, subsampling="3x3")


def test_quantization_bands_and_schedule_roundtrip():
    sched = quantization_bands([[0, 4, 1], [4, 10, 5], [10, 15, 20]], 8)
    assert sched == (1,) * 4 + (5,) * 6 + (20,) * 5
    c = CodecConfig(8, 64, quantization=sched)
    assert not c.is_uniform and c.quantization_factor == 1  # min(schedule) quirk
    d = c.to_dict()
    assert d["quantization_schedule"] == list(sched) and d["quantization_factor"] == 1
    assert CodecConfig.from_dict(d) == c
    with pytest.raises(ValueError):
        quantization_bands([[0, 4, 1]], 8)  # gap
    with pytest.raises(ValueError):
        quantization_bands([[0, 8, 1], [4, 15, 2]], 8)  # overlap


def test_to_dict_keys_match_research_metadata():
    d = CodecConfig(32, 448, quantization=20).to_dict()
    assert set(d) == {"codec", "clevel", "block_size", "use_ycrcb", "subsampling_strategy", "quantization_factor", "image_size", "max_diag"}
    assert CodecConfig.from_dict(d) == CodecConfig(32, 448, quantization=20)
