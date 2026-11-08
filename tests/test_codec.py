import numpy as np
import pytest
import torch
from PIL import Image

import imagecalculator as ic
from imagecalculator.codec import EncodedImage


def test_lossless_full_decode_no_ycrcb(noisy64, block_size):
    enc = ic.compress(noisy64, keep="all", quantization=1, block_size=block_size, image_size=64, use_ycrcb=False)
    out = np.asarray(ic.decompress(enc, size="full")).astype(int)
    # q=1 stores integer coefficients: up to 1 level from coefficient rounding + 1 from uint8 truncation
    assert np.abs(out - np.asarray(noisy64).astype(int)).max() <= 2


def test_lossless_full_decode_ycrcb(noisy64):
    enc = ic.compress(noisy64, keep="all", quantization=1, block_size=8, image_size=64)
    out = np.asarray(ic.decompress(enc, size="full")).astype(int)
    assert np.abs(out - np.asarray(noisy64).astype(int)).max() <= 4   # YCrCb float round-off


def test_truncation_and_native_size(smooth64):
    for k in (1, 3, 8, 12, 15):
        enc = ic.compress(smooth64, keep=k, quantization=5, block_size=8, image_size=64)
        assert enc.keep_k == k and enc.is_truncated == (k < 15)
        m = min(k, 8)
        assert enc.native_m == m and enc.pixels == (64 * m // 8,) * 2
        img = ic.decompress(enc)
        assert img.size == enc.pixels
        assert enc.valid_sizes()[0] == enc.pixels[0] and enc.valid_sizes()[-1] == 64
    full = ic.compress(smooth64, keep="all", quantization=5, block_size=8, image_size=64)
    assert full.truncate(3).payloads == ic.compress(smooth64, keep=3, quantization=5, block_size=8, image_size=64).payloads


def test_keep_fraction_and_cumulative_bytes(smooth64):
    enc = ic.compress(smooth64, keep=0.5, quantization=5, block_size=8, image_size=64)
    assert enc.keep_k == 8
    cb = enc.cumulative_bytes()
    assert cb[-1] == enc.payload_bytes and (np.diff(cb) > 0).all()
    assert enc.nbytes > enc.payload_bytes


def test_size_argument(smooth64):
    enc = ic.compress(smooth64, keep=4, quantization=5, block_size=8, image_size=64)
    assert ic.decompress(enc, size=48).size == (48, 48)
    assert ic.decompress(enc, size="full").size == (64, 64)
    with pytest.raises(ValueError):
        ic.decompress(enc, size=24)   # m=3 < k=4
    with pytest.raises(ValueError):
        ic.decompress(enc, size=50)   # not S*m/N


def test_native_decode_equals_reduced_idct_of_full_plane(smooth64):
    enc = ic.compress(smooth64, keep=4, quantization=1, block_size=8, image_size=64, use_ycrcb=False)
    from imagecalculator.transform import blockify, deblockify, reduced_block_idct
    plane = torch.from_numpy(enc.dense_plane()).unsqueeze(0)
    ref = deblockify(reduced_block_idct(blockify(plane, 8), 4), (32, 32)).clamp(0, 255)
    assert torch.equal(ic.decode(enc, "native"), ref)
    # decoding at a larger m only adds zero coefficients: a 32px decode is the area-mean of nothing extra
    assert ic.decode(enc, size=64).shape[-1] == 64


def test_container_roundtrip(tmp_path, smooth64):
    enc = ic.compress(smooth64, keep=6, quantization=[[0, 3, 1], [3, 15, 10]], block_size=8, image_size=64)
    blob = enc.to_bytes()
    assert len(blob) == enc.nbytes and blob[:4] == b"ICF1"
    back = EncodedImage.from_bytes(blob)
    assert back.config == enc.config and back.payloads == enc.payloads and back.keep_k == 6
    p = tmp_path / "x.icf"
    enc.save(p)
    assert p.stat().st_size == enc.nbytes
    assert ic.load(p).payloads == enc.payloads
    assert np.array_equal(np.asarray(ic.decompress(p)), np.asarray(ic.decompress(enc)))
    with pytest.raises(ValueError):
        EncodedImage.from_bytes(blob[:-1])


def test_policy_transform_and_clevel(smooth64):
    base = ic.compress(smooth64, keep=10, quantization=3, block_size=8, image_size=64)
    same = ic.compress(smooth64, policy=ic.DiagonalPolicy(keep=10, quantization=3), block_size=8, image_size=64)
    assert same.payloads == base.payloads and same.config == base.config and same.transform_label is None

    def zero_d5(d, coeffs):
        return torch.zeros_like(coeffs) if d == 5 else coeffs
    z = ic.compress(smooth64, policy=ic.DiagonalPolicy(keep=10, quantization=3, transform=zero_d5), block_size=8, image_size=64)
    assert z.transform_label == "custom"
    assert (z.coefficients(5) == 0).all() and len(z.payloads[5]) < len(base.payloads[5])
    assert z.payloads[4] == base.payloads[4]
    assert EncodedImage.from_bytes(z.to_bytes()).transform_label == "custom"

    low = ic.compress(smooth64, policy=ic.DiagonalPolicy(keep=10, quantization=3, clevel=1), block_size=8, image_size=64)
    assert [low.coefficients(d).tolist() for d in range(10)] == [base.coefficients(d).tolist() for d in range(10)]
    assert low.payloads != base.payloads


def test_policy_quantization_override_is_recorded(smooth64):
    enc = ic.compress(smooth64, quantization=1, policy=ic.DiagonalPolicy(keep=4, quantization=7), block_size=8, image_size=64)
    assert enc.config.quantization == 7
    assert np.array_equal(np.asarray(ic.decompress(enc)), np.asarray(ic.decompress(ic.compress(smooth64, keep=4, quantization=7, block_size=8, image_size=64))))


def test_savings_and_tokens(tmp_path, smooth64):
    p = tmp_path / "in.png"
    smooth64.save(p)
    enc = ic.compress(p, keep=4, quantization=5, block_size=8, image_size=64)
    sv = enc.savings(p)
    assert sv["tokens"]["reduced"] <= sv["tokens"]["full_image"]
    assert sv["bytes"]["original_file"] == p.stat().st_size and sv["bytes"]["over_the_air"] == enc.nbytes
    assert sv["pixels"]["reduced"] == (32, 32) and sv["keep"] == 4
    assert int(enc.token_estimate()) == enc.token_estimate().tokens


def test_input_types_agree(smooth64):
    arr = np.asarray(smooth64)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    ref = ic.compress(smooth64, keep=5, quantization=2, block_size=8, image_size=64).payloads
    assert ic.compress(arr, keep=5, quantization=2, block_size=8, image_size=64).payloads == ref
    assert ic.compress(t, keep=5, quantization=2, block_size=8, image_size=64).payloads == ref


def test_coefficient_plane_matches_payloads(smooth64):
    cfg = ic.CodecConfig(8, 64, quantization=4)
    enc = ic.Encoder(cfg)
    plane = enc.coefficient_plane(smooth64)[0].reshape(3, -1).numpy()
    e = enc.encode(smooth64)
    idx = ic.diag_support_index(64, 8)
    for d in range(15):
        assert np.array_equal(plane[:, idx[d]], e.coefficients(d))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_device_runs(smooth64):
    enc = ic.compress(smooth64, keep=4, quantization=5, block_size=8, image_size=64, device="cuda:0")
    assert ic.decompress(enc, device="cuda:0").size == (32, 32)
