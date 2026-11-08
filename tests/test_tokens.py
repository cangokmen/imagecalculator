import math

import pytest

from imagecalculator import token_estimate


def test_anthropic_formula():
    e = token_estimate(448, 448)
    assert e.tokens == 268 and e.pixels == 448 * 448 and e.provider == "anthropic"
    assert token_estimate(112, 112).tokens == 17
    assert token_estimate(1000, 1000).tokens == 1334


def test_monotone_and_downscale():
    sizes = [56, 112, 224, 448, 896, 1568, 3000]
    toks = [token_estimate(s, s).tokens for s in sizes]
    assert toks == sorted(toks)
    assert token_estimate(3000, 3000).tokens == token_estimate(1568, 1568).tokens or token_estimate(3000, 3000).tokens <= 1600
    with pytest.raises(ValueError):
        token_estimate(10, 10, provider="nope")


def test_openai_patch_models():
    # docs examples for 1.2x / 2500-patch budget
    assert token_estimate(1024, 1024, "openai:gpt-6-astra", "high").tokens == 1229
    assert token_estimate(2048, 2048, "openai:gpt-6-astra", "high").tokens == 3000
    assert token_estimate(4096, 512, "openai:gpt-6-astra", "high").tokens == 2458
    e = token_estimate(448, 448, "openai")
    assert e.provider == "openai" and e.tokens == math.ceil(14 * 14 * 1.2)
    assert token_estimate(448, 448, "openai:gpt-4.1-mini").tokens == math.ceil(196 * 1.62)
    assert token_estimate(4096, 4096, "openai:gpt-5.5", "low").tokens == math.ceil(16 * 16 * 1.2)
    with pytest.raises(ValueError):
        token_estimate(448, 448, "openai:gpt-5.2", "original")
    with pytest.raises(ValueError):
        token_estimate(65_000, 65_000, "openai:gpt-6-astra", "original")


def test_openai_tile_models():
    assert token_estimate(448, 448, "openai:gpt-4o").tokens == 85 + 170
    assert token_estimate(1024, 1024, "openai:gpt-4o").tokens == 85 + 170 * 4  # -> 768x768, 2x2 tiles
    assert token_estimate(4096, 2048, "openai:gpt-4.1").tokens == 85 + 170 * 6  # -> 1536x768
    assert token_estimate(4096, 4096, "openai:gpt-5.1", "low").tokens == 70
    assert token_estimate(448, 448, "openai:gpt-4o-2024-08-06").tokens == 255
    assert token_estimate(448, 448, "openai:gpt-4o-mini").tokens == 2833 + 5667
    with pytest.raises(ValueError):
        token_estimate(448, 448, "openai:gpt-5-mini")  # not a dated gpt-5 snapshot


def test_google():
    assert token_estimate(112, 112, "google").tokens == 1120 == token_estimate(4000, 3000, "google").tokens
    assert token_estimate(448, 448, "google:gemini-3-flash", "low").tokens == 280
    assert token_estimate(448, 448, "google", "MEDIA_RESOLUTION_ULTRA_HIGH").tokens == 2240
    assert token_estimate(384, 200, "google:gemini-2.5-flash").tokens == 258
    assert token_estimate(960, 540, "google:gemini-2.5-pro").tokens == 258 * 6  # docs: crop unit 360, 3x2
    assert token_estimate(448, 448, "google:gemini-2.5-flash").tokens == 258 * 4  # unit 298 -> 2x2
    with pytest.raises(ValueError):
        token_estimate(448, 448, "google:gemini-2.5-flash", "low")
    with pytest.raises(ValueError):
        token_estimate(448, 448, "anthropic", "low")
