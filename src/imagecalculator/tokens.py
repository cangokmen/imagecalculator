"""Image-token estimates for pixel-priced vision APIs.

These are documented approximations keyed by pixel count, checked against
the providers' public docs on the date given. They are not promises: always
look at the usage numbers in the real response. ``estimate`` also returns
the pixel count so callers can apply their own rule.

``provider`` is ``"anthropic"``, ``"openai"`` or ``"google"``, optionally with a
model after a colon (``"openai:gpt-4o"``, ``"google:gemini-2.5-flash"``); the
bare name uses that provider's default model. ``detail`` is OpenAI's
``detail`` or Gemini's ``media_resolution``; ``None`` means the API default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CHECKED = "checked 2026-10-30"


@dataclass(frozen=True)
class TokenEstimate:
    provider: str
    tokens: int
    pixels: int
    width: int
    height: int
    note: str
    model: str | None = None
    detail: str | None = None

    def __int__(self) -> int:
        return self.tokens


def _fit(w: int, h: int, max_dim: int) -> tuple[int, int]:
    """Scale down (never up) so the long edge is at most ``max_dim``."""
    long_edge = max(w, h)
    if long_edge <= max_dim:
        return w, h
    s = max_dim / long_edge
    return max(1, round(w * s)), max(1, round(h * s))


def _anthropic(width: int, height: int, model: str | None, detail: str | None) -> tuple[int, str]:
    if detail is not None:
        raise ValueError("anthropic has no detail setting")
    # Images with a long edge over 1568 px, or more than ~1.15 MP, are scaled
    # down by the API before tokenization; cost ~= (w * h) / 750.
    w, h = width, height
    long_edge = max(w, h)
    if long_edge > 1568:
        s = 1568 / long_edge
        w, h = int(w * s), int(h * s)
    if w * h > 1_150_000:
        s = math.sqrt(1_150_000 / (w * h))
        w, h = int(w * s), int(h * s)
    return max(1, math.ceil(w * h / 750)), f"tokens ~= ceil(w*h/750) after the API's own downscale ({CHECKED})"


# -- OpenAI ------------------------------------------------------------------
# https://developers.openai.com/api/docs/guides/images-vision
# Patch models: model -> (multiplier, {detail: (max pixel dimension, patch budget or None)}).
_ORIGINAL = (65_535, None)
_OPENAI_PATCH = {
    "gpt-6-astra": (1.2, {"low": (512, None), "high": (65_535, 2500), "original": _ORIGINAL, "auto": _ORIGINAL}),
    **{m: (1.2, {"low": (512, None), "high": (2048, 2500), "original": _ORIGINAL, "auto": _ORIGINAL})
       for m in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")},
    "gpt-5.5": (1.2, {"low": (512, None), "high": (2048, 2500), "original": (6000, 10_000), "auto": (6000, 10_000)}),
    **{m: (1.2, {"low": (2048, 6144), "high": (2048, 2500), "original": (6000, 10_000), "auto": (2048, 2500)})
       for m in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano")},
    "gpt-5.2": (1.2, {d: (2048, 6144) for d in ("low", "high", "auto")}),
    "gpt-4.1-mini": (1.62, {d: (2048, 6144) for d in ("low", "high", "auto")}),
}
# Tile models: model -> (base tokens, tokens per 512 px tile).
_OPENAI_TILE = {
    "gpt-5.1": (70, 140), "gpt-5": (70, 140),
    "gpt-4o": (85, 170), "gpt-4.1": (85, 170),
    "gpt-4o-mini": (2833, 5667),
    "o1": (75, 150), "o1-pro": (75, 150), "o3": (75, 150),
}
_OPENAI_MAX_PATCHES = 30_000


def _openai_patches(w: int, h: int, max_dim: int, budget: int | None) -> int:
    w, h = _fit(w, h, max_dim)
    patches = math.ceil(w / 32) * math.ceil(h / 32)
    if budget is not None and patches > budget:
        shrink = math.sqrt(32 ** 2 * budget / (w * h))
        shrink *= min(math.floor(w * shrink / 32) / (w * shrink / 32),
                      math.floor(h * shrink / 32) / (h * shrink / 32))
        w, h = int(w * shrink), int(h * shrink)
        patches = math.ceil(w / 32) * math.ceil(h / 32)
    if patches > _OPENAI_MAX_PATCHES:
        raise ValueError(f"{patches} patches exceeds OpenAI's {_OPENAI_MAX_PATCHES}-patch limit; the API rejects this image")
    return patches


def _openai(width: int, height: int, model: str | None, detail: str | None) -> tuple[int, str]:
    model = _lookup(model or "gpt-5.5", {**_OPENAI_PATCH, **_OPENAI_TILE}, "openai")
    detail = detail or "auto"
    if model in _OPENAI_PATCH:
        mult, sizing = _OPENAI_PATCH[model]
        if detail not in sizing:
            raise ValueError(f"openai {model} detail must be one of {sorted(sizing)}, got {detail!r}")
        max_dim, budget = sizing[detail]
        patches = _openai_patches(width, height, max_dim, budget)
        cap = f", {budget}-patch budget" if budget else ""
        return math.ceil(patches * mult), f"{model} detail={detail}: ceil(32px patches * {mult}) within {max_dim}px{cap} ({CHECKED})"
    base, per_tile = _OPENAI_TILE[model]
    if detail == "low":
        return base, f"{model} detail=low: fixed {base} tokens ({CHECKED})"
    if detail not in ("high", "auto"):
        raise ValueError(f"openai {model} detail must be one of ['auto', 'high', 'low'], got {detail!r}")
    w, h = _fit(width, height, 2048)
    if min(w, h) > 768:
        s = 768 / min(w, h)
        w, h = (768, int(h * s)) if w <= h else (int(w * s), 768)
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return base + per_tile * tiles, f"{model} detail={detail}: {base} + {per_tile} per 512px tile after fit-2048 / short-side-768 ({CHECKED})"


# -- Google ------------------------------------------------------------------
# Gemini 3 / 4: a fixed budget per image set by media_resolution, independent
# of pixel count (https://ai.google.dev/gemini-api/docs/media-resolution).
_GEMINI_FIXED = {"unspecified": 1120, "low": 280, "medium": 560, "high": 1120, "ultra_high": 2240}


def _google(width: int, height: int, model: str | None, detail: str | None) -> tuple[int, str]:
    model = (model or "gemini-3").lower()
    if not model.startswith("gemini-"):
        raise ValueError(f"unknown google model {model!r}; expected gemini-<version>[...]")
    if model.startswith(("gemini-3", "gemini-4")):
        res = (detail or "unspecified").lower().removeprefix("media_resolution_")
        if res not in _GEMINI_FIXED:
            raise ValueError(f"google media_resolution must be one of {sorted(_GEMINI_FIXED)}, got {detail!r}")
        return _GEMINI_FIXED[res], f"{model} media_resolution={res}: fixed per image regardless of size ({CHECKED})"
    if detail is not None:
        raise ValueError(f"per-image media_resolution is only supported on Gemini 3 and later, not {model}")
    # Earlier Gemini: 258 tokens if both sides <= 384 px, else 258 per crop
    # tile, the crop unit being ~min side / 1.5 clamped to [256, 768].
    if max(width, height) <= 384:
        return 258, f"{model}: 258 tokens for images <= 384px ({CHECKED})"
    unit = min(768, max(256, min(width, height) * 2 // 3))
    tiles = math.ceil(width / unit) * math.ceil(height / unit)
    return 258 * tiles, f"{model}: 258 per {unit}px crop tile ({CHECKED})"


def _lookup(model: str, table: dict, provider: str) -> str:
    """Exact model name, or a dated snapshot of one (``gpt-4o-2024-08-06``)."""
    if model in table:
        return model
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name + "-") and model[len(name) + 1:][:1].isdigit():
            return name
    raise ValueError(f"unknown {provider} model {model!r}; known: {sorted(table)}")


PROVIDERS = {
    "anthropic": _anthropic,
    "openai": _openai,
    "google": _google,
}


def estimate(width: int, height: int, provider: str = "anthropic", detail: str | None = None) -> TokenEstimate:
    name, _, model = provider.partition(":")
    try:
        fn = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}") from None
    w, h = int(width), int(height)
    tokens, note = fn(w, h, model or None, detail)
    return TokenEstimate(provider=name, tokens=tokens, pixels=w * h, width=w, height=h, note=note,
                         model=model or None, detail=detail)
