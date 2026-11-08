"""Tier 1 / tier 2 demo: ask Claude the same question at several `keep` values.

    export ANTHROPIC_API_KEY=...   (or `ant auth login`)
    python examples/compress_for_claude.py photo.jpg "What color is the bus?" --keep 4 8 16 32

Prints, per `keep`: pixels, measured input tokens, bytes over the air, and the
same for a Lanczos resize of the original to the same pixel size encoded as
JPEG at the quality closest in bytes -- so the token claim (tier 1) and the
byte claim (tier 2) are measured, not asserted.
"""
from __future__ import annotations

import argparse
import io
import os

import anthropic
from PIL import Image

import imagecalculator as ic

MODEL = "claude-opus-5-5"


def ask(client: anthropic.Anthropic, img: Image.Image, question: str) -> tuple[str, int]:
    r = client.messages.create(
        model=MODEL, max_tokens=512, output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ic.png_b64(img)}},
            {"type": "text", "text": question},
        ]}],
    )
    if r.stop_reason == "refusal":
        raise RuntimeError(f"refused: {r.stop_details}")
    text = next((b.text for b in r.content if b.type == "text"), "")
    return text.strip(), r.usage.input_tokens


def jpeg_at_bytes(img: Image.Image, target: int) -> tuple[bytes, int]:
    """Smallest-quality JPEG whose size is >= target (the fair byte competitor)."""
    best = None
    for q in range(5, 96, 5):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        b = buf.getvalue()
        best = (b, q)
        if len(b) >= target:
            break
    return best  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("question")
    ap.add_argument("--keep", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--quantization", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=448)
    a = ap.parse_args()

    client = anthropic.Anthropic()
    original = Image.open(a.image).convert("RGB")
    orig_bytes = os.path.getsize(a.image)
    full = original.resize((a.image_size, a.image_size), Image.BILINEAR)
    ans, toks = ask(client, full, a.question)
    print(f"full {a.image_size}px: tokens={toks:4d} bytes={orig_bytes:7d}  -> {ans!r}")
    print()
    print(f"{'keep':>4} {'px':>4} | {'IC tokens':>9} {'IC bytes':>8} | {'resize tokens':>13} {'JPEG bytes':>10} q | answers")
    for k in a.keep:
        enc = ic.compress(a.image, keep=k, quantization=a.quantization, block_size=a.block_size, image_size=a.image_size)
        img = ic.decompress(enc)
        ans_ic, tok_ic = ask(client, img, a.question)
        small = original.resize(enc.pixels, Image.LANCZOS)
        jpg, q = jpeg_at_bytes(small, enc.nbytes)
        ans_rs, tok_rs = ask(client, Image.open(io.BytesIO(jpg)), a.question)
        print(f"{k:>4} {enc.pixels[0]:>4} | {tok_ic:>9} {enc.nbytes:>8} | {tok_rs:>13} {len(jpg):>10} {q:>2} | {ans_ic!r} / {ans_rs!r}")


if __name__ == "__main__":
    main()
