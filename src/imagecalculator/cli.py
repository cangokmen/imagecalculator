"""``imagecalculator compress | decompress | info``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .codec import EncodedImage, compress, decompress


def _parse_quantization(text: str | None):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return int(text)


def _parse_keep(text: str):
    if text == "all":
        return "all"
    if "." in text or text.endswith("%"):
        v = float(text.rstrip("%"))
        return v / 100.0 if text.endswith("%") else v
    return int(text)


def _fmt_bytes(n: int) -> str:
    return f"{n / 1024:.1f} KiB" if n >= 1024 else f"{n} B"


def cmd_compress(a: argparse.Namespace) -> int:
    enc = compress(
        a.image, keep=_parse_keep(a.keep), quantization=_parse_quantization(a.quantization),
        block_size=a.block_size, image_size=a.image_size, use_ycrcb=not a.no_ycrcb,
        subsampling=a.subsampling, clevel=a.clevel, device=a.device,
    )
    out = Path(a.output) if a.output else Path(a.image).with_suffix(".icf")
    enc.save(out)
    _print_info(enc, original=a.image, provider=a.provider, detail=a.detail)
    print(f"wrote {out}")
    return 0


def cmd_decompress(a: argparse.Namespace) -> int:
    size: str | int = a.size if a.size in ("native", "full") else int(a.size)
    img = decompress(a.icf, size=size, device=a.device)
    out = Path(a.output) if a.output else Path(a.icf).with_suffix(".png")
    img.save(out)
    print(f"wrote {out} ({img.size[0]}x{img.size[1]})")
    return 0


def _print_info(enc: EncodedImage, original: str | None = None, provider: str = "anthropic", detail: str | None = None) -> None:
    c = enc.config
    sv = enc.savings(original, provider, detail)
    q = c.quantization if c.is_uniform else f"schedule(min={c.quantization_factor})"
    print(f"config: block {c.block_size}, image {c.image_size}, ycrcb={c.use_ycrcb}, subsampling={c.subsampling}, q={q}, clevel={c.clevel}")
    print(f"keep: {enc.keep_k}/{c.max_diag} diagonals  -> native decode {enc.pixels[0]}x{enc.pixels[1]} px")
    t = sv["tokens"]
    print(f"tokens: {t['reduced']} (full image {t['full_image']}, {t['provider']} estimate, {t['ratio']:.1f}x fewer)")
    b = sv["bytes"]
    if b["original_file"]:
        print(f"bytes over the air: {_fmt_bytes(b['over_the_air'])} (original file {_fmt_bytes(b['original_file'])}, {b['ratio']:.1f}x smaller)")
    else:
        print(f"bytes over the air: {_fmt_bytes(b['over_the_air'])}")
    if enc.transform_label:
        print(f"transform: {enc.transform_label} (not research-store compatible)")


def cmd_info(a: argparse.Namespace) -> int:
    enc = EncodedImage.load(a.icf)
    _print_info(enc, original=a.original, provider=a.provider, detail=a.detail)
    if a.json:
        print(json.dumps(enc.savings(a.original, a.provider, a.detail), indent=2))
    return 0


def _add_token_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="anthropic", help="token estimate: anthropic | openai[:model] | google[:model]")
    p.add_argument("--detail", default=None, help="OpenAI detail or Gemini media_resolution (default: the API default)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imagecalculator", description="Progressive block-DCT image codec")
    p.add_argument("--version", action="version", version=f"imagecalculator {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="truncate + quantize an image into an .icf blob")
    c.add_argument("image")
    c.add_argument("--keep", default="all", help="diagonals to keep: int, fraction (0.25) or percent (25%%), or 'all'")
    c.add_argument("--quantization", "-q", default=None, help="scalar step, or JSON bands [[from,to,q],...]")
    c.add_argument("--block-size", type=int, default=32)
    c.add_argument("--image-size", type=int, default=448)
    c.add_argument("--no-ycrcb", action="store_true")
    c.add_argument("--subsampling", default="1x1", choices=["1x1", "2x2-r", "2x2-c", "4x4"])
    c.add_argument("--clevel", type=int, default=9)
    c.add_argument("--device", default="cpu")
    c.add_argument("-o", "--output")
    _add_token_args(c)
    c.set_defaults(fn=cmd_compress)

    d = sub.add_parser("decompress", help="decode an .icf blob to PNG")
    d.add_argument("icf")
    d.add_argument("--size", default="native", help="native | full | <side px>")
    d.add_argument("--device", default="cpu")
    d.add_argument("-o", "--output")
    d.set_defaults(fn=cmd_decompress)

    i = sub.add_parser("info", help="print config, keep, tokens and bytes")
    i.add_argument("icf")
    i.add_argument("--original", help="original image file for the byte baseline")
    i.add_argument("--json", action="store_true")
    _add_token_args(i)
    i.set_defaults(fn=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
