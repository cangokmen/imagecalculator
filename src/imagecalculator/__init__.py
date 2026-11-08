"""imagecalculator: progressive block-DCT image codec.

Tier 1 -- truncation reduces your token count::

    enc = ic.compress("photo.jpg", keep=8)      # 8 of 63 diagonals -> 112 px for the model
    img = ic.decompress(enc)                    # PIL image at the reduced native size

Tier 2 -- truncation + per-diagonal quantization, YCrCb, chroma subsampling and
entropy coding reduce the bytes over the air (``enc.nbytes``).
"""
from ._version import __version__
from .bands import decode_payload, diag_support_index, encode_payload
from .codec import EncodedImage, Encoder, compress, decode, decompress, load
from .config import IC_BLOCK_SIZES, IC_QUANTIZATION_FACTORS, SUBSAMPLING_STRATEGIES, CodecConfig, quantization_bands
from .image import load_image, load_pil, png_b64, png_bytes, to_pil
from .policy import DiagonalPolicy
from .tokens import TokenEstimate, estimate as token_estimate

__all__ = [
    "__version__",
    "CodecConfig", "quantization_bands", "IC_BLOCK_SIZES", "IC_QUANTIZATION_FACTORS", "SUBSAMPLING_STRATEGIES",
    "DiagonalPolicy",
    "Encoder", "EncodedImage", "compress", "decompress", "decode", "load",
    "load_image", "load_pil", "to_pil", "png_bytes", "png_b64",
    "token_estimate", "TokenEstimate",
    "encode_payload", "decode_payload", "diag_support_index",
]
