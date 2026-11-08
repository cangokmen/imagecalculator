"""Image I/O: anything-to-tensor loading, tensor-to-PIL, PNG bytes."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

ImageLike = Union[str, Path, Image.Image, np.ndarray, Tensor]


def load_pil(image: ImageLike) -> Image.Image:
    """Return an RGB PIL image from a path, PIL image, ``(H, W, 3)`` uint8 array or ``(3, H, W)`` tensor."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) array; got shape {image.shape}")
        return Image.fromarray(np.ascontiguousarray(image.astype(np.uint8)), "RGB")
    if isinstance(image, Tensor):
        t = image.detach().cpu()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"expected a (3, H, W) tensor; got shape {tuple(image.shape)}")
        arr = t.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(np.ascontiguousarray(arr), "RGB")
    raise TypeError(f"unsupported image type {type(image).__name__}")


def load_image(image: ImageLike, size: int, device: torch.device | str = "cpu") -> Tensor:
    """Load, resize to ``size x size`` (PIL bilinear, as the research builder) and return ``(1, 3, S, S)`` float32 in [0, 255]."""
    pil = load_pil(image)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.uint8)  # writable copy
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.float().unsqueeze(0).to(device)


def to_pil(t: Tensor, round: bool = False) -> Image.Image:
    """``(1, 3, H, W)`` or ``(3, H, W)`` float in [0, 255] -> PIL RGB. Truncates like the research reader unless ``round=True``."""
    if t.ndim == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0, 255)
    if round:
        t = torch.round(t)
    arr = t.to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def png_bytes(img: Image.Image | Tensor) -> bytes:
    if isinstance(img, Tensor):
        img = to_pil(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_b64(img: Image.Image | Tensor) -> str:
    """Base64 PNG, ready for an API image content block."""
    return base64.standard_b64encode(png_bytes(img)).decode("ascii")
