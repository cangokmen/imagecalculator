# imagecalculator

A progressive block-DCT image codec for sending images to vision models.

**Truncation reduces your token count.** Keep only the first `k` anti-diagonals
of each DCT block and decode them at the *reduced native size*: the model sees
fewer pixels, and pixel-priced APIs (Claude, GPT, Gemini) charge fewer image
tokens. Tokens scale with `(k/N)^2`.

**Truncation + the other optimizations reduce your wifi cost.** Per-diagonal
quantization, YCrCb with chroma subsampling, support-only (zero-free) storage
and blosc/zstd entropy coding shrink the bytes that leave the device for the
same pixels.

The two levers are independent: `keep` sets the pixel/token budget, the rest
sets bytes per pixel.

```python
import imagecalculator as ic

enc = ic.compress("photo.jpg", keep=8, quantization=20)   # 8 of 63 diagonals (block 32, 448 px)
enc.pixels            # (112, 112)  -> what the model will see
enc.token_estimate()  # ~17 tokens on Anthropic (vs ~268 for the full 448 px image)
enc.nbytes            # bytes over the air (.icf container)
enc.save("photo.icf")

img = ic.decompress(enc)                 # PIL image, 112 x 112, k-point inverse DCT per block
img = ic.decompress(enc, size=224)       # any S*m/N with m >= k
img = ic.decompress(enc, size="full")    # blurry full-size reconstruction
```

### Sending it to Claude

```python
import os, anthropic, imagecalculator as ic

enc = ic.compress("photo.jpg", keep=8, quantization=20)
img = ic.decompress(enc)                                   # receiver side
client = anthropic.Anthropic()
r = client.messages.create(
    model="claude-opus-5-5", max_tokens=1024,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ic.png_b64(img)}},
        {"type": "text", "text": "What color is the bus?"},
    ]}],
)
print("tokens:", r.usage.input_tokens, "(full image: ~268)")                      # tier 1: truncation
print("bytes over wifi:", enc.nbytes, "(original JPEG:", os.path.getsize("photo.jpg"), ")")  # tier 2
```

`examples/compress_for_claude.py` runs the same question at `keep = 4, 8, 16, 32`
and prints measured `usage.input_tokens` and bytes next to a Lanczos resize of
the original to the same pixel size.

### Honest framing

A `k`-point inverse DCT of the low corner is a near-ideal low-pass filter
followed by decimation. It carries the **same information as a good resize**
(Lanczos / area) to that size, not more. What the format adds: one encoding
serves every size, and bytes are competitive with a JPEG of the resized image.

## Parameters

Notation: `N` = `block_size`, `S` = `image_size`, and each block has `2N - 1`
anti-diagonals (63 with the defaults), numbered from 0 (DC) upward.

### `ic.compress(image, *, ...)`

`image` is a file path, a PIL image, an `(H, W, 3)` uint8 array or a
`(3, H, W)` tensor. Everything else is keyword-only.

| Parameter | Default | Values | Effect |
|---|---|---|---|
| `keep` | `"all"` | int in `1..2N-1`, fraction in `(0, 1]`, or `"all"` | Number of anti-diagonals stored per block. A fraction `f` keeps `ceil(f * (2N-1))`. Sets the native decode size `S * min(keep, N) / N` px, and so the token count. |
| `quantization` | `1` | int `>= 1`, a list of `2N-1` ints, or bands `[[from, to, q], ...]` | Step per anti-diagonal; coefficients are stored as `round(coeff / q)`. Bands are half-open `[from, to)` ranges that must cover every diagonal in `[0, 2N-1)` exactly once. Larger steps mean fewer bytes and more loss. |
| `block_size` | `32` | int dividing `image_size` | DCT block side `N`. |
| `image_size` | `448` | positive int | The input is resized (bilinear) to `S x S` before encoding; the aspect ratio is not preserved. |
| `use_ycrcb` | `True` | bool | Convert RGB to YCrCb before the DCT. |
| `subsampling` | `"1x1"` | `"1x1"`, `"2x2-r"`, `"2x2-c"`, `"4x4"` | Chroma averaging: none, pairs of rows, pairs of columns, or 2x2 blocks. Anything but `"1x1"` requires `use_ycrcb=True`. |
| `clevel` | `9` | int in `0..9` | blosc/zstd compression level. Changes bytes and encode time only, never pixels. |
| `policy` | `None` | `ic.DiagonalPolicy` | Per-diagonal control, see below. |
| `device` | `"cpu"` | torch device | Where the transform runs. |

`keep` is the only parameter that changes the token count. The others change
bytes (and, except `clevel`, quality).

`quantization=1` stores the DCT coefficients as rounded integers. A full decode
is then within 2 levels of the resized input (one from coefficient rounding,
one from uint8 truncation), not bit-exact in pixel space.

### `ic.DiagonalPolicy(...)`

| Field | Default | Values | Effect |
|---|---|---|---|
| `keep` | `"all"` | as above | As above. If left at `"all"`, the `keep` argument of `compress` is used. |
| `quantization` | `None` | int `>= 1` or a list of `2N-1` ints | Overrides the `quantization` argument of `compress`; `None` falls back to it. Bands are not accepted here. |
| `clevel` | `None` | int in `0..9` or a list of `2N-1` ints | Compression level per diagonal; `None` falls back to the `clevel` argument. |
| `transform` | `None` | callable `(d, coeffs) -> coeffs` | Hook on the un-quantized float `(3, n_d)` coefficients of diagonal `d`, applied before quantization. |

### `ic.decompress(encoded, size="native", device="cpu")`

`encoded` is an `EncodedImage`, `.icf` bytes, or a path to an `.icf` file.
Returns a PIL image.

| `size` | Output |
|---|---|
| `"native"` (default) | `S * m / N` px with `m = min(keep, N)`: the smallest size that holds everything kept. |
| `"full"` | `S x S` px. |
| int | A side length of the form `S * m / N` for an integer `m` between `min(keep, N)` and `N`. `enc.valid_sizes()` lists them; anything else raises `ValueError`. |

### `enc.token_estimate(provider="anthropic", detail=None)`

| Parameter | Default | Values |
|---|---|---|
| `provider` | `"anthropic"` | `"anthropic"`, `"openai"` or `"openai:<model>"` (default model `gpt-5.5`), `"google"` or `"google:<model>"` (default `gemini-3`) |
| `detail` | `None` (the API default) | OpenAI `detail` (`"low"`, `"high"`, `"auto"`, and `"original"` on models that have it) or Gemini 3+ `media_resolution` (`"low"`, `"medium"`, `"high"`, `"ultra_high"`). Not accepted for Anthropic. |

`enc.savings(original=None, provider="anthropic", detail=None)` takes the same
two arguments plus the original file path (or its size in bytes) for the byte
comparison.

## Per-diagonal control (the adaptivity knobs)

```python
# scalar step, a per-diagonal vector, or [[from, to, q], ...] bands tiling [0, max_diag)
enc = ic.compress("photo.jpg", keep=16, quantization=[[0, 8, 1], [8, 20, 10], [20, 63, 50]])

# anything else per diagonal: a hook on the un-rounded (3, n_d) coefficients
import torch
def dead_zone(d, coeffs):            # drop tiny high-frequency coefficients
    return coeffs if d < 8 else torch.where(coeffs.abs() < 30, torch.zeros_like(coeffs), coeffs)
pol = ic.DiagonalPolicy(keep=16, quantization=20, transform=dead_zone, clevel=9)
enc = ic.compress("photo.jpg", policy=pol)
```

## CLI

```
imagecalculator compress photo.jpg --keep 8 --quantization 20 [--block-size 32 --image-size 448] -o photo.icf
imagecalculator decompress photo.icf [--size native|full|224] -o photo.png
imagecalculator info photo.icf [--original photo.jpg]
```

| Command | Flag | Default | Meaning |
|---|---|---|---|
| `compress` | `--keep` | `all` | Count (`8`), fraction (`0.25`), percent (`25%`) or `all`. |
| | `--quantization`, `-q` | `1` | Scalar step, or JSON bands `'[[0,8,1],[8,63,20]]'`. |
| | `--block-size` | `32` | `block_size`. |
| | `--image-size` | `448` | `image_size`. |
| | `--no-ycrcb` | off | Sets `use_ycrcb=False`. |
| | `--subsampling` | `1x1` | `1x1`, `2x2-r`, `2x2-c` or `4x4`. |
| | `--clevel` | `9` | `0..9`. |
| | `--device` | `cpu` | torch device. |
| | `-o`, `--output` | input name with `.icf` | Output file. |
| `decompress` | `--size` | `native` | `native`, `full` or a side length in px. |
| | `--device` | `cpu` | torch device. |
| | `-o`, `--output` | input name with `.png` | Output file. |
| `info` | `--original` | none | Original image, for the byte comparison. |
| | `--json` | off | Also print the savings as JSON. |
| `compress`, `info` | `--provider` | `anthropic` | `anthropic`, `openai[:model]` or `google[:model]`. |
| | `--detail` | API default | OpenAI `detail` or Gemini `media_resolution`. |

`info` prints the config, then two lines: `tokens: 17 (full image 268, ...)` and
`bytes over the air: 6.1 KiB (original file 182 KiB, ...)`.

Truncation only saves tokens
where cost scales with pixels: Anthropic, OpenAI patch models (gpt-5.2 and
later, gpt-4.1-mini; default `openai` = gpt-5.5) and Gemini 2.5 and earlier. Tile-priced OpenAI
models (gpt-4o, gpt-4.1, gpt-5.1) bill a 448 px image as one 512 px tile, and
Gemini 3/4 bill a fixed count per image (1120 by default), so `keep` doesn't
change their token count, only bytes.

## Install

```
pip install imagecalculator
```

`pip` pulls in a default torch build; to pick the CPU wheel (much smaller),
install torch from the PyTorch index first:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install imagecalculator
```

From a checkout: `pip install -e ".[dev]"` and `OMP_NUM_THREADS=4 pytest -q`.

Dependencies: numpy, torch, blosc (1.x), Pillow.

On a shared machine with many cores, set `OMP_NUM_THREADS` (or
`torch.set_num_threads`) to a small number: the codec's tensors are small, and
32 contending OpenMP threads turn a 2 ms encode into seconds.
