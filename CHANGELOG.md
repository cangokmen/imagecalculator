# Changelog

## 0.1.0 (2026-11-08)

Initial release.

- Progressive block-DCT codec: `compress`, `decompress`, `Encoder`, `EncodedImage`, `.icf` container.
- `keep` truncation to the first `k` anti-diagonals with reduced-native-size decode (`S*k/N` px).
- Per-diagonal quantization (scalar, vector or bands), YCrCb + chroma subsampling, support-only storage, blosc/zstd payloads.
- Token estimators for Anthropic, OpenAI and Google image pricing (`token_estimate`, `EncodedImage.savings`).
- `imagecalculator compress | decompress | info` CLI.
- Byte-exact compatibility with the research Frequency-Store dir stores when no `transform` is set.
