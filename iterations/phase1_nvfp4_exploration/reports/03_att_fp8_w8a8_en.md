# #3 W8A8: Attention key/value FP8 (Pure Quantized GEMM)

## Results (2100 tokens)

| Metric | Value |
|------|-----|
| PPL | 1.5079 (orig 1.5061, delta +0.0018) |
| Top-1 | 99.43% (early 98%, late 100%) |
| VRAM | 2.50 GiB |
| Speed | 6563 tok/s |

## Conclusion

FP8 W8A8 is near-lossless for attention key/value (PPL delta 0.0018, Top-1 99.43%). Activation online quantization to FP8 has minimal error, far better than FP4.

## Comparison with #2

| Metric | #2 FFN NVFP4 W4A4 | #3 Att FP8 W8A8 |
|------|-------------------|-----------------|
| PPL delta | +0.0385 | +0.0018 |
| Top-1 | 96.09% | 99.43% |
| Speed | 2426 tok/s | 6563 tok/s |

FP8 W8A8 is far superior to NVFP4 W4A4 in both accuracy and speed.
