# #2 W4A4: FFN-only NVFP4 Baseline (Pure Quantized GEMM)

## Overview

Only quantize the 1.5B model's FFN key/value to NVFP4, keeping the rest as BF16. Uses W4A4 pure quantized GEMM (FP4xFP4 `_scaled_mm`), without dequantization.

## Configuration

- Model: rwkv7-g1h-1.5B (24 layers)
- Quantization scope: 48 FFN tensors (ffn.key + ffn.value)
- GEMM: `linear_nvfp4` (FP4xFP4->BF16)
- Activation: online quantization to FP4 (fused Triton kernel)

## Results (2100 tokens)

| Metric | Value |
|------|-----|
| PPL | 1.5446 (orig 1.5061, delta +0.0385) |
| Top-1 | 96.09% (early 84%, late 100%) |
| VRAM | 1.61 GiB |
| Speed | 2426 tok/s |
| Compression | 2.25->0.99 GB (2.3x) |

## Comparison with W4A16

| Metric | W4A16 (old) | W4A4 (new) |
|------|------------|-----------|
| PPL delta | +0.0050 | +0.0385 |
| Top-1 | 98.28% | 96.09% |
| Speed | 2542 tok/s | 2426 tok/s |

W4A4 accuracy degradation is significant (PPL delta 7.7x), mainly due to activation online quantization to FP4 introducing errors. However, the last 1600 tokens still achieve 100% Top-1.

## Analysis

- FFN value's W4A4 error is large (ReLU-squared activation distribution is unfavorable for FP4)
- Consider using FP8 W8A8 for FFN value instead of NVFP4 W4A4
- ffn.key can use NVFP4+FP8 residual to compensate for errors
