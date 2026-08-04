# #8 Engineering Report: Inference Engine Adaptation

## Overview

Extend the v3a inference engine to support inference with the complete quantization scheme:
- NVFP4 attention weights (receptance / output / key L4-19)
- FP8 attention weights (key L0-3,L20-23 / value all) -- W8A16
- NVFP4+FP8 residual FFN key weights -- W4A16
- FP8 FFN value weights -- W8A8

## Code Changes

### nvfp4_ops.py (3 changes)

1. `load_nvfp4_weight`: detect and load `.res_fp8` + `.res_fp8_scale`, set qtype=`nvfp4_res_w4a16`
2. `linear_nvfp4_w4a16`: add FP8 residual compensation after dequantization
3. `linear_quantized`: dispatch `nvfp4_res_w4a16` -> `linear_nvfp4_w4a16`

### rwkv7_fast_v3a.py (3 changes)

1. import: add `dequantize_nvfp4`
2. Weight loading: skip `.res_fp8` / `.res_fp8_scale` keys
3. `_deq_att_weight`: support NVFP4 dict weights (dequantization + caching)

## Validation Results

### 1.5B Complete Scheme (First Run)

| Metric | Value |
|------|-----|
| PPL | 1.5111 (orig 1.5061, delta +0.0050) |
| Top-1 | 98.28% |
| CE delta | +0.003323 |
| VRAM | 1.67 GiB |
| Speed (b1tn) | 2542 tok/s |

### Quantization Statistics

- 144 weights: 56 FP8 + 64 NVFP4 + 24 NVFP4+res
- Compression: 2.25 GB -> 1.07 GB (2.1x)
- VRAM usage: 1.67 GiB (original ~3 GiB, ~44% savings)

### Comparison with Component-Level Experiments

| Scheme | PPL delta | Top-1 | Notes |
|------|-----------|-------|------|
| v8 (FFN key only NVFP4) | +0.0044 | 98.81% | FP8 att + BF16 val |
| v12 (FFN key NVFP4+res) | +0.0044 | 99.05% | FP8 att + BF16 val |
| Complete scheme | +0.0050 | 98.28% | NVFP4 att + FP8 val + NVFP4+res FFN |

The complete scheme's Top-1 is 0.77% lower than v12, mainly due to NVFP4 quantization of attention rec/out/key. PPL is nearly lossless.

## Analysis

- PPL delta 0.005 is well below the 0.05 target, excellent accuracy
- Top-1 98.28% is below the 99.5% target, due to low redundancy of 1.5B + error accumulation in the complete scheme
- The 7.2B model has higher redundancy, Top-1 is expected to improve
- attention NVFP4 is the main source of accuracy loss (rec/out + key L4-19)

## Next Steps

#9: 7.2B complete accuracy/speed/VRAM comparison
