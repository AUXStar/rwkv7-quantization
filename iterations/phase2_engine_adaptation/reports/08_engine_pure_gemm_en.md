# #8 Inference Engine Adaptation (Pure Quantized GEMM, W4A4/W8A8)

## Overview

The v3a inference engine is adapted to the pure quantized GEMM scheme, where all quantized weights directly participate in `_scaled_mm` computation without dequantization.
The complete "1.5b" scheme is used to quantize the 1.5B model (att key/value/rec/out + ffn key/value), validating end-to-end inference accuracy and performance.

## Code Changes

### rwkv7_fast_v3a.py

1. **import**: `from nvfp4_ops import is_nvfp4_weight, load_nvfp4_weight, linear_nvfp4, is_fp8_weight, load_fp8_weight, linear_quantized`
2. **Weight loading** (line 279-290):
   - Skip scale keys (`.nf4_b_scale`, `.nvfp4_t_scale`, `.fp8_scale`, `.awq_scale`, `.res_fp8`, `.res_fp8_scale`)
   - NVFP4 weights: `load_nvfp4_weight(z, key, dev, swizzle=True)` (128x4 swizzle for `_scaled_mm`)
   - FP8 weights: `load_fp8_weight(z, key, dev, w8a16=False)` (W8A8, activation online quantization)
3. **`_att_linear`** (line 656-662): dict weights -> `linear_quantized(x, w, out_dtype=DTYPE)`, otherwise take the original path
4. **`cmix_from_mixed`** (line 618-642): when FFN key/value is dict -> `linear_quantized`, bypassing the orig_layout+sparse path
5. **`NVFP4_W4A16 = False`**: global setting, ensuring swizzle is enabled and FP8 uses W8A8

### nvfp4_ops.py

- `linear_nvfp4`: FP4xFP4->BF16 `_scaled_mm`, activation online quantization (fused Triton kernel)
- `linear_fp8`: FP8xFP8->BF16 `_scaled_mm`, activation online quantization
- `linear_nvfp4` (with res): FP4 GEMM + FP8 GEMM residual addition
- `linear_quantized`: dispatcher, selects GEMM path based on qtype

## Validation Results (1.5B, 2099 tokens, pure quantized GEMM)

| Metric | Value |
|------|-----|
| PPL | 1.5304 (orig 1.5061, delta +0.0242) |
| Top-1 | 97.14% |
| CE delta | +0.015961 |
| VRAM | 1.67 GiB |
| Speed | 2669 tok/s |
| File size | 1.82 GB (orig 2.85 GB, 1.56x) |
| Quantized tensors | 144 (88 NVFP4 + 56 FP8 + 24 NVFP4+res) |
| Compression ratio | 2.1x (quantized weight portion) |
| Quantization time | 13.7s |

### Window Analysis

| Window | Top-1 | PPL delta |
|------|-------|-----------|
| 0-100 | 84.00% | +1.6491 |
| 100-300 | 85.50% | +0.3374 |
| 300-500 | 92.50% | +0.1651 |
| 500-700 | 100.00% | +0.0029 |
| 700-1000 | 100.00% | +0.0004 |
| 1000-1500 | 100.00% | +0.0003 |
| 1500-2099 | 100.00% | +0.0002 |

### Error Characteristics

- Errors are concentrated in the warmup phase (0-500 tokens), PPL delta decreases from 1.65 to 0.17
- After 500 tokens, Top-1=100%, PPL delta<0.003
- PPL delta decreases and approaches 0 as the sequence grows, with no state accumulation

## Comparison with Old Scheme (W4A16 Dequantization)

| Metric | Old #8 (W4A16 dequantization) | New #8 (W4A4/W8A8 pure quantization) | Change |
|------|-------------------|----------------------|------|
| PPL delta | +0.0050 | +0.0242 | 4.8x (introduced by activation quantization) |
| Top-1 | 98.28% | 97.14% | -1.14% |
| VRAM | 1.67 GiB | 1.67 GiB | same |
| Speed | 2542 tok/s | 2669 tok/s | +5% |
| Inference method | Dequantize to FP16 then GEMM | Pure quantized GEMM (`_scaled_mm`) | fundamental change |

### Accuracy Difference Analysis

Reasons why W4A4 has lower accuracy than W4A16:
1. **Activation quantization to FP4**: FP4 has only 16 discrete values, introducing quantization error to attention rec/out and FFN key activations
2. **FFN value W8A8**: FP8 activation quantization error is smaller, but still slightly worse than FP16
3. **Errors concentrated in warmup**: quantization error is larger when the state is not yet stable, but completely disappears after convergence

### Speed Improvement Analysis

Pure quantized GEMM is slightly faster than the dequantization scheme:
- Eliminates dequantization overhead (NVFP4->FP16 unpack+scale operations)
- `_scaled_mm` computes directly in the quantized domain, reducing memory transfers
- But activation online quantization introduces some overhead, net speedup ~5%

## Key Confirmations

1. **Pure quantized GEMM is feasible**: PPL delta 0.0242 is well below the 0.05 target
2. **No state accumulation**: fully converges after 500 tokens
3. **VRAM unchanged**: quantized weight storage format is the same, VRAM usage during inference is consistent
4. **Complete engine adaptation**: both attention and FFN paths correctly intercept quantized weights

## Next Steps

#9: 7.2B complete accuracy/speed/VRAM comparison (pure quantized GEMM)
