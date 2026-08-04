# #9 1.5B Complete Accuracy/Speed/VRAM Comparison (Pure Quantized GEMM, W4A4/W8A8)

## Overview

The user requires all tests to be conducted on 1.5B (small models are more sensitive to quantization, and low redundancy better exposes issues).
In the same process, under exactly the same configuration (CMIX_SPARSE=no-fc, same ORIG_LINEAR_GROUPS),
compare the accuracy, speed, and VRAM of the original bf16 vs. pure quantized GEMM (W4A4/W8A8).

## Test Configuration

- Model: rwkv7-g1h-1.5b (24 layers, C=2048, H=32, N=64)
- Text: 2100 tokens (warmup 500 + stable 1600)
- Configuration: EMB_DEVICE=cpu, RKV_MODE=off, CMIX_SPARSE=no-fc, LOWRANK_WEIGHT=both
- Quantization scheme: "1.5b" (see #1 design document v2)

## Quantization Statistics

- 144 quantized tensors: 56 FP8 + 64 NVFP4 + 24 NVFP4+res
- Quantized weight portion: 2.25 GB -> 1.07 GB (2.1x compression)
- File: 2.85 GB -> 1.82 GB (1.56x, including non-quantized weights)
- Quantization time: 11.9s

## Results Comparison (2099 tokens)

| Metric | Original bf16 | Quantized | delta |
|------|-----------|------|-------|
| PPL | 1.5061 | 1.5304 | **+0.0242** |
| Top-1 | -- | 97.14% | -- |
| CE | 0.4095 | 0.4255 | +0.015961 |

| Resource | Original bf16 | Quantized | Change |
|------|-----------|------|------|
| VRAM (load) | 2.69 GiB | 1.70 GiB | **-37%** |
| VRAM (peak) | 3.50 GiB | 2.48 GiB | -29% |
| Speed | 5269 tok/s | 3518 tok/s | **0.67x (-33%)** |
| File | 2.85 GB | 1.82 GB | 1.56x |

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

## Key Finding: Quantized GEMM is Slower on 1.5B

### Speed Comparison (same configuration, no-fc)

| Model | Speed | Relative |
|------|-------|------|
| Original bf16 | 5269 tok/s | 1.0x |
| Quantized (W4A4/W8A8) | 3518 tok/s | **0.67x** |

### Why the 34.5x Speedup on 7.2B is an Illusion

Previously in #9 (7.2B), the quantized model measured 1381 tok/s vs. original 40 tok/s, seemingly a 34.5x speedup.
But this comparison is **not fair**: the 7.2B original model requires 17.35 GiB VRAM, which exceeds a 12GB GPU,
causing it to run at the edge of VRAM overflow (frequent CPU offload), dropping speed to 40 tok/s.
The quantized model only needs 7.90 GiB and fits entirely in VRAM, so it is faster.

**Essentially**: the 7.2B "speedup" comes from VRAM savings allowing the model to go from "doesn't fit" to "fits", not from the algorithm being faster.

### Real Reasons for 1.5B Slowdown

1. **Activation online quantization overhead**: before each quantized GEMM, FP16 activation must be quantized to FP4/FP8
   - FP4 path: quantize + pack + swizzle (fused Triton kernel)
   - FP8 path: per-tensor scale + clamp + cast
   - On C=2048 matrices, the quantization kernel's launch/transfer overhead is a large proportion
2. **`_scaled_mm` vs native custom kernel**:
   - The v3a engine's `linear_orig_rows_exact_f16` etc. are highly tuned custom kernels
   - `_scaled_mm` is not optimally efficient for K=2048, smaller-row matrices
3. **Native fp16 is the optimal path when VRAM is sufficient**: 1.5B fits entirely in VRAM,
   and the native kernel has no extra overhead

### Re-evaluating the Value of Quantization

| Value | Conclusion |
|------|------|
| VRAM savings | True value (-37%, -34% on 7.2B) |
| Speedup | Only beneficial when the model originally doesn't fit in VRAM |
| Accuracy | PPL +0.0242 (meets target), Top-1=100% after warmup |

## Comparison with 7.2B (same scheme)

| Metric | 1.5B | 7.2B |
|------|------|------|
| PPL delta | +0.0242 | +0.0009 |
| Top-1 (overall) | 97.14% | 98.05% |
| VRAM savings | -37% | -34% |
| Speed vs native (fair) | 0.67x | to be tested (native doesn't fit) |
| Compression | 2.1x | 2.1x |

1.5B accuracy metrics are significantly worse than 7.2B (PPL delta 27x larger), validating
the judgment that "small models are more sensitive to quantization" -- 1.5B is a better test benchmark.

## Acceptance Criteria

| Criterion | Target | Actual | Status |
|------|------|------|------|
| PPL delta | <= 0.05 | +0.0242 | Met |
| Top-1 (after warmup) | >= 99.5% | 100% | Met |
| Compression ratio | >= 2x | 2.1x | Met |
| Pure quantized GEMM | no dequantization | _scaled_mm | Met |
| Speed | >= native | 0.67x | Not met |

## Next Optimization Directions

1. **Fused quantized GEMM kernel**: inline activation quantization into the GEMM kernel,
   eliminating intermediate tensor memory round-trips (expected to match or even exceed native)
2. **FP8 path priority**: FP8 quantization overhead (cast+scale) is much smaller than FP4 (pack+swizzle),
   and FP8xFP8 `_scaled_mm` is more efficient; the FP8 usage range can be expanded
3. **Reduce quantization count**: reuse already-quantized activations between consecutive quantized GEMMs (e.g., before and after relu_square)
4. **Accept 0.67x**: if the goal is "running 7.2B on a 12GB GPU", the speed bottleneck is VRAM not algorithm,
   and the quantized model's 1381 tok/s is sufficient for inference scenarios
