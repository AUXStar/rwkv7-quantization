# Operator Rewrite Report: Fused Quantized GEMM Kernel (W4A4/W8A8 Single Kernel)

## Overview

Addressing the pure quantized GEMM speed issue on 1.5B (0.67x vs native fp16), rewrote the quantized GEMM operator.
Rewrote the original "cast->AWQ->quantize kernel->_scaled_mm->scale-fold" multi-launch pipeline
into a **minimum-launch** Triton single kernel, and continuously optimized across three versions.

## Evolution

| Version | Optimization | decode speed | Launches per linear |
|------|---------|------------|------------------|
| Baseline | `_scaled_mm` (quantize+GEMM separated) | 16.9 tok/s | 4-9 |
| v1 | fused GEMM single kernel + hybrid routing | 21.3 tok/s | 2-3 |
| v2 | prep_x (cast+AWQ+amax 1 launch) + GPU-side amax | 64.9 tok/s | 2 |
| **v3** | **FP8 residual fused into main kernel + r/k/v triple projection fusion** | **70.4 tok/s** | **~1.5** |

## Implementation (fused_nvfp4_gemm.py)

### prep_x_kernel (v2)
Fuses cast bf16 + AWQ division + amax (atomic max) into **1 launch**.
amax stays on GPU (computed inside GEMM kernel via `tl.load(amax_ptr)` to get pts=amax/2688), **no D2H sync**.

### fused_nvfp4_gemm_kernel (v1)
x(bf16) -> [in-register] FP4 quantization + FP4xFP4 dot, weights kept as packed FP4 storage (4x bandwidth benefit).
block scale = clamp(max_abs*448/amax, 0.015625, 448) -> fp8, numerically identical to _scaled_mm.

### fused_nvfp4_res_gemm_kernel (v3)
Merges NVFP4 main GEMM + FP8 residual GEMM into **single kernel**: same x_tile does FP4 quantization (per-block)
and FP8 quantization (per-tensor) separately, two accumulators, merged at the end. Residual path reduced from 5+ launches to 0.

### fused_rkv_gemm_kernel (v3)
Merges attention's r(NVFP4)/k(NVFP4|FP8)/v(FP8) three projections into **single GEMM kernel** +
prep3_x (xr/xk/xv three inputs 1 launch). Per-layer attention reduced from 6 launches to 2.
**Bit-exact with separate calls** (max_diff=0).

### Hybrid Routing
- M <= 64 (decode/small batch): fused single kernel (fewer launches)
- M > 64 (prefill): `_scaled_mm` (cuBLAS FP4 kernel more efficient for large M)

## Kernel-Level Speed (M=1, K=2048)

| Scenario | `_scaled_mm` | fused | Speedup |
|------|-------------|-------|------|
| decode N=2048 | 643us | **39us** (after tuning) | 16x |
| decode N=8192 (res) | ~1200us (two paths) | **118us** | 10x |

## Engine-Level Results (1.5B, 2099 tokens)

| Metric | Native fp16 | `_scaled_mm` | **fused v3** |
|------|---------|------------|--------------|
| decode | 163 tok/s | 16.9 | **70.4 (43%)** |
| prefill | 5269 | 3518 | **3434 (flat)** |
| PPL delta | — | +0.0242 | **+0.0242** (unchanged) |
| Top-1 | — | 97.14% | 97.14% |
| VRAM | 2.69G | 1.67G | 1.71G |

## Performance Bottleneck Evolution (profile)

| Stage | Bottleneck |
|------|------|
| Before v1 | **CPU launch-bound**: ~1000 launches/step, CPU 35ms vs GPU 11ms |
| v2 | launches reduced to ~400, GPU computation emerges |
| v3 | **GPU compute-bound**: quantized GEMM accounts for 81% of GPU (rkv 25% + res 27% + fp8 20% + nvfp4 9%) |

After v3, the bottleneck is kernel computation efficiency (low program utilization at M=1 + FP4/E4M3 decode ALU overhead),
not launches. BLOCK config tuning confirms current (16,64,64,4) is already optimal.

## Conclusion

1. **Operator rewrite effective**: decode 16.9->70.4 tok/s (4.2x), reaching 43% of native fp16
2. **Accuracy unchanged**: all kernels numerically identical to _scaled_mm (rkv bit-exact)
3. **Weights not dequantized**: packed FP4 weights decoded directly inside kernel for computation
4. **Prefill unaffected**: hybrid routing ensures large M still takes cuBLAS optimal path

## Next Steps (Diminishing Returns)

1. FP4/E4M3 decode lookup table optimization (exp2->lookup table, reduce ALU overhead)
2. decode multi-request batching (B>1 improves program utilization)
3. head projection FP8 quantization (currently accounts for 7% of decode GPU)
