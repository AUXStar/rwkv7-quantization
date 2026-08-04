# Phase 2: Inference Engine Adaptation and Fused Kernel Development

[中文](README_zh.md) | **English**

> **Issues**: #7-#9 | **Reports**: 5 | **Status**: Completed

## Phase Goals

Integrate the quantized model into the Albatross inference engine (faster3a_2605), develop fused Triton kernels, and complete 1.5B/7.2B benchmark testing.

## Experiment Contents

### #7 Engine Adaptation and Weight Loading
- Modify rwkv7_fast_v3a.py to support automatic quantized weight detection
- Implement is_fp8_weight() / load_fp8_weight() interface
- Handle .fp8_scale key loading and cleanup
- Reports: 08_engine_adaptation.md, 08_engine_pure_gemm.md, 08_quantization_toolchain.md

### #8 Fused Kernel Development
- prep_x fusion: input cast + AWQ + amax completed in one kernel launch
- fused_fp8_hwdot_gemm_kernel: FP8 hardware tensor core dot (tl.dot(fp8, fp8))
- fused_rkv_fp8_kernel: r/k/v three attention projections completed in one kernel
- Shape-aware tile: automatically select optimal BLOCK configuration for different matrix shapes

### #9 Benchmark Testing
- 1.5B model complete benchmark: speed, VRAM, accuracy
- 7.2B model supplementary benchmark
- Reports: 09_benchmark_1_5b_pure_gemm.md, 09b_benchmark_7_2b_supplement.md

## Key Results

1. **Fused kernel 1.84x speedup**: prep_x + FP8 hardware dot + RKV fusion
2. **Shape-aware tile configuration**:
   - att (4096x4096) -> BLOCK=(16,64,64), GROUP=4
   - ffn_key (16384x4096) -> BLOCK=(16,64,128), GROUP=4
   - ffn_val (4096x16384) -> BLOCK=(16,128,256), GROUP=8
3. **CUDA Graph not applicable**: decode step 96 kernel replay produces ~1ms extra overhead > launch savings
4. **Dense path enforced**: quantized model CMIX_SPARSE=off, FFN sparse path incompatible (0% block sparsity)

## Report List

| # | File | Content |
|---|------|------|
| 08 | 08_engine_adaptation.md | Engine adaptation scheme |
| 08 | 08_engine_pure_gemm.md | Pure GEMM path implementation |
| 08 | 08_quantization_toolchain.md | Quantization toolchain integration |
| 09 | 09_benchmark_1_5b_pure_gemm.md | 1.5B benchmark testing |
| 09b | 09b_benchmark_7_2b_supplement.md | 7.2B supplementary benchmark |

## Performance Data

### 1.5B Model (RTX 5070 Ti)

| Metric | Original BF16 | FP8 Quantized |
|------|----------|----------|
| Decode speed | 164.1 t/s | 67.8 t/s |
| VRAM | 2.69 GB | 1.60 GB |
| Top-1 consistency | 100% | 97.85% |

### 7.2B Model

| Metric | Original BF16 | FP8 Quantized |
|------|----------|----------|
| Decode speed | 7.0 t/s | 44.9 t/s (6.4x) |
| VRAM | 13.32 GB | 7.35 GB |
| Top-1 consistency | 100% | 93.75% |
