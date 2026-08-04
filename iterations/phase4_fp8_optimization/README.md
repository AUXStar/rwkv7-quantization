# Phase 4: FP8 Final Scheme and Operator Optimization

[中文](README_zh.md) | **English**

> **Issues**: #15-#16 | **Reports**: 3 | **Status**: Completed, all-FP8 is the final scheme

## Phase Goals

Determine all-FP8 as the final quantization scheme, optimize operator performance, and complete the full evaluation of the 7.2B model. Clean up all NVFP4-related code and rename files.

## Experiment Contents

### #15 Operator Optimization and Performance Tuning
- 7.2B model performance bottleneck analysis: 95% GPU bound, ffn_key_res accounts for 44.6% of GPU time
- Shape-aware tile configuration optimization:
  - att (4096x4096) -> (16,64,64,4): 1.84x vs baseline
  - ffn_key (16384x4096) -> (16,64,128,4): 37% speedup
  - ffn_val (4096x16384) -> (16,128,256,8): 29% speedup
- Split-K parallelism ineffective (atomic add overhead > benefit)
- Report: 15_opt_7b_ops.md

### #16 FP8 Final Scheme Determination
- All-FP8 vs X5 comparison: PPL delta improved 53%, Top-1 +1.29%, VRAM -18.8%
- Code cleanup: delete all NVFP4-related code
- File renaming: nvfp4_ops.py -> fp8_ops.py, fused_nvfp4_gemm.py -> fused_fp8_gemm.py
- Reports: 16_v2_fp8_scheme.md, 16_perf_tuning_report.md

## Key Results

1. **All-FP8 crushes X5**:
   - PPL delta improved 53%
   - Top-1 consistency +1.29pp (93.75% vs 91.02%)
   - VRAM -18.8% (7.35 GB vs 8.54 GB)
   - Speed 1.56x (44.9 vs 28.7 t/s)

2. **Shape-aware tile configuration**:

   | Matrix shape | Scenario | Tile (M,N,K,W) | Optimization effect |
   |----------|------|-----------------|----------|
   | 4096x4096 | att (decode) | (16,64,64,4) | 1.84x |
   | 16384x4096 | ffn_key (decode) | (16,64,128,4) | +37% |
   | 4096x16384 | ffn_value (decode) | (16,128,256,8) | +29% |

3. **Decode speed 44.9 t/s** (7.2B, 6.4x improvement), 18.1ms/token, 74% bandwidth utilization

4. **Code cleanup**:
   - Delete all NVFP4 quantization/dequantization/GEMM code
   - nvfp4_ops.py -> fp8_ops.py
   - fused_nvfp4_gemm.py -> fused_fp8_gemm.py
   - quantize_model.py retains only FP8 scheme

5. **NVFP4+NVFP4 residual scheme (V3) validation**:
   - Top-1 94.92% (same as pure NVFP4)
   - Data volume 1.13 B/elem (higher than FP8's 1.0 B/elem)
   - FP4 residual's 16 discrete values cannot effectively compensate for the main NVFP4 quantization error

## Final Performance Data

### 7.2B Model (RTX 5070 Ti, Blackwell)

| Metric | Original BF16 | FP8 Quantized | Change |
|------|----------|----------|------|
| Decode speed | 7.0 t/s | **44.9 t/s** | 6.4x |
| Prefill speed (1x128) | -- | 1603 t/s | -- |
| VRAM | 13.32 GB | **7.35 GB** | -45% |
| File size | 14.40 GB | **7.96 GB** | -45% |
| Top-1 consistency | 100% | **93.75%** | -6.25% |
| PPL delta (2048) | -- | +0.24% | -- |
| MATH500 | ~55% | **53%** | -2pp |
| GSM8K | ~85% | **83%** | -2pp |

### Concurrency Stress Test (64 concurrent)

| Metric | Value |
|------|-----|
| Total throughput | 473.2 tok/s |
| p50 latency | 51.0s |
| p90 latency | 67.5s |
| Error rate | 0/64 |

## Report List

| # | File | Content |
|---|------|------|
| 15 | 15_opt_7b_ops.md | 7.2B operator optimization |
| 16 | 16_v2_fp8_scheme.md | FP8 final scheme determination |
| 16 | 16_perf_tuning_report.md | Performance tuning report |
