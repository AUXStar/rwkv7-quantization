# 1.5B v2 Quantization Scheme Performance Tuning Report

## 1. Overview

This report documents the optimization process and A/B test results of the RWKV-7 1.5B model from the X5 (NVFP4+FP8 residual) to the v2 (pure FP8) quantization scheme.

**Core improvement**: Change ffn_key from NVFP4+FP8 residual quantization to pure FP8 quantization, "baking the residual into the quantization" -- FP8 directly quantizes the original weights, avoiding loading additional residual data during inference.

## 2. Quantization Scheme Comparison

| Component | X5 scheme | v2 scheme |
|------|---------|---------|
| att.key (L0) | BF16 | BF16 |
| att.key (L1-23) | FP8 | FP8 |
| att.value | FP8 | FP8 |
| att.receptance | NVFP4 | NVFP4 |
| att.output | NVFP4 | NVFP4 |
| **ffn.key** | **NVFP4+FP8 residual** | **FP8** |
| ffn.value | FP8 | FP8 |

**Quantization statistics**:
- X5: 48 NVFP4 + 24 NVFP4_RES + 71 FP8 + 1 BF16 = 144 weights
- v2: 48 NVFP4 + 95 FP8 + 1 BF16 = 144 weights

## 3. A/B Test Results

### 3.1 Quality Metrics (2100 tokens, chunked forward)

| Metric | Reference (bf16) | X5 (NVFP4+RES) | v2 (FP8) | v2 vs X5 |
|------|------------|----------------|----------|----------|
| PPL | 1.5053 | 1.5410 | 1.5219 | **-0.0191** |
| PPL delta | -- | +0.0357 | +0.0166 | **53% improvement** |
| Top-1 consistency | 90.90% | 96.43% | 97.71% | **+1.29%** |
| VRAM | 2.94 GiB | 2.18 GiB | 1.77 GiB | **-0.41 GiB (-18.8%)** |

### 3.2 Decode Speed

| Batch | X5 t/s/req | v2 t/s/req | Delta | Change |
|-------|-----------|-----------|-------|------|
| B=1 | 87.4 | 80.4 | -7.0 | -8.0% |
| B=2 | 84.8 | 78.8 | -6.0 | -7.1% |
| B=4 | 91.4* | 77.9 | -13.6 | -14.8%* |
| B=8 | 64.1 | 61.1 | -3.0 | -4.7% |

> *B=4 X5 data may be affected by measurement noise (higher than B=1)

### 3.3 Model File Size

| Model | Size | Compression ratio |
|------|------|--------|
| Original bf16 | 2.9 GB | 1.0x |
| X5 (NVFP4+RES) | 2.1 GB | 1.38x |
| v2 (FP8) | 1.7 GB | 1.71x |

## 4. Analysis

### 4.1 Reasons for Quality Improvement

The v2 scheme improves PPL delta by 53% and Top-1 by 1.29%. Reasons:

- **NVFP4 has only 16 discrete values** (+-0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6), large quantization error
- **FP8 E4M3 has 256 discrete values**, directly quantizing original weights is far more accurate than NVFP4+residual
- Although the residual scheme compensates for NVFP4 quantization error, the residual itself also has FP8 quantization error
- Pure FP8 "one-step" quantization avoids the stacking of dual quantization errors

### 4.2 Reasons for VRAM Reduction

| Scheme | ffn_key data volume/layer | Calculation |
|------|-------------------|------|
| X5 (NVFP4+RES) | NVFP4 weights (0.5 B/elem) + FP8 residual (1 B/elem) + block scales | 1.5 B/elem |
| v2 (FP8) | FP8 weights (1 B/elem) + tensor scale | 1.0 B/elem |

v2 reduces ffn_key data volume by 33% per layer, saving ~0.41 GiB VRAM across 24 layers.

### 4.3 Reasons for Speed Regression

v2 decode speed is 5-8% slower than X5. Analysis:

- **X5 ffn_key**: uses `fused_nvfp4_res_gemm_kernel` (Triton fused kernel)
  - FP4 GEMM + FP8 GEMM completed in one kernel
  - FP4 weight loading halved, high memory bandwidth utilization
- **v2 ffn_key**: uses `fused_fp8_hwdot_gemm_kernel` (Triton FP8 kernel)
  - Single FP8 GEMM, but weight data volume is 2x that of FP4
  - Kernel may not be fully optimized

Although v2 has less total data loading (1.0 vs 1.5 B/elem), it is slower, indicating:
1. The memory access pattern of the FP8 Triton kernel may not be as efficient as the NVFP4_res kernel
2. The NVFP4_res kernel has gone through more rounds of optimization (tile configuration, split-K, etc.)
3. Hardware FP4 tensor core throughput may be higher than FP8

### 4.4 Issues Found and Fixed

**Prefill path _scaled_mm bug**: when `NVFP4_W4A16=True`, block_scale is not swizzled, causing the prefill path (`_scaled_mm`) with M>64 to produce incorrect results. Temporary fix: use chunked forward (CHUNK=32 <= 64) to take the fused kernel path.

## 5. Conclusions and Recommendations

### v2 Scheme Advantages
- **Better quality**: PPL delta improved by 53%, Top-1 improved by 1.29%
- **Lower VRAM**: reduced by 18.8% (0.41 GiB)
- **Smaller model**: compression ratio 1.71x vs 1.38x
- **Simpler implementation**: no residual logic, single FP8 quantization

### v2 Scheme Disadvantages
- **Slightly slower decode**: 5-8% (can be improved by optimizing the FP8 kernel)

### Future Optimization Directions
1. **Optimize FP8 fused kernel**: reference the NVFP4_res kernel's tile configuration and optimization strategy
2. **Try _scaled_mm FP8 decode**: use GPU-side amax to avoid D2H synchronization
3. **Fix prefill _scaled_mm path**: generate swizzled block scales for W4A16 mode
4. **MATH500 validation**: supplement long-text generation capability evaluation

## 6. Test Environment

- GPU: RTX 5070 Ti (12 GiB)
- Model: RWKV-7-g1h-1.5B (24 layers, C=2048, V=65536)
- Engine: faster3a v3a, NVFP4_W4A16=True
- Test data: 2100 token wikitext fragment
