# Comprehensive Verification Report: Long-Text Verification + Secondary Check + Speed Optimization

## Overview

Secondary verification of #8/#9 results, using 2100 tokens long text (previously 7.2B only 446 tokens), and attempting pre-dequantization speed optimization.

## I. Long-Text Verification (2100 tokens)

### 7.2B Model

| Metric | 446 tokens (old) | 2100 tokens (new) |
|------|-----------------|------------------|
| PPL delta | +0.0149 | **+0.0017** |
| Top-1 | 93.03% | **98.57%** |
| VRAM | 7.87 GiB | 7.87 GiB |
| Speed | 75 tok/s | 91 tok/s |

Window analysis:
| Interval | Top-1 | PPL delta |
|------|-------|-----------|
| 0-500 | 94.00% | +0.0213 |
| 500-1000 | 100.00% | -0.0003 |
| 1000-1500 | 100.00% | +0.0000 |
| 1500-2100 | 100.00% | +0.0000 |

### 1.5B Secondary Check

| Metric | First Run | Second Run (re-quantized) |
|------|------|----------------|
| PPL delta | +0.0050 | +0.0050 |
| Top-1 | 98.28% | 98.28% |
| Speed | 2542 tok/s | 4233 tok/s |

Results are completely identical, reproducibility confirmed.

### Key Findings

Window analysis for both models shows the same pattern:
- **First 500 tokens**: Top-1 ~93-94%, errors concentrated here
- **Last 1600 tokens**: Top-1 **100%**, PPL delta ~0

Reason: Quantization errors mainly occur during the warmup phase when the state has not stabilized. After the state converges, the quantized model is completely consistent with the original model.

## II. Speed Optimization: Pre-Dequantization

### Approach

Dequantize NVFP4/FP8 weights to FP16 during the first forward pass and cache them, then use FP16 GEMM directly for subsequent passes.

### 1.5B Results

| Metric | W4A16 (original) | Pre-dequantization | Change |
|------|------------|---------|------|
| PPL delta | +0.0050 | **+0.0033** | Improved |
| Top-1 | 98.28% | 98.19% | Flat |
| VRAM | 1.67 GiB | 3.70 GiB | +122% |
| Speed | 4233 tok/s | **21977 tok/s** | **5.2x** |

### 7.2B Results

| Metric | W4A16 (original) | Pre-dequantization | Change |
|------|------------|---------|------|
| PPL delta | +0.0017 | **+0.0006** | Improved |
| Top-1 | 98.57% | 98.33% | Flat |
| VRAM | 7.87 GiB | 18.66 GiB | +137% |
| Speed | 91 tok/s | 80 tok/s | -12% |

### Analysis

- 1.5B pre-dequantization is effective: VRAM is manageable (3.70GB), speed 5.2x
- 7.2B pre-dequantization VRAM overload (18.66GB > 12GB GPU), no speed improvement
- Both models' PPL delta improved: because FP8 value upgraded from W8A8 to W8A16 (activation not quantized)

## III. Solution Matrix

| Approach | Accuracy | Speed | VRAM | Applicable |
|------|------|------|------|------|
| W4A16 (original) | Good | Slow | Low | Memory-constrained |
| Pre-dequantization (full) | Better | 1.5B fast/7.2B slow | High | Sufficient memory |
| Pre-dequantization (att only) | Good | Medium | Medium | Balanced |
| fused W4A16 kernel | Good | Fast | Low | Requires development |
| W4A4 + FP8 res | Medium | Fast | Low | Requires development |

## IV. Conclusion

1. **PPL meets target**: 1.5B delta 0.003-0.005, 7.2B delta 0.0006-0.002, well below the 0.05 target
2. **Top-1**: Last 1600 tokens 100%, first 500 tokens ~93%. Overall 98%+
3. **Reproducibility**: 1.5B secondary check completely identical
4. **Speed optimization**: 1.5B pre-dequantization 5.2x speedup; 7.2B requires larger VRAM GPU or fused kernel
5. **Accuracy improvement**: Pre-dequantization upgrades FP8 value from W8A8 to W8A16, further reducing PPL delta
