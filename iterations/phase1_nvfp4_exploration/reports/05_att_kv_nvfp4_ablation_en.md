# #4 Experiment Report: L4-27 key/value NVFP4 Ablation

## Overview

Conducted NVFP4 (E2M1) quantization ablation experiments on the **attention layer key/value projection matrices** of the RWKV-7 1.5B model (24 layers). This follows #2 (FFN NVFP4) and #3 (key/value FP8 validation) to test the core question: **can attention key/value use NVFP4 quantization**.

### Experimental Method

Using **dequantization proxy testing**: weights are quantized to NVFP4 (with AWQ scaling + clip ratio search), dequantized back to BF16, and stored. The model loads via the normal BF16 GEMM path. This method precisely captures the impact of quantization error on quality; actual NVFP4 inference speed/VRAM optimization is deferred to #8.

## Experimental Results

| Test | Scheme | Top1 | PPL delta | VRAM |
|------|------|------|-----------|------|
| Baseline | All BF16 | 100.00% | 0.0000 | — |
| #3 | All FP8 W8A16 | 98.62% | -0.0004 | 2.69 GiB |
| 4a | All-layer att key+value NVFP4 | 98.62% | +0.0065 | 2.69 GiB |
| 4b | L4-19 key+value NVFP4, edge BF16 | 98.81% | +0.0059 | 2.72 GiB |
| 4c | L4-19 key NVFP4 only, value BF16 | 99.67% | +0.0020 | 2.72 GiB |
| 4d | L0 BF16, L1-3 FP8, L4-19 NVFP4, L20-23 FP8 | 98.86% | +0.0055 | 2.72 GiB |

## Key Findings

### 1. NVFP4 is Acceptable for Attention key/value
All-layer NVFP4 (4a) Top1=98.62%, on par with all FP8 (#3). NVFP4 did not cause attention layer quality collapse.

### 2. value is More Sensitive to NVFP4 than key (Contrary to Expectations)
- key NVFP4 only (4c): Top1=99.67%, only 0.33% drop
- key+value NVFP4 (4b): Top1=98.81%, 1.19% drop
- **value contributes approximately 0.86% of the Top1 drop**, the main source of score loss

In the README, att.key is rated (5 stars, most sensitive), att.value rated (4 stars). But actual testing shows value is more sensitive under NVFP4 quantization. Reason analysis:
- **key has L2 normalization protection**: kk = normalize(k * k_k), normalization makes the direction robust to weight perturbation
- **value directly enters state**: v outer_product k_modified writes to state, quantization error has no normalization buffer

### 3. Middle Layers are More Robust than Edge Layers
4b (middle NVFP4) is 0.19% better than 4a (all NVFP4), confirming that L0 and the last layer are more sensitive.

### 4. Final Scheme is Viable
4d (L0 BF16, L1-3 FP8, L4-19 NVFP4, L20-23 FP8) Top1=98.86%, better than all NVFP4. The scheme is effective.

## Correction Suggestions for the Final Quantization Scheme

Based on actual test data, the following adjustments are recommended:
- **att.key can be relaxed to NVFP4** (all middle layers); normalization protection makes it robust
- **att.value should maintain FP8 in key layers**, or use v12 dual quantization (NVFP4+FP8 residual) for compensation
- Original scheme has L4-27 key/value both as NVFP4 -> recommend splitting: key NVFP4, value FP8 or NVFP4+residual

## Next Steps

- #5: L0 value BF16 necessity validation (v_first cross-layer propagation)
- #6: Long-sequence state MSE
- Actual NVFP4 inference engine adaptation deferred to #8
