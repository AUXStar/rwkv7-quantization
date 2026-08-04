# Research Experiments: EAR Metric, Per-Channel Quantization, and Weight Asymmetry Analysis

> **Date**: 2026-08-04
> **Model**: RWKV-7 1.5B (g1h-1.5b-20260710-ctx10240)
> **Hardware**: RTX 5070 Ti (Blackwell, 12GB VRAM)
> **Branch**: `research/ear-and-optimizations`

## Overview

Based on insights from two arXiv papers:
- [SLQ: Simple Linear Quantization](https://arxiv.org/abs/2605.02404) - introduces the EAR (Expected Acceptance Rate) metric
- [A Weight Quantization Study](https://arxiv.org/abs/2505.03803) - surveys per-channel and asymmetric quantization techniques

We conducted three experiments to evaluate potential improvements to our FP8 quantization scheme:

1. **EAR Evaluation**: Measure distribution-level quantization loss beyond Top-1
2. **Per-Channel FP8**: Test per-output-channel scales vs per-tensor
3. **Weight Asymmetry Analysis**: Evaluate potential benefit of asymmetric quantization

---

## 1. EAR (Expected Acceptance Rate) Evaluation

### Method

EAR measures the distribution consistency between original and quantized models:

```
EAR = E_x[ Σ_v min(p_orig(v|x), p_quant(v|x)) ]
```

- EAR ≥ 0.99 → "distribution-lossless"
- EAR ≥ 0.95 → "near-lossless"
- EAR < 0.95 → "significant distribution shift"

We also compute KL divergence, JS divergence, Top-1/Top-5 agreement, and logit differences.

### Results (1.5B model, 10 prompts, 107 tokens)

| Metric | Per-Tensor FP8 | Per-Channel FP8 |
|--------|---------------|-----------------|
| **EAR** | **0.9412** | 0.9262 |
| KL(orig∥quant) | 0.0201 | 0.0376 |
| JS divergence | 0.0049 | 0.0091 |
| Top-1 agreement | 92.52% | **95.33%** |
| Top-5 agreement | 88.97% | 89.91% |
| Max logit diff | 2.956 | 3.350 |
| Prob L1 distance | 0.118 | 0.148 |

### Key Finding

**FP8 quantization achieves EAR = 0.94**, classified as "significant distribution shift" (below the 0.95 threshold). While Top-1 agreement is 92.5%, the full probability distributions diverge more than Top-1 alone suggests. This means:

- Greedy decoding (argmax) is relatively well-preserved
- Sampling-based generation (temperature > 0) may diverge more
- The EAR metric reveals distribution distortion that Top-1 misses

---

## 2. Per-Channel FP8 Quantization

### Method

Per-channel (per-output-channel) FP8 quantization assigns an independent scale to each output channel (row of the weight matrix), instead of a single shared scale for the entire tensor.

**Implementation**:
- Modified `quantize_model.py`: `--per-channel` flag
- Modified `fp8_ops.py`: `_scaled_mm` RowWise mode with `scale_a=(M,1)`, `scale_b=(1,N)`
- Modified `fused_fp8_gemm.py`: Skip fused Triton path for per-channel (use `_scaled_mm` instead)

### Results

| Metric | Per-Tensor | Per-Channel | Change |
|--------|-----------|-------------|--------|
| EAR | **0.9412** | 0.9262 | -1.5pp ↓ |
| Top-1 | 92.52% | **95.33%** | +2.8pp ↑ |
| KL | **0.0201** | 0.0376 | +87% ↑ |
| JS | **0.0049** | 0.0091 | +84% ↑ |
| File size | 1.13 GB | 1.13 GB | same |

### Key Finding

**Per-channel quantization is a trade-off, not an improvement:**
- ✅ Better Top-1 agreement (+2.8pp): per-channel scales preserve argmax ordering
- ❌ Worse distribution similarity (EAR -1.5pp, KL +87%): per-channel scales distort relative magnitudes
- ❌ Worse on all distribution metrics (KL, JS, L1)

**Conclusion**: Per-channel is NOT recommended for production. The Top-1 improvement is offset by worse distribution matching, which affects sampling-based generation. Per-tensor remains the optimal scheme.

---

## 3. Weight Distribution Asymmetry Analysis

### Method

For each weight component (24 layers × 6 components = 144 weights), we computed:
- Mean, std, skewness (mean/std)
- Positive fraction
- Symmetric vs asymmetric FP8 quantization MSE

Asymmetric quantization shifts by the mean before scaling: `q = quantize(w - mean)`, `deq = dequant(q) + mean`

### Results

| Component | Skew | Pos% | MSE Improvement | Verdict |
|-----------|------|------|-----------------|---------|
| att.receptance | -0.0004 | 49.98% | -0.02% | no benefit |
| att.key | -0.0002 | 49.98% | 0.00% | no benefit |
| att.value | 0.0012 | 50.05% | -0.08% | no benefit |
| att.output | -0.0000 | 50.00% | 0.01% | no benefit |
| ffn.key | -0.0053 | 49.80% | -0.32% | no benefit |
| ffn.value | 0.0007 | 50.02% | 0.00% | no benefit |

### Key Finding

**All weight distributions are perfectly symmetric** (|skew| < 0.01, positive fraction ~50%). Asymmetric quantization provides **zero benefit** and actually worsens MSE slightly due to the added zero-point rounding error.

This is expected for well-trained neural network weights - the training process naturally centers weight distributions at zero.

---

## Summary and Recommendations

| Experiment | Result | Action |
|-----------|--------|--------|
| EAR metric | FP8 EAR = 0.94 (below 0.95 threshold) | Acceptable for greedy; monitor for sampling |
| Per-channel FP8 | Top-1 ↑ but EAR/KL ↓ | **Do not adopt** |
| Asymmetric quantization | No benefit (weights already symmetric) | **Do not adopt** |

### Overall Conclusion

The current **per-tensor FP8** scheme remains optimal. Per-channel and asymmetric quantization do not provide net improvements. The EAR metric reveals that while FP8 quantization preserves greedy decoding well (92.5% Top-1), there is measurable distribution shift that could affect sampling-based generation.

### Future Directions

1. **Per-tensor + per-channel hybrid**: Use per-tensor for attention, per-channel for FFN (where Top-1 matters more)
2. **Activation quantization improvement**: The current W8A8 path quantizes activations per-tensor; per-token activation scales could help
3. **Larger model validation**: Verify EAR on 7.2B model (currently only 1.5B tested due to GPU memory)

---

## Files

| File | Description |
|------|-------------|
| `experiments/eval_ear.py` | EAR evaluation script |
| `experiments/analyze_weights.py` | Weight distribution asymmetry analysis |
| `experiments/ear_results_fp8.json` | EAR results (per-tensor FP8) |
| `experiments/ear_results_fp8_perchannel.json` | EAR results (per-channel FP8) |
| `experiments/weight_asymmetry_analysis.json` | Weight asymmetry analysis results |

## References

1. SLQ: Simple Linear Quantization, arXiv:2605.02404
2. A Weight Quantization Study, arXiv:2505.03803
