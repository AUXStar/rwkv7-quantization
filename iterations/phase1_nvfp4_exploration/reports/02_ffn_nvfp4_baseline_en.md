# #2 Experiment Report: 2.9B FFN-only NVFP4 Baseline

## Experiment Overview

Quantized the FFN key/value weights of the RWKV-7 2.9B model to NVFP4 (E2M1), validating toolchain usability and the impact of FFN quantization on accuracy.

## Experiment Configuration

| Item | Value |
|------|-----|
| Model | rwkv7-g1h_preview4673-2.9b-20260701-ctx8192 (2.9B, bf16, C=2560, L=32) |
| Quantization scope | `blocks.*.ffn.key.weight` [10240,2560] + `ffn.value.weight` [2560,10240] |
| Quantized tensor count | 64 (32 layers x 2) |
| Quantization format | NVFP4 E2M1, block_size=16, E4M3 block scale + FP32 tensor scale |
| GEMM path | `torch._scaled_mm` (FP4xFP4->BF16), activation online quantization |
| GPU | RTX 5070 Ti Laptop (sm_120, 12GB VRAM) |
| CUDA | 13.0, PyTorch 2.13.0+cu130 |

## Key Technical Decisions

### 1. cuBLASLt unavailable -> `torch._scaled_mm` available

cuBLASLt 13.3's block scaling does not support mixed precision (FP16xFP4), returning `CUBLAS_STATUS_INVALID_VALUE`. `torch._scaled_mm` supports FP4xFP4->BF16, requiring activation to be online quantized to FP4 before participating in GEMM.

### 2. `to_blocked` swizzle is required

`torch._scaled_mm`'s Blockwise 1x16 scaling requires the scale tensor to undergo 128x4 block reordering (swizzle), not a simple 1D flatten. The `to_blocked()` function converts the (H, W) scale tensor to the 1D flat layout expected by cuBLAS.

### 3. Per-tensor scale folding

`_scaled_mm` does not directly support a per-tensor scale parameter. Solution:
- Fold the per-tensor scale into the block scale during quantization
- Multiply the GEMM output by `A_ts * B_ts` to restore the correct scale

## Experiment Results

### Accuracy Comparison (b1tn, 446 tokens real text)

| Metric | Original | NVFP4 | Delta | Target | Status |
|------|------|-------|-------|------|------|
| mean_loss (CE) | 0.4824 | 0.5177 | +0.0353 | — | — |
| PPL | 1.6199 | 1.6782 | +0.0583 | <=0.05 | Close |
| Top-1 agree | — | — | 96.19% | >=99.5% | Below |
| Top-5 agree | — | — | 85.83% | — | — |
| KL divergence | — | — | 0.025 (mean) | — | — |
| Max abs diff | — | — | 12.03 | — | — |

### VRAM Comparison

| Model | VRAM Usage | Allocated | Savings |
|------|----------|------|------|
| Original bf16 | 6.65 GB | 5.37 GB | — |
| NVFP4 FFN-only | 4.64 GB | 3.13 GB | 2.01 GB (30.2%) |

### Speed Comparison

| Path | Original tok/s | NVFP4 tok/s | Slowdown |
|------|-----------|-------------|------|
| b1tn (T=446) | 983 | 433 | 56% |
| b1tn (T=89) | 251 | — | — |
| b1t1 (decode) | 103 | 14 | 86% |

### Discretization Error Analysis (Single-layer GEMM)

| Test | Max diff | Mean diff | Mean rel diff |
|------|----------|-----------|---------------|
| FFN key (random data) | 0.86 | 0.16 | 101% |
| FFN key (real weights) | 0.081 | 0.014 | 94% |
| FFN value (real weights) | 0.207 | 0.026 | — |

## Analysis and Discussion

### Reasons for Not Meeting Accuracy Targets

1. **Activation online quantization error**: FP4 has only 16 discrete values; the error introduced by activation quantization is the primary error source. The ReLU-squared activation of FFN amplifies errors in certain channels.

2. **Text repetition effect**: The test text is the same paragraph repeated 5 times; the model is in a low-entropy state (high confidence) in the latter half, where small quantization errors are more likely to change top-1 predictions.

3. **PPL delta close to target**: 0.058 vs target 0.05, gap of only 0.008. Considering the short test text (446 tokens), delta may converge on longer texts.

### Speed Bottleneck Analysis

1. **Activation quantization overhead**: `to_nvfp4()` is a pure PyTorch implementation, including reshape/amax/clamp/quantization operations, without using a fused kernel.

2. **Scale swizzle overhead**: `to_blocked()` includes multiple reshape/permute operations, executed on every FFN call.

3. **Extreme b1t1 slowdown**: decode path M=1, each token requires 64 `linear_nvfp4` calls (32 layers x 2), where Python overhead + quantization overhead per call dominates.

### Improvement Directions

1. **Accuracy improvements**:
   - Consider keeping FFN value as FP8 instead of FP4 (value is closer to output, error impact is greater)
   - Try higher per-tensor scale precision
   - Use finer-grained quantization schemes for activations (e.g., per-channel scale)

2. **Speed improvements**:
   - Implement fused CUDA kernel: activation quantization + scale swizzle + GEMM in one step
   - Cache activation block scale swizzle results (when input shape is unchanged)
   - Use CUDA Graph for b1t1 path to eliminate Python overhead

## File Listing

| File | Description |
|------|------|
| `quantize_ffn_nvfp4.py` | Quantization tool: load 2.9B -> quantize FFN -> save .pth+meta |
| `nvfp4_ops.py` | NVFP4 GEMM operator: load + swizzle + online quantization + _scaled_mm |
| `rwkv7_fast_v3a.py` (patched) | v3a inference engine: NVFP4 detection + FFN path replacement |

## Conclusion

The NVFP4 FFN-only quantization toolchain has been validated end-to-end with normal inference. VRAM savings of 30.2% (2.01 GB) meet expectations. In terms of accuracy, PPL delta 0.058 slightly exceeds the 0.05 target, top-1 agreement 96.19% is below the 99.5% target, with the primary error coming from activation online quantization. In terms of speed, b1tn slowdown is 56%, b1t1 slowdown is 86%, requiring fused kernel optimization.

Next steps:
1. Implement fused activation quantization CUDA kernel to improve speed
2. Validate PPL delta convergence on longer texts (2048+ tokens)
3. Try a mixed scheme of FFN key with NVFP4 + FFN value with FP8
