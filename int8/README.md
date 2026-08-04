# INT8 Quantization

## Overview

INT8 (8-bit integer) quantization for RWKV-7, providing an alternative to FP8 with broader hardware compatibility (no FP8 tensor core required).

## Schemes

### 1. Per-Tensor Symmetric INT8 (W8A8)

Simplest scheme, mirrors the FP8 approach:

```
scale = max(|W|) / 127
w_int8 = round(W / scale).clamp(-128, 127).to(int8)
W_approx = w_int8 * scale
```

- **Activation**: online quantized to int8 per-tensor
- **GEMM**: `torch._scaled_mm` with int8 weights, or Triton kernel
- **Compression**: 2x (vs FP16)
- **Hardware**: Any CUDA GPU (SM 6.0+)

### 2. Affine INT8 (MM8-style, per-row + per-column)

Official RWKV quantization format, higher precision:

```
my = amin(W, dim=1)   # per-row offset
W  = W - my
mx = amin(W, dim=0)   # per-col offset
W  = W - mx
rx = amax(W, dim=0)   # per-col scale
W  = W / rx
ry = amax(W, dim=1)   # per-row scale
W  = W / ry
w_u8 = clip(floor(W * 256), 0, 255).to(uint8)
W_approx = (w_u8 + 0.5) * ry * rx + my + mx
```

- **5 buffers**: w_u8, mx, rx, my, ry
- **Compression**: ~2x (scale/offset overhead on small matrices)
- **Precision**: better than per-tensor (dual affine absorbs asymmetry)

### 3. W8A16 (Weight-only INT8)

Weight quantized to int8, activation stays FP16:

- No activation quantization error
- Slower than W8A8 (FP16 GEMM vs INT8 GEMM)
- Use case: when activation quantization hurts quality too much

## Implementation Plan

| Step | File | Description |
|------|------|-------------|
| 1 | `quantize_int8.py` | Quantization tool (per-tensor + affine) |
| 2 | `int8_ops.py` | Weight detection, loading, GEMM dispatch |
| 3 | `fused_int8_gemm.py` | Triton fused kernels (GEMV decode + batched prefill) |
| 4 | Benchmark | PPL, Top-1, MATH500, speed, VRAM on 1.5B model |

## Quantized Components

Same as FP8 scheme (6 linear layers per block):

| Component | Shape (1.5B) | Quantize? |
|-----------|-------------|-----------|
| att.receptance | [2560, 2560] | Yes |
| att.key | [2560, 2560] | Yes |
| att.value | [2560, 2560] | Yes |
| att.output | [2560, 2560] | Yes |
| ffn.key | [2560, 11264] | Yes |
| ffn.value | [11264, 2560] | Yes |
| emb | [65536, 2560] | No |
| head | [65536, 2560] | No |
| LayerNorm | [2560] | No |
| Low-rank (g1/g2/a1/a2/w1/w2/v1/v2) | small | No |

## Hardware Compatibility

| GPU | SM | Per-Tensor INT8 | Affine INT8 | Triton Kernel |
|-----|-----|-----------------|-------------|---------------|
| V100/T4 | 7.0/7.5 | DP4A | DP4A | Row-level |
| A100 | 8.0 | `mm` | Triton | Batched GEMV |
| RTX 4090 | 8.9 | `mm` | Triton | Tensor core |
| RTX 5070 Ti | 12.0 | `mm` | Triton | Tensor core |
| CPU | - | Fallback | Fallback | Dense matmul |
