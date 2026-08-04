# INT4 Quantization

## Overview

INT4 (4-bit integer) quantization for RWKV-7, providing maximum compression (4x vs FP16) at the cost of higher quantization error. Suitable for memory-constrained deployment scenarios.

## Schemes

### 1. Per-Tensor Symmetric INT4 (W4A16)

Simplest 4-bit scheme, weight-only:

```
scale = max(|W|) / 7
w_int4 = round(W / scale).clamp(-8, 7).to(int8)
W_approx = w_int4 * scale
```

- **Packed storage**: two int4 in one uint8 byte: `packed = lo | (hi << 4)`
- **Activation**: stays FP16 (W4A16)
- **Compression**: 4x (vs FP16)
- **Levels**: 16 discrete values `{-8, -7, ..., 6, 7}`

### 2. Affine INT4 (MM4-style, per-row + per-column)

Official RWKV 4-bit format, paired nibble packing:

```
# Same affine as MM8, but 16 levels
my = amin(W, dim=1);  W = W - my
mx = amin(W, dim=0);  W = W - mx
rx = amax(W, dim=0);  W = W / rx
ry = amax(W, dim=1);  W = W / ry
w_u4 = clip(floor(W * 16), 0, 15).to(uint8)
# Pack: two nibbles per byte along output dim
packed = w_u4[:, 0::2] | (w_u4[:, 1::2] << 4)
W_approx = (u4 + 0.5) * ry * rx + my + mx
```

- **7 buffers**: packed, mx, rx_s, my, ry_s, m_orig
- **Compression**: ~4x
- **Precision**: better than per-tensor (dual affine)

### 3. Group-wise INT4

Per-group quantization with configurable group size:

```
# group_size=128: every 128 elements share one (scale, offset)
for g in range(0, K, group_size):
    chunk = W[:, g:g+group_size]
    scale_g = (chunk.max() - chunk.min()) / 15
    offset_g = chunk.min()
    w_u4_g = clip(round((chunk - offset_g) / scale_g), 0, 15)
```

- **group_size options**: 128, 256
- **Precision**: significantly better than per-tensor (finer granularity)
- **Compression**: ~3.5x (group scale/offset overhead)
- **Modules**: selectively applied (e.g., only ffn.key/value or only lm_head)

## Implementation Plan

| Step | File | Description |
|------|------|-------------|
| 1 | `quantize_int4.py` | Quantization tool (per-tensor + affine + group-wise) |
| 2 | `int4_ops.py` | Weight detection, loading, nibble unpack, GEMM dispatch |
| 3 | `fused_int4_gemm.py` | Triton fused kernels (nibble unpack in-register) |
| 4 | Benchmark | PPL, Top-1, MATH500, speed, VRAM on 1.5B model |

## Quantized Components

Same 6 linear layers as FP8/INT8 schemes. Additionally, group-wise quantization can be selectively applied:

| Group Policy | Modules with group-wise | Rationale |
|-------------|------------------------|-----------|
| `all` | All 6 components | Maximum precision |
| `ffn_key` | Only ffn.key | Largest matrix, most sensitive |
| `ffn_key_value` | ffn.key + ffn.value | FFN pair |
| `lm_head` | Only head | Output layer precision |

## Key Design Decisions

1. **W4A16 over W4A4**: Activation stays FP16 — int4 activation (16 levels) causes severe quality loss, confirmed by our NVFP4 experiments (EAR dropped significantly)

2. **Paired nibble along output dim**: `packed[n, b] = u4[n, 2b] | (u4[n, 2b+1] << 4)` — adjacent output channels share a byte, enabling efficient GEMV with single byte load per two outputs

3. **+0.5 centering**: `(u4 + 0.5)` centers the reconstruction at the midpoint of each quantization bin, reducing average error by ~50%

4. **Scale absorption**: `rx_stored = rx / 4.0` absorbs the 16-level factor into the scale, avoiding an extra multiply in the hot path

## Expected Performance (1.5B model, RTX 5070 Ti)

| Metric | FP8 (baseline) | INT4 per-tensor | INT4 affine | INT4 group-128 |
|--------|---------------|-----------------|-------------|----------------|
| Compression | 1.66x | 4.0x | 3.8x | 3.5x |
| VRAM | 1.60 GB | ~0.85 GB | ~0.90 GB | ~0.95 GB |
| Top-1 | 92.52% | ~80%? | ~85%? | ~88%? |
| Decode speed | 67.8 t/s | ? | ? | ? |

> Values with `?` are predictions — actual results will be measured during implementation.
