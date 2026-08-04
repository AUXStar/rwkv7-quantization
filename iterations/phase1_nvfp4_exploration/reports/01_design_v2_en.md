# #1 [Meta] RWKV-7 Layered Quantization Scheme v2 (Pure Quantized GEMM)

## Major Changes

The v1 scheme was based on dequantization inference (W4A16) and has been rejected by the user. v2 switches to **pure quantized GEMM**, where all quantized weights directly participate in `_scaled_mm` computation without dequantization.

### v1 -> v2 Core Changes

| Item | v1 (deprecated) | v2 (current) |
|------|-------------|-----------|
| Inference method | Dequantize to FP16 then GEMM | Pure quantized GEMM (`_scaled_mm`) |
| FFN key | NVFP4 W4A16 | NVFP4+FP8 residual W4A4+W8A8 |
| FFN value | NVFP4 W4A16 | FP8 W8A8 |
| Attention key/value | Layered BF16/FP8/NVFP4 | Unified FP8 W8A8 (L4-19 optional NVFP4) |
| L0 value | BF16 | FP8 (#5 verified lossless) |
| Activation quantization | Not quantized (A16) | Online quantization to FP4/FP8 (A4/A8) |

## Target Models

| Model | Path | Original Size | dtype | Layers | Purpose |
|------|------|----------|-------|------|------|
| 1.5B | `rwkv7-g1h-1.5b-20260710-ctx10240.pth` | 2.97 GB | bf16 | 24 | Experimental validation |
| 2.9B | `rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth` | 5.49 GB | bf16 | 32 | Experimental validation |
| 7.2B | `rwkv7-g1g-7.2b-20260523-ctx8192.pth` | 13.41 GB | bf16 | 32 | Final target |

## Quantization Formats

### NVFP4 (E2M1) — W4A4

- Weights: 4-bit floating-point (1s+2e+1m), per-16-elements shared FP8(E4M3) block scale + FP32 tensor scale
- Activations: online quantization to FP4, using the same block+tensor scale structure
- GEMM: `torch._scaled_mm(fp4, fp4, scale_a, scale_b) -> bf16`
- AWQ: channel scaling (alpha=0.5), weight heuristics (use weight abs mean instead of activation statistics when no calibration data)
- Clip ratio search: 9 candidate values [0.60~1.00], per-block select minimum MSE
- Weight storage: 128x4 swizzle format (required by `_scaled_mm`)

### FP8 (E4M3) — W8A8

- Weights: 8-bit floating-point (1s+4e+3m), per-tensor scale
- Activations: online quantization to FP8
- GEMM: `torch._scaled_mm(fp8, fp8, scale_a, scale_b) -> bf16`

### NVFP4+FP8 Residual — W4A4+W8A8

- Main weight: NVFP4 W4A4 GEMM
- Residual weight: (original weight - NVFP4 dequantized value) quantized to FP8 W8A8
- Two-path GEMM results summed: `out = nvfp4_gemm(x, w) + fp8_gemm(x, residual)`
- Purpose: compensate for NVFP4 W4A4 quantization error (FFN key)

## Layered Scheme (v2, based on #2-#6 experimental conclusions)

### 1.5B (24 layers)

```
component:  key    value  rec    out    ffn_key      ffn_value
L0-3:       fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
L4-19:      nvfp4  fp8    nvfp4  nvfp4  nvfp4+res    fp8
L20-23:     fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
```

### 2.9B / 7.2B (32 layers)

```
component:  key    value  rec    out    ffn_key      ffn_value
L0-3:       fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
L4-27:      nvfp4  fp8    nvfp4  nvfp4  nvfp4+res    fp8
L28-31:     fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
```

### Scheme Selection Rationale

| Component | Choice | Rationale |
|------|------|------|
| att.key | Edge FP8 / Middle NVFP4 | #4: NVFP4 key PPL delta 0.0041 (feasible but slightly worse than FP8); edge layers more sensitive, use FP8 |
| att.value | All FP8 | #4: value more sensitive than key; #3: FP8 W8A8 PPL delta 0.0018 near-lossless |
| att.rec/out | All NVFP4 | #4: rec/out low sensitivity, NVFP4 feasible (scheme C PPL delta 0.0101) |
| ffn.key | NVFP4+FP8 residual | #2: pure NVFP4 W4A4 error large (0.0385); residual compensates error |
| ffn.value | All FP8 | #2: ReLU-squared activation distribution unfavorable for FP4; FP8 W8A8 better precision |
| L0 value | FP8 (not BF16) | #5: FP8 vs BF16 PPL delta difference 0.0003, no need for BF16 |

## Validation Summary (#2-#6, 1.5B, 2100 tokens)

| Issue | Scheme | PPL delta | Top-1 | Speed | VRAM |
|-------|------|-----------|-------|------|------|
| #2 | FFN NVFP4 W4A4 | +0.0385 | 96.09% | 2426 t/s | 1.61G |
| #3 | Att FP8 W8A8 | +0.0018 | 99.43% | 6563 t/s | 2.50G |
| #4A | key NVFP4 L4-19 | +0.0041 | 98.19% | 2458 t/s | 1.54G |
| #4B | All FP8 | +0.0033 | 98.38% | 6132 t/s | 1.60G |
| #5A | L0 value BF16 | +0.0030 | 98.33% | — | 1.57G |
| #5B | L0 value FP8 | +0.0033 | 98.38% | — | 1.60G |
| #6 | Mixed scheme | +0.0242 | 97.14% | 3322 t/s | 1.67G |

### Key Findings

1. **FP8 W8A8 far superior to NVFP4 W4A4**: Both precision (0.0018 vs 0.0385) and speed (6563 vs 2426 t/s) are better
2. **Errors concentrated in warmup phase**: First 500 tokens Top-1 ~84-92%, last 1600 tokens Top-1=100%
3. **No state accumulation**: PPL delta decreases with sequence growth approaching 0 (#6 validation)
4. **L0 value does not need BF16**: FP8 W8A8 is sufficient (#5 validation)
5. **FFN value unsuitable for NVFP4**: ReLU-squared activation distribution unfavorable for FP4 (#2 finding)

## Unquantized Parameters

### Vector parameters (~104KB per layer)
- `blocks.*.att.x_{r,w,k,v,a,g}` [1,1,4096] x6
- `blocks.*.att.{w0,a0,v0,k_k,k_a}` [1,1,4096] x5
- `blocks.*.att.r_k` [64,64]
- `blocks.*.ffn.x_k` [1,1,4096]

### Low-rank weights (~13MB per layer, ~406MB for 32 layers)
- `att.g1/g2` [4096,480]/[480,4096]
- `att.a1/a2` [4096,128]/[128,4096]
- `att.w1/w2` [4096,128]/[128,4096]
- `att.v1/v2` [4096,96]/[96,4096]

### LayerNorm / Global
- `blocks.*.ln{0,1,2}.weight/bias`, `blocks.*.att.ln_x.weight/bias`
- `ln_out.weight/bias`
- `emb.weight` [65536,4096], `head.weight` [65536,4096]

## Toolchain

| File | Function |
|------|------|
| `quantize_model.py` | Unified quantization tool (load -> classify -> quantize -> store+meta) |
| `nvfp4_ops.py` | NVFP4/FP8 GEMM kernel + weight loading + activation quantization |
| `fused_nvfp4_quant.py` | Triton fused kernel (activation quantization + pack + swizzle) |

## Acceptance Criteria

- PPL delta <= 0.05 (1.5B) / <= 0.02 (7.2B)
- Top-1 >= 99.5% (after warmup)
- Compression ratio >= 2x
- Pure quantized GEMM, no dequantization
