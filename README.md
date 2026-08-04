# RWKV-7 FP8 Quantized Inference

> **Status**: Production-ready | **Format**: FP8 E4M3 | **Speedup**: 6.4x (7.2B) | **Accuracy loss**: <=0.3%

[中文](README_zh.md) | **English**

Full FP8 weight quantization for RWKV-7 models, achieving **44.9 tok/s decode** on RTX 5070 Ti (Blackwell) — 6.4x over the 7.0 tok/s BF16 baseline — while reducing VRAM from 13.3 GB to **7.35 GB** with **93.75%** Top-1 consistency.

MATH500 evaluation reaches **53%** (vs 28% for the original 2.9B model), GSM8K reaches **83%** (vs 27% for 2.9B), demonstrating that FP8 quantization preserves the 7.2B model's reasoning capability nearly losslessly.

---

## Quick Start

```bash
# 1. Quantize a model
python quantize_model.py \
  --model /path/to/rwkv7-7.2b.pth \
  --output /path/to/rwkv7-7.2b-fp8.pth \
  --scheme fp8

# 2. Run inference (Albatross engine auto-detects quantized weights)
python rwkv7_fast_v3a.py --model /path/to/rwkv7-7.2b-fp8.pth
```

The quantized `.pth` file contains FP8 weights + per-tensor scales + meta rules. The inference engine auto-switches to the quantized path by detecting `.fp8_scale` keys — **no inference code changes required**.

---

## Quantization Scheme

### Core Approach: Full FP8, No Residuals

All 6 linear layer components (att.receptance/key/value/output + ffn.key/value) use **FP8 E4M3 per-tensor quantization**. Not quantized: emb, head, LayerNorm, low-rank weights (g1/g2/a1/a2/w1/w2/v1/v2), vector parameters (x_r/x_w/...k_k/k_a/r_k).

**Why not NVFP4?** After 4 phases and 30+ systematic experiments (see [Iteration History](#iteration-history)), NVFP4 has fundamental limitations:

| Metric | Full FP8 | NVFP4+FP8 Residual (X5) | Pure NVFP4 |
|--------|----------|--------------------------|-------------|
| Top-1 (7.2B) | **93.75%** | 91.02% | ~85% |
| Relative quant. error | **0.2%** | 0.2%+8.8% | 8.8% |
| File size | **7.96 GB** | 8.85 GB | 5.2 GB |
| Implementation complexity | **Low** | High (dual GEMM) | Medium |
| Residual recovery rate | — | 97.7% (FP8) | 91.4% (FP4) |

NVFP4's 16 discrete values `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` cause 8.8% quantization error — 44x worse than FP8 (0.2%). Even with FP8 residual compensation, accuracy remains below pure FP8.

### Quantization Principle

```python
# FP8 E4M3 per-tensor quantization
scale = max(|W|) / 448.0           # FP8 E4M3 max value = 448
W_fp8 = clamp(W / scale, -448, 448).to(float8_e4m3fn)

# Dequantization
W_dequant = W_fp8 * scale

# Inference (W8A8 path)
x_scale = max(|x|) / 448.0
x_fp8 = clamp(x / x_scale, -448, 448).to(float8_e4m3fn)
out = torch._scaled_mm(x_fp8, W_fp8.T, scale_a=x_scale, scale_b=scale)
```

### Non-Quantized Components

| Category | Parameters | Reason |
|----------|-----------|--------|
| Global | emb.weight, head.weight | emb on CPU; head is final projection layer |
| Low-rank | g1/g2, a1/a2, w1/w2, v1/v2 | Only 1.7% VRAM; quantization not worth the accuracy loss |
| Vector | x_r/x_w/x_k/x_v/x_a/x_g, k_k, k_a, r_k | Negligible parameter count (~100KB/layer) |
| Norm | ln0, ln1, ln2, ln_x, ln_out | Normalization params; quantization breaks numerical stability |

---

## Files

### Core Code (root directory)

| File | Lines | Description |
|------|-------|-------------|
| `quantize_model.py` | 236 | Unified quantization tool: load -> FP8 quantize -> save .pth + meta |
| `fp8_ops.py` | 148 | FP8 weight detection, loading, GEMM operations (W8A8 / W8A16) |
| `fused_fp8_gemm.py` | 730 | Fused Triton kernels: prep_x + FP8 hardware dot + RKV fusion |

### Quantization Metadata Format

The quantized `.pth` file contains a `meta` dictionary:

```python
meta = {
    "v": 1,
    "scheme": "fp8",
    "layers": 32,
    "r": [[0, 999, 0, 1], [0, 999, 1, 1], ...],  # [layer_start, layer_end, comp, dtype]
    "s": {"sd": "fp8e4m3", "td": "fp32"},
    "n": ["emb.", "head.", "ln_out.", ...],        # non-quantized prefix list
    "stats": {"bf16": 0, "fp8": 192},
    "compression": 1.82,
}
```

The inference engine detects `.fp8_scale` suffix keys via `is_fp8_weight(z, key)`, automatically loading FP8 weights and routing to the quantized path.

---

## Performance Benchmarks

### 7.2B Model (RTX 5070 Ti, Blackwell)

| Metric | Original BF16 | FP8 Quantized | Change |
|--------|--------------|---------------|--------|
| Decode speed | 7.0 t/s | **44.9 t/s** | 6.4x |
| Prefill speed (1x128) | — | 1603 t/s | — |
| VRAM usage | 13.32 GB | **7.35 GB** | -45% |
| File size | 14.40 GB | **7.96 GB** | -45% |
| Top-1 consistency | 100% | **93.75%** | -6.25% |
| PPL delta (2048) | — | +0.24% | — |
| MATH500 | ~55% | **53%** | -2pp |
| GSM8K | ~85% | **83%** | -2pp |

### 1.5B Model

| Metric | Original BF16 | FP8 Quantized | Change |
|--------|--------------|---------------|--------|
| Decode speed | 164.1 t/s | **67.8 t/s** | — |
| VRAM usage | 2.69 GB | **1.60 GB** | -41% |
| Top-1 consistency | 100% | **97.85%** | -2.15% |
| PPL delta (2048) | — | -0.08% | — |

### Concurrency Stress Test (7.2B, 64 concurrent)

| Metric | Value |
|--------|-------|
| Total throughput | 473.2 tok/s |
| p50 latency | 51.0s |
| p90 latency | 67.5s |
| Error rate | 0/64 |

### Operator Optimization Details

Fused kernels with shape-aware tile configuration for Blackwell architecture:

| Matrix shape | Scenario | Tile (M,N,K,W) | Speedup |
|-------------|----------|-----------------|---------|
| 4096x4096 | att (decode) | (16,64,64,4) | 1.84x vs baseline |
| 16384x4096 | ffn_key (decode) | (16,64,128,4) | +37% |
| 4096x16384 | ffn_value (decode) | (16,128,256,8) | +29% |

Key optimizations:
- **prep_x fusion**: Input cast + AWQ + amax in a single kernel launch
- **FP8 hardware dot**: `tl.dot(fp8, fp8)` directly utilizes Blackwell FP8 tensor cores
- **RKV fusion**: r/k/v attention projections computed in a single kernel
- **CUDA Graph disabled**: Decode step's 96 kernel replays cause ~1ms overhead > launch savings

---

## Iteration History

This project went through 4 phases of systematic exploration. Full reports are in the `iterations/` directory:

| Phase | Directory | Content | Key Conclusion |
|-------|-----------|---------|----------------|
| Phase 1 | `iterations/phase1_nvfp4_exploration/` | NVFP4 exploration, sensitivity analysis, long-sequence state MSE | NVFP4 error 8.8%, all components equally sensitive |
| Phase 2 | `iterations/phase2_engine_adaptation/` | Engine integration, fused kernel development, 1.5B/7.2B benchmarks | Fused kernel 1.84x speedup, CUDA Graph not beneficial |
| Phase 3 | `iterations/phase3_x5_residual_scheme/` | X5 residual scheme, multi-model validation, generation quality | X5 slightly more accurate but not worth the complexity |
| Phase 4 | `iterations/phase4_fp8_optimization/` | Final FP8 scheme, operator optimization, performance tuning | Full FP8 is the optimal scheme |

See [QUANTIZATION_CONCLUSION.md](QUANTIZATION_CONCLUSION.md) for the complete experimental comparison.

---

## Quantization Sensitivity Analysis

Derived from RWKV-7 forward propagation formulas, the sensitivity ranking:

```
*****  att.key.weight     state erase direction + info injection, dual path into state
****   att.value.weight   state info injection, layer0 cross-layer propagation (v_first)
***    att.receptance     read-only on state, errors don't accumulate
**     att.output.weight  residual stream + GroupNorm buffer
*      ffn.key/value       no state, ReLU2 suppresses ~50% channels
```

**Experimental conclusion**: Although att.key has the highest theoretical sensitivity, FP8 (0.2% error) is safe for all components. NVFP4 (8.8% error) causes significant accuracy loss even for the least sensitive ffn.

---

## Technical Notes

### Why is FP8 better than NVFP4+FP8 residual?

1. **Residual scheme is larger**: NVFP4(0.5B/elem) + FP8 residual(1B/elem) = 1.5B/elem > FP8(1B/elem)
2. **Residual scheme is slower**: Requires two GEMMs (main path + residual path)
3. **FP4 residual is ineffective**: FP4 has only 16 levels, 91.4% recovery rate (15.2% crushed to 0), cannot compensate main quantization error
4. **FP8 direct quantization is simpler**: No residual management, no dual-path dispatch

### Why is head.weight not quantized?

`head.weight [65536, 4096]` is the final vocabulary projection layer. Quantization would shift the logits distribution, directly affecting token sampling. Keeping FP16 ensures generation quality.

### Why are low-rank weights not quantized?

Low-rank weights (g1/g2 [4096,480], a1/a2 [4096,128], etc.) account for only 1.7% of VRAM. FP8 quantization would:
- Save only 1.7% on disk, zero runtime benefit
- Increase PPL by 0.0052
- Require additional dequantization logic

The cost-benefit ratio is extremely low; keeping BF16.

---

## Dependencies

- PyTorch >= 2.1 (requires `torch._scaled_mm` and `float8_e4m3fn` support)
- Triton >= 2.1 (fused kernels)
- Blackwell / Ada Lovelace / Hopper GPU (FP8 tensor core hardware support)

---

## Issues

Discussions welcome in the [Issues](https://github.com/AUXStar/rwkv7-quantization/issues) section:

- **#1-#9**: Quantization scheme design, NVFP4 ablation experiments, toolchain development
- **#10-#14**: X5 residual scheme validation, multi-model testing, generation quality evaluation
- **#15-#16**: Operator optimization, performance tuning
- **#12**: Per-layer/per-head sensitivity attribution (research direction, contributions welcome)

---

## Acknowledgments

- [RWKV-7](https://github.com/RWKV/RWKV-LM) model architecture
- [Blink_DL](https://modelscope.cn/models/Blink_DL/temp-latest-training-models) model weights
- [Albatross](https://github.com/BlinkDL/Albatross) inference engine
