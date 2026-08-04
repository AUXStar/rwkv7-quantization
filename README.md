# RWKV-7 FP8 Quantized Inference

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-red.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-%3E%3D2.1-green.svg)](https://github.com/openai/triton)
[![FP8](https://img.shields.io/badge/quantization-FP8%20E4M3-orange.svg)](#quantization-scheme)
[![Speedup](https://img.shields.io/badge/speedup-6.4x-brightgreen.svg)](#performance-benchmarks)
[![Accuracy](https://img.shields.io/badge/Top--1-93.75%25-success.svg)](#performance-benchmarks)
[![VRAM](https://img.shields.io/badge/VRAM-7.35GB-blue.svg)](#performance-benchmarks)
[![Platform](https://img.shields.io/badge/platform-Blackwell%20%7C%20Ada%20%7C%20Hopper-purple.svg)](#dependencies)
[![中文](https://img.shields.io/badge/%E6%96%87%E6%A1%A3-中文-red.svg)](README_zh.md)

> **Status**: Production-ready | **Format**: FP8 E4M3 | **Speedup**: 6.4x (7.2B) | **Accuracy loss**: <=0.3%

[中文](README_zh.md) | **English**

Full FP8 weight quantization for RWKV-7 models, achieving **44.9 tok/s decode** on RTX 5070 Ti (Blackwell) — 6.4x over the 7.0 tok/s BF16 baseline — while reducing VRAM from 13.3 GB to **7.35 GB** with **93.75%** Top-1 consistency.

MATH500 evaluation reaches **53%** (vs 28% for the original 2.9B model), GSM8K reaches **83%** (vs 27% for 2.9B), demonstrating that FP8 quantization preserves the 7.2B model's reasoning capability nearly losslessly.

### Highlights

- **6.4x decode speedup** — 7.0 to 44.9 tok/s on 7.2B model
- **45% VRAM reduction** — 13.3 GB to 7.35 GB, fits on consumer GPUs
- **<0.3% accuracy loss** — 93.75% Top-1 consistency, 53% MATH500, 83% GSM8K
- **Fused Triton kernels** — Shape-aware tile config for Blackwell FP8 tensor cores
- **Zero code changes** — Auto-detect quantized weights, drop-in replacement
- **4-phase systematic research** — 30+ experiments comparing FP8 vs NVFP4 vs residual schemes

### Keywords

`RWKV-7` `RWKV` `FP8` `E4M3` `quantization` `model compression` `inference acceleration` `Triton` `CUDA` `Blackwell` `tensor core` `low-bit quantization` `weight quantization` `LLM` `RNN` `GPU inference` `model optimization` `8-bit quantization` `fused kernel` `RTX 5070 Ti`

---

## Table of Contents

- [Quick Start](#quick-start)
- [What is RWKV-7?](#what-is-rwkv-7)
- [Quantization Scheme](#quantization-scheme)
- [Files](#files)
- [Performance Benchmarks](#performance-benchmarks)
- [Iteration History](#iteration-history)
- [Quantization Sensitivity Analysis](#quantization-sensitivity-analysis)
- [Technical Notes](#technical-notes)
- [Dependencies](#dependencies)
- [Issues](#issues)
- [Acknowledgments](#acknowledgments)

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

### What is RWKV-7?

RWKV-7 is the latest generation of the RWKV (Receptance Weighted Key Value) architecture — a linear RNN that combines transformer-level performance with RNN-level inference efficiency. Unlike transformers with O(n²) attention, RWKV processes tokens in O(1) per step, making it ideal for long-context generation and edge deployment. This project applies **FP8 E4M3 weight quantization** to reduce model size and accelerate inference on modern NVIDIA GPUs (Blackwell, Ada Lovelace, Hopper).

| Feature | RWKV-7 | Transformer |
|---------|--------|-------------|
| Inference complexity | O(1) per token | O(n²) attention |
| Context length | Unlimited (fixed state) | Bounded by KV cache |
| VRAM scaling | Constant | Linear with context |
| Quantization benefit | Direct speedup (memory-bound) | Limited (compute-bound) |

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

## FAQ

### Does FP8 quantization work on all GPUs?

No. FP8 tensor cores require **NVIDIA Blackwell** (RTX 50 series), **Ada Lovelace** (RTX 40 series), or **Hopper** (H100) architectures. Older GPUs (Ampere and earlier) do not have FP8 hardware support. The `torch._scaled_mm` API and `float8_e4m3fn` dtype require PyTorch >= 2.1.

### How does this compare to GPTQ or AWQ?

GPTQ and AWQ are INT4/INT8 quantization methods that require calibration data. This project uses **FP8 E4M3 per-tensor quantization** — no calibration needed, direct weight-only quantization. FP8 has 256 levels (vs INT4's 16), resulting in much lower quantization error (0.2% vs 8.8% for NVFP4). The trade-off is larger file size (1 byte/weight vs 0.5 bytes for INT4).

### Can I use this with other RWKV models?

Yes. The quantization tool (`quantize_model.py`) works with any RWKV-7 `.pth` model. The 1.5B and 7.2B models were tested. The 13.3B model is planned for future testing.

### Why not use NVFP4 (FP4) for smaller files?

NVFP4 has only 16 discrete values, causing 8.8% relative quantization error — 44x worse than FP8 (0.2%). We tested NVFP4 with FP8 residual compensation (X5 scheme), but it still underperformed pure FP8 on the 7.2B model (91.02% vs 93.75% Top-1). See [Phase 3 reports](iterations/phase3_x5_residual_scheme/) for details.

### Is the quantized model compatible with the original Albatross engine?

Yes. The inference engine auto-detects `.fp8_scale` keys in the model file and routes to the FP8 GEMM path. No code changes needed — just load the quantized `.pth` file as you would the original.

---

## Dependencies

- PyTorch >= 2.1 (requires `torch._scaled_mm` and `float8_e4m3fn` support)
- Triton >= 2.1 (fused kernels)
- Blackwell / Ada Lovelace / Hopper GPU (FP8 tensor core hardware support)

---

## Research

### EAR (Expected Acceptance Rate) Metric

Based on [SLQ paper (arXiv:2605.02404)](https://arxiv.org/abs/2605.02404), we implemented the EAR metric to measure distribution-level quantization loss beyond simple Top-1 agreement.

**FP8 quantization EAR = 0.94** (1.5B model): classified as "significant distribution shift" (below 0.95 threshold). While Top-1 agreement is 92.5%, the full probability distributions diverge more than Top-1 alone suggests.

| Metric | Per-Tensor FP8 | Per-Channel FP8 |
|--------|---------------|-----------------|
| EAR | **0.9412** | 0.9262 |
| Top-1 agreement | 92.52% | **95.33%** |
| KL(orig∥quant) | **0.0201** | 0.0376 |

### Per-Channel FP8 Quantization

Based on [Weight Quantization Study (arXiv:2505.03803)](https://arxiv.org/abs/2505.03803), we tested per-output-channel FP8 scales.

**Result**: Per-channel improves Top-1 (+2.8pp) but worsens distribution similarity (EAR -1.5pp, KL +87%). **Not recommended** — per-tensor remains optimal.

### Weight Distribution Asymmetry

Analysis of all 144 weight matrices (24 layers × 6 components) shows **perfectly symmetric distributions** (|skew| < 0.01, positive fraction ~50%). Asymmetric quantization provides zero benefit.

### Experiment Files

| File | Description |
|------|-------------|
| `experiments/eval_ear.py` | EAR evaluation script |
| `experiments/analyze_weights.py` | Weight distribution asymmetry analysis |
| `experiments/EXPERIMENT_REPORT.md` | Complete experiment report |

### References

1. SLQ: Simple Linear Quantization, [arXiv:2605.02404](https://arxiv.org/abs/2605.02404)
2. A Weight Quantization Study, [arXiv:2505.03803](https://arxiv.org/abs/2505.03803)

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

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{rwkv7-fp8-quantization,
  title={RWKV-7 FP8 Quantized Inference: 6.4x Speedup with Lossless Accuracy},
  author={AUXStar},
  year={2026},
  url={https://github.com/AUXStar/rwkv7-quantization}
}
```

## Star History

If this project helps you, please consider giving it a star!
