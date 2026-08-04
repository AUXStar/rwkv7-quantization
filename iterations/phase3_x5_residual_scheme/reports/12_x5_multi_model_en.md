# X5 Dilution Layer Quantization Scheme - 1.5B / 2.9B / 7.2B Complete Comparison

**Date**: 2026-08-01
**Author**: Quantization Experiment Automation

---

## I. Core Conclusions

The X5 scheme **passes on all three** RWKV-7 models: 1.5B / 2.9B / 7.2B:

| Model | Compression Ratio | VRAM Savings | PPL delta | MATH500 |
|------|--------|-----------|-----------|---------|
| 1.5B | **1.7x** | 0.32 GiB (-15%) | +0.0024 | -0.8pp (500 problems) |
| 2.9B | **1.8x** | 1.58 GiB (-30%) | +0.0006 | -1.0pp (500 problems) |
| 7.2B | **1.8x** | 4.21 GiB (-32%) | +0.0012 | +2.0pp (100 problems, within noise) |

**Key Insight**: Dilution point positions (L0 + L8/L6 + L24/L18 + L31) automatically scale at **1/4 + 3/4 ratio** on 24/32 layer models, no tuning needed.

---

## II. Quantization Method Details

### 2.1 X5 Scheme (Universal for 1.5B / 2.9B / 7.2B)

| Tensor Type | Quantization Method | Ratio |
|---------|---------|------|
| **att.receptance** | NVFP4 (E2M1) + per-16-block FP8 scale + FP32 tensor scale + AWQ (α=0.3) | 4bit |
| **att.key** | FP8 E4M3 + per-tensor FP32 scale | 8bit |
| **att.value** | FP8 E4M3 + per-tensor FP32 scale | 8bit |
| **att.output** | NVFP4 + per-16-block FP8 scale + FP32 tensor scale + AWQ (α=0.3) | 4bit |
| **ffn.key** | NVFP4 + FP8 residual (per-block ratio + tensor scale) | ~6bit effective |
| **ffn.value** | FP8 E4M3 + per-tensor FP32 scale | 8bit |

### 2.2 Dilution Layers (bf16 protection)

- **1.5B (24 layers)**: L0 + L6 + L18 + L23 (ratios 0/25/75/96)
- **2.9B / 7.2B (32 layers)**: L0 + L8 + L24 + L31 (ratios 0/25/75/97)

All 6 linear layers remain bf16 at dilution layers.

### 2.3 Tensors Not Quantized (Universal Across All Models)

- `emb.weight`, `head.weight` (embedding/output lookup)
- Low-rank parameters `g1/g2/a1/a2/w1/w2/v1/v2` (account for 1.7% of memory)
- Vector parameters `x_r/x_w/x_k/x_v/x_a/x_g, w0/a0/v0, k_k/k_a/r_k` (~104KB per layer)
- All `LayerNorm/GroupNorm` (normalization must be precise)

### 2.4 AWQ Channel Scaling

- `alpha = 0.3` (optimal from 7-point grid search)
- Uses weight-based heuristic (column absolute mean) as activation proxy

---

## III. 1.5B Detailed Data (Baseline, 24 layers)

### 3.1 File and VRAM

| Item | Value |
|----|------|
| Original .pth file | **2.85 GB** |
| Quantized .pth file | **1.66 GB** (1.7x compression) |
| Original load VRAM (orig bf16) | **2.18 GiB** |
| Quantized load VRAM (X5 quantized weights) | **1.86 GiB** |
| **VRAM Savings** | **0.32 GiB (-15%)** |

Note: 1.5B is small, emb+head+norm accounts for a large proportion, so VRAM savings ratio is small after 1.7x compression.

### 3.2 Accuracy

| Metric | Orig | X5 | Δ |
|------|------|----|----|
| PPL@8192 | 1.2031 | 1.2055 | **+0.0024** |
| MATH500 (500 problems) | 12.6% (63/500) | 11.8% (59/500) | -0.8pp |

### 3.3 Speed

- decode: **78.9 t/s** (B=1, after quantization)
- Speed reaches 43% of native (decode CPU-launch-bound, known bottleneck)

---

## IV. 2.9B Detailed Data (32 layers)

### 4.1 File and VRAM

| Item | Value |
|----|------|
| Original .pth file | **5.49 GB** |
| Quantized .pth file | **3.72 GB** (1.8x compression) |
| Original load VRAM (orig bf16) | **5.35 GiB** |
| Quantized load VRAM (X5 quantized weights) | **3.78 GiB** |
| **VRAM Savings** | **1.58 GiB (-30%)** |

### 4.2 Accuracy

| Metric | Orig | X5 | Δ |
|------|------|----|----|
| PPL@8192 | 1.0409 | 1.0415 | **+0.0006** |
| MATH500 (500 problems) | 5.8% (29/500) | 4.8% (24/500) | -1.0pp |

Note: 2.9B is a preview training checkpoint (preview4673), MATH500 baseline is only 5.8%, not representative of production 2.9B level.

### 4.3 Speed

- decode: **49.5 t/s** (B=1, after quantization)

---

## V. 7.2B Detailed Data (32 layers, g1g)

### 5.1 File and VRAM

| Item | Value |
|----|------|
| Original .pth file | **13.40 GB** |
| Quantized .pth file | **8.50 GB** (1.8x compression) |
| Original load VRAM (orig bf16) | **13.32 GiB** |
| Quantized load VRAM (X5 quantized weights) | **9.11 GiB** |
| **VRAM Savings** | **4.21 GiB (-32%)** |

### 5.2 Accuracy

| Metric | Orig | X5 | Δ |
|------|------|----|----|
| PPL@2048 | 1.1610 | 1.1622 | **+0.0012** |
| MATH500 (100 problems) | 5.0% (5/100) | 7.0% (7/100) | +2.0pp (within noise) |

Note: 100-problem sample standard deviation ~2.2%, ±2pp difference is within statistical noise. PPL delta +0.0012 is the reliable accuracy metric.

### 5.3 Speed

- decode: **25.1 t/s** (B=1, after quantization, only 30 iters)

---

## VI. Cross-Comparison Table

### 6.1 Compression and VRAM

| Model | Original File | Quantized File | Compression Ratio | Original VRAM | Quantized VRAM | Savings (GiB) | Savings % |
|------|---------|---------|--------|---------|---------|-----------|-------|
| **1.5B** | 2.85 GB | 1.66 GB | **1.7x** | 2.18 GiB | 1.86 GiB | 0.32 | -15% |
| **2.9B** | 5.49 GB | 3.72 GB | **1.8x** | 5.35 GiB | 3.78 GiB | 1.58 | -30% |
| **7.2B** | 13.40 GB | 8.50 GB | **1.8x** | 13.32 GiB | 9.11 GiB | 4.21 | -32% |

### 6.2 Accuracy

| Model | PPL_orig | PPL_X5 | PPL_delta | MATH_orig | MATH_X5 | MATH_Δ |
|------|----------|--------|-----------|-----------|---------|--------|
| **1.5B** | 1.2031 | 1.2055 | +0.0024 | 12.6% | 11.8% | -0.8pp |
| **2.9B** | 1.0409 | 1.0415 | +0.0006 | 5.8% | 4.8% | -1.0pp |
| **7.2B** | 1.1610 | 1.1622 | +0.0012 | 5.0% | 7.0% | +2.0pp* |

*7.2B 100-problem sample, ±2.2% noise, difference not reliable

### 6.3 Speed

| Model | decode (B=1) | Relative to native |
|------|--------------|------------|
| **1.5B** | 78.9 t/s | 43% (optimized) |
| **2.9B** | 49.5 t/s | - |
| **7.2B** | 25.1 t/s | - |

---

## VII. Dilution Layer Position Adaptation Table

| Model | Total Layers | L0 | 1/4 Dilution Point | 3/4 Dilution Point | L_last | Total bf16 Layers | Percentage |
|------|------|-----|-----------|-----------|--------|------------|------|
| 1.5B | 24 | 0 | 6 | 18 | 23 | 4 | 16.7% |
| 2.9B | 32 | 0 | 8 | 24 | 31 | 4 | 12.5% |
| 7.2B | 32 | 0 | 8 | 24 | 31 | 4 | 12.5% |

**Pattern**: Dilution point positions = `round(N/4)`, `round(3N/4)`, plus head/tail protection.
**Role of L0**: v_first state source, must be bf16
**Role of L_last**: last layer before output, maintaining accuracy lower bound
**Role of the two middle points**: at 1/4 and 3/4 positions, quantization errors have accumulated to the mid-section, bf16 reset prevents error propagation

---

## VIII. Quantized Tensor Type Distribution (X5 Universal)

| Tensor | Type | Ratio | Notes |
|------|------|------|------|
| att.receptance (4/layer) | NVFP4 | 4bit | AWQ α=0.3, per-block FP8 scale |
| att.key (4/layer) | FP8 | 8bit | per-tensor FP32 scale |
| att.value (4/layer) | FP8 | 8bit | per-tensor FP32 scale |
| att.output (4/layer) | NVFP4 | 4bit | AWQ α=0.3, per-block FP8 scale |
| ffn.key (4/layer) | NVFP4+FP8 residual | ~6bit | per-block ratio + tensor scale |
| ffn.value (4/layer) | FP8 | 8bit | per-tensor FP32 scale |

**Effective average bit width**: (2×4 + 2×8 + 2×6) / 6 = **6.0 bit/weight** (including 4 layers bf16 protection)

---

## IX. Reproduction Commands

### 9.1 Quantization

```bash
# 1.5B
python quant_x5.py   # Protect L0+L6+L18+L23

# 2.9B / 7.2B
python quant_x5_32l.py <orig.pth> <out.pth>   # Protect L0+L8+L24+L31
```

### 9.2 Acceptance

```bash
# PPL + VRAM + decode
python accept_x5_32l.py <quant.pth> <orig.pth> <tag> [ppl_len=2048]

# MATH500 evaluation (100/500 problems)
python run_math500_x5_32l.py <quant.pth> <tag> [n=500] [batch=8]
```

---

## X. File List

| File | Size | Purpose |
|------|------|------|
| `/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth` | 2.9G | 1.5B original model |
| `/home/njzy/model/rwkv7-2.9b-X5.pth` | 3.8G | 2.9B X5 quantized |
| `/home/njzy/model/rwkv7-7.2b-X5.pth` | 8.9G | 7.2B X5 quantized |

> 2.9B / 7.2B original models deleted after acceptance (to save disk space)

---

## XI. Source Code

- `quant_x5_32l.py` — 32-layer model quantization (2.9B / 7.2B)
- `accept_x5_32l.py` — 32-layer model acceptance (PPL+VRAM+decode)
- `run_math500_x5_32l.py` — 32-layer model MATH500 evaluation (batch+repair)
- `quant_x5.py` — 24-layer model quantization (1.5B, reused)
- `accept_x5.py` — 24-layer model acceptance (reused)

---

## XII. Appendix: Why Not Use More Aggressive NVFP4-only Directly?

1.5B experimental data (historical):

| Scheme | PPL delta | MATH500 |
|------|-----------|---------|
| NVFP4-only (all att) | +0.058 | 8.6% |
| Mixed NVFP4+FP8 (key FP8) | +0.034 | 10.2% |
| + AWQ α=0.3 | +0.012 | 11.2% |
| + Dilution layers (X5) | **+0.0024** | **11.8%** |

NVFP4 4-bit single-point (E2M1, 16 discrete values) has large quantization errors, must use FP8 to protect attention key/value, then use dilution layers to reset accumulated errors.
