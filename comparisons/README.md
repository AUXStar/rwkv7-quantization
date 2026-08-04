# Cross-Scheme Comparison

## Overview

Systematic comparison of all quantization schemes for RWKV-7, evaluated on the same model, hardware, and benchmark suite.

## Schemes Under Comparison

| ID | Scheme | Format | Compression | Hardware Req |
|----|--------|--------|-------------|--------------|
| S0 | FP8 per-tensor (W8A8) | float8_e4m3fn | 2.0x | SM 8.9+ |
| S1 | INT8 per-tensor (W8A8) | int8 symmetric | 2.0x | Any CUDA |
| S2 | INT8 affine (MM8) | uint8 + dual affine | ~1.9x | Any CUDA |
| S3 | INT4 per-tensor (W4A16) | int4 packed | 4.0x | Any CUDA |
| S4 | INT4 affine (MM4) | uint4 + dual affine | ~3.8x | Any CUDA |
| S5 | INT4 group-128 (W4A16) | int4 + per-group scale | ~3.5x | Any CUDA |
| S6 | INT4 group-256 (W4A16) | int4 + per-group scale | ~3.7x | Any CUDA |

## Evaluation Framework

### Three-Layer Evaluation (same as FP8)

| Layer | Metric | Description |
|-------|--------|-------------|
| Distribution | EAR | Expected Acceptance Rate (logit distribution similarity) |
| Task | Top-1 | Next-token prediction agreement with BF16 |
| Task | MATH500 | Greedy decode, pass@1 |
| Task | GSM8K | Greedy decode, pass@1 |
| Task | Code gen | Token match rate, first divergence position |
| Efficiency | Decode speed | tok/s on RTX 5070 Ti |
| Efficiency | VRAM | Peak GPU memory during decode |
| Efficiency | File size | .pth file size on disk |

### Pareto Frontier Analysis

A scheme is on the Pareto frontier if no other scheme is strictly better in ALL dimensions simultaneously. Expected trade-offs:

- **FP8**: best precision, moderate compression, requires modern GPU
- **INT8 affine**: good precision, moderate compression, universal compatibility
- **INT4 group-128**: best compression, lower precision, universal compatibility

### Test Protocol

1. **Model**: RWKV-7 1.5B (small model indicates large model viability)
2. **Hardware**: RTX 5070 Ti (Blackwell, SM 120)
3. **Seed**: fixed for reproducibility
4. **Prompts**: 500 diverse prompts (code, math, prose, dialogue)
5. **Generation**: 128 tokens, greedy decoding
6. **Comparison**: token-by-token agreement with BF16 baseline

## Expected Results Matrix

| Metric | FP8 | INT8 per-tensor | INT8 affine | INT4 per-tensor | INT4 affine | INT4 group-128 |
|--------|-----|-----------------|-------------|-----------------|-------------|----------------|
| EAR | 0.94 | ~0.92 | ~0.95 | ~0.80 | ~0.85 | ~0.88 |
| Top-1 | 92.5% | ~88% | ~93% | ~75% | ~82% | ~86% |
| MATH500 | 53% | ~48% | ~52% | ~35% | ~42% | ~47% |
| File size | 1.72GB | 1.70GB | 1.80GB | 0.85GB | 0.90GB | 0.95GB |
| VRAM | 1.60GB | 1.58GB | 1.68GB | 0.85GB | 0.90GB | 0.95GB |
| Decode | 67.8 t/s | ~70 t/s | ~55 t/s | ~90 t/s | ~75 t/s | ~80 t/s |

> Values with `~` are predictions based on theoretical analysis. Actual results will be measured.

## Key Questions to Answer

1. **Can INT8 affine match FP8 precision?** (Both use 256 levels, but FP8 has hardware tensor cores)
2. **Is INT4 group-128 viable for production?** (16 levels with per-group compensation)
3. **What's the speed/precision trade-off across schemes?**
4. **Does RWKV-7's uniform weight distribution help or hurt INT quantization?**
   - FP8 benefits from symmetry (per-tensor sufficient)
   - INT4 may benefit more from affine (asymmetric compensation)
5. **Which scheme is optimal for each deployment scenario?**
   - High-precision server: FP8 or INT8 affine
   - Memory-constrained edge: INT4 group-wise
   - Legacy GPU (no FP8): INT8 affine
