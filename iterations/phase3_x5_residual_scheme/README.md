# Phase 3: X5 Residual Quantization Scheme and Acceptance Testing

[中文](README_zh.md) | **English**

> **Issues**: #10-#14 | **Reports**: 9 | **Status**: Completed, X5 scheme retired

## Phase Objectives

Design and validate the X5 residual quantization scheme (NVFP4 main path + FP8 residual compensation), conduct multi-model testing and generation quality evaluation. Introduce MATH500 greedy evaluation and Uncheatable Eval anti-memorization evaluation.

## Experiment Content

### #10 X5 Scheme Design and Verification
- Designed residual quantization architecture: main path NVFP4 + residual FP8 compensation
- Implemented residual GEMM kernel in fused_fp8_gemm.py
- Verified numerical correctness: max_diff = 0.0039
- Report: 10_verification_report.md

### #11 Fused GEMM Operators and Generation Quality
- Developed linear_fp8_fused / linear_rkv_fused fused paths
- Generation quality evaluation: 8 sampling methods x 6 prompts
- Reports: 11_fused_gemm_ops.md, 11_generation_quality.md

### #11b MATH500 Cross-Validation
- Introduced MATH500 greedy evaluation (replacing PPL as the sole metric)
- Discovered PPL does not predict MATH500: alpha has no effect on PPL but +2.6pp on problem-solving
- Report: 11b_math500_v2_crosscheck.md

### #12 Multi-Model Verification
- 1.5B and 7.2B dual-model testing
- Per-layer/per-head sensitivity attribution analysis
- Report: 12_x5_multi_model.md

### #13 Novel Corpus Generation Evaluation
- Uncheatable Eval: using novel corpus to prevent model memorization
- Report: 13_novel_generation.md

### #14 Best Practice Summary
- Consolidated all findings, provided recommended scheme
- Reports: 14_best_practice.md, final_report_t1t4.md, final_scheme_m2.md

## Key Findings

1. **PPL does not predict MATH500**: alpha parameter has no effect on PPL but +2.6pp on problem-solving, PPL as a sole metric is insufficient
2. **X5 accuracy slightly higher than full FP8** (1.5B: 99.05% vs 97.85%), but complexity and storage are not worthwhile
3. **Reasoning trajectory diverges early**: quantized model diverges from original model at the 16th token
4. **Residual per-block FP8 outperforms per-tensor**
5. **FP4 residual ineffective**: FP4 has only 16 levels, 91.4% recovery rate (15.2% compressed to 0), cannot compensate for main quantization error
6. **FP8 residual recovery rate 97.7%**: only 1.1% compressed to 0

## Scheme Comparison Data

### 1.5B Model

| Scheme | Top-1 | PPL delta | Decode Speed | VRAM | File Size |
|------|-------|-----------|-------------|------|----------|
| Full FP8 | 97.85% | -0.08% | 67.8 t/s | 1.60 GB | 1.85 GB |
| X5 (NVFP4+FP8) | 99.05% | +0.52% | 73.9 t/s | 1.53 GB | 1.76 GB |

### 7.2B Model

| Scheme | Top-1 | PPL delta | Decode Speed | VRAM | File Size |
|------|-------|-----------|-------------|------|----------|
| Full FP8 | **93.75%** | +0.24% | **44.9 t/s** | **7.35 GB** | **7.96 GB** |
| X5 (NVFP4+FP8) | 91.02% | +0.24% | 28.7 t/s | 8.54 GB | 8.85 GB |

## Report List

| # | File | Content |
|---|------|------|
| 10 | 10_verification_report.md | X5 scheme verification |
| 11 | 11_fused_gemm_ops.md | Fused GEMM operators |
| 11 | 11_generation_quality.md | Generation quality evaluation |
| 11b | 11b_math500_v2_crosscheck.md | MATH500 cross-validation |
| 12 | 12_x5_multi_model.md | Multi-model verification |
| 13 | 13_novel_generation.md | Novel corpus generation evaluation |
| 14 | 14_best_practice.md | Best practice summary |
| — | final_report_t1t4.md | Final report (Task 1-4) |
| — | final_scheme_m2.md | M2 scheme final report |
