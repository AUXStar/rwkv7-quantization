# Phase 1: NVFP4 Quantization Exploration and Sensitivity Analysis

[中文](README_zh.md) | **English**

> **Issues**: #1-#6 | **Reports**: 17 | **Status**: Completed, NVFP4 scheme eliminated

## Phase Objectives

Validate the feasibility of NVFP4 (E2M1) quantization on RWKV-7, determine the quantization sensitivity of each component, and establish the quantization toolchain.

## Experiment Contents

### #1 Quantization Scheme Design and Toolchain Setup
- Define 6 quantizable components: att.receptance/key/value/output + ffn.key/value
- Implement unified quantization tool quantize_model.py, supporting scheme rule tables
- Determine non-quantized components: emb, head, LayerNorm, low-rank weights, vector parameters

### #2 FFN NVFP4 Baseline
- Quantize only ffn.key/value, validate toolchain correctness
- Conclusion: FFN has no state, ReLU2 suppresses ~50% of channels, theoretically safest
- Reports: 02_ffn_nvfp4_baseline.md, 02_ffn_nvfp4_w4a4.md, 02_mixed_nvfp4_fp8.md

### #3 Key/Value FP8 Validation (W8A16)
- Cross-validate community conclusion: W8A16 lossless
- Test FP8 quantization of att.key/value
- Reports: 03_att_fp8_w8a8.md, 03_fused_kernel.md, 04_att_fp8.md

### #4 L4-27 Key/Value NVFP4 Ablation
- Core question: is NVFP4 feasible for middle layers
- Conclusion: NVFP4 relative error 8.82%, nearly identical across all components
- Reports: 04_att_kv_nvfp4_w4a4.md, 05_att_kv_nvfp4_ablation.md

### #5 Layer0 Value Quantization and State Propagation Analysis
- Analyze how quantization error propagates through RWKV state
- Impact of Layer0 value cross-layer propagation (v_first)
- Reports: 05_l0_value_w4a4.md, 06_l0_value_bf16.md, 06_state_mse.md

### #6 Long-sequence State MSE Analysis
- Measure state MSE for 1K-32K token sequences
- Generate visualization charts (state_analysis_plots/)
- Reports: 06_long_seq_w4a4.md, 07_long_seq_state.md, 07_quantization_toolchain_v2.md

## Key Findings

1. **NVFP4 quantization error is uniform**: All 6 components have ~8.82% relative error, no significant sensitivity difference
2. **FP8 (W8A16) is lossless**: att.key/value FP8 quantization PPL delta < 0.01%
3. **State MSE grows linearly with sequence length**: NVFP4 state MSE is significant at 8K+ tokens
4. **Layer0 value affects cross-layer propagation**: v_first mechanism causes L0 value error to propagate to all layers

## Report List

| # | File | Content |
|---|------|------|
| 00 | 00_audit_summary.md | Initial audit and scheme planning |
| 01 | 01_design_v2.md | V2 quantization scheme design |
| 02 | 02_ffn_nvfp4_baseline.md | FFN NVFP4 baseline test |
| 02 | 02_ffn_nvfp4_w4a4.md | FFN W4A4 test |
| 02 | 02_mixed_nvfp4_fp8.md | NVFP4+FP8 mixed test |
| 03 | 03_att_fp8_w8a8.md | Attention FP8 W8A8 test |
| 03 | 03_fused_kernel.md | Fused kernel preliminary design |
| 04 | 04_att_fp8.md | Attention FP8 validation |
| 04 | 04_att_kv_nvfp4_w4a4.md | KV NVFP4 W4A4 ablation |
| 05 | 05_att_kv_nvfp4_ablation.md | KV NVFP4 per-layer ablation |
| 05 | 05_l0_value_w4a4.md | L0 Value W4A4 test |
| 06 | 06_l0_value_bf16.md | L0 Value BF16 comparison |
| 06 | 06_state_mse.md | State MSE analysis |
| 06 | 06_long_seq_w4a4.md | Long-sequence W4A4 test |
| 07 | 07_long_seq_state.md | Long-sequence state propagation |
| 07 | 07_quantization_toolchain_v2.md | Toolchain V2 |

## Visualization Resources

The state_analysis_plots/ directory contains:
- state_mse_heatmap.png — per-layer state MSE heatmap
- state_mse_vs_len.png — State MSE vs sequence length
- state_mse_vs_step.png — State MSE vs step count
- state_cosine_vs_step.png — State cosine similarity vs step count
- state_rel_vs_step.png — State relative error vs step count
