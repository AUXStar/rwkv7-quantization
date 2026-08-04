# Phase 1: NVFP4 量化探索与敏感度分析

> **Issues**: #1-#6 | **报告数**: 17 | **状态**: 已完成，NVFP4 方案被淘汰

## 阶段目标

验证 NVFP4 (E2M1) 量化在 RWKV-7 上的可行性，确定各组件的量化敏感度，建立量化工具链。

## 实验内容

### #1 量化方案设计与工具链搭建
- 定义 6 个可量化组件：att.receptance/key/value/output + ffn.key/value
- 实现统一量化工具 quantize_model.py，支持 scheme 规则表
- 确定不量化组件：emb、head、LayerNorm、低秩权重、向量参数

### #2 FFN NVFP4 基线
- 仅量化 ffn.key/value，验证工具链正确性
- 结论：FFN 无 state，ReLU2 抑制 ~50% 通道，理论最安全
- 报告：02_ffn_nvfp4_baseline.md, 02_ffn_nvfp4_w4a4.md, 02_mixed_nvfp4_fp8.md

### #3 Key/Value FP8 验证 (W8A16)
- 社区结论交叉验证：W8A16 无损
- 测试 att.key/value 的 FP8 量化
- 报告：03_att_fp8_w8a8.md, 03_fused_kernel.md, 04_att_fp8.md

### #4 L4-27 Key/Value NVFP4 消融
- 核心争议点：中间层 NVFP4 是否可行
- 结论：NVFP4 相对误差 8.82%，所有组件几乎相同
- 报告：04_att_kv_nvfp4_w4a4.md, 05_att_kv_nvfp4_ablation.md

### #5 Layer0 Value 量化与 State 传播分析
- 分析量化误差如何通过 RWKV state 传播
- Layer0 value 跨层传播 (v_first) 的影响
- 报告：05_l0_value_w4a4.md, 06_l0_value_bf16.md, 06_state_mse.md

### #6 长序列 State MSE 分析
- 测量 1K-32K token 序列的 state MSE
- 生成可视化图表（state_analysis_plots/）
- 报告：06_long_seq_w4a4.md, 07_long_seq_state.md, 07_quantization_toolchain_v2.md

## 关键发现

1. **NVFP4 量化误差均匀**：所有 6 个组件的相对误差均为 ~8.82%，无显著敏感度差异
2. **FP8 (W8A16) 无损**：att.key/value 的 FP8 量化 PPL delta < 0.01%
3. **State MSE 随序列长度线性增长**：NVFP4 在 8K+ token 时 state MSE 显著
4. **Layer0 value 影响跨层传播**：v_first 机制使 L0 value 误差传播到所有层

## 报告列表

| # | 文件 | 内容 |
|---|------|------|
| 00 | 00_audit_summary.md | 初始审计与方案规划 |
| 01 | 01_design_v2.md | V2 量化方案设计 |
| 02 | 02_ffn_nvfp4_baseline.md | FFN NVFP4 基线测试 |
| 02 | 02_ffn_nvfp4_w4a4.md | FFN W4A4 测试 |
| 02 | 02_mixed_nvfp4_fp8.md | NVFP4+FP8 混合测试 |
| 03 | 03_att_fp8_w8a8.md | Attention FP8 W8A8 测试 |
| 03 | 03_fused_kernel.md | 融合内核初步设计 |
| 04 | 04_att_fp8.md | Attention FP8 验证 |
| 04 | 04_att_kv_nvfp4_w4a4.md | KV NVFP4 W4A4 消融 |
| 05 | 05_att_kv_nvfp4_ablation.md | KV NVFP4 逐层消融 |
| 05 | 05_l0_value_w4a4.md | L0 Value W4A4 测试 |
| 06 | 06_l0_value_bf16.md | L0 Value BF16 对比 |
| 06 | 06_state_mse.md | State MSE 分析 |
| 06 | 06_long_seq_w4a4.md | 长序列 W4A4 测试 |
| 07 | 07_long_seq_state.md | 长序列 State 传播 |
| 07 | 07_quantization_toolchain_v2.md | 工具链 V2 |

## 可视化资源

state_analysis_plots/ 目录包含：
- state_mse_heatmap.png — 各层 state MSE 热力图
- state_mse_vs_len.png — State MSE vs 序列长度
- state_mse_vs_step.png — State MSE vs 步数
- state_cosine_vs_step.png — State 余弦相似度 vs 步数
- state_rel_vs_step.png — State 相对误差 vs 步数
