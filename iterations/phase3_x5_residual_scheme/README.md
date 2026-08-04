# Phase 3: X5 残差量化方案与验收测试

> **Issues**: #10-#14 | **报告数**: 9 | **状态**: 已完成，X5 方案被淘汰

## 阶段目标

设计并验证 X5 残差量化方案（NVFP4 主路径 + FP8 残差补偿），进行多模型测试和生成质量评测。引入 MATH500 greedy 评测和 Uncheatable Eval 防记忆评测。

## 实验内容

### #10 X5 方案设计与验证
- 设计残差量化架构：主路径 NVFP4 + 残差 FP8 补偿
- 实现 fused_fp8_gemm.py 中的残差 GEMM 内核
- 验证数值正确性：max_diff = 0.0039
- 报告：10_verification_report.md

### #11 融合 GEMM 算子与生成质量
- 开发 linear_fp8_fused / linear_rkv_fused 融合路径
- 生成质量评测：8 种采样方法 x 6 个 prompt
- 报告：11_fused_gemm_ops.md, 11_generation_quality.md

### #11b MATH500 交叉验证
- 引入 MATH500 greedy 评测（替代 PPL 单一指标）
- 发现 PPL 不预测 MATH500：alpha 对 PPL 无影响但对解题 +2.6pp
- 报告：11b_math500_v2_crosscheck.md

### #12 多模型验证
- 1.5B 和 7.2B 双模型测试
- 逐层/逐头敏感度归因分析
- 报告：12_x5_multi_model.md

### #13 新语料生成评测
- Uncheatable Eval：使用新语料防止模型记忆
- 报告：13_novel_generation.md

### #14 最佳实践总结
- 汇总所有发现，给出推荐方案
- 报告：14_best_practice.md, final_report_t1t4.md, final_scheme_m2.md

## 关键发现

1. **PPL 不预测 MATH500**：alpha 参数对 PPL 无影响但对解题 +2.6pp，PPL 单一指标不足
2. **X5 精度略高于全 FP8**（1.5B: 99.05% vs 97.85%），但复杂度和存储不划算
3. **推理轨迹早期分叉**：量化模型第 16 token 即与原模型分歧
4. **残差 per-block FP8 优于 per-tensor**
5. **FP4 残差无效**：FP4 仅 16 级，91.4% 回收率（15.2% 被压为 0），无法补偿主量化误差
6. **FP8 残差回收率 97.7%**：仅 1.1% 被压为 0

## 方案对比数据

### 1.5B 模型

| 方案 | Top-1 | PPL delta | Decode 速度 | VRAM | 文件大小 |
|------|-------|-----------|-------------|------|----------|
| 全 FP8 | 97.85% | -0.08% | 67.8 t/s | 1.60 GB | 1.85 GB |
| X5 (NVFP4+FP8) | 99.05% | +0.52% | 73.9 t/s | 1.53 GB | 1.76 GB |

### 7.2B 模型

| 方案 | Top-1 | PPL delta | Decode 速度 | VRAM | 文件大小 |
|------|-------|-----------|-------------|------|----------|
| 全 FP8 | **93.75%** | +0.24% | **44.9 t/s** | **7.35 GB** | **7.96 GB** |
| X5 (NVFP4+FP8) | 91.02% | +0.24% | 28.7 t/s | 8.54 GB | 8.85 GB |

## 报告列表

| # | 文件 | 内容 |
|---|------|------|
| 10 | 10_verification_report.md | X5 方案验证 |
| 11 | 11_fused_gemm_ops.md | 融合 GEMM 算子 |
| 11 | 11_generation_quality.md | 生成质量评测 |
| 11b | 11b_math500_v2_crosscheck.md | MATH500 交叉验证 |
| 12 | 12_x5_multi_model.md | 多模型验证 |
| 13 | 13_novel_generation.md | 新语料生成评测 |
| 14 | 14_best_practice.md | 最佳实践总结 |
| — | final_report_t1t4.md | 最终报告 (Task 1-4) |
| — | final_scheme_m2.md | M2 方案最终报告 |
