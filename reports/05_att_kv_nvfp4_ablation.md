# #4 实验报告：L4-27 key/value NVFP4 消融

## 概述

对 RWKV-7 1.5B 模型（24层）的 **注意力层 key/value 投影矩阵** 进行 NVFP4 (E2M1) 量化消融实验。这是在 #2（FFN NVFP4）和 #3（key/value FP8 验证）之后，测试核心争议点：**注意力 key/value 能否使用 NVFP4 量化**。

### 实验方法

采用 **dequant 代理测试**：将权重量化到 NVFP4（含 AWQ 缩放 + clip ratio 搜索），反量化回 BF16 后存储。模型加载时走正常 BF16 GEMM 路径。此方法精确捕获量化误差对质量的影响，实际 NVFP4 推理速度/显存优化留待 #8。

## 实验结果

| 测试 | 方案 | Top1 | PPL delta | VRAM |
|------|------|------|-----------|------|
| Baseline | 全 BF16 | 100.00% | 0.0000 | — |
| #3 | 全 FP8 W8A16 | 98.62% | -0.0004 | 2.69 GiB |
| 4a | 全层 att key+value NVFP4 | 98.62% | +0.0065 | 2.69 GiB |
| 4b | L4-19 key+value NVFP4, 边缘 BF16 | 98.81% | +0.0059 | 2.72 GiB |
| 4c | L4-19 仅 key NVFP4, value BF16 | 99.67% | +0.0020 | 2.72 GiB |
| 4d | L0 BF16, L1-3 FP8, L4-19 NVFP4, L20-23 FP8 | 98.86% | +0.0055 | 2.72 GiB |

## 关键发现

### 1. NVFP4 对注意力 key/value 可接受
全层 NVFP4（4a）Top1=98.62%，与全 FP8（#3）持平。NVFP4 并未导致注意力层质量崩溃。

### 2. value 比 key 对 NVFP4 更敏感（与预期相反）
- 仅 key NVFP4（4c）：Top1=99.67%，仅掉 0.33%
- key+value NVFP4（4b）：Top1=98.81%，掉 1.19%
- **value 贡献了约 0.86% 的 Top1 下降**，是主要掉分来源

README 中 att.key 评级 ★★★★★（最敏感），att.value 评级 ★★★★。但实测 NVFP4 量化下 value 更敏感。原因分析：
- **key 有 L2 归一化保护**：kk = normalize(k * k_k)，归一化使方向对权重扰动鲁棒
- **value 直接进 state**：v ⊗ k_modified 外积写入 state，量化误差无归一化缓冲

### 3. 中间层比边缘层更鲁棒
4b（中间 NVFP4）比 4a（全 NVFP4）好 0.19%，确认 L0 和末层更敏感。

### 4. 最终方案可行
4d（L0 BF16, L1-3 FP8, L4-19 NVFP4, L20-23 FP8）Top1=98.86%，优于全 NVFP4。方案有效。

## 对最终量化方案的修正建议

基于实测数据，建议调整：
- **att.key 可放宽到 NVFP4**（所有中间层），归一化保护使其鲁棒
- **att.value 在关键层保持 FP8**，或使用 v12 双量化（NVFP4+FP8 residual）补偿
- 原方案 L4-27 key/value 同为 NVFP4 → 建议拆分：key NVFP4, value FP8 或 NVFP4+residual

## 下一步

- #5: L0 value BF16 必要性验证（v_first 跨层传播）
- #6: 长序列 state MSE
- 实际 NVFP4 推理引擎适配留待 #8
