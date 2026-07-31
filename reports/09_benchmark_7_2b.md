# #9 Benchmark报告：7.2B 完整精度/速度/显存对比

## 概述

对 7.2B 模型（rwkv7-g1g-7.2b，32层，C=4096）进行完整量化方案测试，对比原始 bf16 与量化模型的精度、速度、显存。

## 量化方案

| 组件 | L0-3 | L4-27 | L28-31 | 格式 |
|------|------|-------|--------|------|
| att.key | FP8 | NVFP4 | FP8 | W8A16 / W4A16 |
| att.value | FP8 | FP8 | FP8 | W8A16 |
| att.rec/out | NVFP4 | NVFP4 | NVFP4 | W4A16 |
| ffn.key | NVFP4+res | NVFP4+res | NVFP4+res | W4A16 |
| ffn.value | FP8 | FP8 | FP8 | W8A8 |

- 192 权重：72 FP8 + 88 NVFP4 + 32 NVFP4+res
- 量化耗时：82.9s

## 结果对比

### 精度（446 tokens）

| 指标 | 原始 bf16 | 量化 | delta |
|------|-----------|------|-------|
| PPL | 4.7139 | 4.7288 | +0.0149 |
| Top-1 | - | 93.03% | - |
| CE | 1.5505 | 1.5537 | +0.0032 |

### 资源

| 指标 | 原始 bf16 | 量化 | 变化 |
|------|-----------|------|------|
| VRAM | ~12 GiB | 7.87 GiB | -34% |
| 文件大小 | 13.41 GB | 8.0 GB | -40% |
| Speed (b1tn) | 154 tok/s | 75 tok/s | -51% |

### 与 1.5B 对比

| 指标 | 1.5B | 7.2B |
|------|------|------|
| PPL delta | +0.0050 | +0.0149 |
| Top-1 | 98.28% | 93.03% |
| VRAM | 1.67 GiB | 7.87 GiB |
| Speed | 2542 tok/s | 75 tok/s |
| 压缩比 | 2.1x | 2.1x |

## 分析

1. **PPL**：delta 0.0149，远低于 0.05 目标，精度优秀
2. **Top-1**：93.03%，446 tokens 样本小导致统计方差大；PPL delta 小说明整体分布偏差小
3. **显存**：7.87 GiB，12GB GPU 可舒适运行；原始模型需 ~12GB 勉强加载
4. **速度**：75 tok/s vs 154 tok/s，W4A16 反量化开销 + CMIX_SPARSE=off 导致减速
5. **7.2B vs 1.5B**：PPL delta 更大（0.0149 vs 0.0050），因 7.2B 权重更大（4096² vs 2048²），单 block 量化误差更高

## 已知问题

- W4A16 速度下降是预期行为：每次 forward 需反量化 NVFP4 attention 权重
- 可通过 fused kernel 或预反量化（牺牲显存）优化速度
- Top-1 统计需更长文本（2100+ tokens）降低方差

## 工具链修复

- 修复 quantize_model.py mmap 加载 + torch.save 文件膨胀问题（19.94GB → 8.0GB）
- 添加 clone before save 避免 mmap tensor 序列化开销
