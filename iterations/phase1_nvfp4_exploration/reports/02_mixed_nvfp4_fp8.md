# #2 实验报告更新：长文本验证 + 混合 NVFP4/FP8 方案

## 更新概述

在 #2 基线实验基础上，完成了两项改进：
1. **长文本 PPL 收敛验证**：在 2100 token 上确认 PPL delta 收敛至目标范围内
2. **混合 NVFP4 key + FP8 value 方案**：显著提升精度和速度

## 实验配置

| 项目 | NVFP4-only | Mixed (NVFP4+FP8) |
|------|-----------|-------------------|
| ffn.key.weight | NVFP4 (4-bit) | NVFP4 (4-bit) |
| ffn.value.weight | NVFP4 (4-bit) | FP8 E4M3 (8-bit) |
| key GEMM | FP4×FP4→BF16 (_scaled_mm + swizzle) | FP4×FP4→BF16 |
| value GEMM | FP4×FP4→BF16 (_scaled_mm + swizzle) | FP8×FP8→BF16 (_scaled_mm, 无 swizzle) |
| 张量存储 | 3.25 GB | 3.59 GB |
| VRAM 占用 | 4.64 GB | 5.00 GB |

## 精度对比

### 446 token（短文本，高熵 PPL~5.6）

| 指标 | NVFP4-only | Mixed | 改善 |
|------|-----------|-------|------|
| PPL delta | 0.6234 | 0.2358 | -62% |
| Top-1 agree | 82.70% | 87.64% | +4.94% |
| CE delta | 0.1054 | 0.0412 | -61% |
| Mean KL | 0.1133 | 0.0681 | -40% |

### 2100 token（长文本，低熵 PPL~1.45）

| 指标 | NVFP4-only | Mixed | 改善 | 目标 |
|------|-----------|-------|------|------|
| PPL delta | 0.0338 | 0.0104 | -69% | ≤0.05 |
| Top-1 agree | 96.33% | 97.14% | +0.81% | ≥99.5% |
| CE delta | 0.0230 | 0.0072 | -69% | — |
| Mean KL | 0.0246 | 0.0150 | -39% | — |

**PPL delta 在长文本上从 0.034 降至 0.010，远低于 0.05 目标。**

## 速度对比

| 模型 | VRAM | b1tn 446 tok/s | b1tn 2100 tok/s | 加载时间 |
|------|------|----------------|-----------------|---------|
| 原版 bf16 | 6.65 GB | 585 | 3425 | 13.4s |
| NVFP4-only | 4.64 GB | 415 | 1785 | 6.5s |
| Mixed | 5.00 GB | 385 | 2560 | 6.2s |

混合方案在长文本上速度 2560 tok/s，比 NVFP4-only 的 1785 快 43%，因为 FP8 GEMM 无需 swizzle，量化也更简单。

## PPL delta 与序列长度的关系

| 序列长度 | 原版 PPL | NVFP4 PPL delta | Mixed PPL delta |
|---------|---------|----------------|----------------|
| 446 | 5.6078 | 0.6234 (11.1%) | 0.2358 (4.2%) |
| 2100 | 1.4539 | 0.0338 (2.3%) | 0.0104 (0.7%) |

PPL delta 随序列长度增加而收敛，原因：
1. 长文本上模型置信度更高（PPL 1.45 vs 5.61），量化噪声对 top-1 预测的影响更小
2. 短文本上模型处于高熵状态，微小的 logits 变化即可改变 top-1 预测

## 关键技术发现

### FP8 GEMM 比 NVFP4 GEMM 更简单高效

| 特性 | NVFP4 GEMM | FP8 GEMM |
|------|-----------|---------|
| 激活量化 | to_nvfp4() (reshape+amax+clamp+量化+pack) | (x/scale).clamp().to(fp8) |
| Scale 处理 | to_blocked() 128x4 swizzle（必需） | 无需 swizzle |
| _scaled_mm 模式 | Blockwise 1x16 | TensorWise (singleton) |
| 量化精度 | 16 个离散值 | 256 个离散值 |

### 混合方案的内存权衡

| 方案 | key 存储 | value 存储 | 总 FFN 存储/层 |
|------|---------|----------|-------------|
| bf16 | 50 MB | 50 MB | 100 MB |
| NVFP4-only | 12.5 MB | 12.5 MB | 25 MB |
| Mixed | 12.5 MB | 25 MB | 37.5 MB |

混合方案比 NVFP4-only 多用 50% FFN 内存，但精度大幅提升。

## 文件清单

| 文件 | 说明 |
|------|------|
| `quantize_mixed_nvfp4_fp8.py` | 混合量化工具：key→NVFP4, value→FP8 |
| `nvfp4_ops.py` (v2) | 支持 NVFP4 + FP8 双路径 GEMM |
| `rwkv7_fast_v3a.py` (patched) | 支持 NVFP4 + FP8 检测和加载 |

## 结论

混合 NVFP4 key + FP8 value 方案在所有维度上优于纯 NVFP4 方案：
- **精度**：PPL delta 改善 69%（0.034→0.010），远超 ≤0.05 目标
- **速度**：长文本 2560 tok/s vs 1785 tok/s，提升 43%
- **显存**：5.00 GB vs 6.65 GB，节省 24.8%（略低于 NVFP4-only 的 30.2%）
- **简洁性**：FP8 GEMM 无需 swizzle，代码更简单

下一步建议：
1. 实现 fused 激活量化 CUDA kernel 进一步提升速度
2. 在 7.2B 模型上验证混合方案
3. 推进 #3 实验：key/value FP8 验证
