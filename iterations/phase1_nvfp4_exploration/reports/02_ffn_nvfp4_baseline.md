# #2 实验报告：2.9B FFN-only NVFP4 基线

## 实验概述

对 RWKV-7 2.9B 模型的 FFN key/value 权重进行 NVFP4 (E2M1) 量化，验证工具链可用性和 FFN 量化对精度的影响。

## 实验配置

| 项目 | 值 |
|------|-----|
| 模型 | rwkv7-g1h_preview4673-2.9b-20260701-ctx8192 (2.9B, bf16, C=2560, L=32) |
| 量化范围 | `blocks.*.ffn.key.weight` [10240,2560] + `ffn.value.weight` [2560,10240] |
| 量化张量数 | 64 (32层 × 2) |
| 量化格式 | NVFP4 E2M1, block_size=16, E4M3 block scale + FP32 tensor scale |
| GEMM 路径 | `torch._scaled_mm` (FP4×FP4→BF16), 激活在线量化 |
| GPU | RTX 5070 Ti Laptop (sm_120, 12GB VRAM) |
| CUDA | 13.0, PyTorch 2.13.0+cu130 |

## 关键技术决策

### 1. cuBLASLt 不可用 → `torch._scaled_mm` 可用

cuBLASLt 13.3 的块缩放不支持混合精度（FP16×FP4），返回 `CUBLAS_STATUS_INVALID_VALUE`。
`torch._scaled_mm` 支持 FP4×FP4→BF16，需要激活在线量化为 FP4 后才能参与 GEMM。

### 2. `to_blocked` swizzle 是必需的

`torch._scaled_mm` 的 Blockwise 1x16 缩放要求 scale 张量经过 128x4 块重排（swizzle），
不是简单的 1D flatten。`to_blocked()` 函数将 (H, W) scale 张量转换为 cuBLAS 期望的 1D flat 布局。

### 3. per-tensor scale 折叠

`_scaled_mm` 不直接支持 per-tensor scale 参数。解决方案：
- 量化时将 per-tensor scale 折叠到 block scale 中
- GEMM 输出后乘以 `A_ts * B_ts` 恢复正确 scale

## 实验结果

### 精度对比 (b1tn, 446 token 实文)

| 指标 | 原版 | NVFP4 | Delta | 目标 | 状态 |
|------|------|-------|-------|------|------|
| mean_loss (CE) | 0.4824 | 0.5177 | +0.0353 | — | — |
| PPL | 1.6199 | 1.6782 | +0.0583 | ≤0.05 | 接近 |
| Top-1 agree | — | — | 96.19% | ≥99.5% | 低于 |
| Top-5 agree | — | — | 85.83% | — | — |
| KL divergence | — | — | 0.025 (mean) | — | — |
| Max abs diff | — | — | 12.03 | — | — |

### 显存对比

| 模型 | VRAM 使用 | 分配 | 节省 |
|------|----------|------|------|
| 原版 bf16 | 6.65 GB | 5.37 GB | — |
| NVFP4 FFN-only | 4.64 GB | 3.13 GB | 2.01 GB (30.2%) |

### 速度对比

| 路径 | 原版 tok/s | NVFP4 tok/s | 减速 |
|------|-----------|-------------|------|
| b1tn (T=446) | 983 | 433 | 56% |
| b1tn (T=89) | 251 | — | — |
| b1t1 (decode) | 103 | 14 | 86% |

### 离散化误差分析（单层 GEMM）

| 测试 | Max diff | Mean diff | Mean rel diff |
|------|----------|-----------|---------------|
| FFN key (随机数据) | 0.86 | 0.16 | 101% |
| FFN key (真实权重) | 0.081 | 0.014 | 94% |
| FFN value (真实权重) | 0.207 | 0.026 | — |

## 分析与讨论

### 精度未达标的原因

1. **激活在线量化误差**：FP4 只有 16 个离散值，激活量化引入的误差是主要误差源。
   FFN 的 ReLU² 激活会放大某些通道的误差。

2. **文本重复效应**：测试文本是同一段落重复5次，模型在后半段处于低熵状态（高置信度），
   此时小的量化误差更容易改变 top-1 预测。

3. **PPL delta 接近目标**：0.058 vs 目标 0.05，差距仅 0.008。考虑到测试文本较短（446 token），
   更长文本上 delta 可能收敛。

### 速度瓶颈分析

1. **激活量化开销**：`to_nvfp4()` 是纯 PyTorch 实现，包含 reshape/amax/clamp/量化等操作，
   未使用 fused kernel。

2. **Scale swizzle 开销**：`to_blocked()` 包含多次 reshape/permute，每次 FFN 调用都需执行。

3. **b1t1 极端减速**：decode 路径 M=1，每 token 需 64 次 `linear_nvfp4` 调用（32层×2），
   每次调用的 Python 开销 + 量化开销 dominates。

### 改进方向

1. **精度改进**：
   - 考虑对 FFN value 保持 FP8 而非 FP4（value 接近输出，误差影响更大）
   - 尝试更大的 per-tensor scale 精度
   - 对激活使用更精细的量化方案（如 per-channel scale）

2. **速度改进**：
   - 实现 fused CUDA kernel：激活量化 + scale swizzle + GEMM 一步完成
   - 缓存激活的 block scale swizzle 结果（当输入 shape 不变时）
   - 对 b1t1 路径使用 CUDA Graph 消除 Python 开销

## 文件清单

| 文件 | 说明 |
|------|------|
| `quantize_ffn_nvfp4.py` | 量化工具：加载2.9B→量化FFN→存.pth+meta |
| `nvfp4_ops.py` | NVFP4 GEMM 算子：加载+swizzle+在线量化+_scaled_mm |
| `rwkv7_fast_v3a.py` (patched) | v3a 推理引擎：NVFP4检测+FFN路径替换 |

## 结论

NVFP4 FFN-only 量化工具链已跑通，端到端推理正常。显存节省 30.2%（2.01 GB）符合预期。
精度方面 PPL delta 0.058 略超目标 0.05，top-1 agreement 96.19% 低于目标 99.5%，
主要误差来自激活在线量化。速度方面 b1tn 减速 56%，b1t1 减速 86%，需 fused kernel 优化。

下一步建议：
1. 实现 fused 激活量化 CUDA kernel 提升速度
2. 在更长文本（2048+ tokens）上验证 PPL delta 收敛性
3. 尝试 FFN key 用 NVFP4 + FFN value 用 FP8 的混合方案
