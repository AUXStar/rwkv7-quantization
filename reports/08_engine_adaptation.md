# #8 工程报告：推理引擎适配

## 概述

扩展 v3a 推理引擎，支持完整量化方案的推理：
- NVFP4 attention 权重（receptance / output / key L4-19）
- FP8 attention 权重（key L0-3,L20-23 / value 全部）— W8A16
- NVFP4+FP8 residual FFN key 权重 — W4A16
- FP8 FFN value 权重 — W8A8

## 代码改动

### nvfp4_ops.py（3处）

1. `load_nvfp4_weight`：检测并加载 `.res_fp8` + `.res_fp8_scale`，设置 qtype=`nvfp4_res_w4a16`
2. `linear_nvfp4_w4a16`：反量化后加 FP8 residual 补偿
3. `linear_quantized`：dispatch `nvfp4_res_w4a16` → `linear_nvfp4_w4a16`

### rwkv7_fast_v3a.py（3处）

1. import 添加 `dequantize_nvfp4`
2. 权重加载跳过 `.res_fp8` / `.res_fp8_scale` keys
3. `_deq_att_weight` 支持 NVFP4 dict 权重（反量化 + 缓存）

## 验证结果

### 1.5B 完整方案（首次）

| 指标 | 值 |
|------|-----|
| PPL | 1.5111 (orig 1.5061, delta +0.0050) |
| Top-1 | 98.28% |
| CE delta | +0.003323 |
| VRAM | 1.67 GiB |
| Speed (b1tn) | 2542 tok/s |

### 量化统计

- 144 权重：56 FP8 + 64 NVFP4 + 24 NVFP4+res
- 压缩：2.25 GB → 1.07 GB（2.1x）
- VRAM 占用：1.67 GiB（原始 ~3 GiB，节省 ~44%）

### 与分组件实验对比

| 方案 | PPL delta | Top-1 | 说明 |
|------|-----------|-------|------|
| v8 (FFN key only NVFP4) | +0.0044 | 98.81% | FP8 att + BF16 val |
| v12 (FFN key NVFP4+res) | +0.0044 | 99.05% | FP8 att + BF16 val |
| 完整方案 | +0.0050 | 98.28% | NVFP4 att + FP8 val + NVFP4+res FFN |

完整方案 Top-1 比 v12 低 0.77%，主要来自 attention rec/out/key 的 NVFP4 量化。PPL 几乎无损。

## 分析

- PPL delta 0.005 远低于目标 0.05，精度优秀
- Top-1 98.28% 低于 99.5% 目标，因 1.5B 冗余度低 + 完整方案误差累积
- 7.2B 模型冗余度更高，预计 Top-1 会改善
- attention NVFP4 是主要精度损失来源（rec/out + key L4-19）

## 下一步

#9: 7.2B 完整精度/速度/显存对比
