# #2 W4A4: FFN-only NVFP4 基线（纯量化GEMM）

## 概述

仅将1.5B模型的FFN key/value量化为NVFP4，其余BF16。
使用W4A4纯量化GEMM（FP4×FP4 `_scaled_mm`），不反量化。

## 配置

- 模型：rwkv7-g1h-1.5b（24层）
- 量化范围：48个FFN张量（ffn.key + ffn.value）
- GEMM：`linear_nvfp4`（FP4×FP4→BF16）
- 激活：在线量化到FP4（fused Triton kernel）

## 结果（2100 tokens）

| 指标 | 值 |
|------|-----|
| PPL | 1.5446 (orig 1.5061, delta +0.0385) |
| Top-1 | 96.09% (early 84%, late 100%) |
| VRAM | 1.61 GiB |
| Speed | 2426 tok/s |
| 压缩 | 2.25→0.99 GB (2.3x) |

## 与W4A16对比

| 指标 | W4A16（旧） | W4A4（新） |
|------|------------|-----------|
| PPL delta | +0.0050 | +0.0385 |
| Top-1 | 98.28% | 96.09% |
| Speed | 2542 tok/s | 2426 tok/s |

W4A4精度下降明显（PPL delta 7.7x），主因是activation在线量化到FP4引入误差。但后1600 tokens仍100% Top-1。

## 分析

- FFN value的W4A4误差较大（ReLU²后激活分布对FP4不友好）
- 可考虑FFN value用FP8 W8A8替代NVFP4 W4A4
- ffn.key可用NVFP4+FP8 residual补偿误差
