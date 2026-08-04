# Phase 4: FP8 最终方案与算子优化

[English](README.md) | **中文**

> **Issues**: #15-#16 | **报告数**: 3 | **状态**: 已完成，全 FP8 为最终方案

## 阶段目标

确定全 FP8 为最终量化方案，优化算子性能，完成 7.2B 模型的完整评测。清理所有 NVFP4 相关代码，文件重命名。

## 实验内容

### #15 算子优化与性能调优
- 7.2B 模型性能瓶颈分析：95% GPU bound，ffn_key_res 占 44.6% GPU 时间
- Shape-aware tile 配置优化：
  - att (4096x4096) -> (16,64,64,4)：1.84x vs baseline
  - ffn_key (16384x4096) -> (16,64,128,4)：37% 加速
  - ffn_val (4096x16384) -> (16,128,256,8)：29% 加速
- Split-K 并行无效（atomic add 开销 > 收益）
- 报告：15_opt_7b_ops.md

### #16 FP8 最终方案确定
- 全 FP8 vs X5 对比：PPL delta 改善 53%，Top-1 +1.29%，VRAM -18.8%
- 代码清理：删除所有 NVFP4 相关代码
- 文件重命名：nvfp4_ops.py -> fp8_ops.py，fused_nvfp4_gemm.py -> fused_fp8_gemm.py
- 报告：16_v2_fp8_scheme.md, 16_perf_tuning_report.md

## 关键成果

1. **全 FP8 碾压 X5**：
   - PPL delta 改善 53%
   - Top-1 一致性 +1.29pp (93.75% vs 91.02%)
   - VRAM -18.8% (7.35 GB vs 8.54 GB)
   - 速度 1.56x (44.9 vs 28.7 t/s)

2. **Shape-aware tile 配置**：

   | 矩阵形状 | 场景 | Tile (M,N,K,W) | 优化效果 |
   |----------|------|-----------------|----------|
   | 4096x4096 | att (decode) | (16,64,64,4) | 1.84x |
   | 16384x4096 | ffn_key (decode) | (16,64,128,4) | +37% |
   | 4096x16384 | ffn_value (decode) | (16,128,256,8) | +29% |

3. **Decode 速度 44.9 t/s**（7.2B，6.4x 提升），18.1ms/token，74% 带宽利用率

4. **代码清理**：
   - 删除所有 NVFP4 量化/反量化/GEMM 代码
   - nvfp4_ops.py -> fp8_ops.py
   - fused_nvfp4_gemm.py -> fused_fp8_gemm.py
   - quantize_model.py 仅保留 FP8 scheme

5. **NVFP4+NVFP4 残差方案 (V3) 验证**：
   - Top-1 94.92%（与纯 NVFP4 相同）
   - 数据量 1.13 B/elem（高于 FP8 的 1.0 B/elem）
   - FP4 残差的 16 离散值无法有效补偿主 NVFP4 量化误差

## 最终性能数据

### 7.2B 模型 (RTX 5070 Ti, Blackwell)

| 指标 | 原始 BF16 | FP8 量化 | 变化 |
|------|----------|----------|------|
| Decode 速度 | 7.0 t/s | **44.9 t/s** | 6.4x |
| Prefill 速度 (1x128) | — | 1603 t/s | — |
| VRAM | 13.32 GB | **7.35 GB** | -45% |
| 文件大小 | 14.40 GB | **7.96 GB** | -45% |
| Top-1 一致性 | 100% | **93.75%** | -6.25% |
| PPL delta (2048) | — | +0.24% | — |
| MATH500 | ~55% | **53%** | -2pp |
| GSM8K | ~85% | **83%** | -2pp |

### 并发压测 (64 并发)

| 指标 | 值 |
|------|-----|
| 总吞吐 | 473.2 tok/s |
| p50 延迟 | 51.0s |
| p90 延迟 | 67.5s |
| 错误率 | 0/64 |

## 报告列表

| # | 文件 | 内容 |
|---|------|------|
| 15 | 15_opt_7b_ops.md | 7.2B 算子优化 |
| 16 | 16_v2_fp8_scheme.md | FP8 最终方案确定 |
| 16 | 16_perf_tuning_report.md | 性能调优报告 |
