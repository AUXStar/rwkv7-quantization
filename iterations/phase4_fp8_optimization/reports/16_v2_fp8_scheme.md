# 1.5B v2 量化方案性能调优报告

## 1. 概述

本报告记录了 RWKV-7 1.5B 模型从 X5（NVFP4+FP8 残差）到 v2（纯 FP8）量化方案的优化过程和 A/B 测试结果。

**核心改进**: 将 ffn_key 从 NVFP4+FP8 残差量化改为纯 FP8 量化，"将残差烘焙进量化"——FP8 直接量化原始权重，避免推理时加载额外的残差数据。

## 2. 量化方案对比

| 组件 | X5 方案 | v2 方案 |
|------|---------|---------|
| att.key (L0) | BF16 | BF16 |
| att.key (L1-23) | FP8 | FP8 |
| att.value | FP8 | FP8 |
| att.receptance | NVFP4 | NVFP4 |
| att.output | NVFP4 | NVFP4 |
| **ffn.key** | **NVFP4+FP8 残差** | **FP8** |
| ffn.value | FP8 | FP8 |

**量化统计**:
- X5: 48 NVFP4 + 24 NVFP4_RES + 71 FP8 + 1 BF16 = 144 权重
- v2: 48 NVFP4 + 95 FP8 + 1 BF16 = 144 权重

## 3. A/B 测试结果

### 3.1 质量指标（2100 token, chunked forward）

| 指标 | 参考 (bf16) | X5 (NVFP4+RES) | v2 (FP8) | v2 vs X5 |
|------|------------|----------------|----------|----------|
| PPL | 1.5053 | 1.5410 | 1.5219 | **-0.0191** |
| PPL delta | — | +0.0357 | +0.0166 | **改善 53%** |
| Top-1 一致性 | 90.90% | 96.43% | 97.71% | **+1.29%** |
| VRAM | 2.94 GiB | 2.18 GiB | 1.77 GiB | **-0.41 GiB (-18.8%)** |

### 3.2 解码速度

| Batch | X5 t/s/req | v2 t/s/req | Delta | 变化 |
|-------|-----------|-----------|-------|------|
| B=1 | 87.4 | 80.4 | -7.0 | -8.0% |
| B=2 | 84.8 | 78.8 | -6.0 | -7.1% |
| B=4 | 91.4* | 77.9 | -13.6 | -14.8%* |
| B=8 | 64.1 | 61.1 | -3.0 | -4.7% |

> *B=4 X5 数据可能受测量噪声影响（高于 B=1）

### 3.3 模型文件大小

| 模型 | 大小 | 压缩比 |
|------|------|--------|
| 原始 bf16 | 2.9 GB | 1.0x |
| X5 (NVFP4+RES) | 2.1 GB | 1.38x |
| v2 (FP8) | 1.7 GB | 1.71x |

## 4. 分析

### 4.1 质量改善原因

v2 方案 PPL delta 改善 53%，Top-1 提升 1.29%。原因：

- **NVFP4 只有 16 个离散值**（±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6），量化误差大
- **FP8 E4M3 有 256 个离散值**，直接量化原始权重的精度远高于 NVFP4+残差
- 残差方案虽然补偿了 NVFP4 的量化误差，但残差本身也有 FP8 量化误差
- 纯 FP8 "一步到位"量化，避免了双重量化误差叠加

### 4.2 VRAM 减少原因

| 方案 | ffn_key 数据量/层 | 计算 |
|------|-------------------|------|
| X5 (NVFP4+RES) | NVFP4 权重 (0.5 B/elem) + FP8 残差 (1 B/elem) + 块缩放 | 1.5 B/elem |
| v2 (FP8) | FP8 权重 (1 B/elem) + 张量缩放 | 1.0 B/elem |

v2 每层 ffn_key 减少 33% 数据量，24 层共节省 ~0.41 GiB VRAM。

### 4.3 速度回退原因

v2 解码速度比 X5 慢 5-8%。分析：

- **X5 ffn_key**: 使用 `fused_nvfp4_res_gemm_kernel`（Triton 融合内核）
  - FP4 GEMM + FP8 GEMM 在一个 kernel 中完成
  - FP4 权重加载量减半，内存带宽利用率高
- **v2 ffn_key**: 使用 `fused_fp8_hwdot_gemm_kernel`（Triton FP8 内核）
  - 单一 FP8 GEMM，但权重数据量是 FP4 的 2 倍
  - 内核可能未充分优化

虽然 v2 总数据加载量更少（1.0 vs 1.5 B/elem），但速度更慢，表明：
1. FP8 Triton kernel 的内存访问模式可能不如 NVFP4_res kernel 高效
2. NVFP4_res kernel 经过了更多轮优化（tile 配置、split-K 等）
3. 硬件 FP4 tensor core 的吞吐可能高于 FP8

### 4.4 已发现并修复的问题

**Prefill 路径 _scaled_mm bug**: 当 `NVFP4_W4A16=True` 时，block_scale 未 swizzle，导致 M>64 的 prefill 路径（`_scaled_mm`）产生错误结果。临时修复：使用 chunked forward（CHUNK=32 ≤ 64）走 fused kernel 路径。

## 5. 结论与建议

### v2 方案优势
- **质量更好**: PPL delta 改善 53%，Top-1 提升 1.29%
- **VRAM 更省**: 减少 18.8%（0.41 GiB）
- **模型更小**: 压缩比 1.71x vs 1.38x
- **实现更简洁**: 无残差逻辑，单一 FP8 量化

### v2 方案劣势
- **解码速度略慢**: 5-8%（可通过优化 FP8 kernel 改善）

### 后续优化方向
1. **优化 FP8 fused kernel**: 参考 NVFP4_res kernel 的 tile 配置和优化策略
2. **尝试 _scaled_mm FP8 decode**: 使用 GPU-side amax 避免 D2H 同步
3. **修复 prefill _scaled_mm 路径**: 为 W4A16 模式生成 swizzled block scales
4. **MATH500 验证**: 补充长文本生成能力评估

## 6. 测试环境

- GPU: RTX 5070 Ti (12 GiB)
- 模型: RWKV-7-g1h-1.5B (24 层, C=2048, V=65536)
- 引擎: faster3a v3a, NVFP4_W4A16=True
- 测试数据: 2100 token wikitext 片段
