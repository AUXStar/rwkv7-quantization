# Phase 2: 推理引擎适配与融合内核开发

> **Issues**: #7-#9 | **报告数**: 5 | **状态**: 已完成

## 阶段目标

将量化模型集成到 Albatross 推理引擎 (faster3a_2605)，开发融合 Triton 内核，完成 1.5B/7.2B 基准测试。

## 实验内容

### #7 引擎适配与权重加载
- 修改 rwkv7_fast_v3a.py 支持量化权重自动检测
- 实现 is_fp8_weight() / load_fp8_weight() 接口
- 处理 .fp8_scale 键的加载与清理
- 报告：08_engine_adaptation.md, 08_engine_pure_gemm.md, 08_quantization_toolchain.md

### #8 融合内核开发
- prep_x 融合：输入 cast + AWQ + amax 在一个 kernel launch 完成
- fused_fp8_hwdot_gemm_kernel：FP8 硬件 tensor core dot (tl.dot(fp8, fp8))
- fused_rkv_fp8_kernel：r/k/v 三个 attention 投影在一个 kernel 中完成
- Shape-aware tile：针对不同矩阵形状自动选择最优 BLOCK 配置

### #9 基准测试
- 1.5B 模型完整基准：速度、VRAM、精度
- 7.2B 模型补充基准
- 报告：09_benchmark_1_5b_pure_gemm.md, 09b_benchmark_7_2b_supplement.md

## 关键成果

1. **融合内核 1.84x 加速**：prep_x + FP8 硬件 dot + RKV 融合
2. **Shape-aware tile 配置**：
   - att (4096x4096) -> BLOCK=(16,64,64), GROUP=4
   - ffn_key (16384x4096) -> BLOCK=(16,64,128), GROUP=4
   - ffn_val (4096x16384) -> BLOCK=(16,128,256), GROUP=8
3. **CUDA Graph 不适用**：decode 步骤 96 kernel replay 产生 ~1ms 额外开销 > launch 节省
4. **Dense 路径强制**：量化模型 CMIX_SPARSE=off，FFN 稀疏路径不兼容（0% block sparsity）

## 报告列表

| # | 文件 | 内容 |
|---|------|------|
| 08 | 08_engine_adaptation.md | 引擎适配方案 |
| 08 | 08_engine_pure_gemm.md | 纯 GEMM 路径实现 |
| 08 | 08_quantization_toolchain.md | 量化工具链整合 |
| 09 | 09_benchmark_1_5b_pure_gemm.md | 1.5B 基准测试 |
| 09b | 09b_benchmark_7_2b_supplement.md | 7.2B 补充基准 |

## 性能数据

### 1.5B 模型 (RTX 5070 Ti)

| 指标 | 原始 BF16 | FP8 量化 |
|------|----------|----------|
| Decode 速度 | 164.1 t/s | 67.8 t/s |
| VRAM | 2.69 GB | 1.60 GB |
| Top-1 一致性 | 100% | 97.85% |

### 7.2B 模型

| 指标 | 原始 BF16 | FP8 量化 |
|------|----------|----------|
| Decode 速度 | 7.0 t/s | 44.9 t/s (6.4x) |
| VRAM | 13.32 GB | 7.35 GB |
| Top-1 一致性 | 100% | 93.75% |
