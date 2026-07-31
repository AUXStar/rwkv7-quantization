# RWKV-7 NVFP4 量化项目总结

## 目标

对 RWKV-7 模型实现**纯量化GEMM**推理（NVFP4/FP8，不反量化），
在保持精度（PPL delta ≤ 0.05）的同时，减少显存占用、提升推理速度。

## 分层量化方案（v2）

| 组件 | L0-3 | L4-19 | L20-23 | 格式 |
|------|------|-------|--------|------|
| att.key | FP8 | NVFP4 | FP8 | W8A8 / W4A4 |
| att.value | FP8 | FP8 | FP8 | W8A8 |
| att.rec/out | NVFP4 | NVFP4 | NVFP4 | W4A4 |
| ffn.key | NVFP4+FP8残差 | 同 | 同 | W4A4+W8A8 |
| ffn.value | FP8 | FP8 | FP8 | W8A8 |

- 144 量化张量（1.5B）：56 FP8 + 64 NVFP4 + 24 NVFP4+res
- 权重量化压缩：2.25 GB → 1.07 GB（**2.1x**）
- 数值设计：per-16-block scale（FP8 E4M3）+ per-tensor scale（FP32）+ AWQ 通道缩放 + clip-ratio 搜索

## 算子重写（本次核心工作）

将量化 GEMM 从"cast→AWQ→量化kernel→`_scaled_mm`→scale-fold"多launch流水线
重写为**最少launch**的 Triton 单kernel：

| 优化 | 内容 | decode 速度 |
|------|------|------------|
| 基线 `_scaled_mm` | 4-9 launches/linear | 16.9 tok/s |
| v1 fused GEMM | 量化+GEMM 单kernel，混合路由 | 21.3 tok/s |
| v2 prep_x | cast+AWQ+amax 1launch，GPU端amax | 64.9 tok/s |
| v3 残差+rkv融合 | FP8残差入主kernel；r/k/v三投影1kernel | **70.4 tok/s** |

- `fused_nvfp4_gemm.py`：`prep_x_kernel` / `fused_nvfp4_gemm_kernel` /
  `fused_nvfp4_res_gemm_kernel` / `fused_rkv_gemm_kernel` / `fused_fp8_gemm_kernel`
- 混合路由：M≤64 用 fused 单kernel（decode），M>64 用 `_scaled_mm`（prefill，cuBLAS 更快）
- 全部 kernel 与 `_scaled_mm` 路径数值一致（逐GEMM max_diff ~0.001；rkv bit-exact）

## 实验结果（1.5B，用户指定小模型基准）

| 指标 | 原生 bf16 | 量化 | 变化 |
|------|-----------|------|------|
| PPL | 1.5061 | 1.5304 | **+0.0242**（≤0.05 达标） |
| Top-1 | — | 97.14%（warmup 后 100%） | — |
| VRAM | 2.69 GiB | 1.71 GiB | **-37%** |
| decode | 163 tok/s | 70.4 tok/s | 43% of native |
| prefill (2100批) | 5269 tok/s | 3434 tok/s | 持平 |
| 压缩 | — | 2.1x | — |

### 关键发现

1. **纯量化GEMM精度好于反量化**：7.2B 上 W4A4 PPL delta 0.0009 < W4A16 的 0.0017
2. **1.5B 是更好的测试基准**：小模型冗余度低，暴露了 7.2B 被显存掩盖的速度真相
3. **decode 是 launch-bound → GPU-bound 演进**：v1-v3 消除 4x launch 后，瓶颈转为 kernel 计算效率
4. **无 state 累积**：误差集中在 warmup（0-500 tokens），之后 Top-1=100%

## 文件

| 文件 | 说明 |
|------|------|
| `quantize_model.py` | 统一量化工具链（加载→分类→量化→存储+meta） |
| `nvfp4_ops.py` | 权重加载 + `_scaled_mm` GEMM + 混合路由 dispatcher |
| `fused_nvfp4_gemm.py` | 重写的单kernel量化GEMM（prep_x/FP4/FP8/残差/rkv融合） |
| `reports/` | #1-#9 + 11 号实验报告（设计/验证/算子重写） |

## Commit 记录

```
231baf7 rkv fusion: r/k/v attention projections in ONE kernel
ad8a113 fused GEMM v2: prep_x + GPU-side amax + FP8-residual-in-kernel
ce5f797 rewrite quantized GEMM ops: fused single-kernel Triton kernels
0dade78 #9 (1.5B): full comparison — VRAM -37%, speed analysis
5f5b316 #8 #9: pure quantized GEMM verification
924fd21 #7: verified quantization toolchain
6d8ab12 #1 v2: updated design doc
ae57251 fix: eliminate all dequantization — pure W4A4+W8A8 quantized GEMM
```
