# 算子重写报告：Fused量化GEMM Kernel（W4A4/W8A8单kernel）

## 概述

针对1.5B上纯量化GEMM速度问题（0.67x vs 原生fp16），重写了量化GEMM算子。
将原来的"cast→AWQ→量化kernel→_scaled_mm→scale-fold"多launch流水线
重写为**最少launch**的Triton单kernel，并持续优化三版。

## 演进

| 版本 | 优化内容 | decode 速度 | 每linear launch数 |
|------|---------|------------|------------------|
| 基线 | `_scaled_mm`（量化+GEMM分离） | 16.9 tok/s | 4-9 |
| v1 | fused GEMM 单kernel + 混合路由 | 21.3 tok/s | 2-3 |
| v2 | prep_x（cast+AWQ+amax 1launch）+ GPU端amax | 64.9 tok/s | 2 |
| **v3** | **FP8残差融合入主kernel + r/k/v三投影融合** | **70.4 tok/s** | **~1.5** |

## 实现（fused_nvfp4_gemm.py）

### prep_x_kernel（v2）
cast bf16 + AWQ除法 + amax（atomic max）融合为**1次launch**。
amax留在GPU端（GEMM kernel内 `tl.load(amax_ptr)` 计算 pts=amax/2688），**无 D2H sync**。

### fused_nvfp4_gemm_kernel（v1）
x(bf16) → [寄存器内] FP4量化 + FP4×FP4 dot，权重保持packed FP4存储（4x带宽收益）。
block scale = clamp(max_abs*448/amax, 0.015625, 448) → fp8，与_scaled_mm数值一致。

### fused_nvfp4_res_gemm_kernel（v3）
NVFP4主GEMM + FP8残差GEMM合并为**单kernel**：同一x_tile分别做FP4量化（per-block）
和FP8量化（per-tensor），两个accumulator，最后合并。残差路径从5+ launch降到0。

### fused_rkv_gemm_kernel（v3）
attention的r(NVFP4)/k(NVFP4|FP8)/v(FP8)三个投影合并为**单GEMM kernel** +
prep3_x（xr/xk/xv三输入1launch）。每层attention从6 launch降到2。
**与单独调用bit-exact**（max_diff=0）。

### 混合路由
- M ≤ 64（decode/小批量）：fused 单kernel（launch少）
- M > 64（prefill）：`_scaled_mm`（cuBLAS FP4 kernel 大M效率更高）

## Kernel级速度（M=1, K=2048）

| 场景 | `_scaled_mm` | fused | 加速 |
|------|-------------|-------|------|
| decode N=2048 | 643us | **39us**（调参后） | 16x |
| decode N=8192 (res) | ~1200us（两路） | **118us** | 10x |

## 引擎级结果（1.5B, 2099 tokens）

| 指标 | 原生fp16 | `_scaled_mm` | **fused v3** |
|------|---------|------------|--------------|
| decode | 163 tok/s | 16.9 | **70.4（43%）** |
| prefill | 5269 | 3518 | **3434（持平）** |
| PPL delta | — | +0.0242 | **+0.0242**（不变） |
| Top-1 | — | 97.14% | 97.14% |
| VRAM | 2.69G | 1.67G | 1.71G |

## 性能瓶颈演进（profile）

| 阶段 | 瓶颈 |
|------|------|
| v1前 | **CPU launch-bound**：~1000 launches/step，CPU 35ms vs GPU 11ms |
| v2 | launch减至~400，GPU计算浮现 |
| v3 | **GPU计算bound**：量化GEMM占 GPU 81%（rkv 25% + res 27% + fp8 20% + nvfp4 9%） |

v3后瓶颈是kernel的计算效率（M=1时program利用率低 + FP4/E4M3解码ALU开销），
不是launch。BLOCK配置tune确认当前 (16,64,64,4) 已最优。

## 结论

1. **算子重写有效**：decode 16.9→70.4 tok/s（4.2x），达原生fp16的43%
2. **精度不变**：全部kernel与_scaled_mm数值一致（rkv bit-exact）
3. **权重不反量化**：packed FP4权重直接在kernel内解码参与计算
4. **prefill不受影响**：混合路由保证大M仍走cuBLAS最优路径

## 下一步（收益递减）

1. FP4/E4M3解码查表优化（exp2→查表，减少ALU开销）
2. decode多请求批处理（B>1提升program利用率）
3. head投影FP8量化（当前占decode GPU 7%）
