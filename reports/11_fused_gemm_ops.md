# 算子重写报告：Fused量化GEMM Kernel（W4A4/W8A8单kernel）

## 概述

针对1.5B上纯量化GEMM速度问题（0.67x vs 原生fp16），重写了量化GEMM算子。
将原来的"cast→AWQ→量化kernel→_scaled_mm→scale-fold"多launch流水线
融合为**单次Triton kernel**，消除中间张量往返。

## 原路径（多launch，每个量化linear ~4-6次kernel）

```
x(fp16) → cast bf16 → AWQ除法 → fused_nvfp4_quant(量化+pack+swizzle)
        → 写packed A + swizzled scale到显存
        → _scaled_mm(重新读packed A)
        → 乘tensor scale
```

## 新路径（fused单kernel，0中间张量）

```
x(bf16) ──► fused_nvfp4_gemm_kernel（单次launch）
              ├─ 寄存器内: 每16元素block scale计算
              ├─ 寄存器内: FP4 RNE量化（查表）
              ├─ 寄存器内: FP4×FP4 dot（权重保持packed FP4从显存读，4x带宽收益）
              └─ 寄存器内: tensor scale折叠 → 写fp16输出
```

## 实现文件

`fused_nvfp4_gemm.py`（Triton）：
- `fused_nvfp4_gemm_kernel`：W4A4单kernel（激活量化+FP4×FP4 GEMM）
- `fused_fp8_gemm_kernel`：W8A8单kernel（FP8激活量化+FP8×FP8 GEMM）
- `linear_nvfp4_fused` / `linear_fp8_fused`：host wrapper（含AWQ、amax、残差支持）
- M自适应launch配置：M≤4用(16,64,64,4)，否则(64,64,64,4)

数值设计与_scaled_mm路径完全一致：
- block scale = clamp(max_abs/6 × inv_pts, 0.015625, 448) → fp8
- FP4 RNE查表（7级where，与fused_nvfp4_quant相同）
- 权重从packed E2M1/E4M3在kernel内解码（不反量化存储）

## 混合路由（decode/prefill分治）

`nvfp4_ops.linear_quantized_fused`（引擎FUSED_GEMM=True时调用）：

| 场景 | M | 路由 | 原因 |
|------|-----|------|------|
| decode（逐token生成） | 1 | **fused单kernel** | 消除launch开销，3.1x快于_scaled_mm |
| prefill（批量处理） | ≥2100 | **_scaled_mm** | cuBLAS FP4 kernel大M效率更高 |

加载时同时保存swizzled（`block_scale_sw`）和unswizzled（`block_scale`）两种scale布局。

## Kernel级速度对比（fused vs _scaled_mm）

| 场景 | M | `_scaled_mm` | fused | 加速 |
|------|-----|-------------|-------|------|
| decode | 1 | 643us | 206us→**37us**(调参后) | 3.1x→17x |
| small batch | 64 | 296us | 207us | 1.4x |
| prefill | 2100 | 0.43ms | 0.76ms | 0.57x（_scaled_mm胜） |

## 引擎级验证（1.5B, 2099 tokens）

### 精度

| 指标 | _scaled_mm路径 | fused混合路由 |
|------|---------------|--------------|
| PPL delta | +0.0242 | **+0.0242**（prefill走_scaled_mm，逐GEMM验证max_diff~0.001） |
| Top-1 | 97.14% | 97.14% |
| VRAM | 1.67 GiB | 1.71 GiB |

### 速度

| 场景 | 原生fp16 | _scaled_mm | fused混合路由 |
|------|---------|-----------|--------------|
| decode（逐token） | 145.7 tok/s | 16.9 tok/s | **21.3 tok/s（+26%）** |
| prefill（2100批） | 5269 tok/s | 3518 tok/s | **3543 tok/s（持平）** |

## decode瓶颈分析（profile）

fused decode每step ~1000次kernel launch，CPU启动开销(~35ms/step)
远超GPU计算(~11ms/step)：

| 热点 | GPU占比 | 说明 |
|------|--------|------|
| fused_nvfp4_gemm | 36% | 880 calls/10步 |
| fused_fp8_gemm | 35% | 800 calls/10步 |
| amax reduce | 3% | 1680 calls（每个GEMM一次） |
| bf16 cast/AWQ elementwise | 11% | ~7000 calls（每个GEMM约5次） |

**decode是launch-bound**：fp16 GEMM本身（cublas 13us）远快于量化GEMM（37-50us），
但量化GEMM的预处理（cast+AWQ+amax）每次额外3-5次launch，累计开销主导。

## 结论

1. **算子重写有效**：decode +26%（16.9→21.3 tok/s），prefill持平（混合路由取最优）
2. **精度保持**：与_scaled_mm路径数值一致（PPL +0.0242）
3. **权重不反量化**：packed FP4权重直接参与计算，4x带宽收益保留
4. **decode剩余瓶颈是launch开销**，不是kernel本身

## 下一步优化方向

1. **r/k/v三投影融合**：共享x的3个GEMM合成1个kernel，每次decode省10+次launch
2. **残差融合**：nvfp4+fp8残差合并进同一kernel（当前2个kernel）
3. **预处理融合**：bf16 cast + AWQ + amax合并为单kernel，或利用共享x一次性计算
4. **persistent kernel**：跨linear调用的streaming kernel，彻底消除launch开销
