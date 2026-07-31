# #8 推理引擎适配（纯量化GEMM, W4A4/W8A8）

## 概述

v3a推理引擎适配纯量化GEMM方案，所有量化权重直接参与`_scaled_mm`计算，不反量化。
用完整"1.5b"方案量化1.5B模型（att key/value/rec/out + ffn key/value），验证端到端推理精度和性能。

## 代码改动

### rwkv7_fast_v3a.py

1. **import**: `from nvfp4_ops import is_nvfp4_weight, load_nvfp4_weight, linear_nvfp4, is_fp8_weight, load_fp8_weight, linear_quantized`
2. **权重加载** (line 279-290):
   - 跳过 scale keys (`.nf4_b_scale`, `.nvfp4_t_scale`, `.fp8_scale`, `.awq_scale`, `.res_fp8`, `.res_fp8_scale`)
   - NVFP4权重: `load_nvfp4_weight(z, key, dev, swizzle=True)`（128x4 swizzle for `_scaled_mm`）
   - FP8权重: `load_fp8_weight(z, key, dev, w8a16=False)`（W8A8，激活在线量化）
3. **`_att_linear`** (line 656-662): dict权重 → `linear_quantized(x, w, out_dtype=DTYPE)`，否则走原始路径
4. **`cmix_from_mixed`** (line 618-642): FFN key/value为dict时 → `linear_quantized`，绕过orig_layout+sparse路径
5. **`NVFP4_W4A16 = False`**: 全局设置，确保swizzle开启、FP8用W8A8

### nvfp4_ops.py

- `linear_nvfp4`: FP4×FP4→BF16 `_scaled_mm`，激活在线量化（fused Triton kernel）
- `linear_fp8`: FP8×FP8→BF16 `_scaled_mm`，激活在线量化
- `linear_nvfp4` (with res): FP4 GEMM + FP8 GEMM残差相加
- `linear_quantized`: dispatcher，根据qtype选择GEMM路径

## 验证结果（1.5B, 2099 tokens, pure quantized GEMM）

| 指标 | 值 |
|------|-----|
| PPL | 1.5304 (orig 1.5061, delta +0.0242) |
| Top-1 | 97.14% |
| CE delta | +0.015961 |
| VRAM | 1.67 GiB |
| Speed | 2669 tok/s |
| 文件大小 | 1.82 GB (orig 2.85 GB, 1.56x) |
| 量化张量 | 144 (88 NVFP4 + 56 FP8 + 24 NVFP4+res) |
| 压缩率 | 2.1x (权重量化部分) |
| 量化耗时 | 13.7s |

### 窗口分析

| 窗口 | Top-1 | PPL delta |
|------|-------|-----------|
| 0-100 | 84.00% | +1.6491 |
| 100-300 | 85.50% | +0.3374 |
| 300-500 | 92.50% | +0.1651 |
| 500-700 | 100.00% | +0.0029 |
| 700-1000 | 100.00% | +0.0004 |
| 1000-1500 | 100.00% | +0.0003 |
| 1500-2099 | 100.00% | +0.0002 |

### 误差特征

- 误差集中在warmup阶段（0-500 tokens），PPL delta从1.65递减到0.17
- 500 tokens后Top-1=100%，PPL delta<0.003
- PPL delta随序列增长递减趋近0，无state累积

## 与旧方案（W4A16反量化）对比

| 指标 | 旧#8 (W4A16反量化) | 新#8 (W4A4/W8A8纯量化) | 变化 |
|------|-------------------|----------------------|------|
| PPL delta | +0.0050 | +0.0242 | 4.8x（激活量化引入） |
| Top-1 | 98.28% | 97.14% | -1.14% |
| VRAM | 1.67 GiB | 1.67 GiB | 相同 |
| Speed | 2542 tok/s | 2669 tok/s | +5% |
| 推理方式 | 反量化到FP16后GEMM | 纯量化GEMM (`_scaled_mm`) | 根本性变化 |

### 精度差异分析

W4A4比W4A16精度低的原因：
1. **激活量化到FP4**：FP4只有16个离散值，对attention rec/out和FFN key的激活引入量化误差
2. **FFN value的W8A8**：FP8激活量化误差较小，但仍比FP16略差
3. **误差集中在warmup**：state未稳定时量化误差更大，但收敛后完全消失

### 速度提升分析

纯量化GEMM略快于反量化方案：
- 消除了反量化开销（NVFP4→FP16的unpack+scale操作）
- `_scaled_mm`直接在量化域计算，减少内存搬运
- 但激活在线量化引入少量开销，净提速约5%

## 关键确认

1. **纯量化GEMM可行**：PPL delta 0.0242远低于0.05目标
2. **无state累积**：500 tokens后完全收敛
3. **VRAM不变**：量化权重存储格式相同，推理时显存占用一致
4. **引擎适配完整**：attention和FFN路径都正确拦截量化权重

## 下一步

#9: 7.2B完整精度/速度/显存对比（纯量化GEMM）
