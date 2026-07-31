# #7 工程报告：量化工具链

## 概述

构建统一的量化工具链 `quantize_model.py`，支持：
- 加载模型 → 按方案量化 → 存储带 meta 的量化模型
- 支持 BF16 / FP8 / NVFP4 / NVFP4+FP8 residual 四种量化格式
- 基于 #2-#6 实验数据更新的量化方案

## 量化方案（基于实验数据更新）

### 1.5B 模型（24层）

| 组件 | L0-3 | L4-19 | L20-23 | 依据 |
|------|------|-------|--------|------|
| att.key | FP8 | NVFP4 | FP8 | #4: key NVFP4 99.67% |
| att.value | FP8 | FP8 | FP8 | #4: value 比 key 更敏感 |
| att.rec/out | NVFP4 | NVFP4 | NVFP4 | 低敏感度 ★★ |
| ffn.key | NVFP4+res | NVFP4+res | NVFP4+res | #2 v12: 99.05% |
| ffn.value | FP8 | FP8 | FP8 | 近乎无损 |

### 与原 README 方案的区别

| 项目 | 原方案 | 更新方案 | 原因 |
|------|--------|----------|------|
| L0 key/value | BF16 | FP8 | #5: L0 value FP8 = 99.95% |
| att.value | NVFP4 (L4-27) | FP8 | #4: value NVFP4 敏感 |
| ffn.key | NVFP4 | NVFP4+FP8 res | #2 v12: 99.05% |
| ffn.value | NVFP4 | FP8 | FP8 近乎无损，避免 NVFP4 风险 |

## 工具链实现

### 文件
- `quantize_model.py` — 统一量化工具
- CLI: `python quantize_model.py --model ... --output ... --scheme 1.5b`

### 量化格式

**NVFP4**（E2M1 + AWQ + clip ratio）:
```
weight → packed (uint8, K/2 列)
         .nf4_b_scale (fp8_e4m3fn, block scales)
         .nvfp4_t_scale (fp32, tensor scale)
         .awq_scale (fp32, channel scales)
```

**FP8**（E4M3 per-tensor）:
```
weight → .weight (float8_e4m3fn)
         .fp8_scale (fp32, scalar)
```

**NVFP4+FP8 residual**（v12 双量化）:
```
weight → packed (uint8) + scales (同 NVFP4)
         .res_fp8 (float8_e4m3fn, residual)
         .res_fp8_scale (fp32, scalar)
```

### Meta 格式
```python
meta = {
    "v": 1, "scheme": "1.5b", "layers": 24,
    "r": [[layer_start, layer_end, comp, dtype], ...],
    "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},
    "n": [non_quantized_prefixes],
    "stats": {"bf16": 0, "fp8": 56, "nvfp4": 64, "nvfp4_res": 24}
}
```

## 验证结果

- 1.5B 模型量化：10.2 秒完成
- 144 个权重张量正确量化（56 FP8 + 64 NVFP4 + 24 NVFP4+res）
- Tensor 数据：2.25 GB → 1.82 GB（1.24x 压缩，含 non-quantized）
- 量化部分：2.25 GB → 1.07 GB（2.1x 压缩）
- 已知问题：torch.save 序列化开销导致文件偏大（4.07 GB vs 1.82 GB 实际数据）

## 下一步

#8: 推理引擎适配 — 扩展 v3a 引擎支持：
1. NVFP4 attention 权重（rec/key/out）
2. FP8 FFN value 权重
3. NVFP4+FP8 residual FFN key 权重
