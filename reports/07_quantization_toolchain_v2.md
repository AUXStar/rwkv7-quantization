# #7 量化工具链（已验证）

## 概述

`quantize_model.py` 统一量化工具链，支持加载→分类→量化→存储+meta全流程。
已在 #2-#6 实验中验证可靠性和正确性。

## 工具链文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `quantize_model.py` | 统一量化工具（加载→分类→量化→存储+meta） | ~400 |
| `nvfp4_ops.py` | NVFP4/FP8 GEMM kernel + 权重加载 + 激活量化 | ~500 |
| `fused_nvfp4_quant.py` | Triton fused kernel（激活量化+pack+swizzle） | ~300 |

## 量化格式

### NVFP4 (E2M1) — W4A4
```
存储格式:
  weight:          [N, K//2] uint8 (packed FP4 pairs)
  .nf4_b_scale:    [N, K//16] float8_e4m3fn (per-block scale)
  .nvfp4_t_scale:  scalar float32 (per-tensor scale)
  .awq_scale:      [K] float32 (AWQ channel scale)
```

### FP8 (E4M3) — W8A8
```
存储格式:
  weight:          [N, K] float8_e4m3fn
  .fp8_scale:      scalar float32 (per-tensor scale)
```

### NVFP4+FP8残差 — W4A4+W8A8
```
存储格式:
  weight:          [N, K//2] uint8 (packed FP4)
  .nf4_b_scale:    [N, K//16] float8_e4m3fn
  .nvfp4_t_scale:  scalar float32
  .awq_scale:      [K] float32
  .res_fp8:        [N, K] float8_e4m3fn (FP8 residual)
  .res_fp8_scale:  scalar float32
```

## 量化流程

```python
# 1. 加载模型 (mmap)
z = torch.load(model_path, map_location="cpu", mmap=True)

# 2. 检测层数
num_layers = max(int(k.split(".")[1]) for k in z if k.startswith("blocks.")) + 1

# 3. 获取方案
rules = scheme_fn()  # [(layer_start, layer_end, comp, dtype), ...]

# 4. 逐张量分类+量化
for key in z:
    result = classify_weight(key, num_layers)  # → (layer, comp)
    dtype = get_dtype_for(rules, layer, comp)
    if dtype == FP8:    quantize_to_fp8(w)
    if dtype == NVFP4:  quantize_nvfp4(w, awq_scale)  # AWQ + clip ratio搜索
    if dtype == NVFP4_RES: quantize_nvfp4_with_residual(w, awq_scale)

# 5. Clone所有张量（脱离mmap，防止文件膨胀）
# 6. 生成meta字典
# 7. 保存
```

## 支持的方案

| 方案 | 适用模型 | 描述 |
|------|---------|------|
| `1.5b` | 1.5B (24层) | 混合方案：FP8 key边缘+NVFP4中间, FP8 value, NVFP4 rec/out, NVFP4+res ffn_key, FP8 ffn_value |
| `2.9b` | 2.9B/7.2B (32层) | 同上，层范围调整为32层 |
| `experimental` | 任意 | 全NVFP4（极限压缩测试） |
| `fp8` | 任意 | 全FP8（精度基线） |
| 自定义 | 任意 | 通过 `_scheme_override` 参数传入自定义规则 |

## 验证结果（1.5B, "1.5b" scheme）

```
Scheme: 1.5b
Layers: 24
Stats: 56 FP8 + 64 NVFP4 + 24 NVFP4+res = 144 tensors
Original: 2.25 GB → Quantized: 1.07 GB (2.1x compression)
File size: 1.82 GB (含非量化权重)
Quantization time: 10.2s
```

### Meta 数据结构
```python
meta = {
    "v": 1,                    # 版本
    "scheme": "1.5b",          # 方案名
    "layers": 24,              # 层数
    "r": [...],                # 量化规则 (8 entries)
    "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},  # 量化参数
    "n": [...],                # 不量化的前缀 (27 entries)
    "stats": {"bf16": 0, "fp8": 56, "nvfp4": 64, "nvfp4_res": 24},
    "orig_size_gb": 2.25,
    "quant_size_gb": 1.07,
    "compression": 2.1
}
```

## CLI 用法

```bash
# 基本用法
python quantize_model.py --model /path/to/model.pth --output /path/to/quantized.pth

# 指定方案
python quantize_model.py --model ... --output ... --scheme 1.5b
python quantize_model.py --model ... --output ... --scheme 2.9b
python quantize_model.py --model ... --output ... --scheme experimental
python quantize_model.py --model ... --output ... --scheme fp8
```

## 工程要点

1. **mmap加载**：大模型用 `mmap=True` 加载，避免内存溢出
2. **Clone脱离mmap**：保存前clone所有张量，防止文件膨胀（已验证）
3. **AWQ权重启发式**：无校准数据时用权重abs mean代替激活统计
4. **Clip ratio搜索**：9个候选值[0.60~1.00]，逐block选MSE最小
5. **128x4 swizzle**：block scale在加载时swizzle（`load_nvfp4_weight`中处理）
