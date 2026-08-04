# #7 Quantization Toolchain (Validated)

## Overview

`quantize_model.py` is a unified quantization toolchain supporting the full pipeline: load -> classify -> quantize -> store+meta.
It has been validated for reliability and correctness in #2-#6 experiments.

## Toolchain Files

| File | Function | Lines |
|------|------|------|
| `quantize_model.py` | Unified quantization tool (load -> classify -> quantize -> store+meta) | ~400 |
| `nvfp4_ops.py` | NVFP4/FP8 GEMM kernel + weight loading + activation quantization | ~500 |
| `fused_nvfp4_quant.py` | Triton fused kernel (activation quantization + pack + swizzle) | ~300 |

## Quantization Formats

### NVFP4 (E2M1) — W4A4
```
Storage format:
  weight:          [N, K//2] uint8 (packed FP4 pairs)
  .nf4_b_scale:    [N, K//16] float8_e4m3fn (per-block scale)
  .nvfp4_t_scale:  scalar float32 (per-tensor scale)
  .awq_scale:      [K] float32 (AWQ channel scale)
```

### FP8 (E4M3) — W8A8
```
Storage format:
  weight:          [N, K] float8_e4m3fn
  .fp8_scale:      scalar float32 (per-tensor scale)
```

### NVFP4+FP8 Residual — W4A4+W8A8
```
Storage format:
  weight:          [N, K//2] uint8 (packed FP4)
  .nf4_b_scale:    [N, K//16] float8_e4m3fn
  .nvfp4_t_scale:  scalar float32
  .awq_scale:      [K] float32
  .res_fp8:        [N, K] float8_e4m3fn (FP8 residual)
  .res_fp8_scale:  scalar float32
```

## Quantization Pipeline

```python
# 1. Load model (mmap)
z = torch.load(model_path, map_location="cpu", mmap=True)

# 2. Detect number of layers
num_layers = max(int(k.split(".")[1]) for k in z if k.startswith("blocks.")) + 1

# 3. Get scheme
rules = scheme_fn()  # [(layer_start, layer_end, comp, dtype), ...]

# 4. Per-tensor classify + quantize
for key in z:
    result = classify_weight(key, num_layers)  # -> (layer, comp)
    dtype = get_dtype_for(rules, layer, comp)
    if dtype == FP8:    quantize_to_fp8(w)
    if dtype == NVFP4:  quantize_nvfp4(w, awq_scale)  # AWQ + clip ratio search
    if dtype == NVFP4_RES: quantize_nvfp4_with_residual(w, awq_scale)

# 5. Clone all tensors (detach from mmap, prevent file bloat)
# 6. Generate meta dictionary
# 7. Save
```

## Supported Schemes

| Scheme | Applicable Models | Description |
|------|---------|------|
| `1.5b` | 1.5B (24 layers) | Mixed scheme: FP8 key edge + NVFP4 middle, FP8 value, NVFP4 rec/out, NVFP4+res ffn_key, FP8 ffn_value |
| `2.9b` | 2.9B/7.2B (32 layers) | Same as above, layer ranges adjusted for 32 layers |
| `experimental` | Any | All NVFP4 (extreme compression test) |
| `fp8` | Any | All FP8 (accuracy baseline) |
| Custom | Any | Pass custom rules via `_scheme_override` parameter |

## Validation Results (1.5B, "1.5b" scheme)

```
Scheme: 1.5b
Layers: 24
Stats: 56 FP8 + 64 NVFP4 + 24 NVFP4+res = 144 tensors
Original: 2.25 GB -> Quantized: 1.07 GB (2.1x compression)
File size: 1.82 GB (including non-quantized weights)
Quantization time: 10.2s
```

### Meta Data Structure
```python
meta = {
    "v": 1,                    # version
    "scheme": "1.5b",          # scheme name
    "layers": 24,              # number of layers
    "r": [...],                # quantization rules (8 entries)
    "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},  # quantization parameters
    "n": [...],                # non-quantized prefixes (27 entries)
    "stats": {"bf16": 0, "fp8": 56, "nvfp4": 64, "nvfp4_res": 24},
    "orig_size_gb": 2.25,
    "quant_size_gb": 1.07,
    "compression": 2.1
}
```

## CLI Usage

```bash
# Basic usage
python quantize_model.py --model /path/to/model.pth --output /path/to/quantized.pth

# Specify scheme
python quantize_model.py --model ... --output ... --scheme 1.5b
python quantize_model.py --model ... --output ... --scheme 2.9b
python quantize_model.py --model ... --output ... --scheme experimental
python quantize_model.py --model ... --output ... --scheme fp8
```

## Engineering Notes

1. **mmap loading**: Large models loaded with `mmap=True` to avoid out-of-memory
2. **Clone to detach from mmap**: Clone all tensors before saving to prevent file bloat (validated)
3. **AWQ weight heuristics**: Use weight abs mean instead of activation statistics when no calibration data
4. **Clip ratio search**: 9 candidate values [0.60~1.00], per-block select minimum MSE
5. **128x4 swizzle**: Block scale swizzled at load time (handled in `load_nvfp4_weight`)
