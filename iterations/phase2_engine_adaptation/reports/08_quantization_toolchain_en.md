# #7 Engineering Report: Quantization Toolchain

## Overview

Build a unified quantization toolchain `quantize_model.py` that supports:
- Load model -> quantize according to scheme -> store quantized model with meta
- Support four quantization formats: BF16 / FP8 / NVFP4 / NVFP4+FP8 residual
- Quantization scheme updated based on #2-#6 experimental data

## Quantization Scheme (Updated Based on Experimental Data)

### 1.5B Model (24 layers)

| Component | L0-3 | L4-19 | L20-23 | Basis |
|------|------|-------|--------|------|
| att.key | FP8 | NVFP4 | FP8 | #4: key NVFP4 99.67% |
| att.value | FP8 | FP8 | FP8 | #4: value is more sensitive than key |
| att.rec/out | NVFP4 | NVFP4 | NVFP4 | low sensitivity |
| ffn.key | NVFP4+res | NVFP4+res | NVFP4+res | #2 v12: 99.05% |
| ffn.value | FP8 | FP8 | FP8 | nearly lossless |

### Differences from the Original README Scheme

| Item | Original scheme | Updated scheme | Reason |
|------|--------|----------|------|
| L0 key/value | BF16 | FP8 | #5: L0 value FP8 = 99.95% |
| att.value | NVFP4 (L4-27) | FP8 | #4: value NVFP4 sensitive |
| ffn.key | NVFP4 | NVFP4+FP8 res | #2 v12: 99.05% |
| ffn.value | NVFP4 | FP8 | FP8 nearly lossless, avoids NVFP4 risk |

## Toolchain Implementation

### Files
- `quantize_model.py` -- unified quantization tool
- CLI: `python quantize_model.py --model ... --output ... --scheme 1.5b`

### Quantization Formats

**NVFP4** (E2M1 + AWQ + clip ratio):
```
weight -> packed (uint8, K/2 columns)
         .nf4_b_scale (fp8_e4m3fn, block scales)
         .nvfp4_t_scale (fp32, tensor scale)
         .awq_scale (fp32, channel scales)
```

**FP8** (E4M3 per-tensor):
```
weight -> .weight (float8_e4m3fn)
         .fp8_scale (fp32, scalar)
```

**NVFP4+FP8 residual** (v12 dual quantization):
```
weight -> packed (uint8) + scales (same as NVFP4)
         .res_fp8 (float8_e4m3fn, residual)
         .res_fp8_scale (fp32, scalar)
```

### Meta Format
```python
meta = {
    "v": 1, "scheme": "1.5b", "layers": 24,
    "r": [[layer_start, layer_end, comp, dtype], ...],
    "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},
    "n": [non_quantized_prefixes],
    "stats": {"bf16": 0, "fp8": 56, "nvfp4": 64, "nvfp4_res": 24}
}
```

## Validation Results

- 1.5B model quantization: completed in 10.2 seconds
- 144 weight tensors correctly quantized (56 FP8 + 64 NVFP4 + 24 NVFP4+res)
- Tensor data: 2.25 GB -> 1.82 GB (1.24x compression, including non-quantized)
- Quantized portion: 2.25 GB -> 1.07 GB (2.1x compression)
- Known issue: torch.save serialization overhead causes file to be larger than expected (4.07 GB vs 1.82 GB actual data)

## Next Steps

#8: Inference engine adaptation -- extend the v3a engine to support:
1. NVFP4 attention weights (rec/key/out)
2. FP8 FFN value weights
3. NVFP4+FP8 residual FFN key weights
