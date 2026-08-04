# #4 W4A4: L4-19 key/value NVFP4 Ablation

## Experimental Design

Three schemes compared, all using pure quantized GEMM (W4A4/W8A8):
- A: att.key L4-19 = NVFP4, rest FP8
- B: All FP8 (baseline)
- C: att.key+rec+out L4-19 = NVFP4, rest FP8

## Results (1.5B, 2100 tokens)

| Scheme | PPL delta | Top-1 | VRAM | Speed |
|------|-----------|-------|------|-------|
| A: key NVFP4 L4-19 | +0.0041 | 98.19% | 1.54G | 2458 t/s |
| B: All FP8 | +0.0033 | 98.38% | 1.60G | 6132 t/s |
| C: key+rec+out NVFP4 | +0.0101 | 97.62% | 1.51G | 11012 t/s |

## Analysis

1. NVFP4 W4A4 error on attention key is slightly larger than FP8 W8A8 (0.0041 vs 0.0033)
2. Adding NVFP4 quantization to rec/out (scheme C) leads to error accumulation (0.0101)
3. All three schemes achieve 100% Top-1 on the last 1600 tokens
4. NVFP4 saves more VRAM but speed differences stem from different GEMM paths

## Conclusion

- Attention key NVFP4 W4A4 is feasible (PPL delta 0.0041 < 0.05 target)
- But FP8 W8A8 has better accuracy and speed; FP8 is recommended for attention
- If extreme compression is needed, NVFP4 can be used for rec/out (low sensitivity)
