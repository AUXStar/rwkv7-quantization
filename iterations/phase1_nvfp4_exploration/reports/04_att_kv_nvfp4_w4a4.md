# #4 W4A4: L4-19 key/value NVFP4消融

## 实验设计

三个方案对比，全部用纯量化GEMM（W4A4/W8A8）：
- A: att.key L4-19 = NVFP4，其余FP8
- B: 全FP8（基线）
- C: att.key+rec+out L4-19 = NVFP4，其余FP8

## 结果（1.5B, 2100 tokens）

| 方案 | PPL delta | Top-1 | VRAM | Speed |
|------|-----------|-------|------|-------|
| A: key NVFP4 L4-19 | +0.0041 | 98.19% | 1.54G | 2458 t/s |
| B: 全FP8 | +0.0033 | 98.38% | 1.60G | 6132 t/s |
| C: key+rec+out NVFP4 | +0.0101 | 97.62% | 1.51G | 11012 t/s |

## 分析

1. NVFP4 W4A4对attention key的误差略大于FP8 W8A8（0.0041 vs 0.0033）
2. 增加rec/out的NVFP4量化（方案C）误差累积（0.0101）
3. 后1600 tokens三个方案都100% Top-1
4. NVFP4 VRAM更省但速度差异源于GEMM路径不同

## 结论

- attention key NVFP4 W4A4可行（PPL delta 0.0041 < 0.05目标）
- 但FP8 W8A8精度更好且速度更快，attention建议用FP8
- 若需极致压缩，NVFP4可用于rec/out（低敏感度）
