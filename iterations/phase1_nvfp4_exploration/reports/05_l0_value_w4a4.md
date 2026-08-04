# #5 W4A4: L0 value BF16 vs FP8

## 结果（1.5B, 2100 tokens）

| 方案 | PPL delta | Top-1 | VRAM |
|------|-----------|-------|------|
| A: L0 value BF16 | +0.0030 | 98.33% | 1.57G |
| B: L0 value FP8 | +0.0033 | 98.38% | 1.60G |

## 结论

L0 value FP8 W8A8与BF16几乎无差异（PPL delta差0.0003，Top-1差0.05%）。
v_first跨层传播的误差在W8A8下可忽略。L0 value不需要BF16，FP8足够。
