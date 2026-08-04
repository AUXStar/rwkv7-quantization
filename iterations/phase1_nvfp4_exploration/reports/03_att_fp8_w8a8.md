# #3 W8A8: attention key/value FP8（纯量化GEMM）

## 结果（2100 tokens）

| 指标 | 值 |
|------|-----|
| PPL | 1.5079 (orig 1.5061, delta +0.0018) |
| Top-1 | 99.43% (early 98%, late 100%) |
| VRAM | 2.50 GiB |
| Speed | 6563 tok/s |

## 结论

FP8 W8A8对attention key/value近乎无损（PPL delta 0.0018，Top-1 99.43%）。
activation在线量化到FP8误差极小，远好于FP4。

## 与#2对比

| 指标 | #2 FFN NVFP4 W4A4 | #3 Att FP8 W8A8 |
|------|-------------------|-----------------|
| PPL delta | +0.0385 | +0.0018 |
| Top-1 | 96.09% | 99.43% |
| Speed | 2426 tok/s | 6563 tok/s |

FP8 W8A8精度和速度都远优于NVFP4 W4A4。
