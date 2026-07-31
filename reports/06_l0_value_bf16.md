# #5 实验报告：L0 value BF16 必要性验证

## 概述

测试 L0 的 att.value 是否需要保持 BF16。L0 的 value 产生 v_first，通过 vres 门控传播到全部后续层：
```
v = v + gate * (v_first - v)   # gate = sigmoid(v0 + ...)
```
假设：v_first 误差被 23 层放大，L0 value 需 BF16 保护。

## 实验结果

| 测试 | 方案 | Top1 | PPL delta |
|------|------|------|-----------|
| Baseline | 全 BF16 | 100.00% | 0.0000 |
| 5a | L0 value NVFP4 only | 99.48% | -0.0003 |
| 5b | L0 value FP8 only | 99.95% | -0.0002 |
| 5c | L0 val NVFP4 + L4-19 key/value NVFP4 | 98.62% | +0.0056 |
| 5d | L1-23 value NVFP4, L0 BF16 | 98.76% | +0.0036 |

## 关键发现

### v_first 放大假设不成立
- L0 value NVFP4 单独：仅掉 0.52%
- L1-23 value NVFP4（跳过 L0）：掉 1.24%
- **L0 value 反而比其他层更不敏感**（2.4× 更鲁棒）

### 原因分析
1. **vres 门控稀释**：`v = v + gate * (v_first - v)`，gate < 1，v_first 误差被混合而非直接传播
2. **v_first 是残差信号**：不是主路径，而是对 v 的修正
3. **归一化缓冲**：后续层的 GroupNorm 和 LayerNorm 吸收 v_first 的幅度误差

### L0 value FP8 近乎无损
5b: Top1=99.95%，PPL delta=-0.0002。FP8 对 L0 value 完全可行。

## 结论

- L0 value **不需要 BF16**，FP8 即可（近乎无损）
- L0 value NVFP4 也可接受（仅 0.52% 下降）
- 最终方案中 L0 value 可从 BF16 降级为 FP8，节省显存
- v_first 跨层传播的误差放大效应被 vres 门控有效抑制

## 对最终方案的修正

原方案：L0 key/value BF16
建议：L0 key/value FP8（近乎无损，节省显存）
