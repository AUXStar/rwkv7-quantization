# #11 生成质量验收：MATH500 greedy + Uncheatable Eval

## 背景

项目 Hard Constraint 要求：量化验收必须包含 PPL + 长文生成问题求解指标
（MATH500 greedy）+ Uncheatable Eval 新语料防记忆。此前遗漏，本轮补齐。

模型：1.5B 最终量化方案（W4A4/W8A8 混合 + fused kernel，修复 prep3_x bug 后）。

## MATH500 greedy（全 500 题，temperature=0）

| 模型 | 准确率 | 正确题数 |
|------|--------|---------|
| 原始 bf16 | **12.6%** | 63/500 |
| 量化 | **8.0%** | 40/500 |
| **delta** | **-4.6pp** | — |

**验收：≤2pp → FAIL** ❌

### 分析

1. 1.5B 模型本身 MATH500 只有 12.6%（小模型数学能力弱），量化后降至 8.0%
2. **PPL delta 仅 +0.0242（达标），但生成质量下降 4.6pp** ——
   验证了 PPL 指标不够、必须做生成质量验收（hard constraint 的正确性）
3. 与 Uncheatable（压缩率）通过形成对比：压缩率对微小 logits 变化不敏感，
   而 greedy 解题对微小 logits 变化高度敏感（一次 argmax 偏差即答错）
4. 1.5B 小模型冗余度低（用户"小模型对量化更敏感"判断的又一证据）

## Uncheatable Eval（24 文档，4 类语料，chunk=4000）

| 指标 | 原始 | 量化 |
|------|------|------|
| bpb | 0.5453 | 0.5732 |
| 压缩率 | 6.82% | 7.17% |
| **ratio** | — | **105.1%** |

**验收：quant ≥ 99% of orig → PASS** ✅

## 结论

| 验收项 | 结果 | 状态 |
|--------|------|------|
| Uncheatable Eval 压缩率 | ratio 105.1% | ✅ PASS |
| MATH500 greedy | delta -4.6pp | ❌ FAIL |

**部分达标**。量化方案在语言建模（PPL/压缩率）层面保持精度，
但在需要精确 token 选择的数学推理任务上有真实退化（-4.6pp）。

## 下一步优化方向（挽回生成质量）

1. **key 全面 FP8**：L4-19 key 从 NVFP4 升 FP8（#4 显示 value NVFP4 差 +0.016，
   key 的影响主要在长程推理）
2. **FFN key 残差增强**：增大 NVFP4+FP8 残差精度（当前残差为 FP8 per-tensor）
3. **AWQ 强度调优**：clip ratio 搜索范围扩大
4. **中间层 value 试 FP8→NVFP4 的替代**：若 key 保留 NVFP4，可将 budget 移到 value

## 输出物

- `eval_tmp/math500_greedy.json` — 每题 pred/gold 明细
- `eval_tmp/uncheatable_eval.json` — 压缩率明细
- `run_math500.py` / `run_uncheatable.py` — 复现脚本
