# #11 生成质量验收 v2：rwkv库decode交叉验证

## 背景

用户指出可安装 rwkv 库用于 decode 文本。本报告用 rwkv 官方 tokenizer
（`rwkv.utils.PIPELINE`）重跑 MATH500 greedy，交叉验证结论 + 增加生成质量诊断。

## 交叉验证结果（500 题，temperature=0）

| 指标 | v1（Albatross TRIE_TOKENIZER） | v2（rwkv库 PIPELINE） |
|------|-------------------------------|----------------------|
| orig acc | 0.126 (63/500) | **0.126 (63/500)** |
| quant acc | 0.080 (40/500) | **0.080 (40/500)** |
| delta | -0.046 FAIL | **-0.046 FAIL** |

**两个 tokenizer 结果完全一致** → MATH500 退化与 decode 方式无关，是真实结果。

## Tokenizer 对比发现

- **encode 完全一致**（MATH500 全量 prompt 抽样 20/20 相同）
- **decode 差异只在 token 0**：Albatross 的 TRIE_TOKENIZER 显式添加
  `{0: "<|endoftext|>"}`；rwkv 库 (0.8.32) 的 TRIE_TOKENIZER **没有 token 0 定义**，
  decode 遇 0 返回 `\ufffd`（Albatross 是修复版）
- 本评估生成序列不含 token 0，故两者结果一致

## 生成质量诊断（v2 新增，rwkv库decode）

| 指标 | orig | quant |
|------|------|-------|
| 无乱码 (garbled) | 0 | 0 |
| 无法提取答案 (none) | 13 (2.6%) | 11 (2.2%) |
| 平均生成长度 | 232 tokens | 241 tokens |
| 过早停止 (<30 tok) | 7 | 7 |

**诊断结论：排除格式退化假设。** 量化模型生成的文本格式正常、无乱码、
答案可提取率相当、长度相当——退化完全来自**数学推理本身**。

## 失败模式分析（v1 明细）

```
orig对quant错: 41 题  ← 净损失来源
  - 33 题 "最后一个数字错了"（number 提取）
  -  7 题 boxed 答案错
  -  1 题 无法提取
orig错quant对: 18 题  ← 异常收益
净损失: 23 题（63-40）
```

**模式**：微小 logits 扰动导致最终数字决策翻转（41题中33题）。
这不是某一步崩溃，而是长推理链上每一步微小误差的累积，
最终在数字选择上产生 ~5pp 的准确率翻转（12.6%→8.0%）。

## 结论

1. **MATH500 FAIL 确凿**：两种 decode 方式结果一致，诊断排除格式因素
2. 量化方案的真实短板是**长链数学推理精度**（PPL 达标但推理决策受损）
3. 下一步必须针对推理精度优化：key 全面 FP8 / FFN 残差增强 / AWQ 调优

## 输出物

- `eval_tmp/math500_greedy_v2.json` — v2 明细（含长度/乱码/提取方式）
- `run_math500_v2.py` — 复现脚本（rwkv库decode）
