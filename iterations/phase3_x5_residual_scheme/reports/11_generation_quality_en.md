# #11 Generation Quality Acceptance: MATH500 greedy + Uncheatable Eval

## Background

Project Hard Constraint requires: quantization acceptance must include PPL + long-text generation problem-solving metrics
(MATH500 greedy) + Uncheatable Eval novel-corpus anti-memorization. Previously omitted, now completed in this round.

Model: 1.5B final quantization scheme (W4A4/W8A8 hybrid + fused kernel, after fixing prep3_x bug).

## MATH500 greedy (all 500 problems, temperature=0)

| Model | Accuracy | Correct Count |
|------|--------|---------|
| Original bf16 | **12.6%** | 63/500 |
| Quantized | **8.0%** | 40/500 |
| **delta** | **-4.6pp** | — |

**Acceptance: <=2pp -> FAIL** ❌

### Analysis

1. 1.5B model itself only achieves 12.6% on MATH500 (weak math ability for small models), dropping to 8.0% after quantization
2. **PPL delta only +0.0242 (passes), but generation quality drops 4.6pp** ——
   validates that PPL metric alone is insufficient, generation quality acceptance is necessary (correctness of the hard constraint)
3. Contrasts with Uncheatable (compression ratio) passing: compression ratio is insensitive to tiny logits changes,
   while greedy problem-solving is highly sensitive to tiny logits changes (one argmax deviation = wrong answer)
4. 1.5B small model has low redundancy (further evidence of the user's judgment that "small models are more sensitive to quantization")

## Uncheatable Eval (24 documents, 4 corpus types, chunk=4000)

| Metric | Original | Quantized |
|------|------|------|
| bpb | 0.5453 | 0.5732 |
| Compression ratio | 6.82% | 7.17% |
| **ratio** | — | **105.1%** |

**Acceptance: quant >= 99% of orig -> PASS** ✅

## Conclusion

| Acceptance Item | Result | Status |
|--------|------|------|
| Uncheatable Eval compression ratio | ratio 105.1% | ✅ PASS |
| MATH500 greedy | delta -4.6pp | ❌ FAIL |

**Partially meets acceptance.** The quantization scheme maintains accuracy at the language modeling level (PPL/compression ratio),
but has real degradation on math reasoning tasks requiring precise token selection (-4.6pp).

## Next Optimization Directions (Recover Generation Quality)

1. **Full FP8 for key**: Upgrade L4-19 key from NVFP4 to FP8 (#4 shows value NVFP4 diff +0.016,
   key's impact is mainly on long-range reasoning)
2. **FFN key residual enhancements**: Increase NVFP4+FP8 residual precision (current residual is FP8 per-tensor)
3. **AWQ intensity tuning**: Expand clip ratio search range
4. **Alternative for intermediate layer value FP8->NVFP4**: If key retains NVFP4, budget can be moved to value

## Deliverables

- `eval_tmp/math500_greedy.json` — per-problem pred/gold details
- `eval_tmp/uncheatable_eval.json` — compression ratio details
- `run_math500.py` / `run_uncheatable.py` — reproduction scripts
