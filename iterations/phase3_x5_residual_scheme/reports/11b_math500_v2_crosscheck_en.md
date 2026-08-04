# #11 Generation Quality Acceptance v2: rwkv Library Decode Cross-Validation

## Background

User noted that the rwkv library can be installed for decoding text. This report re-runs MATH500 greedy using the rwkv official tokenizer
(`rwkv.utils.PIPELINE`) to cross-validate conclusions and add generation quality diagnostics.

## Cross-Validation Results (500 problems, temperature=0)

| Metric | v1 (Albatross TRIE_TOKENIZER) | v2 (rwkv library PIPELINE) |
|------|-------------------------------|----------------------|
| orig acc | 0.126 (63/500) | **0.126 (63/500)** |
| quant acc | 0.080 (40/500) | **0.080 (40/500)** |
| delta | -0.046 FAIL | **-0.046 FAIL** |

**Both tokenizers produce identical results** -> MATH500 degradation is unrelated to decode method, it is a real result.

## Tokenizer Comparison Findings

- **encode is completely identical** (20/20 same across sampled MATH500 full prompts)
- **decode difference only in token 0**: Albatross's TRIE_TOKENIZER explicitly adds
  `{0: "<|endoftext|>"}`; rwkv library (0.8.32)'s TRIE_TOKENIZER **has no token 0 definition**,
  decode encountering 0 returns `\ufffd` (Albatross is the fixed version)
- This evaluation's generated sequences do not contain token 0, so both produce identical results

## Generation Quality Diagnostics (v2 new, rwkv library decode)

| Metric | orig | quant |
|------|------|-------|
| No garbled text (garbled) | 0 | 0 |
| Unable to extract answer (none) | 13 (2.6%) | 11 (2.2%) |
| Average generation length | 232 tokens | 241 tokens |
| Premature stop (<30 tok) | 7 | 7 |

**Diagnostic conclusion: format degradation hypothesis excluded.** The quantized model's generated text format is normal, no garbled text,
answer extractability is comparable, length is comparable — the degradation comes entirely from **math reasoning itself**.

## Failure Mode Analysis (v1 details)

```
orig correct quant wrong: 41 problems  <- source of net loss
  - 33 problems "last number wrong" (number extraction)
  -  7 problems boxed answer wrong
  -  1 problem unable to extract
orig wrong quant correct: 18 problems  <- anomalous gains
Net loss: 23 problems (63-40)
```

**Pattern**: Tiny logits perturbations cause final number decision flips (33 of 41 problems).
This is not a collapse at any single step, but the accumulation of tiny errors at each step over long reasoning chains,
ultimately producing ~5pp accuracy flips in number selection (12.6%->8.0%).

## Conclusion

1. **MATH500 FAIL is definitive**: both decode methods produce identical results, diagnostics exclude format factors
2. The quantization scheme's real shortcoming is **long-chain math reasoning accuracy** (PPL passes but reasoning decisions are impaired)
3. Next steps must target reasoning accuracy optimization: full FP8 for key / FFN residual enhancement / AWQ tuning

## Deliverables

- `eval_tmp/math500_greedy_v2.json` — v2 details (including length/garbled/extraction method)
- `run_math500_v2.py` — reproduction script (rwkv library decode)
