# Final Quantization Scheme (Task 1-4 Completed) Acceptance Report

## Scheme (Three modifications relative to the original scheme)

| Component | Original Scheme | Final Scheme | Basis |
|------|--------|---------|------|
| att.key | L0-3/L20-23 FP8, L4-19 NVFP4 | **L0 bf16, L1-23 FP8** | T2: full FP8 +0.8pp |
| Low-rank (8x/layer) | bf16 | **bf16 (not quantized)** | User decision (1.7% gain not worthwhile)|
| ffn.key residual | per-tensor FP8 | **per-block FP8 ratio** | Task 3, quantization domain computation |
| AWQ alpha | 0.5 | **0.3** | T4 grid search (MATH500 8.6%->11.2%)|

**Principle implemented**: Full-chain quantization domain execution, **no runtime dequantization**. Residual per-block uses pure FP8xFP8 GEMM
(fused kernel handles arbitrary M; _scaled_mm cannot mix fp32 tensor x-scale + fp8 block w-scale,
so the dispatcher forces fused for residual weights).

## Task 1-4 MATH500 Data (orig=12.6% cached baseline)

| Version | Configuration | MATH500 | delta |
|------|------|---------|-------|
| Baseline | Original scheme (alpha 0.5) | 8.0% | -4.6pp |
| T1 | +low-rank FP8 | 8.6% | -4.0pp |
| T2 | +key full FP8 | 9.4% | -3.2pp |
| T3 | low-rank bf16 + residual per-block | 8.6% | -4.0pp |
| **T4** | **T3 + alpha=0.3** | **11.2%** | **-1.4pp** |

**Key finding**: AWQ alpha has almost no effect on PPL (3.4219 vs 3.4224), but has a
**+2.6pp** impact on MATH500 — PPL cannot predict problem-solving quality.

## Final Acceptance

| Metric | Target | Actual | Status |
|------|------|------|------|
| MATH500 | >=12.0% (gap<=0.6pp) | **11.2%** (gap 1.4pp) | ✗ short by 0.8pp |
| MATH500 acceptance line | <=2pp | 1.4pp | ✅ PASS |
| PPL delta (1024/2048/4096/8192) | <=0.05 | 0.028/0.008/0.005/0.002 | ✅ |
| VRAM | <=2.0 GiB | 1.87 GiB | ✅ |
| decode speed | >=65 t/s | 70.0 t/s | ✅ |
| 8192 state MSE L23 | On par with FP16 | 6.81e-3 | ✅ |

## Test Process Record

1. **T1 low-rank FP8**: Found key missing .weight suffix bug (fixed); per-column scale avoids small K error;
   MATH500 8.6%, 1.7% disk savings, 0 runtime benefit -> user decision to revert to bf16
2. **T2 key full FP8**: 9.4%, effective but insufficient
3. **T3 residual per-block**: Initial version failed due to **Albatross/rwkv7-quantization's nvfp4_ops.py copy drift**
   causing loader to not recognize per-block format -> generation completely broke; fixed after sync. PPL 1.5338 (best) but MATH500 8.6%
4. **T4 AWQ alpha search**: Searched alpha in [0.1,0.2,0.3,0.4,0.5,0.7,0.9] on MATH500-dist PPL,
   alpha=0.3 optimal -> MATH500 **11.2%** (major leap)

## Next Steps Optional (Pursuing 12.0%)

1. Run MATH500 directly with alpha=0.2/0.4 (PPL does not predict MATH500, may be better)
2. Restore low-rank FP8 + alpha=0.3 (T2 data +0.8pp was measured under alpha=0.5, may stack under alpha=0.3)
3. Accept 11.2% (already passes the 2pp acceptance line)

## Deliverables

- `eval_tmp/math500_T4.json` — T4 per-problem details
- `eval_tmp/final_acceptance.json` — final acceptance metrics
- `run_math500_quant.py` / `run_final_accept.py` / `run_t4_search.py` — reproduction scripts
