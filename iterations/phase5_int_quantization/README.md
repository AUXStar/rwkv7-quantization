# Phase 5: INT Quantization Exploration

## Scope

Extend the quantization toolkit beyond FP8 to integer formats (INT8, INT4), covering per-tensor, affine, and group-wise schemes. Evaluate on the same 1.5B model with the same benchmark suite to enable direct comparison.

## Issues

### Issue #17: INT8 Per-Tensor Baseline
- Implement `int8/quantize_int8.py` (per-tensor symmetric scheme)
- Implement `int8/int8_ops.py` (weight detection, loading, GEMM)
- Benchmark: PPL, Top-1, EAR, decode speed, VRAM
- Compare with FP8 per-tensor (same 8-bit width, different format)

### Issue #18: INT8 Affine (MM8-style)
- Extend `int8/quantize_int8.py` with affine scheme
- Implement `int8/fused_int8_gemm.py` (Triton GEMV with in-register dequant)
- Benchmark: same suite
- Key question: does dual affine beat per-tensor for RWKV-7's symmetric weights?

### Issue #19: INT4 Per-Tensor Baseline
- Implement `int4/quantize_int4.py` (per-tensor symmetric, paired nibble)
- Implement `int4/int4_ops.py` (nibble unpack, GEMM)
- Benchmark: same suite
- Key question: is 16 levels enough for RWKV-7 linear layers?

### Issue #20: INT4 Affine (MM4-style)
- Extend `int4/quantize_int4.py` with affine scheme
- Implement `int4/fused_int4_gemm.py` (Triton GEMV with nibble unpack)
- Benchmark: same suite
- Key question: does affine compensation help with only 16 levels?

### Issue #21: INT4 Group-wise Quantization
- Extend `int4/quantize_int4.py` with group-wise scheme (group_size=128, 256)
- Implement group-wise Triton GEMV
- Benchmark: same suite
- Key question: what group size gives the best precision/speed trade-off?

### Issue #22: Cross-Scheme Comparison & Pareto Analysis
- Run all schemes on identical prompts and hardware
- Generate comparison tables and charts
- Identify Pareto frontier
- Write final recommendation per deployment scenario

## Timeline

| Step | Issue | Deliverable |
|------|-------|-------------|
| 1 | #17 | INT8 per-tensor quantized model + benchmark report |
| 2 | #18 | INT8 affine quantized model + benchmark report |
| 3 | #19 | INT4 per-tensor quantized model + benchmark report |
| 4 | #20 | INT4 affine quantized model + benchmark report |
| 5 | #21 | INT4 group-wise quantized model + benchmark report |
| 6 | #22 | Cross-scheme comparison report + Pareto analysis |

## Reports

Each issue produces a report in this directory following the naming convention:

```
17_int8_per_tensor.md
18_int8_affine.md
19_int4_per_tensor.md
20_int4_affine.md
21_int4_groupwise.md
22_cross_scheme_comparison.md
```

## Hypothesis

Based on Phase 1-4 findings:
1. RWKV-7 weights are naturally symmetric (skewness < 0.01), so affine's asymmetry advantage may be minimal
2. Layer sensitivity is uniform (CV < 1%), so per-component scheme selection won't help
3. INT4's 16 levels will cause significant EAR drop (cf. NVFP4's 8.8% error rate)
4. Group-wise INT4 (group=128) may recover most of the precision loss
5. INT8 should be competitive with FP8 on precision, but slower without hardware int8 tensor cores
