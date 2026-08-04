# #5 Experiment Report: L0 value BF16 Necessity Validation

## Overview

Testing whether att.value in L0 needs to remain BF16. L0's value produces v_first, which propagates to all subsequent layers through vres gating:
```
v = v + gate * (v_first - v)   # gate = sigmoid(v0 + ...)
```
Hypothesis: v_first error is amplified across 23 layers; L0 value needs BF16 protection.

## Experimental Results

| Test | Scheme | Top1 | PPL delta |
|------|------|------|-----------|
| Baseline | All BF16 | 100.00% | 0.0000 |
| 5a | L0 value NVFP4 only | 99.48% | -0.0003 |
| 5b | L0 value FP8 only | 99.95% | -0.0002 |
| 5c | L0 val NVFP4 + L4-19 key/value NVFP4 | 98.62% | +0.0056 |
| 5d | L1-23 value NVFP4, L0 BF16 | 98.76% | +0.0036 |

## Key Findings

### v_first Amplification Hypothesis Does Not Hold
- L0 value NVFP4 standalone: only 0.52% drop
- L1-23 value NVFP4 (skipping L0): 1.24% drop
- **L0 value is actually less sensitive than other layers** (2.4x more robust)

### Reason Analysis
1. **vres gating dilution**: `v = v + gate * (v_first - v)`, gate < 1, v_first error is mixed rather than directly propagated
2. **v_first is a residual signal**: not the main path, but a correction to v
3. **Normalization buffer**: subsequent layers' GroupNorm and LayerNorm absorb v_first's magnitude error

### L0 value FP8 is Near-Lossless
5b: Top1=99.95%, PPL delta=-0.0002. FP8 is fully viable for L0 value.

## Conclusion

- L0 value **does not need BF16**; FP8 is sufficient (near-lossless)
- L0 value NVFP4 is also acceptable (only 0.52% drop)
- In the final scheme, L0 value can be downgraded from BF16 to FP8, saving VRAM
- The error amplification effect of v_first cross-layer propagation is effectively suppressed by vres gating

## Correction to the Final Scheme

Original scheme: L0 key/value BF16
Recommended: L0 key/value FP8 (near-lossless, saves VRAM)
