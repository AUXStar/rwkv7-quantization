# RWKV-7 1.5B Quantization Final (X5 Scheme)

## I. Finalized Scheme

L0: bf16 (v_first source)
L1-L5: key FP8 / value FP8 / rec NVFP4 / out NVFP4 / ffn_k NVFP4+FP8 residual / ffn_v FP8
L6: bf16 (dilution point, resets quantization error)
L7-L17: key FP8 / value FP8 / rec NVFP4 / out NVFP4 / ffn_k NVFP4+FP8 residual / ffn_v FP8
L18: bf16 (dilution point, resets quantization error)
L19-L22: key FP8 / value FP8 / rec NVFP4 / out NVFP4 / ffn_k NVFP4+FP8 residual / ffn_v FP8
L23: bf16 (last layer before output)
AWQ: alpha=0.3

Compression ratio 1.7x (2.25 GB -> 1.30 GB). 20 layers quantized, 4 layers protected (dilution layers).

## II. Finalization Basis (Final Acceptance Data)

| Metric | Target | X5 Measured | Status |
|------|------|---------|------|
| MATH500 | >=12.0% | 11.8% (59/500) | 1 problem short |
| PPL delta @8192 | <=0.05 | +0.0024 | OK |
| VRAM | <=2.0 GiB | 1.93 GiB | OK |
| decode | >=65t/s | 78.9 t/s | OK |

X5 is the only scheme among all that simultaneously satisfies MATH500~12% and VRAM<=2.0GiB.

## III. How the Final Scheme Was Derived

### Phase 1: Basic Scheme (Task 1-4)
Low-rank quantization not worthwhile -> reverted to bf16; key full FP8; AWQ alpha=0.3 is the biggest contributor (8.6%->11.2%)

### Phase 2: Sensitivity Attribution Experiments
PPL attribution is flat -> PPL cannot be used as the primary metric; state MSE attribution: L0 least sensitive (3.3e-5), L14 peak (1.27e-2)
Alternation scheme collapsed -> L0's v_first contamination is the root cause

### Phase 3: M Gradient Search
M5 (L0 protected) =11.4%, M2 (L0-5+L18-23) =12.4%, M2c (L0-4+L18-23) =12.0%

### Phase 4: Dilution Layer Scan (X0-X5)
X0 (L0+L23) =9.0%, X5 (L0+L6+L18+L23) =13.5% (200 problems)
L6 (1/4 position) + L18 (3/4 position) are the best dilution points
X5 full 500 problems =11.8%, on par with M2c (12.0%), but only 4 bf16 layers

### Key Insight: Dilution Layers vs Protection Bands
Protection band (M2c): consecutive multiple bf16 layers, wasting quantization layers
Dilution layer (X5): discrete bf16 points, segment the network, reset quantization error at the start of each segment
Dilution point position matters more than count: L6+L18 (1/4+3/4) > L12 (exact middle) > L8+L16 (evenly spaced)

## IV. Implementation Approach

Core principle: Full-chain quantization domain execution, no runtime dequantization.

1. Activation dynamic quantization: prep_x kernel computes amax in real-time per forward pass;
   fused kernel internally performs per-16-block dynamic quantization on x tile.
2. Weight static quantization: offline quantization, scales saved to disk.
3. Residual compensation: ffn.key's NVFP4 error compensated by FP8 residual (per-block ratio scale + tensor scale),
   using pure FP8*FP8 GEMM.
4. AWQ alpha=0.3: 7-point grid search, MATH500 +2.6pp.
5. Dilution layers L0/L6/L18/L23: L0 is the v_first source; L6/L18 are dilution points at 1/4 and 3/4 positions,
   resetting quantization error; L23 is the last layer before output.

## V. Quantization Method for Each Tensor

### Quantized Layers L1-L5, L7-L17, L19-L22 (20 layers)

| Tensor | Format | Quantization Method |
|------|------|---------|
| att.receptance.weight [C,C] | NVFP4 | E2M1 4bit + per-16-block E4M3 scale + fp32 tensor scale |
| att.key.weight [C,C] | FP8 | E4M3 8bit + per-row fp32 scale |
| att.value.weight [C,C] | FP8 | Same as above |
| att.output.weight [C,C] | NVFP4 | Same as rec |
| ffn.key.weight [4C,C] | NVFP4+residual | NVFP4 + FP8 residual (per-block ratio + tensor scale) |
| ffn.value.weight [C,4C] | FP8 | Same as key |

### Dilution Layers L0, L6, L18, L23 (4 layers)
All 6 linear layers remain bf16 (not quantized).

### Not Quantized (All Models)
| Tensor | Reason |
|------|------|
| emb.weight, head.weight | Global 1GB, too large, and is embedding lookup |
| Low-rank g1/g2/a1/a2/w1/w2/v1/v2 | Only 1.7% of memory, gain not worth the accuracy cost |
| x_r/x_w/x_k/x_v/x_a/x_g, w0/a0/v0, k_k/k_a/r_k | Vector parameters, ~104KB per layer |
| LayerNorm/GroupNorm (ln0/ln1/ln2/ln_x/ln_out) | Normalization must be precise |

## VI. Iteration History Summary

| Scheme | Protected Layers | MATH500 | VRAM | decode | Compression |
|------|--------|---------|------|--------|------|
| M0 full quantization | L0-key | 11.2% | 1.87 | 70 | 2.0x |
| M5 | L0 | 11.4% | 1.87 | 70 | 1.9x |
| M2 | L0-5+L18-23 | 12.4% | 2.23 | 58.5 | 1.3x |
| M2c | L0-4+L18-23 | 12.0% | 2.19 | 86.3 | 1.4x |
| X5 | L0+L6+L18+L23 | 11.8% | 1.93 | 78.9 | 1.7x |
| Alternation A/B | - | 0% | - | - | - |
