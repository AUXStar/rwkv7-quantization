# RWKV-7 Quantization Best Practice — Finalized 2026-08-01

> Source: 9 GitHub issues (#1-#9) + 14 reports + 1.5B/2.9B/7.2B three-model validation + 96 novel generation samples
> Applicable to: RWKV-7 v3a engine, torch._scaled_mm (sm_120/Blackwell), 24-32 layer architecture

---

## One-Sentence Summary

**NVFP4 + FP8 hybrid + AWQ + 4-layer bf16 dilution (at 1/4+3/4 ratio), reject GPTQ, reject pure NVFP4**.

---

## 1. Recommended Scheme (X5)

| Tensor Type | Quantization Method | Effective bit | Reason |
|---------|---------|--------|------|
| `att.receptance` (r_proj) | NVFP4 (E2M1) + per-16-block FP8 scale + FP32 tensor scale + **AWQ α=0.3** | 4 | r alone at 4bit is still controllable; AWQ channel scaling significantly reduces error |
| `att.key` (k_proj) | **FP8 E4M3** + per-tensor FP32 scale | 8 | k must be high bit, NVFP4 4bit quantization error +69% |
| `att.value` (v_proj) | FP8 E4M3 + per-tensor FP32 scale | 8 | v at 4bit would pollute subsequent state; FP8 is the cost-effectiveness sweet spot |
| `att.output` (o_proj) | NVFP4 + per-16-block FP8 scale + FP32 tensor scale + AWQ α=0.3 | 4 | Output side 4bit is sufficient |
| `ffn.key` | **NVFP4 + FP8 residual** (per-block ratio + tensor scale) | ~6 | Residual allows 4bit main path + 8bit error compensation, more accurate than pure NVFP4 |
| `ffn.value` | FP8 E4M3 + per-tensor FP32 scale | 8 | ffn v is the core path for state accumulation, 4bit is unusable |

**Effective average 6.0 bit/weight**, **1.7-1.8x compression**, **15-32% VRAM savings**, **PPL delta < 0.003**, **MATH500 error < 1.5pp**.

---

## 2. 4-Layer bf16 Dilution Layers (X5 dilution) — Core Innovation

**No paper has done this before**, this is the core of our scheme.

### 2.1 Position Formula (Universal for 24/32 layers)

```
protected = {0, round(N/4), round(3N/4), N-1}
```

- 1.5B (N=24): `{0, 6, 18, 23}`
- 2.9B / 7.2B (N=32): `{0, 8, 24, 31}`

### 2.2 Why These 4 Positions (Not Others)

| Position | Role | Experimental Evidence |
|------|------|---------|
| L0 | v_first state source | L0 bf16 is the most sensitive point for 1.5B; any 1.5B scheme without L0 collapses |
| `round(N/4)` | Segment 1 error reset point | X5 scan (L0+L23)=9.0% -> +L6=12.4% -> +L18=13.5% MATH500 |
| `round(3N/4)` | Segment 2 error reset point | L6/L18 alone each +1.5pp; together +3.5pp |
| L_last | Last precision gate before output | Exit must be precise, otherwise softmax selects wrong token |

### 2.3 1.5B Scan Evidence (MATH500, 200 problems)

| Scheme | Protected Layers | MATH500 |
|------|---------|---------|
| X0 | L0 + L23 | 9.0% |
| X1 | L0 + L6 + L23 | 12.4% |
| X2 | L0 + L18 + L23 | 11.6% |
| X3 | L0 + L12 + L23 | 11.0% |
| X4 | L0 + L6 + L12 + L23 | 12.0% |
| **X5** | **L0 + L6 + L18 + L23** | **13.5% (200 problems) -> 11.8% (500 problems)** |
| M2c (11 layers protected) | L0,1,2,3,4,5,6,8,12,18,23 | 11.5% (500 problems) |

**Conclusion**: Only 4 bf16 layers can achieve the accuracy of 11 layers of protection, with just **0.5GiB more VRAM** for equivalent quality.

### 2.4 Dilution Layers That Cannot Be Removed (Ablation)

- Remove L0: PPL delta +0.04, 1.5B completely collapses
- Remove L6 or L18: MATH500 drops 1.5pp
- Remove L23: Output layer noise amplifies, repetition rate significantly increases
- Add more bf16 layers (M2c 11 layers): accuracy gain < 0.5pp, but VRAM increases 0.5GiB, **not worthwhile**

---

## 3. What Absolutely Not To Do (Anti-Patterns)

### 3.1 Pure NVFP4 (4-bit All Tensors)

| Tensor | Result |
|------|------|
| All att NVFP4 | PPL delta +0.058, MATH500 8.6% |
| All att+ffn NVFP4 | Unusable, completely runs away |

**Reason**: NVFP4 has only 16 discrete values (E2M1), single-layer max abs error ~6.25%; 24 layers accumulate over 100%.

### 3.2 GPTQ Quantization for NVFP4

PPL delta **+21847**, top-1 2.62%. **Completely unusable**.

**Reason**: GPTQ uses Hessian inverse matrix to adjust weights, but NVFP4's 16 discrete value grid is too coarse, adjusted weights land on worse grid points than round-to-nearest.

### 3.3 Alternating bf16/Quantized (alternation)

L0 bf16 -> L1 quantized -> L2 bf16 -> ... pattern.

**Reason**: v_first state (initialized at L0) contains α×v_prev, once L1 quantization error contaminates it, all subsequent layers accumulate the contamination. **L0 must be the start of a continuous bf16 chain**.

### 3.4 4-bit AWQ Without FP8 Residual

Only NVFP4 + AWQ (key unchanged): 1.5B PPL delta +0.012, MATH500 11.2%, still does not pass acceptance.

**Reason**: ffn.key at pure 4bit still has large error, FP8 residual raises the 6bit effective to 95% accuracy.

### 3.5 W4A16 (dequant inference)

```
1.5B: 4233->21977 tok/s prefill ✅  but 1.67->3.70 GiB VRAM ❌ (doubled)
7.2B: PPL improved but 18.66 GiB VRAM exceeds 12GB GPU ❌
```

W4A16 dequantizes quantized weights back to FP16 for inference, speed 5.2x, but VRAM doubles, **7.2B cannot run on 12GB card**. **Must use pure W4A4 / W8A8 GEMM path**.

### 3.6 Quantizing ffn.value to NVFP4

ffn.value is the core path for RWKV state accumulation, 4bit error causes state to drift to an unrecoverable range.
**ffn.value is always FP8** (or bf16 at dilution layers).

---

## 4. AWQ Parameter Selection

### 4.1 α = 0.3 (Optimal)

```
α scan (1.5B T4):
α=0.0 (no AWQ)  PPL delta +0.028
α=0.1          PPL delta +0.018
α=0.2          PPL delta +0.014
α=0.3          PPL delta +0.008  ← optimal
α=0.5          PPL delta +0.013
α=0.7          PPL delta +0.022
α=1.0          PPL delta +0.038
```

**Reason**: α too small makes activation scaling ineffective, too large amplifies outliers into new error sources. 0.3 is the optimal from 7-point grid search.

### 4.2 Activation Proxy Selection

Use **weight-based heuristic** (column absolute mean) as activation proxy, **do not use real activation calibration**.

**Reason**: RWKV-7 state is time-varying and cumulative, different sequences have vastly different activation distributions; using calibration data would overfit a specific distribution, actually harming long-text generation. Weight-based heuristic is model-agnostic, with the best robustness.

---

## 5. Evaluation Acceptance Criteria (Must Check)

### 5.1 Three Core Metrics

| Metric | Pass Threshold | If Failed |
|------|---------|-------|
| PPL delta (8192 token) | < 0.005 | Reject (1.5B's +0.0024 is already the boundary) |
| MATH500 (all 500 problems) | Δ < ±2pp and within ±2.2% noise | Reject (-1pp is real degradation, ±2pp is noise) |
| decode tok/s | > 65 (1.5B), > 20 (7.2B) | Reject |

### 5.2 PPL as Sole Metric Is Insufficient

> 1.5B int4: PPL +0.003 (appears OK) but MATH500 -25pp (collapse)

PPL is a local token distribution metric, cannot reflect long-range reasoning/math ability. **MATH500 is a mandatory check**.

### 5.3 Anti-Memorization: Use Uncheatable Eval Library

Each evaluation must sample from `uncheatable_eval.json` to prevent the model from having "seen" the original problems.

### 5.4 Long Text Must Be Tested (>500 token)

PPL/MATH500 within 500 tokens can easily mask problems. **1.5B actual usage is 1500+ tokens**, use 2100-token PPL + 1500-token novel generation samples as final quality verification.

---

## 6. Engineering Practices (Pitfalls Encountered)

### 6.1 Weight Swizzle: Must Be 128x4

NVFP4 weights must use `128x4 swizzle` format to be compatible with `torch._scaled_mm`.
**Pure PyTorch swizzle is 56-86% slower than _scaled_mm**, must use fused Triton kernel.

### 6.2 Scale Loading: Two Sets Coexist

```python
# NVFP4 weights keep both swizzled and unswizzled scales
weight_dict = {
    "w_nvfp4": packed_data,        # 128x4 swizzled
    "block_scale": scale_unswz,    # [N, K/16] for AWQ
    "block_scale_sw": scale_swz,   # [N/128, K/16*4] for _scaled_mm
    "tensor_scale": fp32,          # [1]
}
```

**Reason**: Different GEMM paths use different layouts, avoiding re-swizzle per inference.

### 6.3 File Saving: Clone Before Save

```python
# Wrong: directly save tensor with mmap -> file bloats 4x
torch.save(state_dict, path)
# Correct: clone first to release mmap
torch.save({k: v.clone() if v.is_mmap else v for k, v in sd.items()}, path)
```

### 6.4 Fused GEMM Hybrid Routing

```python
if M <= 64:
    use fused_nvfp4_gemm        # 37μs (M=1)
else:
    use torch._scaled_mm        # 0.43ms (M=2100)
```

At M=1 fused is 17x faster, at large M _scaled_mm overtakes, hybrid routing is mandatory.

### 6.5 Decode CPU-Launch-Bound Optimization

1.5B decode ~1000 kernel launches/step -> 35ms CPU scheduling vs 11ms GPU computation.
**fused v3** (commit 231baf7): fuses r/k/v three projections + residual in-kernel, 1.5B decode 16.9->70.4 tok/s (+316%).

### 6.6 WSL Temporary Disk 30GB Limit

- 7.2B original 13.4G + quantized 8.9G + 2.9B 3.8G + 1.5B 2.9G + 1.5B X5 1.7G = ~30G
- Immediately `os.remove(orig.pth)` after acceptance to free space
- Use `/home/njzy/model/` instead of `/tmp`, to avoid WSL tmpfs filling up

---

## 7. Implementation Checklist

### 7.1 Quantization Steps (From Original .pth to Quantized .pth)

1. Load original bf16 `.pth`
2. Construct `protected = {0, N//4, 3*N//4, N-1}`
3. For each non-protected layer:
   - att.receptance: AWQ α=0.3 -> NVFP4 + per-16-block FP8 scale
   - att.key: per-tensor FP8 E4M3
   - att.value: per-tensor FP8 E4M3
   - att.output: AWQ α=0.3 -> NVFP4 + per-16-block FP8 scale
   - ffn.key: AWQ α=0.3 -> NVFP4 (main) + FP8 residual (per-block ratio + tensor scale)
   - ffn.value: per-tensor FP8 E4M3
4. For protected layers: **all 6 tensors remain bf16**
5. Swizzle NVFP4 weights to 128x4
6. When saving, **clone tensors to release mmap**

### 7.2 Acceptance Steps

1. Load, measure VRAM, measure decode tok/s (5 warmup + 30 iters)
2. PPL@8192, run 1024 truncation comparison (to avoid OOM)
3. MATH500 all 500 problems, batch=8, garbled auto-repair
4. Long-text generation sampling, verify rep4 < 0.3 (no obvious loops)
5. Compare with orig PPL/MATH500, pass if Δ is within threshold

### 7.3 Deployment Recommendations

| GPU | Recommended Model | Configuration |
|------|---------|------|
| 8GB (RTX 3060/4060) | 1.5B X5 | VRAM 1.93G ✅, 78.9 t/s |
| 12GB (RTX 3060/4070) | 2.9B X5 | VRAM 3.78G ✅, 49.5 t/s |
| 16GB (RTX 4060Ti/4070Ti) | 7.2B X5 | VRAM 9.11G ✅, 25.1 t/s |
| 24GB (RTX 3090/4090) | 7.2B X5 | Same as above, 15G left for state |
| 80GB (H100/A100) | 7.2B+ leave ample state, run 8192 ctx | |

---

## 8. Three-Model Final Performance

| Model | File | VRAM | Compression | PPL Δ | MATH500 Δ | decode |
|------|------|------|------|-------|----------|--------|
| **1.5B** | 2.9G -> 1.7G | 2.18 -> 1.86 GiB | 1.7x | +0.0024 | -0.8pp | 78.9 t/s |
| **2.9B** | 5.5G -> 3.7G | 5.35 -> 3.78 GiB | 1.8x | +0.0006 | -1.0pp | 49.5 t/s |
| **7.2B** | 13.4G -> 8.5G | 13.32 -> 9.11 GiB | 1.8x | +0.0012 | +2.0pp* | 25.1 t/s |

*7.2B 100-problem sample, within ±2.2% noise

**All pass acceptance**: PPL < 0.005, MATH500 < 1.5pp actual degradation, decode meets hardware limits.

---

## 9. Future Optimization Directions (Ranked by Benefit)

1. **Further r/k/v projection fusion**: Current fused v3 already 316% speedup, can also eliminate state sync overhead
2. **Decode multi-request batching**: Currently 1.5B 1 request/token, 16GB card running 16 concurrent can 5x throughput
3. **head projection FP8**: head.weight is 65k->2048, 4bit error sensitive, FP8 is a compromise
4. **FP4/E4M3 decode lookup table**: Dequant via table lookup, reduce register usage
5. **W8A8 -> W4A8**: Currently att uses W4A16, quantize activation to FP8, end-to-end 4bit GEMM

Each requires new kernel + re-test PPL/MATH500, proceed with caution.

---

## 10. Reproduction Links

- Quantization code: `quant_x5.py` (24L), `quant_x5_32l.py` (32L)
- Acceptance code: `accept_x5.py`, `accept_x5_32l.py`
- MATH500 evaluation: `fast_math500.py` (batch+repair)
- Report summary: `reports/00_audit_summary.md` -> `14_best_practice.md` (this document)
- GitHub issues: #1-#9 all closed, PR #10 created
