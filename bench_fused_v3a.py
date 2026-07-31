#!/usr/bin/env python3
"""Comprehensive benchmark: fused NVFP4 kernel in v3a inference.

Tests:
1. Prefill (b1tn) speed at various sequence lengths
2. Decode (b1t1) speed
3. VRAM usage
4. Logits quality (PPL, Top-1 agreement, CE delta) vs original
"""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")

import torch
import json
import time
import math
import os

from rwkv7_fast_v3a import RWKV7, load_extensions, DTYPE, select_path
import rwkv7_fast_v3a as engine

# ============================================================================
# Configuration
# ============================================================================
MIXED_MODEL = "/home/njzy/model/rwkv7-g1h-2.9b-mixed-nvfp4-fp8.pth"
ORIG_MODEL = "/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"

engine.WKV_MODE = "fp16"
engine.EMB_DEVICE = "cpu"
engine.RKV_MODE = "off"
engine.CMIX_SPARSE = "no-fc"
engine.LOWRANK_WEIGHT = "both"
engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}

print("=" * 80)
print("  Comprehensive Benchmark: Fused NVFP4 Kernel + Mixed Model")
print("=" * 80)

# ============================================================================
# Load model
# ============================================================================
load_extensions(engine.WKV_MODE)

t0 = time.time()
engine.MODEL_PATH = MIXED_MODEL
model = RWKV7()
t1 = time.time()
load_time = t1 - t0
vram_after_load = torch.cuda.memory_allocated() / 2**30
print(f"\nModel: {MIXED_MODEL}")
print(f"Load time: {load_time:.1f}s")
print(f"VRAM after load: {vram_after_load:.2f} GiB")

# Load test tokens
with open(f"{EVAL_DIR}/test_446.json") as f:
    tokens_446 = json.load(f)["tokens"]
with open(f"{EVAL_DIR}/test_2100.json") as f:
    tokens_2100 = json.load(f)["tokens"]

print(f"Test tokens: 446 and 2100")

# ============================================================================
# Speed benchmarks
# ============================================================================
def bench_prefill(model, tokens, n_warmup=3, n_iter=5):
    """Benchmark prefill (b1tn) path."""
    T = len(tokens)
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    
    # Warmup
    for _ in range(n_warmup):
        state = model.zero_state(1)
        model.forward(tok_tensor, state)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(n_iter):
        state = model.zero_state(1)
        torch.cuda.synchronize()
        t0 = time.time()
        model.forward(tok_tensor, state)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    return {
        "T": T, "avg_ms": avg * 1000, "min_ms": min_t * 1000, "max_ms": max_t * 1000,
        "toks_per_s": T / avg, "times": times
    }

def bench_decode(model, tokens, n_warmup=10, n_iter=100):
    """Benchmark decode (b1t1) path."""
    n_decode = min(len(tokens), n_iter + n_warmup)
    
    # Warmup
    state = model.zero_state(1)
    for i in range(min(n_warmup, n_decode)):
        tok = torch.tensor([[tokens[i]]], dtype=torch.long)
        model.forward(tok, state)
    torch.cuda.synchronize()
    
    # Measure
    start_idx = n_warmup
    n_measure = min(n_iter, n_decode - n_warmup)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(start_idx, start_idx + n_measure):
        tok = torch.tensor([[tokens[i]]], dtype=torch.long)
        model.forward(tok, state)
    torch.cuda.synchronize()
    t1 = time.time()
    
    elapsed = t1 - t0
    return {
        "n_tokens": n_measure, "elapsed_ms": elapsed * 1000,
        "toks_per_s": n_measure / elapsed,
        "ms_per_tok": elapsed / n_measure * 1000
    }

print("\n" + "=" * 80)
print("  Speed Benchmarks")
print("=" * 80)

# Prefill benchmarks
prefill_results = {}
for T_test in [20, 128, 446, 2100]:
    tokens = tokens_446[:T_test] if T_test <= 446 else tokens_2100[:T_test]
    r = bench_prefill(model, tokens)
    prefill_results[T_test] = r
    print(f"\n  Prefill T={T_test:4d}: avg={r['avg_ms']:.1f}ms  min={r['min_ms']:.1f}ms  "
          f"max={r['max_ms']:.1f}ms  throughput={r['toks_per_s']:.0f} tok/s")

# Decode benchmark
print()
decode_r = bench_decode(model, tokens_446, n_warmup=10, n_iter=100)
print(f"  Decode (b1t1): {decode_r['n_tokens']} tokens in {decode_r['elapsed_ms']:.1f}ms  "
      f"throughput={decode_r['toks_per_s']:.0f} tok/s  latency={decode_r['ms_per_tok']:.2f} ms/tok")

# VRAM after benchmarks
vram_peak = torch.cuda.max_memory_allocated() / 2**30
print(f"\n  VRAM: after_load={vram_after_load:.2f} GiB, peak={vram_peak:.2f} GiB")

# ============================================================================
# Quality benchmarks: generate logits and compare with original
# ============================================================================
print("\n" + "=" * 80)
print("  Quality Benchmarks: Logits Comparison")
print("=" * 80)

def generate_logits(model, tokens, label):
    """Generate logits using forward_all_logits."""
    T = len(tokens)
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = model.zero_state(1)
    
    # Use forward_all_logits to get [B, T, V]
    out = model.forward_all_logits(tok_tensor, state)
    
    # Save: out is [1, T, V], squeeze to [T, V]
    logits = out[0].float().cpu()
    print(f"  [{label}] Generated {logits.shape[0]} logits, shape={logits.shape}")
    return logits

def compare_logits(orig_path, quant_logits, tokens_path, label):
    """Compare quantized logits with original."""
    lo = torch.load(orig_path, map_location="cpu").float()
    lq = quant_logits
    
    with open(tokens_path) as f:
        tokens = json.load(f)["tokens"]
    n = min(lo.shape[0], lq.shape[0])
    lo = lo[:n]
    lq = lq[:n]
    targets = torch.tensor(tokens[1:n+1], dtype=torch.long)
    
    diff = lo - lq
    mse = (diff ** 2).mean().item()
    max_diff = diff.abs().max().item()
    
    top1_o = lo.argmax(dim=-1)
    top1_q = lq.argmax(dim=-1)
    top1_agree = (top1_o == top1_q).float().mean().item()
    
    ce_o = torch.nn.functional.cross_entropy(lo, targets, reduction="mean").item()
    ce_q = torch.nn.functional.cross_entropy(lq, targets, reduction="mean").item()
    ppl_o = math.exp(ce_o)
    ppl_q = math.exp(ce_q)
    
    kl = (torch.softmax(lq, dim=-1) * (torch.log_softmax(lq, dim=-1) - torch.log_softmax(lo, dim=-1))).sum(dim=-1)
    
    return {
        "label": label, "n": n,
        "mse": mse, "max_diff": max_diff,
        "top1_agree": top1_agree,
        "ce_o": ce_o, "ce_q": ce_q, "ce_delta": ce_q - ce_o,
        "ppl_o": ppl_o, "ppl_q": ppl_q, "ppl_delta": ppl_q - ppl_o,
        "mean_kl": kl.mean().item(),
    }

# Generate logits with fused kernel
print("\n  Generating logits with fused kernel...")
logits_446_fused = generate_logits(model, tokens_446, "fused_446")
logits_2100_fused = generate_logits(model, tokens_2100, "fused_2100")

# Compare with original
r_446 = compare_logits(f"{EVAL_DIR}/logits_orig_446.pt", logits_446_fused, f"{EVAL_DIR}/test_446.json", "Mixed+Fused 446")
r_2100 = compare_logits(f"{EVAL_DIR}/logits_orig_2100.pt", logits_2100_fused, f"{EVAL_DIR}/test_2100.json", "Mixed+Fused 2100")

# Also load previous mixed logits (without fused kernel) for comparison
print("\n  Loading previous mixed logits (without fused kernel)...")
logits_446_prev = torch.load(f"{EVAL_DIR}/logits_mixed_446.pt", map_location="cpu").float()
logits_2100_prev = torch.load(f"{EVAL_DIR}/logits_mixed_2100.pt", map_location="cpu").float()

# Save fused logits for future reference
torch.save(logits_446_fused, f"{EVAL_DIR}/logits_fused_446.pt")
torch.save(logits_2100_fused, f"{EVAL_DIR}/logits_fused_2100.pt")
print(f"  Fused logits saved to {EVAL_DIR}/")

# Compare previous mixed with original
r_446_prev = compare_logits(f"{EVAL_DIR}/logits_orig_446.pt", logits_446_prev, f"{EVAL_DIR}/test_446.json", "Mixed-prev 446")
r_2100_prev = compare_logits(f"{EVAL_DIR}/logits_orig_2100.pt", logits_2100_prev, f"{EVAL_DIR}/test_2100.json", "Mixed-prev 2100")

# Compare fused vs previous (should be nearly identical - same algorithm, just faster)
n446 = min(logits_446_fused.shape[0], logits_446_prev.shape[0])
n2100 = min(logits_2100_fused.shape[0], logits_2100_prev.shape[0])
diff_fused_prev_446 = (logits_446_fused[:n446] - logits_446_prev[:n446]).abs()
diff_fused_prev_2100 = (logits_2100_fused[:n2100] - logits_2100_prev[:n2100]).abs()
print(f"\n  Fused vs prev mixed logits diff (446):  max={diff_fused_prev_446.max().item():.8f}  mean={diff_fused_prev_446.mean().item():.10f}")
print(f"  Fused vs prev mixed logits diff (2100): max={diff_fused_prev_2100.max().item():.8f}  mean={diff_fused_prev_2100.mean().item():.10f}")

# Print quality results
print(f"\n{'='*100}")
print(f"  Quality Results: 446 tokens")
print(f"{'='*100}")
print(f"  {'Model':<25} {'PPL':<10} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Mean KL':<14} {'Max diff':<12}")
print(f"  {'-'*95}")
print(f"  {'Original':<25} {r_446['ppl_o']:<10.4f} {'—':<12} {'—':<10} {'—':<12} {'—':<14} {'—':<12}")
for r in [r_446_prev, r_446]:
    print(f"  {r['label']:<25} {r['ppl_q']:<10.4f} {r['ppl_delta']:<12.4f} {r['top1_agree']*100:<10.2f} {r['ce_delta']:<12.6f} {r['mean_kl']:<14.10f} {r['max_diff']:<12.6f}")

print(f"\n{'='*100}")
print(f"  Quality Results: 2100 tokens")
print(f"{'='*100}")
print(f"  {'Model':<25} {'PPL':<10} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Mean KL':<14} {'Max diff':<12}")
print(f"  {'-'*95}")
print(f"  {'Original':<25} {r_2100['ppl_o']:<10.4f} {'—':<12} {'—':<10} {'—':<12} {'—':<14} {'—':<12}")
for r in [r_2100_prev, r_2100]:
    print(f"  {r['label']:<25} {r['ppl_q']:<10.4f} {r['ppl_delta']:<12.4f} {r['top1_agree']*100:<10.2f} {r['ce_delta']:<12.6f} {r['mean_kl']:<14.10f} {r['max_diff']:<12.6f}")

# Acceptance check (2100 tokens)
print(f"\n{'='*100}")
print(f"  Acceptance (2100 tokens)")
print(f"{'='*100}")
for r in [r_2100_prev, r_2100]:
    print(f"  {r['label']}:")
    print(f"    PPL delta <= 0.05:     {'PASS' if abs(r['ppl_delta']) <= 0.05 else 'FAIL'} ({abs(r['ppl_delta']):.4f})")
    print(f"    Top-1 agree >= 99.5%:  {'PASS' if r['top1_agree'] >= 0.995 else 'FAIL'} ({r['top1_agree']*100:.2f}%)")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*100}")
print(f"  Summary")
print(f"{'='*100}")
print(f"  Model: {os.path.basename(MIXED_MODEL)}")
print(f"  Load time: {load_time:.1f}s")
print(f"  VRAM: {vram_after_load:.2f} GiB (loaded), {vram_peak:.2f} GiB (peak)")
print(f"\n  Speed (fused kernel):")
for T, r in prefill_results.items():
    print(f"    Prefill T={T:4d}: {r['toks_per_s']:.0f} tok/s ({r['avg_ms']:.1f}ms)")
print(f"    Decode:      {decode_r['toks_per_s']:.0f} tok/s ({decode_r['ms_per_tok']:.2f}ms/tok)")
print(f"\n  Quality (2100 tokens, fused kernel):")
print(f"    PPL delta:    {r_2100['ppl_delta']:.4f}")
print(f"    Top-1 agree:  {r_2100['top1_agree']*100:.2f}%")
print(f"    CE delta:     {r_2100['ce_delta']:.6f}")
print(f"    Max logits diff: {r_2100['max_diff']:.6f}")

# Save results
results = {
    "model": MIXED_MODEL,
    "load_time_s": load_time,
    "vram_gib": vram_after_load,
    "vram_peak_gib": vram_peak,
    "prefill": {str(T): {"toks_per_s": r["toks_per_s"], "avg_ms": r["avg_ms"]} for T, r in prefill_results.items()},
    "decode": {"toks_per_s": decode_r["toks_per_s"], "ms_per_tok": decode_r["ms_per_tok"]},
    "quality_2100": {
        "ppl_delta": r_2100["ppl_delta"],
        "top1_agree": r_2100["top1_agree"],
        "ce_delta": r_2100["ce_delta"],
    },
}

with open(f"{EVAL_DIR}/bench_fused_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {EVAL_DIR}/bench_fused_results.json")
print(f"\n{'='*80}")
print(f"  BENCHMARK COMPLETE")
print(f"{'='*80}")
