#!/usr/bin/env python3
"""Benchmark: NVFP4 W4A16 (weight-only quantization, FP16 activation).

Tests the hypothesis that activation quantization (FP4) was the main precision
bottleneck in W4A4. By keeping activations at FP16 and only quantizing weights
to NVFP4, we should see dramatically improved Top-1 agreement.

Uses the same mixed model file (NVFP4 key + FP8 value), but with:
- NVFP4 key: W4A16 path (dequantize weight → BF16, FP16 GEMM)
- FP8 value: W8A8 path (unchanged, _scaled_mm with online activation quant)

Comparison points:
- #2 Mixed+Fused (W4A4): PPL delta 0.0176, Top-1 96.90%
- #3 FP8-Att (W8A16):    PPL delta 0.0009, Top-1 99.71%
"""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")

import torch
import json
import time
import math
import os

import rwkv7_fast_v3a as engine
from rwkv7_fast_v3a import RWKV7, load_extensions, DTYPE

# ============================================================================
# Configuration: enable W4A16 mode
# ============================================================================
MIXED_MODEL = "/home/njzy/model/rwkv7-g1h-2.9b-mixed-nvfp4-fp8.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"

engine.NVFP4_W4A16 = True   # KEY: enable weight-only NVFP4 (no activation quantization)
engine.WKV_MODE = "fp16"
engine.EMB_DEVICE = "cpu"
engine.RKV_MODE = "off"
engine.CMIX_SPARSE = "no-fc"
engine.LOWRANK_WEIGHT = "both"
engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}

print("=" * 80)
print("  Benchmark: NVFP4 W4A16 (Weight-Only Quantization, FP16 Activation)")
print("=" * 80)
print(f"  NVFP4_W4A16 = {engine.NVFP4_W4A16}")
print(f"  Model: {os.path.basename(MIXED_MODEL)}")

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
print(f"\n  Load time: {load_time:.1f}s")
print(f"  VRAM after load: {vram_after_load:.2f} GiB")

# Verify W4A16 is active
sample_key = "blocks.0.ffn.key.weight"
w = model.z[sample_key]
if isinstance(w, dict):
    print(f"  FFN key qtype: {w['qtype']} (expect nvfp4_w4a16)")
    print(f"  FFN key weight: {w['weight'].shape} {w['weight'].dtype}")
    print(f"  FFN key block_scale: {w['block_scale'].shape} {w['block_scale'].dtype}")

sample_val = "blocks.0.ffn.value.weight"
wv = model.z[sample_val]
if isinstance(wv, dict):
    print(f"  FFN value qtype: {wv['qtype']} (expect fp8)")

# Load test tokens
with open(f"{EVAL_DIR}/test_446.json") as f:
    tokens_446 = json.load(f)["tokens"]
with open(f"{EVAL_DIR}/test_2100.json") as f:
    tokens_2100 = json.load(f)["tokens"]

# ============================================================================
# Speed benchmarks
# ============================================================================
def bench_prefill(model, tokens, n_warmup=3, n_iter=5):
    T = len(tokens)
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    for _ in range(n_warmup):
        state = model.zero_state(1)
        model.forward(tok_tensor, state)
    torch.cuda.synchronize()
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
    return {"T": T, "avg_ms": avg * 1000, "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000, "toks_per_s": T / avg}

def bench_decode(model, tokens, n_warmup=10, n_iter=100):
    n_decode = min(len(tokens), n_iter + n_warmup)
    state = model.zero_state(1)
    for i in range(min(n_warmup, n_decode)):
        tok = torch.tensor([[tokens[i]]], dtype=torch.long)
        model.forward(tok, state)
    torch.cuda.synchronize()
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
    return {"n_tokens": n_measure, "elapsed_ms": elapsed * 1000,
            "toks_per_s": n_measure / elapsed, "ms_per_tok": elapsed / n_measure * 1000}

print("\n" + "=" * 80)
print("  Speed Benchmarks")
print("=" * 80)

prefill_results = {}
for T_test in [20, 128, 446, 2100]:
    tokens = tokens_446[:T_test] if T_test <= 446 else tokens_2100[:T_test]
    r = bench_prefill(model, tokens)
    prefill_results[T_test] = r
    print(f"\n  Prefill T={T_test:4d}: avg={r['avg_ms']:.1f}ms  min={r['min_ms']:.1f}ms  "
          f"max={r['max_ms']:.1f}ms  throughput={r['toks_per_s']:.0f} tok/s")

decode_r = bench_decode(model, tokens_446, n_warmup=10, n_iter=100)
print(f"\n  Decode (b1t1): {decode_r['n_tokens']} tokens in {decode_r['elapsed_ms']:.1f}ms  "
      f"throughput={decode_r['toks_per_s']:.0f} tok/s  latency={decode_r['ms_per_tok']:.2f} ms/tok")

vram_peak = torch.cuda.max_memory_allocated() / 2**30
print(f"\n  VRAM: after_load={vram_after_load:.2f} GiB, peak={vram_peak:.2f} GiB")

# ============================================================================
# Quality benchmarks
# ============================================================================
print("\n" + "=" * 80)
print("  Quality Benchmarks: Logits Comparison")
print("=" * 80)

def generate_logits(model, tokens, label):
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = model.zero_state(1)
    out = model.forward_all_logits(tok_tensor, state)
    logits = out[0].float().cpu()
    print(f"  [{label}] Generated {logits.shape[0]} logits, shape={logits.shape}")
    return logits

def compare_logits(orig_path, quant_logits, tokens_path, label):
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
    return {"label": label, "n": n, "mse": mse, "max_diff": max_diff,
            "top1_agree": top1_agree, "ce_o": ce_o, "ce_q": ce_q,
            "ce_delta": ce_q - ce_o, "ppl_o": ppl_o, "ppl_q": ppl_q,
            "ppl_delta": ppl_q - ppl_o, "mean_kl": kl.mean().item()}

print("\n  Generating logits with W4A16 model...")
logits_446 = generate_logits(model, tokens_446, "w4a16_446")
logits_2100 = generate_logits(model, tokens_2100, "w4a16_2100")

# Save logits
torch.save(logits_446, f"{EVAL_DIR}/logits_w4a16_446.pt")
torch.save(logits_2100, f"{EVAL_DIR}/logits_w4a16_2100.pt")

# Compare with original
r_446 = compare_logits(f"{EVAL_DIR}/logits_orig_446.pt", logits_446, f"{EVAL_DIR}/test_446.json", "W4A16 446")
r_2100 = compare_logits(f"{EVAL_DIR}/logits_orig_2100.pt", logits_2100, f"{EVAL_DIR}/test_2100.json", "W4A16 2100")

# Print quality results
print(f"\n{'='*100}")
print(f"  Quality Results: 446 tokens")
print(f"{'='*100}")
print(f"  {'Model':<25} {'PPL':<10} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Mean KL':<14} {'Max diff':<12}")
print(f"  {'-'*95}")
print(f"  {'Original':<25} {r_446['ppl_o']:<10.4f} {'—':<12} {'—':<10} {'—':<12} {'—':<14} {'—':<12}")
print(f"  {r_446['label']:<25} {r_446['ppl_q']:<10.4f} {r_446['ppl_delta']:<12.4f} {r_446['top1_agree']*100:<10.2f} {r_446['ce_delta']:<12.6f} {r_446['mean_kl']:<14.10f} {r_446['max_diff']:<12.6f}")

print(f"\n{'='*100}")
print(f"  Quality Results: 2100 tokens")
print(f"{'='*100}")
print(f"  {'Model':<25} {'PPL':<10} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Mean KL':<14} {'Max diff':<12}")
print(f"  {'-'*95}")
print(f"  {'Original':<25} {r_2100['ppl_o']:<10.4f} {'—':<12} {'—':<10} {'—':<12} {'—':<14} {'—':<12}")
print(f"  {r_2100['label']:<25} {r_2100['ppl_q']:<10.4f} {r_2100['ppl_delta']:<12.4f} {r_2100['top1_agree']*100:<10.2f} {r_2100['ce_delta']:<12.6f} {r_2100['mean_kl']:<14.10f} {r_2100['max_diff']:<12.6f}")

# Acceptance check
print(f"\n{'='*100}")
print(f"  Acceptance (2100 tokens)")
print(f"{'='*100}")
print(f"  {r_2100['label']}:")
print(f"    PPL delta <= 0.05:     {'PASS' if abs(r_2100['ppl_delta']) <= 0.05 else 'FAIL'} ({abs(r_2100['ppl_delta']):.4f})")
print(f"    Top-1 agree >= 99.5%:  {'PASS' if r_2100['top1_agree'] >= 0.995 else 'FAIL'} ({r_2100['top1_agree']*100:.2f}%)")

# Comparison table
print(f"\n{'='*100}")
print(f"  Comparison: W4A4 vs W4A16 (FFN NVFP4+FP8 mixed model)")
print(f"{'='*100}")
print(f"  {'Mode':<20} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Prefill 2100':<14} {'Decode':<10}")
print(f"  {'-'*80}")

# Load W4A4 results for comparison
w4a4_path = f"{EVAL_DIR}/bench_fused_results.json"
if os.path.exists(w4a4_path):
    with open(w4a4_path) as f:
        w4a4 = json.load(f)
    print(f"  {'W4A4 (fused)':<20} {w4a4['quality_2100']['ppl_delta']:<12.4f} {w4a4['quality_2100']['top1_agree']*100:<10.2f} {w4a4['quality_2100']['ce_delta']:<12.6f} {w4a4['prefill']['2100']['toks_per_s']:<14.0f} {w4a4['decode']['toks_per_s']:<10.0f}")

print(f"  {'W4A16 (this)':<20} {r_2100['ppl_delta']:<12.4f} {r_2100['top1_agree']*100:<10.2f} {r_2100['ce_delta']:<12.6f} {prefill_results[2100]['toks_per_s']:<14.0f} {decode_r['toks_per_s']:<10.0f}")

# Save results
results = {
    "model": MIXED_MODEL,
    "mode": "W4A16",
    "load_time_s": load_time,
    "vram_gib": vram_after_load,
    "vram_peak_gib": vram_peak,
    "prefill": {str(T): {"toks_per_s": r["toks_per_s"], "avg_ms": r["avg_ms"]} for T, r in prefill_results.items()},
    "decode": {"toks_per_s": decode_r["toks_per_s"], "ms_per_tok": decode_r["ms_per_tok"]},
    "quality_2100": {"ppl_delta": r_2100["ppl_delta"], "top1_agree": r_2100["top1_agree"], "ce_delta": r_2100["ce_delta"]},
}
with open(f"{EVAL_DIR}/bench_w4a16_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {EVAL_DIR}/bench_w4a16_results.json")
