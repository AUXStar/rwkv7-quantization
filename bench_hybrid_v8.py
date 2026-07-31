#!/usr/bin/env python3
"""Benchmark: Hybrid v8 — AWQ+clip NVFP4 FFN key + BF16 value + FP8 W8A16 att."""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")

import torch
import json
import time
import math
import os

import rwkv7_fast_v3a as engine
from rwkv7_fast_v3a import RWKV7, load_extensions, DTYPE

HYBRID_MODEL = "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v8.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"

engine.NVFP4_W4A16 = True
engine.FP8_W8A16 = False
engine.WKV_MODE = "fp16"
engine.EMB_DEVICE = "cpu"
engine.RKV_MODE = "off"
engine.CMIX_SPARSE = "no-fc"
engine.LOWRANK_WEIGHT = "both"
engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}

print("=" * 80)
print("  Benchmark: Hybrid v8 (AWQ+clip NVFP4 W4A16 key + BF16 val + FP8 att)")
print("=" * 80)

load_extensions(engine.WKV_MODE)
t0 = time.time()
engine.MODEL_PATH = HYBRID_MODEL
model = RWKV7()
t1 = time.time()
load_time = t1 - t0
vram_after_load = torch.cuda.memory_allocated() / 2**30
print(f"\n  Load time: {load_time:.1f}s, VRAM: {vram_after_load:.2f} GiB")

for k in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight"]:
    w = model.z[k]
    if isinstance(w, dict):
        bs = w['block_scale']
        ts = w['tensor_scale']
        ts_info = f"scalar={ts.item():.6f}" if ts.dim() == 0 else f"per-channel[{ts.shape[0]}]"
        awq_info = f", awq=[{w['awq_scale'].shape[0]}]" if 'awq_scale' in w else ""
        print(f"  {k}: qtype={w['qtype']}, bs={list(bs.shape)}, ts={ts_info}{awq_info}")
    else:
        print(f"  {k}: {w.shape} {w.dtype} (BF16 path)")

with open(f"{EVAL_DIR}/test_446.json") as f:
    tokens_446 = json.load(f)["tokens"]
with open(f"{EVAL_DIR}/test_2100.json") as f:
    tokens_2100 = json.load(f)["tokens"]

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
    print(f"\n  Prefill T={T_test:4d}: avg={r['avg_ms']:.1f}ms  throughput={r['toks_per_s']:.0f} tok/s")

decode_r = bench_decode(model, tokens_446, n_warmup=10, n_iter=100)
print(f"\n  Decode (b1t1): {decode_r['toks_per_s']:.0f} tok/s  latency={decode_r['ms_per_tok']:.2f} ms/tok")

vram_peak = torch.cuda.max_memory_allocated() / 2**30
print(f"\n  VRAM: after_load={vram_after_load:.2f} GiB, peak={vram_peak:.2f} GiB")

print("\n" + "=" * 80)
print("  Quality Benchmarks")
print("=" * 80)

def generate_logits(model, tokens, label):
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = model.zero_state(1)
    out = model.forward_all_logits(tok_tensor, state)
    logits = out[0].float().cpu()
    print(f"  [{label}] {logits.shape[0]} logits")
    return logits

def compare_logits(orig_path, quant_logits, tokens_path, label):
    lo = torch.load(orig_path, map_location="cpu").float()
    lq = quant_logits
    with open(tokens_path) as f:
        tokens = json.load(f)["tokens"]
    n = min(lo.shape[0], lq.shape[0])
    lo, lq = lo[:n], lq[:n]
    targets = torch.tensor(tokens[1:n+1], dtype=torch.long)
    diff = lo - lq
    mse = (diff ** 2).mean().item()
    max_diff = diff.abs().max().item()
    top1_agree = (lo.argmax(dim=-1) == lq.argmax(dim=-1)).float().mean().item()
    ce_o = torch.nn.functional.cross_entropy(lo, targets, reduction="mean").item()
    ce_q = torch.nn.functional.cross_entropy(lq, targets, reduction="mean").item()
    ppl_o, ppl_q = math.exp(ce_o), math.exp(ce_q)
    kl = (torch.softmax(lq, dim=-1) * (torch.log_softmax(lq, dim=-1) - torch.log_softmax(lo, dim=-1))).sum(dim=-1)
    return {"label": label, "n": n, "mse": mse, "max_diff": max_diff,
            "top1_agree": top1_agree, "ce_delta": ce_q - ce_o,
            "ppl_o": ppl_o, "ppl_q": ppl_q, "ppl_delta": ppl_q - ppl_o, "mean_kl": kl.mean().item()}

print("\n  Generating logits...")
logits_446 = generate_logits(model, tokens_446, "v8_446")
logits_2100 = generate_logits(model, tokens_2100, "v8_2100")
torch.save(logits_446, f"{EVAL_DIR}/logits_hybrid_v8_446.pt")
torch.save(logits_2100, f"{EVAL_DIR}/logits_hybrid_v8_2100.pt")

r_446 = compare_logits(f"{EVAL_DIR}/logits_orig_446.pt", logits_446, f"{EVAL_DIR}/test_446.json", "Hybrid-v8 446")
r_2100 = compare_logits(f"{EVAL_DIR}/logits_orig_2100.pt", logits_2100, f"{EVAL_DIR}/test_2100.json", "Hybrid-v8 2100")

print(f"\n{'='*100}")
print(f"  Quality Results: 446 tokens")
print(f"{'='*100}")
print(f"  {'Model':<30} {'PPL':<10} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Mean KL':<14}")
print(f"  {'-'*80}")
print(f"  {'Original':<30} {r_446['ppl_o']:<10.4f} {'-':<12} {'-':<10} {'-':<12} {'-':<14}")
print(f"  {r_446['label']:<30} {r_446['ppl_q']:<10.4f} {r_446['ppl_delta']:<12.4f} {r_446['top1_agree']*100:<10.2f} {r_446['ce_delta']:<12.6f} {r_446['mean_kl']:<14.10f}")

print(f"\n{'='*100}")
print(f"  Quality Results: 2100 tokens")
print(f"{'='*100}")
print(f"  {'Model':<30} {'PPL':<10} {'PPL delta':<12} {'Top-1%':<10} {'CE delta':<12} {'Mean KL':<14}")
print(f"  {'-'*80}")
print(f"  {'Original':<30} {r_2100['ppl_o']:<10.4f} {'-':<12} {'-':<10} {'-':<12} {'-':<14}")
print(f"  {r_2100['label']:<30} {r_2100['ppl_q']:<10.4f} {r_2100['ppl_delta']:<12.4f} {r_2100['top1_agree']*100:<10.2f} {r_2100['ce_delta']:<12.6f} {r_2100['mean_kl']:<14.10f}")

print(f"\n{'='*100}")
print(f"  Acceptance (2100 tokens)")
print(f"{'='*100}")
print(f"  PPL delta <= 0.05:     {'PASS' if abs(r_2100['ppl_delta']) <= 0.05 else 'FAIL'} ({abs(r_2100['ppl_delta']):.4f})")
print(f"  Top-1 agree >= 99.0%:  {'PASS' if r_2100['top1_agree'] >= 0.99 else 'FAIL'} ({r_2100['top1_agree']*100:.2f}%)")
print(f"  Top-1 agree >= 99.5%:  {'PASS' if r_2100['top1_agree'] >= 0.995 else 'FAIL'} ({r_2100['top1_agree']*100:.2f}%)")

print(f"\n{'='*100}")
print(f"  Full Comparison (2100 tokens)")
print(f"{'='*100}")
print(f"  {'Mode':<55} {'PPL delta':<12} {'Top-1%':<10} {'Prefill':<10} {'Decode':<10}")
print(f"  {'-'*100}")

for name, path in [
    ("v3 (NVFP4 key+BF16 val+FP8 att)", f"{EVAL_DIR}/bench_hybrid_v3_results.json"),
    ("v5 (NVFP4 bs=8 key+BF16 val+FP8 att)", f"{EVAL_DIR}/bench_hybrid_v5_results.json"),
    ("v6 (NVFP4 clip-opt key+BF16 val+FP8 att)", f"{EVAL_DIR}/bench_hybrid_v6_results.json"),
    ("v7 (AWQ+NVFP4 key+BF16 val+FP8 att)", f"{EVAL_DIR}/bench_hybrid_v7_results.json"),
    ("W4A16+W8A16 (both weight-only)", f"{EVAL_DIR}/bench_w4a16_w8a16_results.json"),
    ("FP8-Att W8A16 (att only)", f"{EVAL_DIR}/bench_att_fp8_results.json"),
]:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"  {name:<55} {d['quality_2100']['ppl_delta']:<12.4f} {d['quality_2100']['top1_agree']*100:<10.2f} {d['prefill']['2100']['toks_per_s']:<10.0f} {d['decode']['toks_per_s']:<10.0f}")

print(f"  {'v8 (AWQ+clip+NVFP4 key+BF16 val+FP8 att)':<55} {r_2100['ppl_delta']:<12.4f} {r_2100['top1_agree']*100:<10.2f} {prefill_results[2100]['toks_per_s']:<10.0f} {decode_r['toks_per_s']:<10.0f}")

results = {
    "model": HYBRID_MODEL,
    "mode": "Hybrid-v8-awq-clip",
    "load_time_s": load_time, "vram_gib": vram_after_load, "vram_peak_gib": vram_peak,
    "prefill": {str(T): {"toks_per_s": r["toks_per_s"], "avg_ms": r["avg_ms"]} for T, r in prefill_results.items()},
    "decode": {"toks_per_s": decode_r["toks_per_s"], "ms_per_tok": decode_r["ms_per_tok"]},
    "quality_446": {"ppl_delta": r_446["ppl_delta"], "top1_agree": r_446["top1_agree"], "ce_delta": r_446["ce_delta"]},
    "quality_2100": {"ppl_delta": r_2100["ppl_delta"], "top1_agree": r_2100["top1_agree"], "ce_delta": r_2100["ce_delta"]},
}
with open(f"{EVAL_DIR}/bench_hybrid_v8_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {EVAL_DIR}/bench_hybrid_v8_results.json")
