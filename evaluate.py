#!/usr/bin/env python3
# coding=utf-8
"""Unified EAR/Top-1 evaluator for quantized RWKV-7."""
from __future__ import annotations
import argparse, gc, json, os, sys, time
import torch, torch.nn.functional as F

ENGINE = os.environ.get("RWKV_ENGINE", "/home/njzy/test/Albatross/faster3a_2607")
VOCAB = os.environ.get("RWKV_VOCAB", "/home/njzy/RWKV-Server/rwkv_vocab_v20230424.txt")

PROMPTS = ["The capital of France is", "In machine learning, gradient descent is", "def fibonacci(n):", "To solve 2x+5=11", "Binary search is O(", "Largest planet is", "Water boils at 100", "Quick brown fox", "Python list comprehension", "Derivative of x^2", "Sort a list in Python", "Binary search function", "Pythagorean theorem", "import torch", "AI in 2025", "Neural network 3 layers", "Chemical symbol gold", "Photosynthesis converts", "Square root of 144"]

def load_model_logits(path, prompts):
    """Load RWKV-7 engine and get logits for prompts."""
    sys.path.insert(0, ENGINE)
    import rwkv7_fast_v3a as v3a
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import rwkv
    tokenizer = TRIE_TOKENIZER(VOCAB)
    v3a.MODEL_PATH = path
    v3a.WKV_MODE = "fp16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "off"
    v3a.LOWRANK_WEIGHT = "transpose"
    v3a.ORIG_LINEAR_GROUPS = {"head"}
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    all_logits = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)[:512]
        inp = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
        state = model.zero_state(1)
        with torch.no_grad():
            lg = model.forward_all_logits(inp, state)
            if lg.dim() == 3: lg = lg[0]
        all_logits.append(lg.cpu().float())
    del model
    for m in list(sys.modules.keys()):
        if any(x in m for x in ["fp8_ops","fused_fp8","rwkv7_fast"]):
            del sys.modules[m]
    gc.collect(); torch.cuda.empty_cache()
    return all_logits

def compute_ear(logs_o, logs_q):
    """Compute weighted-average EAR and Top-1."""
    total_ear = total_t = total_top1 = 0.0
    for lo, lq in zip(logs_o, logs_q):
        mn = min(lo.size(0), lq.size(0))
        lo, lq = lo[:mn], lq[:mn]
        p_o, p_q = F.softmax(lo, -1), F.softmax(lq, -1)
        total_ear += torch.minimum(p_o, p_q).sum().item()
        total_top1 += (lo.argmax(-1) == lq.argmax(-1)).float().sum().item()
        total_t += mn
    return total_ear / total_t, total_top1 / total_t

def measure_speed(path, warmup=3, steps=20):
    """Measure decode speed in tok/s."""
    sys.path.insert(0, ENGINE)
    import rwkv7_fast_v3a as v3a
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import rwkv
    tokenizer = TRIE_TOKENIZER(VOCAB)
    v3a.MODEL_PATH = path
    v3a.WKV_MODE = "fp16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "off"
    v3a.LOWRANK_WEIGHT = "transpose"
    v3a.ORIG_LINEAR_GROUPS = {"head"}
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    prompt = "Write a Python function"
    tokens = tokenizer.encode(prompt)[:128]
    inp = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
    state = model.zero_state(1)
    with torch.no_grad():
        model.forward(inp, state)
    # Warmup
    for _ in range(warmup):
        state = model.zero_state(1)
        with torch.no_grad():
            model.forward(inp, state)
    # Measure
    torch.cuda.synchronize()
    t1 = time.time()
    del model
    for m in list(sys.modules.keys()):
        if any(x in m for x in ["fp8_ops","fused_fp8","rwkv7_fast"]):
            del sys.modules[m]
    gc.collect(); torch.cuda.empty_cache()
    return steps / (t1 - t0)

def evaluate(baseline_path, quant_paths):
    """Evaluate one or more quantized models against baseline."""
    print(f"Loading baseline: {baseline_path}")
    logs_base = load_model_logits(baseline_path, PROMPTS)
    print(f"  Got {len(logs_base)} prompts")

    # Measure baseline speed
    print("Measuring baseline speed...")
    base_speed = measure_speed(baseline_path)
    print(f"  Baseline speed: {base_speed:.1f} tok/s")

    results = []
    for qp in quant_paths:
        name = os.path.basename(qp).replace(".pth","")
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        logs_q = load_model_logits(qp, PROMPTS)
        ear, top1 = compute_ear(logs_base, logs_q)
        speed = measure_speed(qp)
        vram = torch.cuda.max_memory_allocated() / 1e9
        file_gb = os.path.getsize(qp) / 1e9

        r = {"name": name, "path": qp, "ear": ear, "top1": top1,
             "speed": speed, "vram_gb": round(vram,2), "file_gb": round(file_gb,2),
             "speedup": round(speed/base_speed, 2)}
        results.append(r)
        print(f"  EAR:    {ear:.6f}")
        print(f"  Top-1:  {top1*100:.2f}%")
        print(f"  Speed:  {speed:.1f} tok/s ({r['speedup']}x)")
        print(f"  VRAM:   {vram:.2f} GB")
        print(f"  Size:   {file_gb:.2f} GB")

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Name':30s} {'EAR':>8s} {'Top-1':>7s} {'Speed':>8s} {'VRAM':>6s} {'Size':>6s}")
    print(f"{'='*70}")
    print(f"{'(baseline)':30s} {'1.000000':>8s} {'100.00%':>7s} {base_speed:>7.1f}s")
    for r in results:
        print(f"{r['name']:30s} {r['ear']:8.6f} {r['top1']*100:6.2f}% {r['speed']:7.1f}s {r['vram_gb']:5.2f}G {r['file_gb']:5.2f}G")

    return results

def main():
    p = argparse.ArgumentParser(description="RWKV-7 Unified Evaluator")
    p.add_argument("-b", "--baseline", required=True, help="Baseline model .pth")
    p.add_argument("-q", "--quantized", nargs="+", required=True, help="Quantized model(s)")
    p.add_argument("-o", "--output", help="Save results to JSON")
    args = p.parse_args()
    results = evaluate(args.baseline, args.quantized)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
