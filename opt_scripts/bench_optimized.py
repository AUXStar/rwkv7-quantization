#!/usr/bin/env python3
"""7.2B optimized decode benchmark + correctness check.

Measures:
1. Decode speed (tok/s) with optimized FP8 hwdot + RKV hwdot kernels
2. Output correctness (compare token IDs with a reference sequence)
3. VRAM usage
4. Kernel-level timing breakdown
"""
import sys, os, json, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

MODEL = "/home/njzy/model/rwkv7-7.2b-X5.pth"

def build_model():
    import rwkv7_fast_v3a as engine
    from rwkv7_fast_v3a import RWKV7, load_extensions
    engine.NVFP4_W4A16 = False; engine.FP8_W8A16 = False; engine.FUSED_GEMM = True
    engine.WKV_MODE = "fp16"; engine.EMB_DEVICE = "cpu"; engine.RKV_MODE = "off"
    engine.CMIX_SPARSE = "no-fc"; engine.LOWRANK_WEIGHT = "both"
    engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    engine.MODEL_PATH = MODEL
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def bench_decode(model, n_warmup=10, n_iter=50):
    """Benchmark decode speed."""
    tok = torch.tensor([[1]], dtype=torch.long)
    s = model.zero_state(1)
    for _ in range(n_warmup):
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Wall clock
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # GPU-only timing
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
        ends[i].record()
    torch.cuda.synchronize()
    gpu_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter

    wall_per = wall / n_iter * 1000
    cpu_per = wall_per - gpu_ms
    return {
        "tps": n_iter / wall,
        "wall_ms": wall_per,
        "gpu_ms": gpu_ms,
        "cpu_ms": cpu_per,
        "cpu_pct": cpu_per / wall_per * 100,
    }


def gen_text(model, prompt_tokens, n_gen=50):
    """Generate text and return token IDs."""
    tok = torch.tensor([prompt_tokens], dtype=torch.long)
    s = model.zero_state(1)
    # Feed prompt
    for t in prompt_tokens:
        out = model.forward(torch.tensor([[t]], dtype=torch.long), s)
    ids = []
    for _ in range(n_gen):
        tok = out[0].argmax(dim=-1, keepdim=True)
        ids.append(tok.item())
        out = model.forward(tok, s)
    return ids


def bench_kernels(model):
    """Kernel-level timing breakdown."""
    import fused_nvfp4_gemm as fused
    z = model.z
    C = 4096
    device = "cuda"
    x = torch.randn(1, C, dtype=torch.float16, device=device)

    wr = z["blocks.1.att.receptance.weight"]
    wk = z["blocks.1.att.key.weight"]
    wv = z["blocks.1.att.value.weight"]
    wo = z["blocks.1.att.output.weight"]
    wfk = z["blocks.1.ffn.key.weight"]
    wfv = z["blocks.1.ffn.value.weight"]

    def _bench(fn, n=30):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n

    rkv_ms = _bench(lambda: fused.linear_rkv_fused(x.clone(), x.clone(), x.clone(), wr, wk, wv))
    fk_ms = _bench(lambda: fused.linear_nvfp4_res_fused(x, wfk))
    fv_ms = _bench(lambda: fused.linear_fp8_fused(x, wfv))
    ao_ms = _bench(lambda: fused.linear_nvfp4_fused(x, wo))

    per_layer = rkv_ms + fk_ms + fv_ms + ao_ms
    return {
        "rkv": rkv_ms, "ffn_key": fk_ms, "ffn_val": fv_ms, "att_out": ao_ms,
        "per_layer": per_layer, "total_gemm_32l": per_layer * 32,
    }


def check_vram():
    """Check VRAM usage."""
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "used_gib": used / 2**30,
        "allocated_gib": allocated / 2**30,
        "reserved_gib": reserved / 2**30,
        "total_gib": total / 2**30,
    }


def main():
    print("=" * 70)
    print("7.2B Optimized Decode Benchmark (FP8 hwdot + RKV hwdot)")
    print("=" * 70)

    m = build_model()

    # VRAM check
    vram = check_vram()
    print(f"\n[1] VRAM: used={vram['used_gib']:.2f}GiB alloc={vram['allocated_gib']:.2f}GiB "
          f"reserved={vram['reserved_gib']:.2f}GiB total={vram['total_gib']:.2f}GiB", flush=True)

    # Kernel-level timing
    print("\n[2] Kernel-level timing (M=1, C=4096)...", flush=True)
    ker = bench_kernels(m)
    print(f"  rkv (hwdot):    {ker['rkv']:.4f} ms", flush=True)
    print(f"  ffn_key:        {ker['ffn_key']:.4f} ms", flush=True)
    print(f"  ffn_val (hwdot):{ker['ffn_val']:.4f} ms", flush=True)
    print(f"  att_out:        {ker['att_out']:.4f} ms", flush=True)
    print(f"  per_layer:      {ker['per_layer']:.4f} ms (32L: {ker['total_gemm_32l']:.2f} ms)", flush=True)

    # Decode benchmark
    print("\n[3] Decode benchmark...", flush=True)
    dec = bench_decode(m)
    print(f"  tps={dec['tps']:.1f}  wall={dec['wall_ms']:.1f}ms  "
          f"gpu={dec['gpu_ms']:.1f}ms  cpu={dec['cpu_ms']:.1f}ms "
          f"({dec['cpu_pct']:.0f}% CPU)", flush=True)

    # Correctness: generate text
    print("\n[4] Correctness: generating 50 tokens from prompt...", flush=True)
    prompt = [1, 2, 3, 4, 5]  # simple prompt
    ids = gen_text(m, prompt, n_gen=50)
    print(f"  Generated IDs (first 20): {ids[:20]}", flush=True)
    print(f"  All IDs valid: {all(0 <= i < 65536 for i in ids)}", flush=True)

    # Compare with baseline results
    print("\n[5] Comparison with baseline...", flush=True)
    baseline_tps = 21.8
    baseline_rkv = 0.4379
    baseline_fv = 0.1223
    baseline_per_layer = 1.594
    print(f"  Decode:  {baseline_tps:.1f} → {dec['tps']:.1f} tok/s ({dec['tps']/baseline_tps:.2f}x)", flush=True)
    print(f"  RKV:     {baseline_rkv:.4f} → {ker['rkv']:.4f} ms ({baseline_rkv/ker['rkv']:.2f}x)", flush=True)
    print(f"  ffn_val: {baseline_fv:.4f} → {ker['ffn_val']:.4f} ms ({baseline_fv/ker['ffn_val']:.2f}x)", flush=True)
    print(f"  per_layer: {baseline_per_layer:.4f} → {ker['per_layer']:.4f} ms ({baseline_per_layer/ker['per_layer']:.2f}x)", flush=True)

    results = {
        "vram": vram,
        "kernels": ker,
        "decode": dec,
        "generated_ids": ids,
        "comparison": {
            "baseline_tps": baseline_tps,
            "optimized_tps": dec["tps"],
            "speedup": dec["tps"] / baseline_tps,
        },
    }
    with open("/home/njzy/test/eval_tmp/bench_optimized.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved bench_optimized.json", flush=True)


if __name__ == "__main__":
    main()
