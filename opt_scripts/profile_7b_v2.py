#!/usr/bin/env python3
"""7.2B decode profile v2: 修复属性名 + tile config 扫描。

关键发现 (v1): 7.2B 是 95% GPU bound (cpu=2.2ms, gpu=38.9ms)
→ 优化方向是 kernel 计算效率, 不是减少 launch
"""
import sys, os, json, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch
import triton

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
    tok = torch.tensor([[1]], dtype=torch.long)
    s = model.zero_state(1)
    for _ in range(n_warmup):
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

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
    return {"tps": n_iter / wall, "wall_ms": wall_per, "gpu_ms": gpu_ms,
            "cpu_ms": cpu_per, "cpu_pct": cpu_per / wall_per * 100}


def bench_kernels(model):
    """用 model.z 里的真实权重测 kernel 速度."""
    import fused_nvfp4_gemm as fused
    z = model.z
    C = 4096
    device = "cuda"
    x = torch.randn(1, C, dtype=torch.float16, device=device)

    # L1 是非 protected 层 (X5: protected={0,8,24,31}), 权重是 dict 格式
    wr = z["blocks.1.att.receptance.weight"]  # NVFP4
    wk = z["blocks.1.att.key.weight"]          # FP8 (X5)
    wv = z["blocks.1.att.value.weight"]        # FP8
    wfk = z["blocks.1.ffn.key.weight"]         # NVFP4+res
    wfv = z["blocks.1.ffn.value.weight"]       # FP8

    # 确认都是 dict
    for name, w in [("att_r", wr), ("att_k", wk), ("att_v", wv), ("ffn_k", wfk), ("ffn_v", wfv)]:
        if not isinstance(w, dict):
            print(f"  WARNING: {name} is {type(w).__name__}, not dict!", flush=True)

    def _bench(fn, n=30):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n

    xr = x.clone(); xk = x.clone(); xv = x.clone()
    rkv_ms = _bench(lambda: fused.linear_rkv_fused(xr, xk, xv, wr, wk, wv))
    fk_ms = _bench(lambda: fused.linear_nvfp4_res_fused(x, wfk))
    fv_ms = _bench(lambda: fused.linear_fp8_fused(x, wfv))

    per_layer = rkv_ms + fk_ms + fv_ms
    return {
        "rkv_fused": rkv_ms, "ffn_key_res": fk_ms, "ffn_value_fp8": fv_ms,
        "per_layer_ms": per_layer, "total_gemm_ms": per_layer * 32,
    }


def scan_tile_config(model):
    """扫描不同 BLOCK 配置对 7.2B (C=4096, M=1) 的影响."""
    import fused_nvfp4_gemm as fused
    C = 4096
    device = "cuda"
    x = torch.randn(1, C, dtype=torch.float16, device=device)
    wfk = model.z["blocks.1.ffn.key.weight"]  # NVFP4+res (最重的 kernel)

    configs = [
        (16, 64, 64, 4),    # 当前 default
        (16, 128, 64, 4),
        (16, 128, 128, 4),
        (16, 256, 64, 4),
        (16, 64, 128, 4),
        (16, 128, 64, 8),
        (16, 256, 128, 8),
        (16, 64, 256, 4),
        (16, 128, 256, 8),
        (16, 64, 64, 8),
        (16, 256, 64, 8),
        (16, 128, 128, 8),
    ]

    results = []
    for bm, bn, bk, nw in configs:
        orig_cfg = fused._nvfp4_cfg_for
        fused._nvfp4_cfg_for = lambda M, c=(bm, bn, bk, nw): c
        try:
            for _ in range(3): fused.linear_nvfp4_res_fused(x, wfk)
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
            for i in range(30):
                starts[i].record(); fused.linear_nvfp4_res_fused(x, wfk); ends[i].record()
            torch.cuda.synchronize()
            ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / 30
            results.append({"config": f"({bm},{bn},{bk},{nw})", "ms": round(ms, 4)})
            print(f"  ({bm:>2},{bn:>3},{bk:>3},{nw}w) -> {ms:.4f} ms", flush=True)
        except Exception as e:
            results.append({"config": f"({bm},{bn},{bk},{nw})", "ms": -1, "err": str(e)[:60]})
            print(f"  ({bm:>2},{bn:>3},{bk:>3},{nw}w) -> ERR: {str(e)[:50]}", flush=True)
        finally:
            fused._nvfp4_cfg_for = orig_cfg
    return results


def main():
    print("=" * 70)
    print("7.2B Decode Profile v2 — opt-7b-ops")
    print("=" * 70)

    m = build_model()

    print("\n[1] Decode baseline...", flush=True)
    dec = bench_decode(m)
    print(f"  tps={dec['tps']:.1f}  wall={dec['wall_ms']:.1f}ms  "
          f"gpu={dec['gpu_ms']:.1f}ms  cpu={dec['cpu_ms']:.1f}ms "
          f"({dec['cpu_pct']:.0f}% CPU)", flush=True)

    print("\n[2] Kernel-level timing (per layer, M=1, C=4096)...", flush=True)
    ker = bench_kernels(m)
    print(f"  rkv_fused:      {ker['rkv_fused']:.4f} ms", flush=True)
    print(f"  ffn_key_res:    {ker['ffn_key_res']:.4f} ms", flush=True)
    print(f"  ffn_value_fp8:  {ker['ffn_value_fp8']:.4f} ms", flush=True)
    print(f"  per_layer:      {ker['per_layer_ms']:.4f} ms", flush=True)
    print(f"  32L total GEMM: {ker['total_gemm_ms']:.2f} ms", flush=True)
    print(f"  GEMM/GPU:       {ker['total_gemm_ms']/dec['gpu_ms']*100:.0f}%", flush=True)

    print("\n[3] Tile config scan (ffn_key_res, M=1, C=4096)...", flush=True)
    tiles = scan_tile_config(m)
    valid = [t for t in tiles if t["ms"] > 0]
    if valid:
        best = min(valid, key=lambda t: t["ms"])
        cur = [t for t in tiles if t["config"] == "(16,64,64,4)"][0]
        print(f"\n  当前: {cur['config']} -> {cur['ms']:.4f} ms", flush=True)
        print(f"  最优: {best['config']} -> {best['ms']:.4f} ms "
              f"({cur['ms']/best['ms']:.2f}x)", flush=True)

    out = {"decode": dec, "kernels": ker, "tiles": tiles}
    with open("/home/njzy/test/eval_tmp/profile_7b_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved profile_7b_v2.json", flush=True)


if __name__ == "__main__":
    main()
