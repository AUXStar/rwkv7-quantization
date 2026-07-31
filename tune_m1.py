#!/usr/bin/env python3
"""Tune fused kernel for M=1 (decode) with various block configs."""
import sys, os, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

from quantize_model import quantize_nvfp4, compute_awq_scale, ALPHA
import fused_nvfp4_gemm as fused

torch.manual_seed(0)


def bench(fn, n_iters=500):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6  # us


def main():
    K, N = 2048, 2048
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    awq = compute_awq_scale(w, act_stats=None, alpha=ALPHA)
    packed, bs, ts, _ = quantize_nvfp4(w, awq, per_channel_ts=False)
    packed, bs, ts = packed.to("cuda"), bs.to("cuda"), ts.to("cuda")
    xb = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.05
    pts = torch.tensor(0.01, dtype=torch.float32, device="cuda")

    print("=== M=1 decode config sweep (K=2048, N=2048) ===")
    best = None
    for bm, bn, bk, nw in [(64,64,64,4), (64,128,64,4), (64,128,64,8), (16,128,64,4),
                           (16,256,64,4), (32,128,64,4), (1,128,64,4), (16,128,128,4),
                           (64,64,32,4), (16,64,64,4)]:
        try:
            out = torch.empty(1, N, dtype=torch.float16, device="cuda")
            t = bench(lambda: fused._launch_nvfp4_custom(xb, packed, bs.view(torch.uint8), ts, pts, out, bm, bn, bk, nw))
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} warps={nw}: {t:7.1f}us")
            if best is None or t < best[0]:
                best = (t, bm, bn, bk, nw)
        except Exception as e:
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} warps={nw}: FAIL {e}")
    print(f"\nBest: {best[1:]} at {best[0]:.1f}us")

    # FP8 path M=1
    print("\n=== FP8 M=1 (K=2048, N=2048) ===")
    from quantize_model import quantize_to_fp8
    wf, wts = quantize_to_fp8(w)
    wf, wts = wf.to("cuda"), wts.to("cuda")
    xs = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    for bm, bn, bk, nw in [(64,64,64,4), (16,128,64,4), (16,256,64,4)]:
        try:
            out = torch.empty(1, N, dtype=torch.float16, device="cuda")
            t = bench(lambda: fused._launch_fp8_custom(xb, wf.view(torch.uint8), wts, xs, out, bm, bn, bk, nw))
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} warps={nw}: {t:7.1f}us")
        except Exception as e:
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} warps={nw}: FAIL {e}")


if __name__ == "__main__":
    main()
