#!/usr/bin/env python3
"""Bench fused kernel at engine prefill shape M=2100, try different block configs."""
import sys, os, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

from quantize_model import quantize_nvfp4, compute_awq_scale, ALPHA
import fused_nvfp4_gemm as fused
from nvfp4_ops import linear_nvfp4, _get_mx

torch.manual_seed(0)
mx = _get_mx()


def bench(fn, n_iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e3  # ms


def main():
    M, K, N = 2100, 2048, 2048
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    awq = compute_awq_scale(w, act_stats=None, alpha=ALPHA)
    packed, bs, ts, _ = quantize_nvfp4(w, awq, per_channel_ts=False)
    packed, bs, ts = packed.to("cuda"), bs.to("cuda"), ts.to("cuda")
    w_ref = {"weight": packed, "block_scale": mx.to_blocked(bs).contiguous(), "tensor_scale": ts, "qtype": "nvfp4"}
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05
    xb = x.to(torch.bfloat16)

    t_ref = bench(lambda: linear_nvfp4(x, w_ref, out_dtype=torch.float16))
    print(f"_scaled_mm path: {t_ref:.2f} ms")

    pts = torch.tensor(0.01, dtype=torch.float32, device="cuda")
    for bm, bn, bk, nw in [(64,64,64,4), (64,64,64,8), (64,128,64,8), (128,64,64,8),
                           (64,64,128,4), (32,64,64,4), (64,64,32,4), (128,128,64,8)]:
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")
        t = bench(lambda: fused._launch_nvfp4_custom(xb, packed, bs.view(torch.uint8), ts, pts, out, bm, bn, bk, nw))
        print(f"fused BM={bm:>3} BN={bn:>3} BK={bk:>3} warps={nw}: {t:.2f} ms")

    w16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    t_native = bench(lambda: torch.mm(x, w16))
    print(f"cublas fp16: {t_native:.2f} ms")


if __name__ == "__main__":
    main()
