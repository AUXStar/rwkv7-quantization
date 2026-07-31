#!/usr/bin/env python3
"""Kernel-level speed benchmark: fused GEMM vs _scaled_mm path.

Shapes matching v3a engine:
- decode: M=1, K=2048, N=2048 (att) / N=8192 (ffn)
- prefill: M=64, K=2048
"""
import sys, os, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

from quantize_model import quantize_nvfp4, quantize_to_fp8, compute_awq_scale, ALPHA
import fused_nvfp4_gemm as fused
from nvfp4_ops import linear_nvfp4, linear_fp8, _get_mx

torch.manual_seed(0)
mx = _get_mx()


def bench(fn, n_iters=200):
    for _ in range(20):  # warmup (includes triton compile)
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6  # us


def make_weights(N, K, res=False):
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    awq = compute_awq_scale(w, act_stats=None, alpha=ALPHA)
    packed, bs, ts, _ = quantize_nvfp4(w, awq, per_channel_ts=False)
    packed = packed.to("cuda")
    bs = bs.to("cuda")
    ts = ts.to("cuda")
    if res:
        from quantize_model import quantize_nvfp4_with_residual
        _, _, _, _, res_fp8, res_scale = quantize_nvfp4_with_residual(w, awq)
        res_fp8 = res_fp8.to("cuda")
        res_scale = res_scale.to("cuda")
        return packed, bs, ts, res_fp8, res_scale
    return packed, bs, ts


def bench_nvfp4(M, K, N):
    packed, bs, ts = make_weights(N, K)
    w_ref = {"weight": packed, "block_scale": mx.to_blocked(bs).contiguous(), "tensor_scale": ts, "qtype": "nvfp4"}
    w_fus = {"weight": packed, "block_scale": bs, "tensor_scale": ts, "qtype": "nvfp4"}
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05

    t_ref = bench(lambda: linear_nvfp4(x, w_ref, out_dtype=torch.float16))
    t_fus = bench(lambda: fused.linear_nvfp4_fused(x, w_fus, out_dtype=torch.float16))

    # native fp16 GEMM reference (cuBLAS)
    w_f16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    t_native = bench(lambda: torch.mm(x, w_f16))
    print(f"nvfp4 M={M:>3} K={K:>5} N={N:>5}: _scaled_mm={t_ref:7.1f}us  fused={t_fus:7.1f}us  "
          f"speedup={t_ref/t_fus:.2f}x  (cublas_f16={t_native:7.1f}us)")


def bench_fp8(M, K, N):
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    w_fp8, w_scale = quantize_to_fp8(w)
    w_fp8 = w_fp8.to("cuda")
    w_scale = w_scale.to("cuda")
    w_info = {"weight": w_fp8, "tensor_scale": w_scale, "qtype": "fp8"}
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05

    t_ref = bench(lambda: linear_fp8(x, w_info, out_dtype=torch.float16))
    t_fus = bench(lambda: fused.linear_fp8_fused(x, w_info, out_dtype=torch.float16))
    print(f"fp8   M={M:>3} K={K:>5} N={N:>5}: _scaled_mm={t_ref:7.1f}us  fused={t_fus:7.1f}us  "
          f"speedup={t_ref/t_fus:.2f}x")


def bench_res(M, K, N):
    packed, bs, ts, res_fp8, res_scale = make_weights(N, K, res=True)
    w_ref = {"weight": packed, "block_scale": mx.to_blocked(bs).contiguous(), "tensor_scale": ts,
             "qtype": "nvfp4", "res_fp8": res_fp8, "res_fp8_scale": res_scale}
    w_fus = {"weight": packed, "block_scale": bs, "tensor_scale": ts,
             "qtype": "nvfp4_res", "res_fp8": res_fp8, "res_fp8_scale": res_scale}
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05

    t_ref = bench(lambda: linear_nvfp4(x, w_ref, out_dtype=torch.float16))
    t_fus = bench(lambda: fused.linear_nvfp4_fused(x, w_fus, out_dtype=torch.float16))
    print(f"res   M={M:>3} K={K:>5} N={N:>5}: _scaled_mm={t_ref:7.1f}us  fused={t_fus:7.1f}us  "
          f"speedup={t_ref/t_fus:.2f}x")


if __name__ == "__main__":
    print("=== kernel-level speed: fused vs _scaled_mm path ===")
    for M in (1, 64):
        bench_nvfp4(M, 2048, 2048)
        bench_nvfp4(M, 2048, 8192)
        bench_fp8(M, 2048, 2048)
        bench_res(M, 2048, 8192)
        print()
