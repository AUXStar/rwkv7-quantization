#!/usr/bin/env python3
"""Standalone precision test: fused GEMM kernels vs _scaled_mm path.

Validates numerics match before engine integration.
Shapes: decode (M=1), prefill (M=64), K=2048, N=2048/8192.
Covers: nvfp4, nvfp4+res (fused single kernel), fp8.
"""
import sys, os
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

from quantize_model import quantize_nvfp4, quantize_to_fp8, quantize_nvfp4_with_residual, compute_awq_scale, ALPHA
import fused_nvfp4_gemm as fused
from nvfp4_ops import linear_nvfp4, linear_fp8

torch.manual_seed(0)


def test_nvfp4(M, K, N, has_awq=True, has_res=False):
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    awq = compute_awq_scale(w, act_stats=None, alpha=ALPHA)

    if has_res:
        packed, bs, ts, awq_s, res_fp8, res_scale = quantize_nvfp4_with_residual(w, awq)
        res_fp8 = res_fp8.to("cuda")
        res_scale = res_scale.to("cuda")
    else:
        packed, bs, ts, _ = quantize_nvfp4(w, awq, per_channel_ts=False)
        res_fp8, res_scale = None, None

    packed = packed.to("cuda")
    bs = bs.to("cuda")          # [N, K//16] fp8 (UNswizzled)
    ts = ts.to("cuda")

    # Reference: _scaled_mm path (swizzled scales)
    from nvfp4_ops import _get_mx
    mx = _get_mx()
    bs_sw = mx.to_blocked(bs).contiguous()

    w_info_ref = {"weight": packed, "block_scale": bs_sw, "tensor_scale": ts, "qtype": "nvfp4"}
    if has_res:
        w_info_ref["res_fp8"] = res_fp8
        w_info_ref["res_fp8_scale"] = res_scale

    # Fused path (unswizzled scales)
    w_info_fused = {"weight": packed, "block_scale": bs, "tensor_scale": ts, "qtype": "nvfp4_res" if has_res else "nvfp4"}
    if has_res:
        w_info_fused["res_fp8"] = res_fp8
        w_info_fused["res_fp8_scale"] = res_scale

    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05

    out_ref = linear_nvfp4(x, w_info_ref, out_dtype=torch.float16)
    if has_res:
        out_fus = fused.linear_nvfp4_res_fused(x, w_info_fused, out_dtype=torch.float16)
    else:
        out_fus = fused.linear_nvfp4_fused(x, w_info_fused, out_dtype=torch.float16)

    diff = (out_ref.float() - out_fus.float()).abs()
    rel = diff / (out_ref.float().abs() + 1e-6)
    label = "res" if has_res else "nvfp4"
    print(f"[{label:>5} M={M:>4} K={K:>5} N={N:>5} awq={has_awq}] "
          f"max_diff={diff.max().item():.6f} mean={diff.mean().item():.8f} "
          f"max_rel={rel.max().item():.6f}")
    ok = diff.max().item() < max(0.02, out_ref.float().abs().max().item() * 0.02)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_fp8(M, K, N):
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
    w_fp8, w_scale = quantize_to_fp8(w)
    w_fp8 = w_fp8.to("cuda")
    w_scale = w_scale.to("cuda")

    w_info = {"weight": w_fp8, "tensor_scale": w_scale, "qtype": "fp8"}
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05

    out_ref = linear_fp8(x, w_info, out_dtype=torch.float16)
    out_fus = fused.linear_fp8_fused(x, w_info, out_dtype=torch.float16)

    diff = (out_ref.float() - out_fus.float()).abs()
    rel = diff / (out_ref.float().abs() + 1e-6)
    print(f"[fp8   M={M:>4} K={K:>5} N={N:>5}] "
          f"max_diff={diff.max().item():.6f} mean={diff.mean().item():.8f} "
          f"max_rel={rel.max().item():.6f}")
    ok = diff.max().item() < max(0.02, out_ref.float().abs().max().item() * 0.02)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for M in (1, 64):
        all_ok &= test_nvfp4(M, 2048, 2048, has_awq=True, has_res=False)
        all_ok &= test_nvfp4(M, 2048, 8192, has_awq=True, has_res=True)
        all_ok &= test_fp8(M, 2048, 2048)
    print("\n=== ALL PASS ===" if all_ok else "\n=== SOME FAILED ===")
