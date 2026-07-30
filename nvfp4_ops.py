#!/usr/bin/env python3
"""NVFP4 GEMM operations for RWKV-7 v3a inference.

Provides:
- is_nvfp4_weight: detect NVFP4 quantized weights in checkpoint
- load_nvfp4_weight: load + swizzle block scale during model init
- linear_nvfp4: on-the-fly input quantization + _scaled_mm GEMM
"""
import torch
import os
import importlib.util

_mx = None

def _get_mx():
    """Load mx_utils from torch._vendor.quack (bypass __init__ which needs cutlass)."""
    global _mx
    if _mx is None:
        quack_dir = os.path.join(os.path.dirname(torch.__file__), '_vendor', 'quack')
        mx_path = os.path.join(quack_dir, 'mx_utils.py')
        spec = importlib.util.spec_from_file_location('_mx_utils', mx_path)
        _mx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mx)
    return _mx


def is_nvfp4_weight(z, key):
    """Check if a weight key has NVFP4 quantization (has .nf4_b_scale sibling)."""
    return (key + ".nf4_b_scale") in z


def load_nvfp4_weight(z, key, dev):
    """Load NVFP4 weight: packed uint8 + pre-swizzled block scale + tensor scale.
    
    Removes the .nf4_b_scale and .nvfp4_t_scale keys from z.
    Returns a dict with weight, block_scale (swizzled 1D), tensor_scale (scalar).
    """
    mx = _get_mx()
    w = z[key].to(device=dev).contiguous()           # [N, K//2] uint8
    bs = z[key + ".nf4_b_scale"].to(device=dev).contiguous()  # [N, K//16] float8_e4m3fn
    ts = z[key + ".nvfp4_t_scale"].to(device=dev)    # scalar float32
    bs_swizzled = mx.to_blocked(bs)                   # 1D flat, swizzled for cuBLAS
    del z[key + ".nf4_b_scale"]
    del z[key + ".nvfp4_t_scale"]
    return {
        "weight": w,
        "block_scale": bs_swizzled,
        "tensor_scale": ts,
    }


def linear_nvfp4(x, weight_info, out_dtype=torch.float16):
    """NVFP4 GEMM: quantize input on-the-fly, use torch._scaled_mm.
    
    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_nvfp4_weight
        out_dtype: output dtype (default fp16 for v3a compatibility)
    
    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    mx = _get_mx()
    w = weight_info["weight"]              # [N, K//2] uint8
    w_bs = weight_info["block_scale"]      # 1D swizzled float8_e4m3fn
    w_ts = weight_info["tensor_scale"]     # scalar float32

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    # to_nvfp4 requires bf16 or fp32
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    # Quantize input to NVFP4
    amax_x = x_2d.abs().max()
    if amax_x > 0:
        ts_x = mx.nvfp4_per_tensor_scale(amax_x)
    else:
        ts_x = torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)
    x_packed, x_bs, x_ts = mx.to_nvfp4(x_2d, block_size=16, per_tensor_scale=ts_x)
    x_bs_swizzled = mx.to_blocked(x_bs)

    # View as float4_e2m1fn_x2 and run _scaled_mm
    a_fp4 = x_packed.view(torch.float4_e2m1fn_x2)
    b_fp4 = w.view(torch.float4_e2m1fn_x2)

    out = torch._scaled_mm(a_fp4, b_fp4.t(),
        scale_a=x_bs_swizzled, scale_b=w_bs,
        out_dtype=torch.bfloat16)

    # Fold per-tensor scales
    out = out * x_ts * w_ts

    # Reshape and convert dtype
    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out
