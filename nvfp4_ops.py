#!/usr/bin/env python3
"""NVFP4 + FP8 GEMM operations for RWKV-7 v3a inference.

Provides:
- is_nvfp4_weight: detect NVFP4 quantized weights (has .nf4_b_scale sibling)
- is_fp8_weight: detect FP8 quantized weights (has .fp8_scale sibling)
- load_nvfp4_weight: load + swizzle block scale during model init
- load_fp8_weight: load FP8 weight + per-tensor scale
- linear_nvfp4: NVFP4 GEMM (FP4×FP4→BF16) with online activation quantization
- linear_fp8: FP8 GEMM (FP8×FP8→BF16) with online activation quantization
- linear_quantized: dispatcher that picks the right GEMM based on weight_info
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


# ========== Detection ==========

def is_nvfp4_weight(z, key):
    """Check if a weight key has NVFP4 quantization (has .nf4_b_scale sibling)."""
    return (key + ".nf4_b_scale") in z

def is_fp8_weight(z, key):
    """Check if a weight key has FP8 quantization (has .fp8_scale sibling)."""
    return (key + ".fp8_scale") in z


# ========== Loading ==========

def load_nvfp4_weight(z, key, dev):
    """Load NVFP4 weight: packed uint8 + pre-swizzled block scale + tensor scale.
    
    Removes the .nf4_b_scale and .nvfp4_t_scale keys from z.
    Returns a dict with weight, block_scale (swizzled 1D), tensor_scale (scalar), qtype="nvfp4".
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
        "qtype": "nvfp4",
    }

def load_fp8_weight(z, key, dev):
    """Load FP8 weight: float8_e4m3fn + per-tensor scale.
    
    Removes the .fp8_scale key from z.
    Returns a dict with weight, tensor_scale (scalar), qtype="fp8".
    """
    w = z[key].to(device=dev).contiguous()            # [N, K] float8_e4m3fn
    scale = z[key + ".fp8_scale"].to(device=dev)      # scalar float32
    del z[key + ".fp8_scale"]
    return {
        "weight": w,
        "tensor_scale": scale,
        "qtype": "fp8",
    }


# ========== GEMM ==========

FP8_E4M3_MAX = 448.0

def linear_nvfp4(x, weight_info, out_dtype=torch.float16):
    """NVFP4 GEMM: quantize input on-the-fly to FP4, use torch._scaled_mm.
    
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

    a_fp4 = x_packed.view(torch.float4_e2m1fn_x2)
    b_fp4 = w.view(torch.float4_e2m1fn_x2)

    out = torch._scaled_mm(a_fp4, b_fp4.t(),
        scale_a=x_bs_swizzled, scale_b=w_bs,
        out_dtype=torch.bfloat16)

    out = out * x_ts * w_ts

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_fp8(x, weight_info, out_dtype=torch.float16):
    """FP8 GEMM: quantize input on-the-fly to FP8, use torch._scaled_mm.
    
    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_fp8_weight
        out_dtype: output dtype (default fp16 for v3a compatibility)
    
    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w = weight_info["weight"]              # [N, K] float8_e4m3fn
    w_scale = weight_info["tensor_scale"]  # scalar float32

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    # Quantize input to FP8 E4M3 with per-tensor scale
    amax_x = x_2d.abs().max()
    if amax_x > 0:
        x_scale = (amax_x / FP8_E4M3_MAX).float()
    else:
        x_scale = torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)

    x_fp8 = (x_2d.float() / x_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    out = torch._scaled_mm(x_fp8, w.t(),
        scale_a=x_scale.reshape(1),
        scale_b=w_scale.reshape(1),
        out_dtype=torch.bfloat16)

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized(x, weight_info, out_dtype=torch.float16):
    """Dispatcher: pick the right GEMM based on qtype in weight_info."""
    qtype = weight_info.get("qtype", "nvfp4")
    if qtype == "fp8":
        return linear_fp8(x, weight_info, out_dtype)
    return linear_nvfp4(x, weight_info, out_dtype)
