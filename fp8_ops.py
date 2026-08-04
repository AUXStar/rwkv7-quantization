#!/usr/bin/env python3
"""FP8 quantized GEMM operations for RWKV-7 v3a inference.

Provides:
- is_fp8_weight: detect FP8 quantized weights (has .fp8_scale sibling)
- load_fp8_weight: load FP8 weight + per-tensor scale
- linear_fp8: FP8 GEMM (FP8×FP8→BF16) with online activation quantization
- (已移除) 反量化路径已删除：FP8 权重永远保持 FP8，只走张量核
- linear_quantized: dispatcher that picks the right GEMM based on weight_info
"""
import torch

# ============================================================================
# Detection
# ============================================================================

def is_fp8_weight(z, key):
    """Check if a weight key has FP8 quantization (has .fp8_scale sibling)."""
    return (key + ".fp8_scale") in z


# ============================================================================
# Loading
# ============================================================================

def load_fp8_weight(z, key, dev):
    """Load FP8 weight: float8_e4m3fn + per-tensor scale (W8A8, 禁止反量化).

    FP8 权重只允许保持 FP8 域，由 FP8 张量核（_scaled_mm / tl.dot(fp8,fp8)）计算，
    绝不反量化回 fp16/bf16 再 GEMM。

    Removes the .fp8_scale key from z.
    Returns a dict with weight, tensor_scale (scalar), qtype.
    """
    w = z[key].to(device=dev).contiguous()            # [N, K] float8_e4m3fn
    scale = z[key + ".fp8_scale"].to(device=dev)      # scalar float32
    del z[key + ".fp8_scale"]
    return {
        "weight": w,
        "tensor_scale": scale,
        "qtype": "fp8",  # 固定 W8A8 真量化域，禁止 w8a16 反量化
    }


# ============================================================================
# GEMM
# ============================================================================

FP8_E4M3_MAX = 448.0

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
    # 禁止反量化：FP8 权重永远走 FP8 张量核（_scaled_mm），保持 FP8 域
    return linear_fp8(x, weight_info, out_dtype)


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Hybrid dispatcher for quantized GEMMs.

    Dispatches to fused single-kernel GEMM in fused_fp8_gemm module.
    """
    from fused_fp8_gemm import linear_quantized_fused as _dispatch
    return _dispatch(x, weight_info, out_dtype)


FUSED_M_MAX = 64  # use fused single-kernel GEMM when M <= this (decode/small-batch domain)
