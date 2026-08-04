#!/usr/bin/env python3
"""FP8 quantized GEMM operations for RWKV-7 v3a inference.

Provides:
- is_fp8_weight: detect FP8 quantized weights (has .fp8_scale sibling)
- load_fp8_weight: load FP8 weight + per-tensor scale
- linear_fp8: FP8 GEMM (FP8×FP8→BF16) with online activation quantization
- linear_fp8_w8a16: FP8 W8A16 GEMM (dequant weight→BF16, FP16 activation)
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

def load_fp8_weight(z, key, dev, w8a16=False):
    """Load FP8 weight: float8_e4m3fn + scale (per-tensor or per-channel).

    Args:
        z: weight dict
        key: weight key
        dev: target device
        w8a16: if True, use W8A16 path (weight-only, FP16 activation).
               if False, use W8A8 path (both weight and activation quantized).

    Removes the .fp8_scale key from z.
    Returns a dict with weight, tensor_scale, qtype.
    Scale is scalar (per-tensor) or [N] tensor (per-channel).
    """
    w = z[key].to(device=dev).contiguous()            # [N, K] float8_e4m3fn
    scale = z[key + ".fp8_scale"].to(device=dev)      # scalar or [N] float32
    del z[key + ".fp8_scale"]
    return {
        "weight": w,
        "tensor_scale": scale,
        "qtype": "fp8_w8a16" if w8a16 else "fp8",
    }


# ============================================================================
# GEMM
# ============================================================================

FP8_E4M3_MAX = 448.0

def linear_fp8(x, weight_info, out_dtype=torch.float16):
    """FP8 GEMM: quantize input on-the-fly to FP8, use torch._scaled_mm.

    Supports both per-tensor (scalar scale) and per-channel ([N] scale).

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_fp8_weight
        out_dtype: output dtype (default fp16 for v3a compatibility)

    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w = weight_info["weight"]              # [N, K] float8_e4m3fn
    w_scale = weight_info["tensor_scale"]  # scalar or [N] float32

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

    # Handle per-tensor (scalar) vs per-channel ([N]) weight scale
    # _scaled_mm RowWise mode requires scale_a=(M,1) and scale_b=(1,N)
    M = x_2d.size(0)
    if w_scale.dim() == 0:
        # Per-tensor: both scales are singletons (TensorWise mode)
        scale_a = x_scale.reshape(1)
        scale_b = w_scale.reshape(1)
    else:
        # Per-channel: RowWise mode, scale_a=(M,1), scale_b=(1,N)
        scale_a = x_scale.reshape(1).expand(M, 1).contiguous()
        scale_b = w_scale.reshape(1, -1).contiguous()

    out = torch._scaled_mm(x_fp8, w.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16)

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_fp8_w8a16(x, weight_info, out_dtype=torch.float16):
    """FP8 W8A16 GEMM: dequantize weight to FP16, then FP16 GEMM.

    Weight-only quantization: activations stay at FP16 precision.
    Eliminates FP8 activation quantization error.

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_fp8_weight (w8a16=True)
        out_dtype: output dtype

    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w = weight_info["weight"]              # [N, K] float8_e4m3fn
    w_scale = weight_info["tensor_scale"]  # scalar float32

    # Dequantize weight: FP8 -> FP16, then multiply by scale
    # Per-channel: w_scale is [N], broadcast over [N, K]
    w_fp16 = w.to(torch.float16) * w_scale.to(torch.float16).unsqueeze(-1) if w_scale.dim() > 0 else w.to(torch.float16) * w_scale.to(torch.float16)

    # Reshape input
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.bfloat16:
        x_2d = x_2d.to(torch.float16)

    # FP16 GEMM
    out = torch.mm(x_2d, w_fp16.t())  # [M, N]

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized(x, weight_info, out_dtype=torch.float16):
    """Dispatcher: pick the right GEMM based on qtype in weight_info."""
    qtype = weight_info.get("qtype", "fp8")
    if qtype == "fp8_w8a16":
        return linear_fp8_w8a16(x, weight_info, out_dtype)
    return linear_fp8(x, weight_info, out_dtype)


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Hybrid dispatcher for quantized GEMMs.

    Dispatches to fused single-kernel GEMM in fused_fp8_gemm module.
    """
    from fused_fp8_gemm import linear_quantized_fused as _dispatch
    return _dispatch(x, weight_info, out_dtype)


FUSED_M_MAX = 64  # use fused single-kernel GEMM when M <= this (decode/small-batch domain)
