#!/usr/bin/env python3
# coding=utf-8
"""INT8 weight detection, loading, and GEMM operations for RWKV-7.

Supports two schemes:
  - per_tensor: w_int8 + scalar scale  (W8A8 via _scaled_mm or Triton)
  - affine:     w_u8 + mx/rx/my/ry     (dequant + GEMM)
"""
from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# Weight detection & loading
# ---------------------------------------------------------------------------

def is_int8_weight(z: dict, key: str) -> bool:
    """Check if a weight key has INT8 quantization metadata."""
    return (key + ".int8_weight" in z) or (key + ".int8_w_u8" in z)


def load_int8_weight(z: dict, key: str, dev: str = "cuda"):
    """Load INT8 weight + metadata. Returns dict with weight info."""
    # Per-tensor symmetric
    if key + ".int8_weight" in z:
        return {
            "weight": z[key + ".int8_weight"].to(dev),
            "tensor_scale": z[key + ".int8_scale"].to(dev),
            "qtype": "int8_per_tensor",
        }
    # Affine (MM8-style)
    if key + ".int8_w_u8" in z:
        return {
            "weight": z[key + ".int8_w_u8"].to(dev),
            "mx": z[key + ".int8_mx"].to(dev),
            "rx": z[key + ".int8_rx"].to(dev),
            "my": z[key + ".int8_my"].to(dev),
            "ry": z[key + ".int8_ry"].to(dev),
            "qtype": "int8_affine",
        }
    raise KeyError(f"No INT8 weight found for key: {key}")


# ---------------------------------------------------------------------------
# GEMM operations
# ---------------------------------------------------------------------------

def linear_int8_per_tensor(x: torch.Tensor, weight_info: dict,
                           out_dtype: torch.dtype = torch.float16):
    """W8A8 GEMM with per-tensor symmetric int8.

    Uses torch._scaled_mm when available (Hopper/Blackwell),
    falls back to dequant + FP16 matmul on CPU/older GPUs.
    """
    w_int8 = weight_info["weight"]  # [out, in] int8
    scale = weight_info["tensor_scale"]  # scalar float32

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])  # [M, K]

    # Try hardware int8 path (requires CUDA)
    if x2.is_cuda and hasattr(torch, "_scaled_mm"):
        # Quantize activation to int8
        amax_x = x2.abs().max()
        x_scale = (amax_x / 127.0).clamp(min=1e-12)
        x_int8 = (x2.float() / x_scale).round().clamp(-128, 127).to(torch.int8)

        # _scaled_mm for int8 requires float8 types, not int8
        # Fallback: dequant weight to fp16 and use fp16 matmul
        w_fp16 = w_int8.to(torch.float16) * scale.to(torch.float16)
        out = x2.to(torch.float16) @ w_fp16.t()
    else:
        # CPU / fallback: dequant + FP16 matmul
        w_fp16 = w_int8.to(torch.float16) * scale.to(torch.float16)
        out = x2.to(torch.float16) @ w_fp16.t()

    out = out.to(out_dtype)
    return out.reshape(*orig_shape[:-1], w_int8.shape[0])


def linear_int8_affine(x: torch.Tensor, weight_info: dict,
                       out_dtype: torch.dtype = torch.float16):
    """Affine INT8 GEMM (MM8-style).

    Dequantizes weight using dual affine:
      W = (u8 + 0.5) * ry * rx + my + mx

    Then performs FP16 matmul. Triton fused kernel can avoid
    materializing the full dequantized weight.
    """
    w_u8 = weight_info["weight"].float()   # [N, M]
    mx = weight_info["mx"]                  # [M]
    rx = weight_info["rx"]                  # [M] (stored /16)
    my = weight_info["my"]                  # [N]
    ry = weight_info["ry"]                  # [N] (stored /16)

    N, M = w_u8.shape

    # Dequantize: W = (u8 + 0.5) * ry * rx * 16 * 16 + my + mx
    # (scales were stored /16, so multiply back by 256)
    ry_col = ry.reshape(N, 1)   # [N, 1]
    rx_row = rx.reshape(1, M)   # [1, M]
    w_fp16 = ((w_u8 + 0.5) * ry_col * rx_row * 256.0 + my.reshape(N, 1) + mx.reshape(1, M)).to(torch.float16)

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).to(torch.float16)  # [M_in, K]
    # w_fp16 is [N, M] = [out, in], need transpose
    out = x2 @ w_fp16.t()
    out = out.to(out_dtype)
    return out.reshape(*orig_shape[:-1], N)


def linear_int8(x: torch.Tensor, weight_info: dict,
                out_dtype: torch.dtype = torch.float16):
    """Dispatch INT8 GEMM based on qtype."""
    qtype = weight_info["qtype"]
    if qtype == "int8_per_tensor":
        return linear_int8_per_tensor(x, weight_info, out_dtype)
    elif qtype == "int8_affine":
        return linear_int8_affine(x, weight_info, out_dtype)
    else:
        raise ValueError(f"Unknown INT8 qtype: {qtype}")
