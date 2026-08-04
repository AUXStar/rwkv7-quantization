#!/usr/bin/env python3
# coding=utf-8
"""INT4 weight detection, loading, nibble unpack, and GEMM for RWKV-7.

Supports three schemes:
  - per_tensor: packed uint8 + scalar scale (W4A16)
  - affine:     packed uint8 + mx/rx/my/ry (MM4-style dequant)
  - groupwise:  packed uint8 + per-group scales/zeros
"""
from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# Nibble unpack utilities
# ---------------------------------------------------------------------------

def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack paired nibble uint8 -> int8.

    packed: [N, M//2] uint8
    Returns: [N, M] int8 (values 0-15, unsigned)

    low nibble = packed & 0xF
    high nibble = (packed >> 4) & 0xF
    """
    lo = packed & 0x0F          # [N, M//2]
    hi = (packed >> 4) & 0x0F   # [N, M//2]
    # Interleave: [lo[0], hi[0], lo[1], hi[1], ...]
    out = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return out.to(torch.int8)


# ---------------------------------------------------------------------------
# Weight detection & loading
# ---------------------------------------------------------------------------

def is_int4_weight(z: dict, key: str) -> bool:
    """Check if a weight key has INT4 quantization metadata."""
    return key + ".int4_packed" in z


def load_int4_weight(z: dict, key: str, dev: str = "cuda"):
    """Load INT4 weight + metadata. Returns dict with weight info."""
    if key + ".int4_packed" not in z:
        raise KeyError(f"No INT4 weight found for key: {key}")

    packed = z[key + ".int4_packed"].to(dev)
    info = {"packed": packed}

    if key + ".int4_scale" in z:
        # Per-tensor symmetric
        info["scale"] = z[key + ".int4_scale"].to(dev)
        info["qtype"] = "int4_per_tensor"
    elif key + ".int4_rx" in z:
        # Affine (MM4-style)
        info["mx"] = z[key + ".int4_mx"].to(dev)
        info["rx"] = z[key + ".int4_rx"].to(dev)
        info["my"] = z[key + ".int4_my"].to(dev)
        info["ry"] = z[key + ".int4_ry"].to(dev)
        info["m_orig"] = z[key + ".int4_m_orig"]
        info["qtype"] = "int4_affine"
    elif key + ".int4_scales" in z:
        # Group-wise
        info["scales"] = z[key + ".int4_scales"].to(dev)
        info["zeros"] = z[key + ".int4_zeros"].to(dev)
        info["group_size"] = z[key + ".int4_group_size"]
        info["qtype"] = "int4_groupwise"
    else:
        raise KeyError(f"Incomplete INT4 metadata for key: {key}")

    return info


# ---------------------------------------------------------------------------
# GEMM operations (reference paths; Triton fused kernels in fused_int4_gemm.py)
# ---------------------------------------------------------------------------

def linear_int4_per_tensor(x: torch.Tensor, weight_info: dict,
                           out_dtype: torch.dtype = torch.float16):
    """W4A16 GEMM with per-tensor symmetric int4.

    Dequantize weight to FP16, then FP16 matmul.
    """
    packed = weight_info["packed"]   # [N, M//2] uint8
    scale = weight_info["scale"]     # scalar fp32

    w_u4 = unpack_int4(packed).float()  # [N, M] values 0-15
    # Symmetric: original values were [-8, 7], stored as unsigned [0, 15]
    # Need to convert: signed = unsigned - 8 (for symmetric range)
    # Actually, per-tensor symmetric stores: lo = (w_q[0::2] & 0xF)
    # where w_q was clamped to [-8, 7] and then & 0xF converts:
    #   -8 -> 8, -7 -> 9, ..., -1 -> 15, 0 -> 0, ..., 7 -> 7
    # So we need to convert back: if val > 7, val -= 16
    w_signed = torch.where(w_u4 > 7, w_u4 - 16, w_u4)
    w_fp16 = (w_signed * scale).to(torch.float16)  # [N, M] = [out, in]

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).to(torch.float16)  # [batch, K]
    out = x2 @ w_fp16.t()  # [batch, N]
    return out.to(out_dtype).reshape(*orig_shape[:-1], w_fp16.shape[0])


def linear_int4_affine(x: torch.Tensor, weight_info: dict,
                       out_dtype: torch.dtype = torch.float16):
    """Affine INT4 GEMM (MM4-style).

    Dequantize: W = (u4 + 0.5) * ry * rx * 16 + my + mx
    (scales stored /4, so multiply back by 4*4=16)
    """
    packed = weight_info["packed"]  # [N, M_pad//2] uint8
    mx = weight_info["mx"]          # [M_pad]
    rx = weight_info["rx"]          # [M_pad] (stored /4)
    my = weight_info["my"]          # [N]
    ry = weight_info["ry"]          # [N] (stored /4)
    m_orig = weight_info["m_orig"]

    w_u4 = unpack_int4(packed).float()  # [N, M_pad] values 0-15
    N, M_pad = w_u4.shape

    # Dequantize
    ry_col = ry.reshape(N, 1)    # [N, 1]
    rx_row = rx.reshape(1, M_pad)  # [1, M_pad]
    w_fp16 = ((w_u4 + 0.5) * ry_col * rx_row * 16.0 + my.reshape(N, 1) + mx.reshape(1, M_pad))
    w_fp16 = w_fp16[:, :m_orig].to(torch.float16)  # trim padding

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).to(torch.float16)
    out = x2 @ w_fp16.t()
    return out.to(out_dtype).reshape(*orig_shape[:-1], N)


def linear_int4_groupwise(x: torch.Tensor, weight_info: dict,
                          out_dtype: torch.dtype = torch.float16):
    """Group-wise INT4 GEMM (W4A16).

    Dequantize with per-group scale and zero point.
    """
    packed = weight_info["packed"]      # [N, M//2] uint8
    scales = weight_info["scales"]      # [N, n_groups]
    zeros = weight_info["zeros"]        # [N, n_groups]
    group_size = weight_info["group_size"]

    w_u4 = unpack_int4(packed).float()  # [N, M] values 0-15
    N, M = w_u4.shape
    n_groups = M // group_size

    # Reshape to groups and dequantize
    w_grouped = w_u4.reshape(N, n_groups, group_size)
    s = scales.reshape(N, n_groups, 1)
    z = zeros.reshape(N, n_groups, 1)
    w_fp16 = (w_grouped * s + z).reshape(N, M).to(torch.float16)

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).to(torch.float16)
    out = x2 @ w_fp16.t()
    return out.to(out_dtype).reshape(*orig_shape[:-1], N)


def linear_int4(x: torch.Tensor, weight_info: dict,
                out_dtype: torch.dtype = torch.float16):
    """Dispatch INT4 GEMM based on qtype."""
    qtype = weight_info["qtype"]
    if qtype == "int4_per_tensor":
        return linear_int4_per_tensor(x, weight_info, out_dtype)
    elif qtype == "int4_affine":
        return linear_int4_affine(x, weight_info, out_dtype)
    elif qtype == "int4_groupwise":
        return linear_int4_groupwise(x, weight_info, out_dtype)
    else:
        raise ValueError(f"Unknown INT4 qtype: {qtype}")
