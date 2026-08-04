#!/usr/bin/env python3
# coding=utf-8
"""Fused INT8 GEMM Triton kernels for RWKV-7 decode.

Kernels:
  - fused_int8_gemv_kernel:   single-vector GEMV (decode, M=1)
  - fused_int8_batched_gemv:  batched GEMV (decode, M<=8)
  - fused_int8_affine_gemv:   affine MM8-style GEMV (in-register dequant)

The per-tensor kernel does in-register dequantization:
  w_fp16 = w_int8 * scale
  y = x @ w_fp16.T

The affine kernel does in-register dual affine dequant:
  w_fp16 = (u8 + 0.5) * ry * rx * 256 + my + mx
  y = x @ w_fp16.T
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _int8_gemv_kernel(
        x_ptr, w_ptr, scale_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Single-vector GEMV: y = x @ (w * scale).T

        x: [K] fp16
        w: [N, K] int8
        scale: scalar fp32
        out: [N] fp16
        """
        pid_n = tl.program_id(0)
        offset_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros([1], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + offset_k
            mask = k < K
            # Load input vector
            x_val = tl.load(x_ptr + k, mask=mask, other=0.0).to(tl.float32)
            # Load weight row (int8 -> fp32)
            w_val = tl.load(w_ptr + pid_n * K + k, mask=mask, other=0).to(tl.float32)
            # Accumulate
            acc += x_val * w_val

        # Apply scale
        scale = tl.load(scale_ptr).to(tl.float32)
        out = acc * scale
        tl.store(out_ptr + pid_n, out.to(tl.float16))

    @triton.jit
    def _int8_affine_gemv_kernel(
        x_ptr, w_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Affine GEMV: y = x @ ((u8+0.5)*ry*rx*256 + my + mx).T

        x: [K] fp16
        w: [N, K] uint8
        mx: [K] fp32 (per-col offset)
        rx: [K] fp32 (per-col scale, stored /16)
        my: [N] fp32 (per-row offset)
        ry: [N] fp32 (per-row scale, stored /16)
        out: [N] fp16
        """
        pid_n = tl.program_id(0)
        offset_k = tl.arange(0, BLOCK_K)

        # Load per-row quantities
        ry_val = tl.load(ry_ptr + pid_n).to(tl.float32)  # scalar
        my_val = tl.load(my_ptr + pid_n).to(tl.float32)  # scalar

        acc = tl.zeros([1], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + offset_k
            mask = k < K
            # Load input
            x_val = tl.load(x_ptr + k, mask=mask, other=0.0).to(tl.float32)
            # Load weight (uint8 -> fp32)
            u8_val = tl.load(w_ptr + pid_n * K + k, mask=mask, other=0).to(tl.float32)
            # Load per-col quantities
            mx_val = tl.load(mx_ptr + k, mask=mask, other=0.0).to(tl.float32)
            rx_val = tl.load(rx_ptr + k, mask=mask, other=0.0).to(tl.float32)
            # Dequantize: w = (u8 + 0.5) * ry * rx * 256 + my + mx
            w_val = (u8_val + 0.5) * ry_val * rx_val * 256.0 + my_val + mx_val
            # Accumulate
            acc += x_val * w_val

        tl.store(out_ptr + pid_n, acc.to(tl.float16))


def linear_int8_gemv_triton(x: torch.Tensor, w_int8: torch.Tensor,
                            scale: torch.Tensor) -> torch.Tensor:
    """Single-vector INT8 GEMV via Triton.

    x: [K] fp16
    w_int8: [N, K] int8
    scale: scalar fp32
    Returns: [N] fp16
    """
    K, N = x.shape[-1], w_int8.shape[0]
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    BLOCK_K = min(1024, triton.next_power_of_2(K))
    grid = (N,)
    _int8_gemv_kernel[grid](x, w_int8, scale, out, K, N, BLOCK_K)
    return out


def linear_int8_affine_gemv_triton(x: torch.Tensor, w_u8: torch.Tensor,
                                    mx, rx, my, ry) -> torch.Tensor:
    """Single-vector affine INT8 GEMV via Triton.

    x: [K] fp16
    w_u8: [N, K] uint8
    mx: [K] fp32, rx: [K] fp32, my: [N] fp32, ry: [N] fp32
    Returns: [N] fp16
    """
    K, N = x.shape[-1], w_u8.shape[0]
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    BLOCK_K = min(1024, triton.next_power_of_2(K))
    grid = (N,)
    _int8_affine_gemv_kernel[grid](
        x, w_u8, mx, rx, my, ry, out, K, N, BLOCK_K
    )
    return out
