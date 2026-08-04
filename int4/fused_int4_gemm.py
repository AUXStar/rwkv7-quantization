#!/usr/bin/env python3
# coding=utf-8
"""Fused INT4 GEMM Triton kernels for RWKV-7 decode.

Kernels:
  - fused_int4_gemv_kernel:    single-vector GEMV with nibble unpack
  - fused_int4_affine_gemv:    affine MM4-style GEMV (in-register dequant)
  - fused_int4_groupwise_gemv: group-wise GEMV (per-group scale/zero)

All kernels do in-register nibble unpacking to avoid materializing
the full dequantized weight, reducing memory traffic by 2x.
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
    def _int4_gemv_kernel(
        x_ptr, packed_ptr, scale_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Single-vector INT4 GEMV: y = x @ (unpack(packed) * scale).T

        x: [K] fp16
        packed: [N, K//2] uint8 (paired nibble along K)
        scale: scalar fp32
        out: [N] fp16

        Note: weight stored as [N, K] = [out, in], packed along K.
        Each byte holds two int4 values from adjacent K positions.
        """
        pid_n = tl.program_id(0)
        offset_k = tl.arange(0, BLOCK_K // 2)
        acc = tl.zeros([1], dtype=tl.float32)

        scale_val = tl.load(scale_ptr).to(tl.float32)

        for k_start in range(0, K // 2, BLOCK_K // 2):
            k = k_start + offset_k
            mask = k < K // 2

            # Load one packed byte -> two int4 values
            byte_val = tl.load(packed_ptr + pid_n * (K // 2) + k,
                               mask=mask, other=0).to(tl.int32)
            lo = (byte_val & 0xF).to(tl.float32)    # low nibble
            hi = ((byte_val >> 4) & 0xF).to(tl.float32)  # high nibble

            # Convert to signed: values 8-15 represent -8 to -1
            lo_signed = tl.where(lo > 7, lo - 16, lo)
            hi_signed = tl.where(hi > 7, hi - 16, hi)

            # Load corresponding input values
            x_lo = tl.load(x_ptr + (k * 2), mask=mask, other=0.0).to(tl.float32)
            x_hi = tl.load(x_ptr + (k * 2 + 1), mask=mask, other=0.0).to(tl.float32)

            # Accumulate
            acc += x_lo * lo_signed * scale_val + x_hi * hi_signed * scale_val

        tl.store(out_ptr + pid_n, acc.to(tl.float16))

    @triton.jit
    def _int4_affine_gemv_kernel(
        x_ptr, packed_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Affine INT4 GEMV: y = x @ ((u4+0.5)*ry*rx*16 + my + mx).T

        x: [K] fp16
        packed: [N, K//2] uint8
        mx: [K] fp32, rx: [K] fp32 (stored /4)
        my: [N] fp32, ry: [N] fp32 (stored /4)
        out: [N] fp16
        """
        pid_n = tl.program_id(0)
        offset_k = tl.arange(0, BLOCK_K // 2)

        ry_val = tl.load(ry_ptr + pid_n).to(tl.float32)
        my_val = tl.load(my_ptr + pid_n).to(tl.float32)

        acc = tl.zeros([1], dtype=tl.float32)

        for k_start in range(0, K // 2, BLOCK_K // 2):
            k = k_start + offset_k
            mask = k < K // 2

            byte_val = tl.load(packed_ptr + pid_n * (K // 2) + k,
                               mask=mask, other=0).to(tl.float32)
            lo = byte_val & 0xF
            hi = (byte_val >> 4) & 0xF

            # Load per-col quantities
            mx_lo = tl.load(mx_ptr + (k * 2), mask=mask, other=0.0).to(tl.float32)
            mx_hi = tl.load(mx_ptr + (k * 2 + 1), mask=mask, other=0.0).to(tl.float32)
            rx_lo = tl.load(rx_ptr + (k * 2), mask=mask, other=0.0).to(tl.float32)
            rx_hi = tl.load(rx_ptr + (k * 2 + 1), mask=mask, other=0.0).to(tl.float32)

            # Dequantize: w = (u4 + 0.5) * ry * rx * 16 + my + mx
            w_lo = (lo + 0.5) * ry_val * rx_lo * 16.0 + my_val + mx_lo
            w_hi = (hi + 0.5) * ry_val * rx_hi * 16.0 + my_val + mx_hi

            x_lo = tl.load(x_ptr + (k * 2), mask=mask, other=0.0).to(tl.float32)
            x_hi = tl.load(x_ptr + (k * 2 + 1), mask=mask, other=0.0).to(tl.float32)

            acc += x_lo * w_lo + x_hi * w_hi

        tl.store(out_ptr + pid_n, acc.to(tl.float16))

    @triton.jit
    def _int4_groupwise_gemv_kernel(
        x_ptr, packed_ptr, scales_ptr, zeros_ptr, out_ptr,
        K: tl.constexpr, N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Group-wise INT4 GEMV: y = x @ (unpack(packed) * scale_g + zero_g).T

        x: [K] fp16
        packed: [N, K//2] uint8
        scales: [N, K//GROUP_SIZE] fp32
        zeros:  [N, K//GROUP_SIZE] fp32
        out: [N] fp16
        """
        pid_n = tl.program_id(0)
        n_groups = K // GROUP_SIZE
        offset_k = tl.arange(0, BLOCK_K // 2)
        acc = tl.zeros([1], dtype=tl.float32)

        for k_start in range(0, K // 2, BLOCK_K // 2):
            k = k_start + offset_k
            mask = k < K // 2

            byte_val = tl.load(packed_ptr + pid_n * (K // 2) + k,
                               mask=mask, other=0).to(tl.float32)
            lo = byte_val & 0xF
            hi = (byte_val >> 4) & 0xF

            # Determine group index for each position
            g_lo = (k * 2) // GROUP_SIZE
            g_hi = (k * 2 + 1) // GROUP_SIZE

            # Load group scale and zero
            scale_lo = tl.load(scales_ptr + pid_n * n_groups + g_lo,
                               mask=mask, other=0.0).to(tl.float32)
            zero_lo = tl.load(zeros_ptr + pid_n * n_groups + g_lo,
                              mask=mask, other=0.0).to(tl.float32)
            scale_hi = tl.load(scales_ptr + pid_n * n_groups + g_hi,
                               mask=mask, other=0.0).to(tl.float32)
            zero_hi = tl.load(zeros_ptr + pid_n * n_groups + g_hi,
                              mask=mask, other=0.0).to(tl.float32)

            # Dequantize
            w_lo = lo * scale_lo + zero_lo
            w_hi = hi * scale_hi + zero_hi

            x_lo = tl.load(x_ptr + (k * 2), mask=mask, other=0.0).to(tl.float32)
            x_hi = tl.load(x_ptr + (k * 2 + 1), mask=mask, other=0.0).to(tl.float32)

            acc += x_lo * w_lo + x_hi * w_hi

        tl.store(out_ptr + pid_n, acc.to(tl.float16))


def linear_int4_gemv_triton(x: torch.Tensor, packed: torch.Tensor,
                            scale: torch.Tensor) -> torch.Tensor:
    """Single-vector INT4 GEMV via Triton (per-tensor)."""
    K, N = x.shape[-1], packed.shape[0]
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    BLOCK_K = min(1024, triton.next_power_of_2(K))
    grid = (N,)
    _int4_gemv_kernel[grid](x, packed, scale, out, K, N, BLOCK_K)
    return out


def linear_int4_affine_gemv_triton(x, packed, mx, rx, my, ry) -> torch.Tensor:
    """Single-vector affine INT4 GEMV via Triton."""
    K, N = x.shape[-1], packed.shape[0]
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    BLOCK_K = min(1024, triton.next_power_of_2(K))
    grid = (N,)
    _int4_affine_gemv_kernel[grid](
        x, packed, mx, rx, my, ry, out, K, N, BLOCK_K
    )
    return out


def linear_int4_groupwise_gemv_triton(x, packed, scales, zeros,
                                       group_size: int) -> torch.Tensor:
    """Single-vector group-wise INT4 GEMV via Triton."""
    K, N = x.shape[-1], packed.shape[0]
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    BLOCK_K = min(1024, triton.next_power_of_2(K))
    grid = (N,)
    _int4_groupwise_gemv_kernel[grid](
        x, packed, scales, zeros, out, K, N, group_size, BLOCK_K
    )
    return out
