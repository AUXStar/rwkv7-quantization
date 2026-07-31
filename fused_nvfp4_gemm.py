#!/usr/bin/env python3
"""Fused quantized GEMM kernels (minimal-launch, no intermediate memory).

v2: launch-count optimization for decode (CPU-launch-bound):

  per quantized linear:  prep_x (1 launch: cast+AWQ+amax) + GEMM (1 launch)
  NVFP4+FP8 residual:    prep_x (1 launch) + fused res GEMM (1 launch)
  (was 4-9 launches per linear before this rewrite)

Design:
  prep_x:   x(fp16/bf16) + AWQ scale -> x_awq(bf16) + amax (GPU scalar)
  fused_nvfp4_gemm_kernel:  reads amax_ptr (no D2H sync), in-register FP4
            quantization + FP4×FP4 dot; weights stay packed FP4 in memory.
  fused_nvfp4_res_gemm_kernel: FP4 main GEMM + FP8 residual GEMM in ONE kernel
            (same x tile reused, one launch).
  fused_fp8_gemm_kernel: in-register FP8 quant (per-tensor from amax) + FP8×FP8.

Numerics match the validated _scaled_mm path:
- activation per-16-block scale: clamp(max_abs*448/amax, 0.015625, 448) -> fp8
- FP4 RNE lookup identical to fused_nvfp4_quant.py
- weights decoded in-register from packed E2M1 codes / E4M3 bytes
"""
import torch
import triton
import triton.language as tl

F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
F8E4M3_MIN = 0.015625
NVFP4_TS_DIV = 448.0 * 6.0


# ============================================================================
# Input preparation: cast + AWQ + amax in one kernel
# ============================================================================

@triton.jit
def prep_x_kernel(
    x_ptr,          # [M, K] fp16/bf16
    awq_ptr,        # [K] fp32 or null (has_awq=False)
    out_ptr,        # [M, K] bf16 (AWQ applied)
    amax_ptr,       # scalar fp32 (atomic max accumulator, init 0)
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    has_awq: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(x_ptr + pid_m * stride_xm + offs_k * stride_xk, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    if has_awq:
        awq = tl.load(awq_ptr + offs_k, mask=mask, other=1.0)
        xf = xf / awq
    tl.store(out_ptr + pid_m * stride_om + offs_k * stride_ok, xf.to(tl.bfloat16), mask=mask)
    row_max = tl.max(tl.abs(xf))
    tl.atomic_max(amax_ptr, row_max)


def prep_x(x, awq_scale=None, out=None):
    """cast(AWQ(x)) -> bf16 + amax (GPU scalar). One launch.

    x: [M, K] fp16/bf16 (contiguous)
    Returns (x_awq_bf16, amax) both on GPU.
    """
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype != torch.float16 and x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    if x2.stride(0) != x2.size(1) or x2.stride(1) != 1:
        x2 = x2.contiguous()
    M, K = x2.shape

    if out is None or out.shape != x2.shape:
        out = torch.empty(x2.shape, dtype=torch.bfloat16, device=x2.device)
    amax = torch.zeros(1, dtype=torch.float32, device=x2.device)

    BLOCK_K = triton.next_power_of_2(K)
    grid = (M,)
    prep_x_kernel[grid](
        x2, awq_scale, out, amax,
        M, K,
        x2.stride(0), x2.stride(1),
        out.stride(0), out.stride(1),
        has_awq=awq_scale is not None,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out, amax


# ============================================================================
# RKV fused prep: cast + AWQ + amax for xr/xk/xv in ONE launch
# ============================================================================

@triton.jit
def prep3_x_kernel(
    xr_ptr, xk_ptr, xv_ptr,
    awq_r_ptr, awq_k_ptr,
    or_ptr, ok_ptr, ov_ptr,
    amax_r_ptr, amax_k_ptr, amax_v_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    AWQ_R: tl.constexpr, AWQ_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K

    xr = tl.load(xr_ptr + pid_m * stride_xm + offs_k * stride_xk, mask=mask, other=0.0).to(tl.float32)
    if AWQ_R:
        awq = tl.load(awq_r_ptr + offs_k, mask=mask, other=1.0)
        xr = xr / awq
    tl.store(or_ptr + pid_m * stride_om + offs_k * stride_ok, xr.to(tl.bfloat16), mask=mask)
    tl.atomic_max(amax_r_ptr, tl.max(tl.abs(xr)))

    xk = tl.load(xk_ptr + pid_m * stride_xm + offs_k * stride_xk, mask=mask, other=0.0).to(tl.float32)
    if AWQ_K:
        awq = tl.load(awq_k_ptr + offs_k, mask=mask, other=1.0)
        xk = xk / awq
    tl.store(ok_ptr + pid_m * stride_om + offs_k * stride_ok, xk.to(tl.bfloat16), mask=mask)
    tl.atomic_max(amax_k_ptr, tl.max(tl.abs(xk)))

    xv = tl.load(xv_ptr + pid_m * stride_xm + offs_k * stride_xk, mask=mask, other=0.0).to(tl.float32)
    tl.store(ov_ptr + pid_m * stride_om + offs_k * stride_ok, xv.to(tl.bfloat16), mask=mask)
    tl.atomic_max(amax_v_ptr, tl.max(tl.abs(xv)))


def prep3_x(xr, xk, xv, awq_r=None, awq_k=None):
    """prep_x for three attention inputs in ONE launch.

    Returns (xr_a, xk_a, xv_a, amax_r, amax_k, amax_v) all bf16/fp32 on GPU.
    """
    outs = []
    amaxs = []
    for xx in (xr, xk, xv):
        x2 = xx.reshape(-1, xx.shape[-1])
        if x2.dtype != torch.float16 and x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if x2.stride(0) != x2.size(1) or x2.stride(1) != 1:
            x2 = x2.contiguous()
        outs.append(torch.empty(x2.shape, dtype=torch.bfloat16, device=x2.device))
        amaxs.append(torch.zeros(1, dtype=torch.float32, device=x2.device))
    M, K = outs[0].shape
    BLOCK_K = triton.next_power_of_2(K)
    grid = (M,)
    prep3_x_kernel[grid](
        xr, xk, xv,
        awq_r, awq_k,
        outs[0], outs[1], outs[2],
        amaxs[0], amaxs[1], amaxs[2],
        M, K,
        xr.stride(0), xr.stride(1),
        outs[0].stride(0), outs[0].stride(1),
        AWQ_R=awq_r is not None, AWQ_K=awq_k is not None,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return outs[0], outs[1], outs[2], amaxs[0], amaxs[1], amaxs[2]


# ============================================================================
# FP4 helpers (in-kernel)
# ============================================================================

@triton.jit
def _fp4_rne(x):
    """FP4 E2M1 round-to-nearest-even code lookup (matches fused_nvfp4_quant)."""
    sign = tl.where(x < 0, 1, 0).to(tl.uint8)
    a = tl.abs(x)
    code = tl.where(a <= 0.25, 0,
           tl.where(a < 0.75, 1,
           tl.where(a <= 1.25, 2,
           tl.where(a < 1.75, 3,
           tl.where(a <= 2.5, 4,
           tl.where(a < 3.5, 5,
           tl.where(a <= 5.0, 6, 7))))))).to(tl.uint8)
    return sign * 8 + code


@triton.jit
def _fp4_val(code):
    """Dequantize FP4 E2M1 code (uint8 0-15) to fp32 value (exact)."""
    exp = (code >> 1) & 3
    man = code & 1
    sign = (code >> 3) & 1
    mag = tl.where(exp == 0, man * 0.5,
                   (1.0 + 0.5 * man) * tl.exp2(exp.to(tl.float32) - 1.0))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _e4m3_val(u8):
    """Dequantize E4M3 byte (uint8, IEEE e4m3fn bit layout) to fp32."""
    sign = (u8 >> 7) & 1
    exp = (u8 >> 3) & 0xF
    man = u8 & 0x7
    mag = tl.where(exp == 0, man * (1.0 / 512.0),
                   (1.0 + man / 8.0) * tl.exp2(exp.to(tl.float32) - 7.0))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _expand_blocks(s, BLOCK_M: tl.constexpr, NKB: tl.constexpr, BLOCK_K: tl.constexpr):
    """Expand [BLOCK_M, NKB] per-16-block scales to [BLOCK_M, BLOCK_K]."""
    s_r = tl.reshape(s, (BLOCK_M, NKB, 1))
    s_b = tl.broadcast_to(s_r, (BLOCK_M, NKB, 16))
    return tl.reshape(s_b, (BLOCK_M, BLOCK_K))


@triton.jit
def _expand_blocks_n(s, BLOCK_N: tl.constexpr, NKB: tl.constexpr, BLOCK_K: tl.constexpr):
    """Expand [BLOCK_N, NKB] per-16-block scales to [BLOCK_N, BLOCK_K]."""
    s_r = tl.reshape(s, (BLOCK_N, NKB, 1))
    s_b = tl.broadcast_to(s_r, (BLOCK_N, NKB, 16))
    return tl.reshape(s_b, (BLOCK_N, BLOCK_K))


# ============================================================================
# Fused NVFP4 (FP4×FP4) GEMM kernel — reads amax_ptr (no D2H sync)
# ============================================================================

@triton.jit
def fused_nvfp4_gemm_kernel(
    x_ptr,           # [M, K] bf16 (AWQ already applied by prep_x)
    w_ptr,           # [N, K//2] uint8 packed FP4
    w_bs_ptr,        # [N, K//16] uint8 (bit-view of float8_e4m3fn block scales)
    w_ts_ptr,        # fp32 scalar (weight per-tensor scale)
    amax_ptr,        # fp32 scalar (activation amax from prep_x)
    out_ptr,         # [M, N] fp16 output
    M, N, K,
    stride_xm,
    stride_wn,       # = K // 2
    stride_wbsn,     # = K // 16
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    offs_nkb = tl.arange(0, BLOCK_K // 16)
    NKB: tl.constexpr = BLOCK_K // 16

    amax_v = tl.maximum(tl.load(amax_ptr), 1e-12)
    inv_pts = 2688.0 / amax_v   # = 1/(amax/2688)
    w_ts_v = tl.load(w_ts_ptr)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = x_tile.to(tl.float32)  # [BM, BK]

        # --- activation FP4 quantization (matches fused_nvfp4_quant) ---
        x_r = tl.reshape(x32, (BLOCK_M, NKB, 16))
        max_abs = tl.max(tl.abs(x_r), axis=2)  # [BM, NKB]
        scaled_bs = max_abs / 6.0 * inv_pts
        scaled_bs = tl.minimum(tl.maximum(scaled_bs, 0.015625), 448.0)
        bs_a = scaled_bs.to(tl.float8e4nv).to(tl.float32)  # [BM, NKB] fp8-rounded
        recip = inv_pts / bs_a
        x_scaled = x32 * _expand_blocks(recip, BLOCK_M, NKB, BLOCK_K)
        x_scaled = tl.minimum(tl.maximum(x_scaled, -6.0), 6.0)
        code_a = _fp4_rne(x_scaled)  # [BM, BK] uint8
        a_val = _fp4_val(code_a)     # [BM, BK] fp32
        a_eff = (a_val * _expand_blocks(bs_a, BLOCK_M, NKB, BLOCK_K)).to(tl.float16)

        # --- weight side: packed FP4 + fp8 block scales ---
        w_packed = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start // 2 + offs_k2)[None, :],
                           mask=(offs_n[:, None] < N) & ((k_start // 2 + offs_k2[None, :]) < K // 2), other=0)
        lo = w_packed & 0xF
        hi = (w_packed >> 4) & 0xF
        w_codes = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))  # [BN, BK] uint8
        b_val = _fp4_val(w_codes)  # [BN, BK] fp32
        w_bs_u8 = tl.load(w_bs_ptr + offs_n[:, None] * stride_wbsn + (k_start // 16 + offs_nkb)[None, :],
                          mask=(offs_n[:, None] < N) & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0)
        bs_b = _e4m3_val(w_bs_u8)  # [BN, NKB] fp32
        b_eff = (b_val * _expand_blocks_n(bs_b, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)

        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    # fold per-tensor scales: pts = amax/2688
    acc = acc * (amax_v / 2688.0 * w_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Fused NVFP4 + FP8 residual GEMM kernel (ONE launch, x tile reused)
# ============================================================================

@triton.jit
def fused_nvfp4_res_gemm_kernel(
    x_ptr,           # [M, K] bf16 (AWQ applied)
    w_ptr,           # [N, K//2] uint8 packed FP4
    w_bs_ptr,        # [N, K//16] uint8 (fp8 block scales)
    w_ts_ptr,        # fp32 scalar (main per-tensor scale)
    res_w_ptr,       # [N, K] uint8 (fp8 residual weights)
    res_ts_ptr,      # fp32 scalar (residual per-tensor scale)
    amax_ptr,        # fp32 scalar (activation amax)
    out_ptr,         # [M, N] fp16 output
    M, N, K,
    stride_xm,
    stride_wn, stride_wbsn, stride_resn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    offs_nkb = tl.arange(0, BLOCK_K // 16)
    NKB: tl.constexpr = BLOCK_K // 16

    amax_v = tl.maximum(tl.load(amax_ptr), 1e-12)
    inv_pts = 2688.0 / amax_v
    inv_xs = 448.0 / amax_v    # for residual FP8 path: xs = amax/448
    w_ts_v = tl.load(w_ts_ptr)
    res_ts_v = tl.load(res_ts_ptr)

    acc_main = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_res = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = x_tile.to(tl.float32)  # [BM, BK]

        # --- main: FP4 path ---
        x_r = tl.reshape(x32, (BLOCK_M, NKB, 16))
        max_abs = tl.max(tl.abs(x_r), axis=2)
        scaled_bs = max_abs / 6.0 * inv_pts
        scaled_bs = tl.minimum(tl.maximum(scaled_bs, 0.015625), 448.0)
        bs_a = scaled_bs.to(tl.float8e4nv).to(tl.float32)
        recip = inv_pts / bs_a
        x_scaled = tl.minimum(tl.maximum(x32 * _expand_blocks(recip, BLOCK_M, NKB, BLOCK_K), -6.0), 6.0)
        a_val = _fp4_val(_fp4_rne(x_scaled))
        a_eff = (a_val * _expand_blocks(bs_a, BLOCK_M, NKB, BLOCK_K)).to(tl.float16)

        w_packed = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start // 2 + offs_k2)[None, :],
                           mask=(offs_n[:, None] < N) & ((k_start // 2 + offs_k2[None, :]) < K // 2), other=0)
        lo = w_packed & 0xF
        hi = (w_packed >> 4) & 0xF
        w_codes = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
        b_val = _fp4_val(w_codes)
        w_bs_u8 = tl.load(w_bs_ptr + offs_n[:, None] * stride_wbsn + (k_start // 16 + offs_nkb)[None, :],
                          mask=(offs_n[:, None] < N) & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0)
        bs_b = _e4m3_val(w_bs_u8)
        b_eff = (b_val * _expand_blocks_n(bs_b, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)
        acc_main = tl.dot(a_eff, tl.trans(b_eff), acc_main)

        # --- residual: FP8 path (same x tile, per-tensor scale) ---
        x_fp8 = tl.minimum(tl.maximum(x32 * inv_xs, -448.0), 448.0)
        a_r = x_fp8.to(tl.float8e4nv).to(tl.float32).to(tl.float16)  # [BM, BK]
        res_u8 = tl.load(res_w_ptr + offs_n[:, None] * stride_resn + (k_start + offs_k)[None, :],
                         mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_r = _e4m3_val(res_u8).to(tl.float16)  # [BN, BK]
        acc_res = tl.dot(a_r, tl.trans(b_r), acc_res)

    out = acc_main * (amax_v / 2688.0 * w_ts_v) + acc_res * (amax_v / 448.0 * res_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Fused FP8 (W8A8) GEMM kernel — reads amax_ptr
# ============================================================================

@triton.jit
def fused_fp8_gemm_kernel(
    x_ptr,           # [M, K] bf16
    w_ptr,           # [N, K] uint8 (bit-view of float8_e4m3fn)
    w_ts_ptr,        # fp32 scalar (weight per-tensor scale)
    amax_ptr,        # fp32 scalar (activation amax)
    out_ptr,         # [M, N] fp16 output
    M, N, K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    amax_v = tl.maximum(tl.load(amax_ptr), 1e-12)
    inv_xs = 448.0 / amax_v
    w_ts_v = tl.load(w_ts_ptr)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = tl.minimum(tl.maximum(x_tile.to(tl.float32) * inv_xs, -448.0), 448.0)
        a_eff = x32.to(tl.float8e4nv).to(tl.float32).to(tl.float16)  # [BM, BK]

        w_u8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                       mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_eff = _e4m3_val(w_u8).to(tl.float16)  # [BN, BK]

        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    acc = acc * (amax_v / 448.0 * w_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Fused RKV GEMM kernel: r(NVFP4) + k(NVFP4|FP8) + v(FP8) in ONE launch
# ============================================================================

@triton.jit
def fused_rkv_gemm_kernel(
    xr_ptr, xk_ptr, xv_ptr,      # [M, K] bf16 (AWQ applied by prep3_x)
    wr_ptr, wr_bs_ptr, wts_r_ptr,  # r: NVFP4 (packed FP4 + fp8 block scales)
    wk_ptr, wk_bs_ptr, wts_k_ptr,  # k: NVFP4 (if K_IS_FP4) or FP8
    wv_ptr, wts_v_ptr,             # v: FP8
    amax_r_ptr, amax_k_ptr, amax_v_ptr,
    or_ptr, ok_ptr, ov_ptr,
    M, N, K,
    stride_xm,
    stride_wn, stride_wbsn, stride_wk, stride_wv, stride_om,
    K_IS_FP4: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    offs_nkb = tl.arange(0, BLOCK_K // 16)
    NKB: tl.constexpr = BLOCK_K // 16

    amax_r = tl.maximum(tl.load(amax_r_ptr), 1e-12)
    amax_k = tl.maximum(tl.load(amax_k_ptr), 1e-12)
    amax_v = tl.maximum(tl.load(amax_v_ptr), 1e-12)
    inv_pts_r = 2688.0 / amax_r
    inv_pts_k = 2688.0 / amax_k
    inv_xs_k = 448.0 / amax_k
    inv_xs_v = 448.0 / amax_v
    wts_r = tl.load(wts_r_ptr)
    wts_k = tl.load(wts_k_ptr)
    wts_v = tl.load(wts_v_ptr)

    acc_r = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_k = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        xmask = (offs_m[:, None] < M) & kmask[None, :]
        wmask = (offs_n[:, None] < N)

        # ---------- r: NVFP4 (FP4×FP4) ----------
        xr = tl.load(xr_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        xr32 = xr.to(tl.float32)
        xr_r = tl.reshape(xr32, (BLOCK_M, NKB, 16))
        max_r = tl.max(tl.abs(xr_r), axis=2)
        s_r = tl.minimum(tl.maximum(max_r / 6.0 * inv_pts_r, 0.015625), 448.0).to(tl.float8e4nv).to(tl.float32)
        xr_s = tl.minimum(tl.maximum(xr32 * _expand_blocks(inv_pts_r / s_r, BLOCK_M, NKB, BLOCK_K), -6.0), 6.0)
        a_r = (_fp4_val(_fp4_rne(xr_s)) * _expand_blocks(s_r, BLOCK_M, NKB, BLOCK_K)).to(tl.float16)
        w_rp = tl.load(wr_ptr + offs_n[:, None] * stride_wn + (k_start // 2 + offs_k2)[None, :],
                       mask=wmask & ((k_start // 2 + offs_k2[None, :]) < K // 2), other=0)
        w_rc = tl.reshape(tl.join(w_rp & 0xF, (w_rp >> 4) & 0xF), (BLOCK_N, BLOCK_K))
        w_rbs = _e4m3_val(tl.load(wr_bs_ptr + offs_n[:, None] * stride_wbsn + (k_start // 16 + offs_nkb)[None, :],
                                  mask=wmask & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0))
        b_r = (_fp4_val(w_rc) * _expand_blocks_n(w_rbs, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)
        acc_r = tl.dot(a_r, tl.trans(b_r), acc_r)

        # ---------- k: NVFP4 or FP8 ----------
        xk = tl.load(xk_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        xk32 = xk.to(tl.float32)
        if K_IS_FP4:
            xk_r = tl.reshape(xk32, (BLOCK_M, NKB, 16))
            max_k = tl.max(tl.abs(xk_r), axis=2)
            s_k = tl.minimum(tl.maximum(max_k / 6.0 * inv_pts_k, 0.015625), 448.0).to(tl.float8e4nv).to(tl.float32)
            xk_s = tl.minimum(tl.maximum(xk32 * _expand_blocks(inv_pts_k / s_k, BLOCK_M, NKB, BLOCK_K), -6.0), 6.0)
            a_k = (_fp4_val(_fp4_rne(xk_s)) * _expand_blocks(s_k, BLOCK_M, NKB, BLOCK_K)).to(tl.float16)
            w_kp = tl.load(wk_ptr + offs_n[:, None] * stride_wn + (k_start // 2 + offs_k2)[None, :],
                           mask=wmask & ((k_start // 2 + offs_k2[None, :]) < K // 2), other=0)
            w_kc = tl.reshape(tl.join(w_kp & 0xF, (w_kp >> 4) & 0xF), (BLOCK_N, BLOCK_K))
            w_kbs = _e4m3_val(tl.load(wk_bs_ptr + offs_n[:, None] * stride_wbsn + (k_start // 16 + offs_nkb)[None, :],
                                      mask=wmask & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0))
            b_k = (_fp4_val(w_kc) * _expand_blocks_n(w_kbs, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)
            acc_k = tl.dot(a_k, tl.trans(b_k), acc_k)
        else:
            a_k = tl.minimum(tl.maximum(xk32 * inv_xs_k, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32).to(tl.float16)
            w_k = _e4m3_val(tl.load(wk_ptr + offs_n[:, None] * stride_wk + (k_start + offs_k)[None, :],
                                    mask=wmask & kmask[None, :], other=0)).to(tl.float16)
            acc_k = tl.dot(a_k, tl.trans(w_k), acc_k)

        # ---------- v: FP8 (W8A8) ----------
        xv = tl.load(xv_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        a_v = tl.minimum(tl.maximum(xv.to(tl.float32) * inv_xs_v, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32).to(tl.float16)
        w_v = _e4m3_val(tl.load(wv_ptr + offs_n[:, None] * stride_wv + (k_start + offs_k)[None, :],
                                mask=wmask & kmask[None, :], other=0)).to(tl.float16)
        acc_v = tl.dot(a_v, tl.trans(w_v), acc_v)

    # fold scales
    out_r = (acc_r * (amax_r / 2688.0 * wts_r)).to(tl.float16)
    if K_IS_FP4:
        out_k = (acc_k * (amax_k / 2688.0 * wts_k)).to(tl.float16)
    else:
        out_k = (acc_k * (amax_k / 448.0 * wts_k)).to(tl.float16)
    out_v = (acc_v * (amax_v / 448.0 * wts_v)).to(tl.float16)

    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(or_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_r, mask=omask)
    tl.store(ok_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_k, mask=omask)
    tl.store(ov_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_v, mask=omask)


def linear_rkv_fused(xr, xk, xv, wr_info, wk_info, wv_info, out_dtype=torch.float16):
    """Fused r/k/v attention projections: 2 launches (prep3_x + GEMM).

    wr_info: NVFP4 dict (qtype nvfp4_fused)
    wk_info: NVFP4 (nvfp4_fused) or FP8 (fp8) dict
    wv_info: FP8 dict
    Returns (r, k, v) each shaped like xr (e.g. [B, T, C]).
    """
    orig_shape = xr.shape
    xr_a, xk_a, xv_a, amax_r, amax_k, amax_v = prep3_x(
        xr, xk, xv,
        wr_info.get("awq_scale", None),
        wk_info.get("awq_scale", None),
    )
    M, K = xr_a.shape
    N = wr_info["weight"].size(0)

    or_ = torch.empty(M, N, dtype=torch.float16, device=xr_a.device)
    ok_ = torch.empty(M, N, dtype=torch.float16, device=xr_a.device)
    ov_ = torch.empty(M, N, dtype=torch.float16, device=xr_a.device)

    k_is_fp4 = wk_info.get("qtype", "fp8") == "nvfp4_fused"
    if k_is_fp4:
        wk_arg, wk_bs_arg = wk_info["weight"], wk_info["block_scale"].view(torch.uint8)
    else:
        wk_arg, wk_bs_arg = wk_info["weight"].view(torch.uint8), wk_info["weight"].view(torch.uint8)  # bs unused
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_rkv_gemm_kernel[grid](
        xr_a, xk_a, xv_a,
        wr_info["weight"], wr_info["block_scale"].view(torch.uint8), wr_info["tensor_scale"],
        wk_arg, wk_bs_arg, wk_info["tensor_scale"],
        wv_info["weight"].view(torch.uint8), wv_info["tensor_scale"],
        amax_r, amax_k, amax_v,
        or_, ok_, ov_,
        M, N, K,
        xr_a.stride(0),
        wr_info["weight"].stride(0), wr_info["block_scale"].stride(0),
        wk_arg.stride(0) if not k_is_fp4 else wr_info["weight"].stride(0),
        wv_info["weight"].stride(0),
        or_.stride(0),
        K_IS_FP4=k_is_fp4,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )
    if out_dtype != torch.float16:
        or_, ok_, ov_ = or_.to(out_dtype), ok_.to(out_dtype), ov_.to(out_dtype)
    or_ = or_.reshape(*orig_shape[:-1], N)
    ok_ = ok_.reshape(*orig_shape[:-1], N)
    ov_ = ov_.reshape(*orig_shape[:-1], N)
    return or_, ok_, ov_


# ============================================================================
# Host wrappers
# ============================================================================

def _nvfp4_cfg_for(M):
    """M-adaptive launch config (tuned on 1.5B, C=2048)."""
    if M <= 4:
        return (16, 64, 64, 4)    # decode
    return (64, 64, 64, 4)        # small batch / prefill chunk


def linear_nvfp4_fused(x, weight_info, out_dtype=torch.float16):
    """Fused NVFP4 GEMM (W4A4): prep_x + single-kernel GEMM.

    weight_info must have unswizzled block scales ([N, K//16]).
    """
    w = weight_info["weight"]                # [N, K//2] uint8 packed
    w_bs = weight_info["block_scale"]        # [N, K//16] float8_e4m3fn
    w_ts = weight_info["tensor_scale"]       # scalar fp32

    x_awq, amax = prep_x(x, weight_info.get("awq_scale", None))
    M, K = x_awq.shape
    N = w.size(0)

    out = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_nvfp4_gemm_kernel[grid](
        x_awq, w, w_bs.view(torch.uint8), w_ts, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0), w_bs.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_nvfp4_res_fused(x, weight_info, out_dtype=torch.float16):
    """Fused NVFP4+FP8 residual GEMM (W4A4+W8A8): ONE kernel.

    weight_info: unswizzled block scales + res_fp8 + res_fp8_scale.
    """
    w = weight_info["weight"]
    w_bs = weight_info["block_scale"]
    w_ts = weight_info["tensor_scale"]
    res_w = weight_info["res_fp8"]           # [N, K] fp8
    res_ts = weight_info["res_fp8_scale"]

    x_awq, amax = prep_x(x, weight_info.get("awq_scale", None))
    M, K = x_awq.shape
    N = w.size(0)

    out = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_nvfp4_res_gemm_kernel[grid](
        x_awq, w, w_bs.view(torch.uint8), w_ts,
        res_w.view(torch.uint8), res_ts, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0), w_bs.stride(0), res_w.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_fp8_fused(x, weight_info_or_w, res_scale=None, out_dtype=torch.float16):
    """Fused FP8 GEMM (W8A8): prep_x + single-kernel GEMM.

    Accepts either a weight_info dict (from load_fp8_weight) or raw (w, scale).
    """
    if isinstance(weight_info_or_w, dict):
        w = weight_info_or_w["weight"]            # [N, K] float8_e4m3fn
        w_ts = weight_info_or_w["tensor_scale"]
    else:
        w = weight_info_or_w
        w_ts = res_scale

    x_awq, amax = prep_x(x)
    M, K = x_awq.shape
    N = w.size(0)

    out = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_fp8_gemm_kernel[grid](
        x_awq, w.view(torch.uint8), w_ts, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Hybrid dispatcher for quantized GEMMs (used by the engine).

    - M <= FUSED_M_MAX (decode/small batch): fused single-kernel GEMMs.
    - M > FUSED_M_MAX (prefill/large batch): _scaled_mm path
      (swizzled scales from block_scale_sw) — cuBLAS wins for large M.
    """
    from nvfp4_ops import linear_nvfp4, linear_fp8
    qtype = weight_info.get("qtype", "nvfp4")
    M = x.numel() // x.size(-1)
    if M <= FUSED_M_MAX:
        if qtype == "fp8":
            return linear_fp8_fused(x, weight_info, out_dtype)
        if qtype == "nvfp4_res_fused":
            return linear_nvfp4_res_fused(x, weight_info, out_dtype)
        # nvfp4_fused
        return linear_nvfp4_fused(x, weight_info, out_dtype)
    # Large M: _scaled_mm (needs swizzled scales)
    wi = dict(weight_info)
    if qtype in ("nvfp4_fused", "nvfp4_res_fused"):
        wi["block_scale"] = wi["block_scale_sw"]
        wi["qtype"] = "nvfp4_res" if qtype == "nvfp4_res_fused" else "nvfp4"
        return linear_nvfp4(x, wi, out_dtype)
    if qtype == "fp8":
        return linear_fp8(x, wi, out_dtype)
    return linear_nvfp4(x, wi, out_dtype)


FUSED_M_MAX = 64  # use fused single-kernel GEMM when M <= this (decode/small-batch domain)
