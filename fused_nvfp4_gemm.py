#!/usr/bin/env python3
"""Fused quantized GEMM kernels (single-launch, no intermediate memory).

Rewrites the W4A4/W8A8 GEMM path: instead of
  x → cast/awq → fused_quant_kernel → packed A → _scaled_mm → scale-fold
(3-4 launches + 2 memory round-trips), this fuses everything into ONE kernel:

  fused_nvfp4_gemm: x(bf16) → [in-register] AWQ + FP4 quant + block scales
                          → FP4×FP4 dot (weights stay packed FP4 in memory)
  fused_fp8_gemm:   x(bf16) → [in-register] FP8 quant (per-tensor scale)
                          → FP8×FP8 dot (weights stay FP8 in memory)

Numerics match the validated _scaled_mm path:
- activation per-16-block scale: clamp(max_abs/6 * inv_pts, 0.015625, 448) → fp8
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
# Fused NVFP4 (FP4×FP4) GEMM kernel
# ============================================================================

@triton.jit
def fused_nvfp4_gemm_kernel(
    x_ptr,           # [M, K] bf16 (AWQ already applied)
    w_ptr,           # [N, K//2] uint8 packed FP4
    w_bs_ptr,        # [N, K//16] uint8 (bit-view of float8_e4m3fn block scales)
    w_ts,            # fp32 scalar (weight per-tensor scale)
    pts,             # fp32 scalar (activation per-tensor scale = amax/2688)
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

    pts_v = tl.load(pts)
    w_ts_v = tl.load(w_ts)
    inv_pts = 1.0 / pts_v
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = x_tile.to(tl.float32)  # [BM, BK]

        # --- activation quantization (matches fused_nvfp4_quant) ---
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

    # fold per-tensor scales
    acc = acc * (pts_v * w_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Fused FP8 (W8A8) GEMM kernel
# ============================================================================

@triton.jit
def fused_fp8_gemm_kernel(
    x_ptr,           # [M, K] bf16
    w_ptr,           # [N, K] uint8 (bit-view of float8_e4m3fn)
    w_ts,            # fp32 scalar (weight per-tensor scale)
    x_scale,         # fp32 scalar (activation scale = amax/448)
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

    w_ts_v = tl.load(w_ts)
    xs_v = tl.load(x_scale)
    inv_xs = 1.0 / xs_v
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = x_tile.to(tl.float32) * inv_xs
        x32 = tl.minimum(tl.maximum(x32, -448.0), 448.0)
        x_q = x32.to(tl.float8e4nv).to(tl.float32)  # fp8-rounded
        a_eff = x_q.to(tl.float16)  # [BM, BK]

        w_u8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                       mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_val = _e4m3_val(w_u8)  # [BN, BK] fp32
        b_eff = b_val.to(tl.float16)

        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    acc = acc * (xs_v * w_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Host wrappers
# ============================================================================

def _launch_nvfp4(x_awq, w, w_bs, w_ts, pts, out):
    M, K = x_awq.shape
    N = w.shape[0]
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_nvfp4_gemm_kernel[grid](
        x_awq, w, w_bs, w_ts, pts, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0), w_bs.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )


def _nvfp4_cfg_for(M):
    """M-adaptive launch config (tuned on 1.5B, C=2048)."""
    if M <= 4:
        return (16, 64, 64, 4)    # decode
    return (64, 64, 64, 4)        # small batch / prefill chunk


def _launch_nvfp4_custom(x_awq, w, w_bs, w_ts, pts, out, bm, bn, bk, nw):
    M, K = x_awq.shape
    N = w.shape[0]
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_nvfp4_gemm_kernel[grid](
        x_awq, w, w_bs, w_ts, pts, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0), w_bs.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )


def _launch_fp8(x_awq, w, w_ts, x_scale, out):
    M, K = x_awq.shape
    N = w.shape[0]
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_fp8_gemm_kernel[grid](
        x_awq, w, w_ts, x_scale, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )


def _launch_fp8_custom(x_awq, w, w_ts, x_scale, out, bm, bn, bk, nw):
    M, K = x_awq.shape
    N = w.shape[0]
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_fp8_gemm_kernel[grid](
        x_awq, w, w_ts, x_scale, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )


def linear_nvfp4_fused(x, weight_info, out_dtype=torch.float16):
    """Fused NVFP4 GEMM (W4A4): single kernel, no intermediate memory.

    weight_info must have unswizzled block scales ([N, K//16]).
    """
    w = weight_info["weight"]                # [N, K//2] uint8 packed
    w_bs = weight_info["block_scale"]        # [N, K//16] float8_e4m3fn
    w_ts = weight_info["tensor_scale"]       # scalar fp32

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    awq_scale = weight_info.get("awq_scale", None)
    if awq_scale is not None:
        x_2d = x_2d / awq_scale.to(x_2d.dtype)

    M, K = x_2d.shape
    N = w.size(0)

    amax = x_2d.abs().max()
    pts = (amax.to(torch.float32) / NVFP4_TS_DIV) if amax > 0 else torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)

    out = torch.empty(M, N, dtype=torch.float16, device=x_2d.device)
    _launch_nvfp4(x_2d, w, w_bs.view(torch.uint8), w_ts, pts, out)

    # FP8 residual if present (NVFP4+FP8 residual scheme)
    if "res_fp8" in weight_info:
        res = linear_fp8_fused(x_2d, weight_info["res_fp8"], weight_info["res_fp8_scale"], out_dtype)
        out = out + res.to(out.dtype)

    out = out.reshape(*orig_shape[:-1], N)
    return out


def linear_fp8_fused(x, weight_info_or_w, res_scale=None, out_dtype=torch.float16):
    """Fused FP8 GEMM (W8A8): single kernel.

    Accepts either a weight_info dict (from load_fp8_weight) or raw (w, scale).
    """
    if isinstance(weight_info_or_w, dict):
        w = weight_info_or_w["weight"]            # [N, K] float8_e4m3fn
        w_ts = weight_info_or_w["tensor_scale"]
    else:
        w = weight_info_or_w
        w_ts = res_scale

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    M, K = x_2d.shape
    N = w.size(0)

    amax = x_2d.abs().max()
    x_scale = (amax.to(torch.float32) / F8E4M3_MAX) if amax > 0 else torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)

    out = torch.empty(M, N, dtype=torch.float16, device=x_2d.device)
    _launch_fp8(x_2d, w.view(torch.uint8), w_ts, x_scale, out)

    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Fused dispatcher: pick the right single-kernel GEMM based on qtype."""
    qtype = weight_info.get("qtype", "nvfp4")
    if qtype == "fp8":
        return linear_fp8_fused(x, weight_info, out_dtype)
    if qtype == "nvfp4_res":
        return linear_nvfp4_fused(x, weight_info, out_dtype)
    # nvfp4 (and others) use fused FP4 GEMM
    return linear_nvfp4_fused(x, weight_info, out_dtype)
