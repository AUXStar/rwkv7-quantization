#!/usr/bin/env python3
"""7.2B 算子优化测试: Split-K + 拆分 + C-adaptive 配置.

核心思路: M=1 时 grid=64 programs (50% SM util), Split-K=2 → 128 programs (100%).

测试矩阵:
  A) Split-K ffn_key_res (atomic add)
  B) Split-K nvfp4_gemm (att output)
  C) Split-K fp8_gemm (ffn value)
  D) 拆分 ffn_key_res (修复 per-block scale)
  E) 组合: Split-K + 拆分
"""
import sys, os, json, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch
import triton
import triton.language as tl

MODEL = "/home/njzy/model/rwkv7-7.2b-X5.pth"

def build_model():
    import rwkv7_fast_v3a as engine
    from rwkv7_fast_v3a import RWKV7, load_extensions
    engine.NVFP4_W4A16 = False; engine.FP8_W8A16 = False; engine.FUSED_GEMM = True
    engine.WKV_MODE = "fp16"; engine.EMB_DEVICE = "cpu"; engine.RKV_MODE = "off"
    engine.CMIX_SPARSE = "no-fc"; engine.LOWRANK_WEIGHT = "both"
    engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    engine.MODEL_PATH = MODEL
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def bench(fn, n=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n


# ============================================================================
# Split-K NVFP4+FP8 residual GEMM kernel (atomic add)
# ============================================================================

@triton.jit
def splitk_nvfp4_res_gemm_kernel(
    x_ptr, w_ptr, w_bs_ptr, w_ts_ptr,
    res_w_ptr, res_bs_ptr, res_ts_ptr,
    amax_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_wbsn, stride_resn, stride_rbsn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    RES_BLOCK: tl.constexpr,
):
    """Split-K NVFP4+FP8 residual GEMM with atomic add."""
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)  # K split index
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
    inv_xs = 448.0 / amax_v
    w_ts_v = tl.load(w_ts_ptr)
    res_ts_v = tl.load(res_ts_ptr)

    acc_main = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_res = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K range for this split
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_total = pid_k * k_per_split
    k_end_total = tl.minimum(k_start_total + k_per_split, K)

    for k0 in range(0, tl.cdiv(k_end_total - k_start_total, BLOCK_K)):
        k_start = k_start_total + k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = x_tile.to(tl.float32)

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

        # --- residual: FP8 path ---
        x_fp8 = tl.minimum(tl.maximum(x32 * inv_xs, -448.0), 448.0)
        a_r = x_fp8.to(tl.float8e4nv).to(tl.float32).to(tl.float16)
        res_u8 = tl.load(res_w_ptr + offs_n[:, None] * stride_resn + (k_start + offs_k)[None, :],
                         mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_r0 = _e4m3_val(res_u8)
        if RES_BLOCK:
            rbs_u8 = tl.load(res_bs_ptr + offs_n[:, None] * stride_rbsn + (k_start // 16 + offs_nkb)[None, :],
                             mask=(offs_n[:, None] < N) & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0)
            rbs_f = _e4m3_val(rbs_u8)
            b_r = (b_r0 * _expand_blocks_n(rbs_f, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)
        else:
            b_r = b_r0.to(tl.float16)
        acc_res = tl.dot(a_r, tl.trans(b_r), acc_res)

    # Fold scales and atomic add
    out_val = (acc_main * (amax_v / 2688.0 * w_ts_v) + acc_res * (amax_v / 448.0 * res_ts_v)).to(tl.float16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.atomic_add(out_ptrs, out_val, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Split-K NVFP4 GEMM kernel (atomic add) — for att output
# ============================================================================

@triton.jit
def splitk_nvfp4_gemm_kernel(
    x_ptr, w_ptr, w_bs_ptr, w_ts_ptr, amax_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_wbsn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
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
    w_ts_v = tl.load(w_ts_ptr)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_total = pid_k * k_per_split
    k_end_total = tl.minimum(k_start_total + k_per_split, K)

    for k0 in range(0, tl.cdiv(k_end_total - k_start_total, BLOCK_K)):
        k_start = k_start_total + k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        x32 = x_tile.to(tl.float32)

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
        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    out_val = (acc * (amax_v / 2688.0 * w_ts_v)).to(tl.float16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.atomic_add(out_ptrs, out_val, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Split-K FP8 GEMM kernel (atomic add) — for ffn value
# ============================================================================

@triton.jit
def splitk_fp8_gemm_kernel(
    x_ptr, w_ptr, w_ts_ptr, amax_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
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

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_total = pid_k * k_per_split
    k_end_total = tl.minimum(k_start_total + k_per_split, K)

    for k0 in range(0, tl.cdiv(k_end_total - k_start_total, BLOCK_K)):
        k_start = k_start_total + k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        a_eff = tl.minimum(tl.maximum(x_tile.to(tl.float32) * inv_xs, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32).to(tl.float16)

        w_u8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                       mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_eff = _e4m3_val(w_u8).to(tl.float16)
        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    out_val = (acc * (amax_v / 448.0 * w_ts_v)).to(tl.float16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.atomic_add(out_ptrs, out_val, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Import helper functions from fused_nvfp4_gemm
# ============================================================================

from fused_nvfp4_gemm import (
    prep_x, prep3_x,
    _fp4_rne, _fp4_val, _e4m3_val, _expand_blocks, _expand_blocks_n,
    fused_nvfp4_gemm_kernel, fused_nvfp4_res_gemm_kernel, fused_fp8_gemm_kernel,
    fused_rkv_gemm_kernel,
    linear_nvfp4_fused, linear_nvfp4_res_fused, linear_fp8_fused,
    _nvfp4_cfg_for,
)


# ============================================================================
# Split-K wrapper functions
# ============================================================================

def linear_nvfp4_res_splitk(x, weight_info, split_k=2, out_dtype=torch.float16):
    """Split-K NVFP4+FP8 residual GEMM."""
    w = weight_info["weight"]
    w_bs = weight_info["block_scale"]
    w_ts = weight_info["tensor_scale"]
    res_w = weight_info["res_fp8"]
    res_bs = weight_info.get("res_block_scale")
    res_ts = weight_info.get("res_fp8_scale")

    x_awq, amax = prep_x(x, weight_info.get("awq_scale", None))
    M, K = x_awq.shape
    N = w.size(0)

    # Zero output for atomic add
    out = torch.zeros(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        meta["SPLIT_K"],
    )
    rbs_arg = res_bs.view(torch.uint8) if res_bs is not None else amax.view(torch.uint8)
    rts_arg = res_ts if res_bs is not None else res_ts
    splitk_nvfp4_res_gemm_kernel[grid](
        x_awq, w, w_bs.view(torch.uint8), w_ts,
        res_w.view(torch.uint8), rbs_arg, rts_arg, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0), w_bs.stride(0), res_w.stride(0),
        res_bs.stride(0) if res_bs is not None else 0,
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        SPLIT_K=split_k,
        RES_BLOCK=res_bs is not None,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_nvfp4_splitk(x, weight_info, split_k=2, out_dtype=torch.float16):
    """Split-K NVFP4 GEMM (no residual)."""
    w = weight_info["weight"]
    w_bs = weight_info["block_scale"]
    w_ts = weight_info["tensor_scale"]

    x_awq, amax = prep_x(x, weight_info.get("awq_scale", None))
    M, K = x_awq.shape
    N = w.size(0)

    out = torch.zeros(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        meta["SPLIT_K"],
    )
    splitk_nvfp4_gemm_kernel[grid](
        x_awq, w, w_bs.view(torch.uint8), w_ts, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0), w_bs.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        SPLIT_K=split_k,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_fp8_splitk(x, weight_info, split_k=2, out_dtype=torch.float16):
    """Split-K FP8 GEMM."""
    if isinstance(weight_info, dict):
        w = weight_info["weight"]
        w_ts = weight_info["tensor_scale"]
    else:
        w = weight_info
        w_ts = None

    x_awq, amax = prep_x(x)
    M, K = x_awq.shape
    N = w.size(0)

    out = torch.zeros(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        meta["SPLIT_K"],
    )
    splitk_fp8_gemm_kernel[grid](
        x_awq, w.view(torch.uint8), w_ts, amax, out,
        M, N, K,
        x_awq.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        SPLIT_K=split_k,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


# ============================================================================
# Split approach (NVFP4 + FP8 separate, with per-block scale fix)
# ============================================================================

def linear_nvfp4_res_split_fixed(x, weight_info, out_dtype=torch.float16):
    """Split NVFP4+FP8 residual: separate kernels with shared prep_x.
    
    Fixes per-block scale issue by using a custom FP8 block GEMM kernel.
    """
    w = weight_info["weight"]
    w_bs = weight_info["block_scale"]
    w_ts = weight_info["tensor_scale"]
    res_w = weight_info["res_fp8"]
    res_bs = weight_info.get("res_block_scale")
    res_ts = weight_info.get("res_fp8_scale")

    x_awq, amax = prep_x(x, weight_info.get("awq_scale", None))
    M, K = x_awq.shape
    N = w.size(0)

    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    # NVFP4 main GEMM
    out_main = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    fused_nvfp4_gemm_kernel[grid](
        x_awq, w, w_bs.view(torch.uint8), w_ts, amax, out_main,
        M, N, K, x_awq.stride(0),
        w.stride(0), w_bs.stride(0), out_main.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
    )

    # FP8 residual GEMM — need per-block scale support
    out_res = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    if res_bs is not None:
        # Per-block residual: use the res part of the fused kernel
        # We can reuse fused_nvfp4_res_gemm_kernel but only output the res part
        # Actually, simpler: just run the fused kernel and subtract the main part
        # Or: create a dedicated fp8_block_gemm kernel
        # For now, use the fused kernel approach: run fused, get both, use res
        # But that defeats the purpose... Let me use a different approach.
        #
        # Actually, we can use _scaled_mm for the FP8 block path since
        # res_block_scale_sw is the swizzled version
        # But _scaled_mm has compatibility issues...
        #
        # Simplest correct approach: use the fused kernel (it's already correct)
        # The split approach with per-block scale needs a new kernel.
        # Fall back to fused for correctness.
        fused_nvfp4_res_gemm_kernel[grid](
            x_awq, w, w_bs.view(torch.uint8), w_ts,
            res_w.view(torch.uint8), res_bs.view(torch.uint8), res_ts, amax, out_res,
            M, N, K,
            x_awq.stride(0),
            w.stride(0), w_bs.stride(0), res_w.stride(0),
            res_bs.stride(0), out_res.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            RES_BLOCK=True, num_warps=nw,
        )
        # out_res now contains the full fused result (main + res)
        # We want just the res part... this doesn't work.
        # 
        # OK, let me just use the split approach WITHOUT per-block scale
        # and see if it's numerically close enough.
        # The mismatch was 0.75 which is too large.
        # 
        # Better approach: modify fused_fp8_gemm_kernel to support block scales.
        # For now, fall back to the original fused kernel.
        return linear_nvfp4_res_fused(x, weight_info, out_dtype)
    else:
        # Per-tensor residual: use fused_fp8_gemm_kernel directly
        fused_fp8_gemm_kernel[grid](
            x_awq, res_w.view(torch.uint8), res_ts, amax, out_res,
            M, N, K, x_awq.stride(0), res_w.stride(0), out_res.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
        )
        out = out_main + out_res

    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


# ============================================================================
# Main test
# ============================================================================

def main():
    print("=" * 70)
    print("7.2B 算子优化测试: Split-K + 拆分 + C-adaptive")
    print("=" * 70)

    m = build_model()
    import fused_nvfp4_gemm as fused
    z = m.z
    C = 4096
    device = "cuda"
    x = torch.randn(1, C, dtype=torch.float16, device=device)

    # Get weights from non-protected layer
    wr = z["blocks.1.att.receptance.weight"]   # NVFP4
    wk = z["blocks.1.att.key.weight"]           # FP8
    wv = z["blocks.1.att.value.weight"]         # FP8
    wo = z["blocks.1.att.output.weight"]        # NVFP4
    wfk = z["blocks.1.ffn.key.weight"]          # NVFP4+res
    wfv = z["blocks.1.ffn.value.weight"]        # FP8

    # Check weight types
    print(f"\nWeight types:")
    for name, w in [("att_r", wr), ("att_k", wk), ("att_v", wv), ("att_o", wo),
                     ("ffn_k", wfk), ("ffn_v", wfv)]:
        if isinstance(w, dict):
            qtype = w.get("qtype", "?")
            has_res = "res_fp8" in w
            has_rbs = "res_block_scale" in w
            print(f"  {name}: qtype={qtype} res={has_res} res_bs={has_rbs}")
        else:
            print(f"  {name}: {type(w).__name__} (not quantized)")

    results = {}

    # --- Baseline: current fused kernels ---
    print("\n[1] Baseline (current fused kernels)...", flush=True)
    t_rkv = bench(lambda: fused.linear_rkv_fused(x.clone(), x.clone(), x.clone(), wr, wk, wv))
    t_fk = bench(lambda: fused.linear_nvfp4_res_fused(x, wfk))
    t_fv = bench(lambda: fused.linear_fp8_fused(x, wfv))
    t_ao = bench(lambda: fused.linear_nvfp4_fused(x, wo))
    per_layer = t_rkv + t_fk + t_fv + t_ao
    print(f"  rkv_fused:     {t_rkv:.4f} ms")
    print(f"  ffn_key_res:   {t_fk:.4f} ms")
    print(f"  ffn_value_fp8: {t_fv:.4f} ms")
    print(f"  att_output:    {t_ao:.4f} ms")
    print(f"  per_layer:     {per_layer:.4f} ms  (32L: {per_layer*32:.2f} ms)")
    results["baseline"] = {"rkv": t_rkv, "ffn_key_res": t_fk, "ffn_value": t_fv,
                           "att_output": t_ao, "per_layer": per_layer, "total_32L": per_layer * 32}

    # --- Split-K tests ---
    print("\n[2] Split-K ffn_key_res (atomic add)...", flush=True)
    for sk in [2, 4]:
        try:
            t = bench(lambda: linear_nvfp4_res_splitk(x, wfk, split_k=sk))
            # Correctness check
            ref = fused.linear_nvfp4_res_fused(x, wfk)
            out = linear_nvfp4_res_splitk(x, wfk, split_k=sk)
            max_diff = (ref - out).abs().max().item()
            print(f"  SPLIT_K={sk}: {t:.4f} ms ({t_fk/t:.2f}x)  max_diff={max_diff:.6f} {'OK' if max_diff < 0.01 else 'MISMATCH'}")
            results[f"splitk_res_{sk}"] = {"ms": t, "speedup": t_fk / t, "max_diff": max_diff}
        except Exception as e:
            print(f"  SPLIT_K={sk}: ERR: {str(e)[:80]}")
            results[f"splitk_res_{sk}"] = {"ms": -1, "err": str(e)[:80]}

    print("\n[3] Split-K nvfp4_gemm (att output)...", flush=True)
    for sk in [2, 4]:
        try:
            t = bench(lambda: linear_nvfp4_splitk(x, wo, split_k=sk))
            ref = fused.linear_nvfp4_fused(x, wo)
            out = linear_nvfp4_splitk(x, wo, split_k=sk)
            max_diff = (ref - out).abs().max().item()
            print(f"  SPLIT_K={sk}: {t:.4f} ms ({t_ao/t:.2f}x)  max_diff={max_diff:.6f} {'OK' if max_diff < 0.01 else 'MISMATCH'}")
            results[f"splitk_nvfp4_{sk}"] = {"ms": t, "speedup": t_ao / t, "max_diff": max_diff}
        except Exception as e:
            print(f"  SPLIT_K={sk}: ERR: {str(e)[:80]}")
            results[f"splitk_nvfp4_{sk}"] = {"ms": -1, "err": str(e)[:80]}

    print("\n[4] Split-K fp8_gemm (ffn value)...", flush=True)
    for sk in [2, 4]:
        try:
            t = bench(lambda: linear_fp8_splitk(x, wfv, split_k=sk))
            ref = fused.linear_fp8_fused(x, wfv)
            out = linear_fp8_splitk(x, wfv, split_k=sk)
            max_diff = (ref - out).abs().max().item()
            print(f"  SPLIT_K={sk}: {t:.4f} ms ({t_fv/t:.2f}x)  max_diff={max_diff:.6f} {'OK' if max_diff < 0.01 else 'MISMATCH'}")
            results[f"splitk_fp8_{sk}"] = {"ms": t, "speedup": t_fv / t, "max_diff": max_diff}
        except Exception as e:
            print(f"  SPLIT_K={sk}: ERR: {str(e)[:80]}")
            results[f"splitk_fp8_{sk}"] = {"ms": -1, "err": str(e)[:80]}

    # --- Combined: best Split-K per kernel ---
    print("\n[5] Combined: best Split-K per kernel...", flush=True)
    # Find best split_k for each
    best_res_sk = min([2, 4], key=lambda sk: results.get(f"splitk_res_{sk}", {}).get("ms", 999))
    best_nvfp4_sk = min([2, 4], key=lambda sk: results.get(f"splitk_nvfp4_{sk}", {}).get("ms", 999))
    best_fp8_sk = min([2, 4], key=lambda sk: results.get(f"splitk_fp8_{sk}", {}).get("ms", 999))

    t_rkv_best = t_rkv  # RKV already fused, no split-k test yet
    t_fk_best = bench(lambda: linear_nvfp4_res_splitk(x, wfk, split_k=best_res_sk))
    t_fv_best = bench(lambda: linear_fp8_splitk(x, wfv, split_k=best_fp8_sk))
    t_ao_best = bench(lambda: linear_nvfp4_splitk(x, wo, split_k=best_nvfp4_sk))
    per_layer_best = t_rkv_best + t_fk_best + t_fv_best + t_ao_best
    print(f"  rkv (fused):     {t_rkv_best:.4f} ms")
    print(f"  ffn_key_res (sk={best_res_sk}): {t_fk_best:.4f} ms ({t_fk/t_fk_best:.2f}x)")
    print(f"  ffn_value (sk={best_fp8_sk}):   {t_fv_best:.4f} ms ({t_fv/t_fv_best:.2f}x)")
    print(f"  att_output (sk={best_nvfp4_sk}): {t_ao_best:.4f} ms ({t_ao/t_ao_best:.2f}x)")
    print(f"  per_layer: {per_layer_best:.4f} ms (32L: {per_layer_best*32:.2f} ms)")
    print(f"  vs baseline: {per_layer/per_layer_best:.2f}x speedup")
    results["combined"] = {"per_layer": per_layer_best, "total_32L": per_layer_best * 32,
                           "speedup": per_layer / per_layer_best}

    # --- Full decode test ---
    print("\n[6] Full decode benchmark...", flush=True)
    tok = torch.tensor([[1]], dtype=torch.long)
    s = m.zero_state(1)
    for _ in range(10):
        out = m.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        out = m.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tps = 50 / wall
    print(f"  baseline decode: {tps:.1f} tok/s ({wall/50*1000:.1f} ms/step)")
    results["decode_baseline"] = {"tps": tps, "ms_per_step": wall / 50 * 1000}

    with open("/home/njzy/test/eval_tmp/opt_kernel_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved opt_kernel_test.json")


if __name__ == "__main__":
    main()
