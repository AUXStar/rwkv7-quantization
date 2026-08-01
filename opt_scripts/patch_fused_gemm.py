#!/usr/bin/env python3
"""Patch fused_nvfp4_gemm.py to add FP8 hardware dot kernels and update wrappers.

Changes:
1. Add fused_fp8_hwdot_gemm_kernel (tl.dot(fp8,fp8)) after fused_fp8_gemm_kernel
2. Add fused_rkv_hwdot_kernel (FP8 hwdot for k/v) after fused_rkv_gemm_kernel
3. Modify linear_fp8_fused to use hwdot kernel
4. Modify linear_rkv_fused to dispatch to hwdot when k is FP8
"""
import re

FILE = "/home/njzy/test/Albatross/faster3a_2605/fused_nvfp4_gemm.py"

with open(FILE, "r") as f:
    src = f.read()

# ============================================================================
# 1. Add fused_fp8_hwdot_gemm_kernel before the RKV section
# ============================================================================

FP8_HWDOT_KERNEL = '''
# ============================================================================
# Fused FP8 (W8A8) GEMM kernel — hardware FP8 tensor core dot (tl.dot(fp8,fp8))
# ============================================================================

@triton.jit
def fused_fp8_hwdot_gemm_kernel(
    x_ptr,           # [M, K] bf16
    w_ptr,           # [N, K] float8_e4m3fn (NOT uint8 — pass directly!)
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
    """FP8 GEMM using hardware FP8 tensor cores: tl.dot(fp8, fp8).

    2x throughput vs FP16 dot on Ada (RTX 40xx). Identical numerics to
    fused_fp8_gemm_kernel because FP8 values are exactly representable in FP16.
    """
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
        # Quantize activation to FP8 — keep as FP8 for dot
        a_fp8 = tl.minimum(tl.maximum(x_tile.to(tl.float32) * inv_xs, -448.0), 448.0).to(tl.float8e4nv)

        # Load weight directly as FP8 (no decode!)
        w_fp8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                        mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0)

        # FP8 tensor core dot (2x faster than FP16 on Ada)
        acc = tl.dot(a_fp8, tl.trans(w_fp8), acc)

    acc = acc * (amax_v / 448.0 * w_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


'''

# Insert before RKV section
rkv_marker = '# ============================================================================\n# Fused RKV GEMM kernel: r(NVFP4) + k(NVFP4|FP8) + v(FP8) in ONE launch'
assert rkv_marker in src, "Cannot find RKV marker"
src = src.replace(rkv_marker, FP8_HWDOT_KERNEL + rkv_marker, 1)

# ============================================================================
# 2. Add fused_rkv_hwdot_kernel after fused_rkv_gemm_kernel, before linear_rkv_fused
# ============================================================================

RKV_HWDOT_KERNEL = '''
# ============================================================================
# Fused RKV GEMM kernel with FP8 hardware dot for k/v (tl.dot(fp8,fp8))
# ============================================================================

@triton.jit
def fused_rkv_hwdot_kernel(
    xr_ptr, xk_ptr, xv_ptr,      # [M, K] bf16
    wr_ptr, wr_bs_ptr, wts_r_ptr,  # r: NVFP4 (packed [N, K//2])
    wk_ptr, wts_k_ptr,             # k: FP8 ([N, K] float8_e4m3fn)
    wv_ptr, wts_v_ptr,             # v: FP8 ([N, K] float8_e4m3fn)
    amax_r_ptr, amax_k_ptr, amax_v_ptr,
    or_ptr, ok_ptr, ov_ptr,
    M, N, K,
    stride_xm,
    stride_wr, stride_wbs,         # r weight strides (K//2 and K//16)
    stride_wk, stride_wv,          # k/v weight strides (K each)
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """RKV fused: r (NVFP4 FP16 dot) + k (FP8 hw dot) + v (FP8 hw dot).

    k/v use tl.dot(fp8, fp8) for 2x tensor core throughput on Ada.
    """
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

        # ---------- r: NVFP4 (FP4×FP4, FP16 dot) ----------
        xr = tl.load(xr_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        xr32 = xr.to(tl.float32)
        xr_r = tl.reshape(xr32, (BLOCK_M, NKB, 16))
        max_r = tl.max(tl.abs(xr_r), axis=2)
        s_r = tl.minimum(tl.maximum(max_r / 6.0 * inv_pts_r, 0.015625), 448.0).to(tl.float8e4nv).to(tl.float32)
        xr_s = tl.minimum(tl.maximum(xr32 * _expand_blocks(inv_pts_r / s_r, BLOCK_M, NKB, BLOCK_K), -6.0), 6.0)
        a_r = (_fp4_val(_fp4_rne(xr_s)) * _expand_blocks(s_r, BLOCK_M, NKB, BLOCK_K)).to(tl.float16)
        w_rp = tl.load(wr_ptr + offs_n[:, None] * stride_wr + (k_start // 2 + offs_k2)[None, :],
                       mask=wmask & ((k_start // 2 + offs_k2[None, :]) < K // 2), other=0)
        w_rc = tl.reshape(tl.join(w_rp & 0xF, (w_rp >> 4) & 0xF), (BLOCK_N, BLOCK_K))
        w_rbs = _e4m3_val(tl.load(wr_bs_ptr + offs_n[:, None] * stride_wbs + (k_start // 16 + offs_nkb)[None, :],
                                  mask=wmask & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0))
        b_r = (_fp4_val(w_rc) * _expand_blocks_n(w_rbs, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)
        acc_r = tl.dot(a_r, tl.trans(b_r), acc_r)

        # ---------- k: FP8 (hardware FP8 dot) ----------
        xk = tl.load(xk_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        a_k_fp8 = tl.minimum(tl.maximum(xk.to(tl.float32) * inv_xs_k, -448.0), 448.0).to(tl.float8e4nv)
        w_k_fp8 = tl.load(wk_ptr + offs_n[:, None] * stride_wk + (k_start + offs_k)[None, :],
                          mask=wmask & kmask[None, :], other=0.0)
        acc_k = tl.dot(a_k_fp8, tl.trans(w_k_fp8), acc_k)

        # ---------- v: FP8 (hardware FP8 dot) ----------
        xv = tl.load(xv_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        a_v_fp8 = tl.minimum(tl.maximum(xv.to(tl.float32) * inv_xs_v, -448.0), 448.0).to(tl.float8e4nv)
        w_v_fp8 = tl.load(wv_ptr + offs_n[:, None] * stride_wv + (k_start + offs_k)[None, :],
                          mask=wmask & kmask[None, :], other=0.0)
        acc_v = tl.dot(a_v_fp8, tl.trans(w_v_fp8), acc_v)

    out_r = (acc_r * (amax_r / 2688.0 * wts_r)).to(tl.float16)
    out_k = (acc_k * (amax_k / 448.0 * wts_k)).to(tl.float16)
    out_v = (acc_v * (amax_v / 448.0 * wts_v)).to(tl.float16)

    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(or_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_r, mask=omask)
    tl.store(ok_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_k, mask=omask)
    tl.store(ov_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_v, mask=omask)


'''

# Insert before linear_rkv_fused
rkv_fused_marker = 'def linear_rkv_fused('
assert rkv_fused_marker in src, "Cannot find linear_rkv_fused marker"
src = src.replace(rkv_fused_marker, RKV_HWDOT_KERNEL + rkv_fused_marker, 1)

# ============================================================================
# 3. Modify linear_fp8_fused to use hwdot kernel
# ============================================================================

# Replace the kernel call in linear_fp8_fused
old_fp8_call = '''    fused_fp8_gemm_kernel[grid](
        x_awq, w.view(torch.uint8), w_ts, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )'''

new_fp8_call = '''    fused_fp8_hwdot_gemm_kernel[grid](
        x_awq, w, w_ts, amax, out,
        M, N, K,
        x_awq.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )'''

assert old_fp8_call in src, "Cannot find old FP8 kernel call"
src = src.replace(old_fp8_call, new_fp8_call, 1)

# ============================================================================
# 4. Modify linear_rkv_fused to dispatch to hwdot when k is FP8
# ============================================================================

# Find the linear_rkv_fused function and add hwdot dispatch
old_rkv_body = '''    k_is_fp4 = wk_info.get("qtype", "fp8") == "nvfp4_fused"
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
    )'''

new_rkv_body = '''    k_is_fp4 = wk_info.get("qtype", "fp8") == "nvfp4_fused"
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    if not k_is_fp4:
        # k is FP8: use hardware FP8 dot for k/v (2x tensor core throughput)
        fused_rkv_hwdot_kernel[grid](
            xr_a, xk_a, xv_a,
            wr_info["weight"], wr_info["block_scale"].view(torch.uint8), wr_info["tensor_scale"],
            wk_info["weight"], wk_info["tensor_scale"],
            wv_info["weight"], wv_info["tensor_scale"],
            amax_r, amax_k, amax_v,
            or_, ok_, ov_,
            M, N, K,
            xr_a.stride(0),
            wr_info["weight"].stride(0), wr_info["block_scale"].stride(0),
            wk_info["weight"].stride(0), wv_info["weight"].stride(0),
            or_.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=nw,
        )
    else:
        # k is NVFP4: use original kernel with K_IS_FP4 path
        wk_arg, wk_bs_arg = wk_info["weight"], wk_info["block_scale"].view(torch.uint8)
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
            wr_info["weight"].stride(0),
            wv_info["weight"].stride(0),
            or_.stride(0),
            K_IS_FP4=True,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=nw,
        )'''

assert old_rkv_body in src, "Cannot find old RKV body"
src = src.replace(old_rkv_body, new_rkv_body, 1)

# ============================================================================
# Write output
# ============================================================================

with open(FILE, "w") as f:
    f.write(src)

print("Patch applied successfully!")
print("Changes:")
print("  1. Added fused_fp8_hwdot_gemm_kernel (tl.dot(fp8,fp8))")
print("  2. Added fused_rkv_hwdot_kernel (FP8 hwdot for k/v)")
print("  3. linear_fp8_fused now uses hwdot kernel")
print("  4. linear_rkv_fused dispatches to hwdot when k is FP8")
