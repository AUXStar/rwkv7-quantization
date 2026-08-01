#!/usr/bin/env python3
"""7.2B 算子优化 v2: FP8 硬件 tensor core + 拆分修正.

核心优化:
  A) FP8 硬件 dot: tl.dot(fp8, fp8) 替代 decode→fp16→dot (per-tensor FP8 paths)
  B) 拆分 ffn_key_res: NVFP4 main + FP8 block residual, 共享 prep_x
  C) C-adaptive 配置: _nvfp4_cfg_for(M, C)
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


from fused_nvfp4_gemm import (
    prep_x, prep3_x,
    _fp4_rne, _fp4_val, _e4m3_val, _expand_blocks, _expand_blocks_n,
    fused_nvfp4_gemm_kernel, fused_nvfp4_res_gemm_kernel, fused_fp8_gemm_kernel,
    fused_rkv_gemm_kernel,
    linear_nvfp4_fused, linear_nvfp4_res_fused, linear_fp8_fused,
    linear_rkv_fused,
    _nvfp4_cfg_for,
)


# ============================================================================
# OPTIMIZED: FP8 hardware tensor core dot (per-tensor FP8)
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
    """FP8 GEMM using hardware FP8 tensor cores: tl.dot(fp8, fp8)."""
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


def linear_fp8_hwdot(x, weight_info, out_dtype=torch.float16):
    """FP8 GEMM with hardware tensor core dot."""
    if isinstance(weight_info, dict):
        w = weight_info["weight"]            # [N, K] float8_e4m3fn
        w_ts = weight_info["tensor_scale"]
    else:
        w = weight_info
        w_ts = None

    x_awq, amax = prep_x(x)
    M, K = x_awq.shape
    N = w.size(0)

    out = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    # Pass w as float8_e4m3fn (NOT view as uint8) so Triton loads as float8e4nv
    fused_fp8_hwdot_gemm_kernel[grid](
        x_awq, w, w_ts, amax, out,
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


# ============================================================================
# OPTIMIZED: FP8 block GEMM (for per-block residual, FP16 dot)
# ============================================================================

@triton.jit
def fused_fp8_block_gemm_kernel(
    x_ptr,           # [M, K] bf16
    w_ptr,           # [N, K] uint8 (fp8 weight)
    w_bs_ptr,        # [N, K//16] uint8 (fp8 block scales)
    w_ts_ptr,        # fp32 scalar (per-tensor scale)
    amax_ptr,        # fp32 scalar (activation amax)
    out_ptr,         # [M, N] fp16 output
    M, N, K,
    stride_xm, stride_wn, stride_wbsn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """FP8 GEMM with per-block scales (FP16 dot — block scales need decode)."""
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
    offs_nkb = tl.arange(0, BLOCK_K // 16)
    NKB: tl.constexpr = BLOCK_K // 16

    amax_v = tl.maximum(tl.load(amax_ptr), 1e-12)
    inv_xs = 448.0 / amax_v
    w_ts_v = tl.load(w_ts_ptr)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k0 * BLOCK_K
        kmask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :],
                         mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        a_eff = tl.minimum(tl.maximum(x_tile.to(tl.float32) * inv_xs, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32).to(tl.float16)

        w_u8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                       mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_val = _e4m3_val(w_u8)  # [BN, BK] fp32

        w_bs_u8 = tl.load(w_bs_ptr + offs_n[:, None] * stride_wbsn + (k_start // 16 + offs_nkb)[None, :],
                          mask=(offs_n[:, None] < N) & ((k_start // 16 + offs_nkb[None, :]) < K // 16), other=0)
        bs_f = _e4m3_val(w_bs_u8)  # [BN, NKB] fp32
        b_eff = (b_val * _expand_blocks_n(bs_f, BLOCK_N, NKB, BLOCK_K)).to(tl.float16)

        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    acc = acc * (amax_v / 448.0 * w_ts_v)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# OPTIMIZED: Split ffn_key_res with shared prep_x + correct per-block scale
# ============================================================================

def linear_nvfp4_res_split_v2(x, weight_info, out_dtype=torch.float16):
    """Split NVFP4+FP8 residual: shared prep_x + separate GEMMs.
    
    Main: fused_nvfp4_gemm_kernel (no residual, FP4 only)
    Residual: fused_fp8_block_gemm_kernel (per-block FP8, correct)
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

    # NVFP4 main GEMM (FP4 path, no residual)
    out_main = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    fused_nvfp4_gemm_kernel[grid](
        x_awq, w, w_bs.view(torch.uint8), w_ts, amax, out_main,
        M, N, K, x_awq.stride(0),
        w.stride(0), w_bs.stride(0), out_main.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
    )

    # FP8 residual GEMM with per-block scales
    out_res = torch.empty(M, N, dtype=torch.float16, device=x_awq.device)
    if res_bs is not None:
        fused_fp8_block_gemm_kernel[grid](
            x_awq, res_w.view(torch.uint8), res_bs.view(torch.uint8), res_ts, amax, out_res,
            M, N, K, x_awq.stride(0),
            res_w.stride(0), res_bs.stride(0), out_res.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
        )
    else:
        # Per-tensor residual: use hardware FP8 dot
        fused_fp8_hwdot_gemm_kernel[grid](
            x_awq, res_w, res_ts, amax, out_res,
            M, N, K, x_awq.stride(0),
            res_w.stride(0), out_res.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
        )

    out = out_main + out_res
    out = out.reshape(*x.shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


# ============================================================================
# OPTIMIZED: RKV with FP8 hardware dot for k/v paths
# ============================================================================

@triton.jit
def fused_rkv_hwdot_kernel(
    xr_ptr, xk_ptr, xv_ptr,      # [M, K] bf16
    wr_ptr, wr_bs_ptr, wts_r_ptr,  # r: NVFP4 (packed [N, K//2])
    wk_ptr, wts_k_ptr,             # k: FP8 ([N, K])
    wv_ptr, wts_v_ptr,             # v: FP8 ([N, K])
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
    """RKV fused: r (NVFP4 FP16 dot) + k (FP8 hw dot) + v (FP8 hw dot)."""
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


def linear_rkv_hwdot(xr, xk, xv, wr_info, wk_info, wv_info, out_dtype=torch.float16):
    """RKV fused with FP8 hardware dot for k/v."""
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

    bm, bn, bk, nw = _nvfp4_cfg_for(M)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
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
    if out_dtype != torch.float16:
        or_, ok_, ov_ = or_.to(out_dtype), ok_.to(out_dtype), ov_.to(out_dtype)
    or_ = or_.reshape(*orig_shape[:-1], N)
    ok_ = ok_.reshape(*orig_shape[:-1], N)
    ov_ = ov_.reshape(*orig_shape[:-1], N)
    return or_, ok_, ov_


# ============================================================================
# Main test
# ============================================================================

def main():
    print("=" * 70)
    print("7.2B 算子优化 v2: FP8 硬件 dot + 拆分修正")
    print("=" * 70)

    m = build_model()
    z = m.z
    C = 4096
    device = "cuda"
    x = torch.randn(1, C, dtype=torch.float16, device=device)

    wr = z["blocks.1.att.receptance.weight"]
    wk = z["blocks.1.att.key.weight"]
    wv = z["blocks.1.att.value.weight"]
    wo = z["blocks.1.att.output.weight"]
    wfk = z["blocks.1.ffn.key.weight"]
    wfv = z["blocks.1.ffn.value.weight"]

    results = {}

    # --- Baseline ---
    print("\n[1] Baseline (current kernels)...", flush=True)
    t_rkv = bench(lambda: linear_rkv_fused(x.clone(), x.clone(), x.clone(), wr, wk, wv))
    t_fk = bench(lambda: linear_nvfp4_res_fused(x, wfk))
    t_fv = bench(lambda: linear_fp8_fused(x, wfv))
    t_ao = bench(lambda: linear_nvfp4_fused(x, wo))
    per_layer = t_rkv + t_fk + t_fv + t_ao
    print(f"  rkv:       {t_rkv:.4f} ms")
    print(f"  ffn_key:   {t_fk:.4f} ms")
    print(f"  ffn_val:   {t_fv:.4f} ms")
    print(f"  att_out:   {t_ao:.4f} ms")
    print(f"  per_layer: {per_layer:.4f} ms (32L: {per_layer*32:.2f} ms)")
    results["baseline"] = {"rkv": t_rkv, "ffn_key": t_fk, "ffn_val": t_fv, "att_out": t_ao}

    # --- FP8 hardware dot: ffn_value ---
    print("\n[2] FP8 HW dot: ffn_value...", flush=True)
    try:
        t = bench(lambda: linear_fp8_hwdot(x, wfv))
        ref = linear_fp8_fused(x, wfv)
        out = linear_fp8_hwdot(x, wfv)
        max_diff = (ref - out).abs().max().item()
        print(f"  current: {t_fv:.4f} ms → hwdot: {t:.4f} ms ({t_fv/t:.2f}x)  max_diff={max_diff:.6f}")
        results["fp8_hwdot_fv"] = {"ms": t, "speedup": t_fv/t, "max_diff": max_diff}
    except Exception as e:
        print(f"  ERR: {str(e)[:100]}")
        results["fp8_hwdot_fv"] = {"err": str(e)[:100]}

    # --- FP8 hardware dot: RKV (k/v paths) ---
    print("\n[3] FP8 HW dot: RKV (k/v)...", flush=True)
    try:
        t = bench(lambda: linear_rkv_hwdot(x.clone(), x.clone(), x.clone(), wr, wk, wv))
        ref_r, ref_k, ref_v = linear_rkv_fused(x.clone(), x.clone(), x.clone(), wr, wk, wv)
        out_r, out_k, out_v = linear_rkv_hwdot(x.clone(), x.clone(), x.clone(), wr, wk, wv)
        max_diff_r = (ref_r - out_r).abs().max().item()
        max_diff_k = (ref_k - out_k).abs().max().item()
        max_diff_v = (ref_v - out_v).abs().max().item()
        print(f"  current: {t_rkv:.4f} ms → hwdot: {t:.4f} ms ({t_rkv/t:.2f}x)")
        print(f"  max_diff: r={max_diff_r:.6f} k={max_diff_k:.6f} v={max_diff_v:.6f}")
        results["rkv_hwdot"] = {"ms": t, "speedup": t_rkv/t,
                                 "max_diff_r": max_diff_r, "max_diff_k": max_diff_k, "max_diff_v": max_diff_v}
    except Exception as e:
        print(f"  ERR: {str(e)[:100]}")
        results["rkv_hwdot"] = {"err": str(e)[:100]}

    # --- Split ffn_key_res with correct per-block scale ---
    print("\n[4] Split ffn_key_res (correct per-block scale)...", flush=True)
    try:
        t = bench(lambda: linear_nvfp4_res_split_v2(x, wfk))
        ref = linear_nvfp4_res_fused(x, wfk)
        out = linear_nvfp4_res_split_v2(x, wfk)
        max_diff = (ref - out).abs().max().item()
        print(f"  current: {t_fk:.4f} ms → split: {t:.4f} ms ({t_fk/t:.2f}x)  max_diff={max_diff:.6f}")
        results["ffn_key_split"] = {"ms": t, "speedup": t_fk/t, "max_diff": max_diff}
    except Exception as e:
        print(f"  ERR: {str(e)[:100]}")
        results["ffn_key_split"] = {"err": str(e)[:100]}

    # --- Combined: all optimizations ---
    print("\n[5] Combined: all optimizations...", flush=True)
    # Use best version of each kernel
    t_rkv_opt = results.get("rkv_hwdot", {}).get("ms", t_rkv)
    t_fk_opt = results.get("ffn_key_split", {}).get("ms", t_fk)
    t_fv_opt = results.get("fp8_hwdot_fv", {}).get("ms", t_fv)
    t_ao_opt = t_ao  # att_output is NVFP4, no FP8 optimization
    per_layer_opt = t_rkv_opt + t_fk_opt + t_fv_opt + t_ao_opt
    print(f"  rkv:       {t_rkv_opt:.4f} ms ({t_rkv/t_rkv_opt:.2f}x)")
    print(f"  ffn_key:   {t_fk_opt:.4f} ms ({t_fk/t_fk_opt:.2f}x)")
    print(f"  ffn_val:   {t_fv_opt:.4f} ms ({t_fv/t_fv_opt:.2f}x)")
    print(f"  att_out:   {t_ao_opt:.4f} ms (1.00x)")
    print(f"  per_layer: {per_layer_opt:.4f} ms (32L: {per_layer_opt*32:.2f} ms)")
    print(f"  vs baseline: {per_layer/per_layer_opt:.2f}x speedup")
    print(f"  est decode: {39.7 * per_layer_opt/per_layer:.1f} ms → {1000/(39.7 * per_layer_opt/per_layer):.1f} tok/s")
    results["combined"] = {"per_layer": per_layer_opt, "speedup": per_layer/per_layer_opt}

    # --- Full decode benchmark (baseline) ---
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
    results["decode_baseline"] = {"tps": tps, "ms_per_step": wall/50*1000}

    with open("/home/njzy/test/eval_tmp/fp8_hwdot_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved fp8_hwdot_test.json")


if __name__ == "__main__":
    main()
