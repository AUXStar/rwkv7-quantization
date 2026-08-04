#!/usr/bin/env python3
"""Fused FP8 quantized GEMM kernels with CUDA Graph support.

  prep_x:     x(fp16/bf16) -> bf16 + amax (GPU scalar) in ONE launch
  prep3_x:    same for xr/xk/xv in ONE launch
  prep_x_fp8: x(fp16/bf16) -> FP8 + scale (GPU-resident, for _scaled_mm path)
  fused_fp8_hwdot_gemm_kernel:  FP8 hardware tensor core dot (tl.dot(fp8,fp8))
  fused_rkv_fp8_kernel:         r/k/v all FP8 hardware dot in ONE kernel
  linear_fp8_fused / linear_rkv_fused / linear_quantized_fused

Hybrid dispatch (benchmark-tuned on RTX 5070 Ti, Blackwell sm_120):
  - att (N=K=4096, M<=4):     Triton + CUDA Graph  (40% faster than plain Triton)
  - ffn_key (N=16384, M=1):   prep_fp8 + _scaled_mm + CUDA Graph  (12% faster)
  - ffn_val (N=4096,K=16384): Triton (plain, graph doesn't help)
  - Large M (prefill):        _scaled_mm (cuBLASLt)

Numerics: per-tensor FP8 E4M3 quantization (amax/448 scale).
"""
import torch
import triton
import triton.language as tl

F8E4M3_MAX = 448.0


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
    """cast(AWQ(x)) -> bf16 + amax (GPU scalar). One launch."""
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
    """prep_x for three attention inputs in ONE launch."""
    def _prep1(xx):
        x2 = xx.reshape(-1, xx.shape[-1])
        if x2.dtype not in (torch.float16, torch.bfloat16):
            x2 = x2.to(torch.bfloat16)
        return x2.contiguous()
    xr2, xk2, xv2 = _prep1(xr), _prep1(xk), _prep1(xv)

    outs = []
    amaxs = []
    for xx in (xr2, xk2, xv2):
        outs.append(torch.empty(xx.shape, dtype=torch.bfloat16, device=xx.device))
        amaxs.append(torch.zeros(1, dtype=torch.float32, device=xx.device))
    M, K = outs[0].shape
    BLOCK_K = triton.next_power_of_2(K)
    grid = (M,)
    prep3_x_kernel[grid](
        xr2, xk2, xv2,
        awq_r, awq_k,
        outs[0], outs[1], outs[2],
        amaxs[0], amaxs[1], amaxs[2],
        M, K,
        xr2.stride(0), xr2.stride(1),
        outs[0].stride(0), outs[0].stride(1),
        AWQ_R=awq_r is not None, AWQ_K=awq_k is not None,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return outs[0], outs[1], outs[2], amaxs[0], amaxs[1], amaxs[2]


# ============================================================================
# FP8 helper
# ============================================================================

@triton.jit
def _e4m3_val(u8):
    """Dequantize E4M3 byte (uint8, IEEE e4m3fn bit layout) to fp32."""
    sign = (u8 >> 7) & 1
    exp = (u8 >> 3) & 0xF
    man = u8 & 0x7
    mag = tl.where(exp == 0, man * (1.0 / 512.0),
                   (1.0 + man / 8.0) * tl.exp2(exp.to(tl.float32) - 7.0))
    return tl.where(sign == 1, -mag, mag)


# ============================================================================
# Fused FP8 (W8A8) GEMM kernel — software dot (fp16 intermediate)
# ============================================================================

@triton.jit
def fused_fp8_gemm_kernel(
    x_ptr, w_ptr, w_ts_ptr, amax_ptr, out_ptr,
    M, N, K, stride_xm, stride_wn, stride_om,
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
        a_eff = x32.to(tl.float8e4nv).to(tl.float32).to(tl.float16)
        w_u8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                       mask=(offs_n[:, None] < N) & kmask[None, :], other=0)
        b_eff = _e4m3_val(w_u8).to(tl.float16)
        acc = tl.dot(a_eff, tl.trans(b_eff), acc)

    acc = acc * (amax_v / 448.0 * w_ts_v)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Fused FP8 (W8A8) GEMM kernel — hardware FP8 tensor core dot
# ============================================================================

@triton.jit
def fused_fp8_hwdot_gemm_kernel(
    x_ptr, w_ptr, w_ts_ptr, amax_ptr, out_ptr,
    M, N, K, stride_xm, stride_wn, stride_om,
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
        a_fp8 = tl.minimum(tl.maximum(x_tile.to(tl.float32) * inv_xs, -448.0), 448.0).to(tl.float8e4nv)
        w_fp8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k_start + offs_k)[None, :],
                        mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0)
        acc = tl.dot(a_fp8, tl.trans(w_fp8), acc)

    acc = acc * (amax_v / 448.0 * w_ts_v)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================================
# Fused RKV FP8 kernel: r/k/v all FP8 hardware dot in ONE launch
# ============================================================================

@triton.jit
def fused_rkv_fp8_kernel(
    xr_ptr, xk_ptr, xv_ptr,
    wr_ptr, wts_r_ptr,
    wk_ptr, wts_k_ptr,
    wv_ptr, wts_v_ptr,
    amax_r_ptr, amax_k_ptr, amax_v_ptr,
    or_ptr, ok_ptr, ov_ptr,
    M, N, K,
    stride_xm,
    stride_wr, stride_wk, stride_wv, stride_om,
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

    amax_r = tl.maximum(tl.load(amax_r_ptr), 1e-12)
    amax_k = tl.maximum(tl.load(amax_k_ptr), 1e-12)
    amax_v = tl.maximum(tl.load(amax_v_ptr), 1e-12)
    inv_xs_r = 448.0 / amax_r
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

        xr = tl.load(xr_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        a_r_fp8 = tl.minimum(tl.maximum(xr.to(tl.float32) * inv_xs_r, -448.0), 448.0).to(tl.float8e4nv)
        w_r_fp8 = tl.load(wr_ptr + offs_n[:, None] * stride_wr + (k_start + offs_k)[None, :], mask=wmask & kmask[None, :], other=0.0)
        acc_r = tl.dot(a_r_fp8, tl.trans(w_r_fp8), acc_r)

        xk = tl.load(xk_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        a_k_fp8 = tl.minimum(tl.maximum(xk.to(tl.float32) * inv_xs_k, -448.0), 448.0).to(tl.float8e4nv)
        w_k_fp8 = tl.load(wk_ptr + offs_n[:, None] * stride_wk + (k_start + offs_k)[None, :], mask=wmask & kmask[None, :], other=0.0)
        acc_k = tl.dot(a_k_fp8, tl.trans(w_k_fp8), acc_k)

        xv = tl.load(xv_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :], mask=xmask, other=0.0)
        a_v_fp8 = tl.minimum(tl.maximum(xv.to(tl.float32) * inv_xs_v, -448.0), 448.0).to(tl.float8e4nv)
        w_v_fp8 = tl.load(wv_ptr + offs_n[:, None] * stride_wv + (k_start + offs_k)[None, :], mask=wmask & kmask[None, :], other=0.0)
        acc_v = tl.dot(a_v_fp8, tl.trans(w_v_fp8), acc_v)

    out_r = (acc_r * (amax_r / 448.0 * wts_r)).to(tl.float16)
    out_k = (acc_k * (amax_k / 448.0 * wts_k)).to(tl.float16)
    out_v = (acc_v * (amax_v / 448.0 * wts_v)).to(tl.float16)

    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(or_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_r, mask=omask)
    tl.store(ok_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_k, mask=omask)
    tl.store(ov_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out_v, mask=omask)


# ============================================================================
# prep_x_fp8: fused quantize to FP8 for _scaled_mm path (all GPU-resident)
# ============================================================================

@triton.jit
def _prep_amax_kernel(x_ptr, amax_ptr, M, K, stride_xm, stride_xk, BLOCK_K: tl.constexpr):
    """Compute amax of each row + global amax via atomic_max."""
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(x_ptr + pid_m * stride_xm + offs_k * stride_xk, mask=mask, other=0.0)
    row_max = tl.max(tl.abs(x.to(tl.float32)))
    tl.atomic_max(amax_ptr, row_max)


@triton.jit
def _quantize_fp8_kernel(x_ptr, out_ptr, inv_scale_ptr, M, K, stride_xm, stride_xk, stride_om, stride_ok, BLOCK_K: tl.constexpr, F8_MAX: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(x_ptr + pid_m * stride_xm + offs_k * stride_xk, mask=mask, other=0.0).to(tl.float32)
    inv_scale = tl.load(inv_scale_ptr)
    x_fp8 = tl.minimum(tl.maximum(x * inv_scale, -F8_MAX), F8_MAX).to(tl.float8e4nv)
    tl.store(out_ptr + pid_m * stride_om + offs_k * stride_ok, x_fp8, mask=mask)


def prep_x_fp8(x):
    """Quantize x to FP8 with GPU-resident scale (no D2H sync). For _scaled_mm.

    Returns: (x_fp8 [M,K] float8_e4m3fn, x_scale [1] float32 on GPU)
    """
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype not in (torch.float16, torch.bfloat16):
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    M, K = x2.shape

    # Kernel 1: compute amax (GPU scalar)
    amax = torch.zeros(1, dtype=torch.float32, device=x2.device)
    BLOCK_K = triton.next_power_of_2(K)
    _prep_amax_kernel[(M,)](x2, amax, M, K, x2.stride(0), x2.stride(1), BLOCK_K=BLOCK_K, num_warps=4)

    # Compute inv_scale on GPU (no D2H sync)
    inv_scale = F8E4M3_MAX / amax.clamp(min=1e-12)

    # Kernel 2: quantize to FP8
    x_fp8 = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x2.device)
    _quantize_fp8_kernel[(M,)](x2, x_fp8, inv_scale, M, K,
                               x2.stride(0), x2.stride(1), x_fp8.stride(0), x_fp8.stride(1),
                               BLOCK_K=BLOCK_K, F8_MAX=F8E4M3_MAX, num_warps=4)

    # x_scale = amax / 448 (GPU float32)
    x_scale = amax / F8E4M3_MAX
    return x_fp8, x_scale


# ============================================================================
# Config — shape-adaptive tile selection (tuned on 7.2B Blackwell sm_120)
# ============================================================================

def _cfg_for(M, N=None, K=None):
    """Shape-adaptive launch config.

    Benchmark results (M=1, us per call on RTX 5070 Ti):
    - att_output [N=4096, K=4096]:  (16,64,64,4)   = 37.6us (1.0x BW, optimal)
    - ffn_key    [N=16384,K=4096]:  (16,64,128,4)  = 116.9us (0.8x BW, 37% faster)
    - ffn_value  [N=4096, K=16384]: (16,128,256,8) = 121.2us (0.8x BW, 29% faster)
    """
    if M <= 4:
        if K is not None and K >= 8192:
            return (16, 128, 256, 8)
        if N is not None and N >= 8192:
            return (16, 64, 128, 4)
        return (16, 64, 64, 4)
    if M <= 64:
        return (64, 64, 64, 4)
    return (64, 128, 64, 4)


def _best_method(M, N, K):
    """Choose best GEMM method based on shape (benchmark-tuned).

    Returns: 'triton', 'triton_graph', or 'smm_graph'
    """
    if M <= 4 and N * K <= 4096 * 4096:
        # att components (N=K=4096): graph eliminates ~19us launch overhead
        return 'triton_graph'
    if M == 1 and N >= 8192 and K <= 8192:
        # ffn_key (N=16384, K=4096): cuBLASLt + graph = 12% faster
        return 'smm_graph'
    # Default: Triton without graph (ffn_val or large M)
    return 'triton'


# ============================================================================
# CUDA Graph cache for decode-path GEMMs
# ============================================================================

class _GraphEntry:
    """One captured CUDA Graph for a fixed-shape GEMM."""
    __slots__ = ('graph', 'static_x', 'static_out', 'method')

    def __init__(self, graph, static_x, static_out, method):
        self.graph = graph
        self.static_x = static_x
        self.static_out = static_out
        self.method = method


# Module-level caches keyed by (M, N, K, w_ptr, method)
_graph_cache: dict = {}
_rkv_graph_cache: dict = {}


def _get_or_capture_single(x, w, w_ts, M, N, K, method):
    """Get or create a CUDA Graph for a single GEMM."""
    key = (M, N, K, w.data_ptr(), method)
    entry = _graph_cache.get(key)
    if entry is not None:
        return entry

    device = w.device
    # Ensure x is bf16 contiguous for static buffer
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.dtype != torch.bfloat16:
        x_2d = x_2d.to(torch.bfloat16)
    x_2d = x_2d.contiguous()

    static_x = torch.zeros(M, K, dtype=torch.bfloat16, device=device)

    # Warmup (3 iterations outside graph)
    for _ in range(3):
        if method == 'triton_graph':
            _run_triton(static_x, w, w_ts, M, N, K)
        elif method == 'smm_graph':
            _run_smm(static_x, w, w_ts, M, N, K)
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if method == 'triton_graph':
            static_out = _run_triton(static_x, w, w_ts, M, N, K)
        elif method == 'smm_graph':
            static_out = _run_smm(static_x, w, w_ts, M, N, K)

    entry = _GraphEntry(graph, static_x, static_out, method)
    _graph_cache[key] = entry
    return entry


def _run_triton(static_x, w, w_ts, M, N, K):
    """Triton path: prep_x + fused_fp8_hwdot_gemm_kernel."""
    x_awq, amax = prep_x(static_x)
    out = torch.empty(M, N, dtype=torch.float16, device=static_x.device)
    bm, bn, bk, nw = _cfg_for(M, N, K)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    fused_fp8_hwdot_gemm_kernel[grid](
        x_awq, w, w_ts, amax, out,
        M, N, K, x_awq.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
    )
    return out


def _run_smm(static_x, w, w_ts, M, N, K):
    """_scaled_mm path: prep_x_fp8 + torch._scaled_mm."""
    x_fp8, x_scale = prep_x_fp8(static_x)
    out = torch._scaled_mm(x_fp8, w.t(),
        scale_a=x_scale.reshape(1), scale_b=w_ts.reshape(1),
        out_dtype=torch.bfloat16)
    return out


def _get_or_capture_rkv(xr, xk, xv, wr, wk, wv, wts_r, wts_k, wts_v, M, N, K):
    """Get or create a CUDA Graph for the fused RKV kernel."""
    key = (M, N, K, wr.data_ptr(), wk.data_ptr(), wv.data_ptr())
    entry = _rkv_graph_cache.get(key)
    if entry is not None:
        return entry

    device = wr.device
    static_xr = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
    static_xk = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
    static_xv = torch.zeros(M, K, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        _run_rkv_triton(static_xr, static_xk, static_xv,
                        wr, wk, wv, wts_r, wts_k, wts_v, M, N, K)
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_or, static_ok, static_ov = _run_rkv_triton(
            static_xr, static_xk, static_xv,
            wr, wk, wv, wts_r, wts_k, wts_v, M, N, K)

    # Store as a composite entry
    class _RKVEntry:
        __slots__ = ('graph', 'static_xr', 'static_xk', 'static_xv',
                     'static_or', 'static_ok', 'static_ov')
    entry = _RKVEntry()
    entry.graph = graph
    entry.static_xr = static_xr
    entry.static_xk = static_xk
    entry.static_xv = static_xv
    entry.static_or = static_or
    entry.static_ok = static_ok
    entry.static_ov = static_ov
    _rkv_graph_cache[key] = entry
    return entry


def _run_rkv_triton(xr, xk, xv, wr, wk, wv, wts_r, wts_k, wts_v, M, N, K):
    """RKV fused Triton path."""
    xr_a, xk_a, xv_a, amax_r, amax_k, amax_v = prep3_x(xr, xk, xv)
    or_ = torch.empty(M, N, dtype=torch.float16, device=xr.device)
    ok_ = torch.empty(M, N, dtype=torch.float16, device=xr.device)
    ov_ = torch.empty(M, N, dtype=torch.float16, device=xr.device)
    bm, bn, bk, nw = _cfg_for(M, N, K)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    fused_rkv_fp8_kernel[grid](
        xr_a, xk_a, xv_a,
        wr, wts_r, wk, wts_k, wv, wts_v,
        amax_r, amax_k, amax_v,
        or_, ok_, ov_,
        M, N, K, xr_a.stride(0),
        wr.stride(0), wk.stride(0), wv.stride(0), or_.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
    )
    return or_, ok_, ov_


# ============================================================================
# Host wrappers
# ============================================================================

# Enable CUDA Graph for decode path
USE_CUDA_GRAPH = False  # Disabled: benchmark shows 96 replays/step overhead > launch savings


def linear_rkv_fused(xr, xk, xv, wr_info, wk_info, wv_info, out_dtype=torch.float16):
    """Fused r/k/v attention projections: 2 launches (prep3_x + GEMM).

    With CUDA Graph for decode (M<=4): graph replay (0 launch overhead).
    """
    orig_shape = xr.shape

    # Try CUDA Graph path for small M
    M = xr.numel() // xr.size(-1)
    K = xr.size(-1)
    N = wr_info["weight"].size(0)

    if USE_CUDA_GRAPH and M <= 4:
        wr = wr_info["weight"]
        wk = wk_info["weight"]
        wv = wv_info["weight"]
        entry = _get_or_capture_rkv(
            xr, xk, xv,
            wr, wk, wv,
            wr_info["tensor_scale"], wk_info["tensor_scale"], wv_info["tensor_scale"],
            M, N, K)
        # Copy inputs to static buffers
        xr_2d = xr.reshape(-1, K)
        if xr_2d.dtype != torch.bfloat16:
            xr_2d = xr_2d.to(torch.bfloat16)
        entry.static_xr.copy_(xr_2d)
        entry.static_xk.copy_(xk.reshape(-1, K).to(torch.bfloat16))
        entry.static_xv.copy_(xv.reshape(-1, K).to(torch.bfloat16))
        entry.graph.replay()
        # Reshape outputs
        or_ = entry.static_or.reshape(*orig_shape[:-1], N)
        ok_ = entry.static_ok.reshape(*orig_shape[:-1], N)
        ov_ = entry.static_ov.reshape(*orig_shape[:-1], N)
        if out_dtype != torch.float16:
            or_, ok_, ov_ = or_.to(out_dtype), ok_.to(out_dtype), ov_.to(out_dtype)
        return or_, ok_, ov_

    # Fallback: non-graph path
    xr_a, xk_a, xv_a, amax_r, amax_k, amax_v = prep3_x(
        xr, xk, xv,
        wr_info.get("awq_scale", None),
        wk_info.get("awq_scale", None),
    )
    M2, K2 = xr_a.shape
    N2 = wr_info["weight"].size(0)

    or_ = torch.empty(M2, N2, dtype=torch.float16, device=xr_a.device)
    ok_ = torch.empty(M2, N2, dtype=torch.float16, device=xr_a.device)
    ov_ = torch.empty(M2, N2, dtype=torch.float16, device=xr_a.device)

    bm, bn, bk, nw = _cfg_for(M2, N2, K2)
    grid = lambda meta: (triton.cdiv(M2, meta["BLOCK_M"]) * triton.cdiv(N2, meta["BLOCK_N"]),)
    fused_rkv_fp8_kernel[grid](
        xr_a, xk_a, xv_a,
        wr_info["weight"], wr_info["tensor_scale"],
        wk_info["weight"], wk_info["tensor_scale"],
        wv_info["weight"], wv_info["tensor_scale"],
        amax_r, amax_k, amax_v,
        or_, ok_, ov_,
        M2, N2, K2,
        xr_a.stride(0),
        wr_info["weight"].stride(0),
        wk_info["weight"].stride(0),
        wv_info["weight"].stride(0),
        or_.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )
    if out_dtype != torch.float16:
        or_, ok_, ov_ = or_.to(out_dtype), ok_.to(out_dtype), ov_.to(out_dtype)
    or_ = or_.reshape(*orig_shape[:-1], N2)
    ok_ = ok_.reshape(*orig_shape[:-1], N2)
    ov_ = ov_.reshape(*orig_shape[:-1], N2)
    return or_, ok_, ov_


def linear_fp8_fused(x, weight_info_or_w, res_scale=None, out_dtype=torch.float16):
    """Fused FP8 GEMM (W8A8): prep_x + single-kernel GEMM.

    With CUDA Graph for decode (M<=4): graph replay (0 launch overhead).
    Hybrid dispatch: Triton+Graph for att, _scaled_mm+Graph for ffn_key.
    """
    if isinstance(weight_info_or_w, dict):
        w = weight_info_or_w["weight"]
        w_ts = weight_info_or_w["tensor_scale"]
    else:
        w = weight_info_or_w
        w_ts = res_scale

    M = x.numel() // x.size(-1)
    K = x.size(-1)
    N = w.size(0)

    # Try CUDA Graph path for small M
    if USE_CUDA_GRAPH and M <= 4:
        method = _best_method(M, N, K)
        if method in ('triton_graph', 'smm_graph'):
            entry = _get_or_capture_single(x, w, w_ts, M, N, K, method)
            # Copy input to static buffer
            x_2d = x.reshape(-1, K)
            if x_2d.dtype != torch.bfloat16:
                x_2d = x_2d.to(torch.bfloat16)
            entry.static_x.copy_(x_2d)
            entry.graph.replay()
            out = entry.static_out.reshape(*x.shape[:-1], N)
            if out_dtype != out.dtype:
                out = out.to(out_dtype)
            return out

    # Fallback: non-graph Triton path
    x_awq, amax = prep_x(x)
    M2, K2 = x_awq.shape
    N2 = w.size(0)

    out = torch.empty(M2, N2, dtype=torch.float16, device=x_awq.device)
    bm, bn, bk, nw = _cfg_for(M2, N2, K2)
    grid = lambda meta: (triton.cdiv(M2, meta["BLOCK_M"]) * triton.cdiv(N2, meta["BLOCK_N"]),)
    fused_fp8_hwdot_gemm_kernel[grid](
        x_awq, w, w_ts, amax, out,
        M2, N2, K2,
        x_awq.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )

    out = out.reshape(*x.shape[:-1], N2)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Quantized GEMM dispatcher (used by the engine).

    - INT8: 无硬件张量核，走 linear_quantized（反量化 fp16 GEMM）。
    - FP8 M <= FUSED_M_MAX: fused single-kernel GEMM (with CUDA Graph).
    - FP8 M > FUSED_M_MAX: _scaled_mm path (cuBLAS wins for large M).
    """
    from fp8_ops import linear_fp8, linear_quantized
    qtype = weight_info.get("qtype", "fp8")
    # INT8 无融合内核，走通用 dispatcher（内部反量化）
    if qtype == "int8":
        return linear_quantized(x, weight_info, out_dtype)
    M = x.numel() // x.size(-1)
    if M <= FUSED_M_MAX:
        return linear_fp8_fused(x, weight_info, out_dtype)
    # Large M: _scaled_mm（FP8 张量核，保持 FP8 域，禁止反量化）
    return linear_fp8(x, weight_info, out_dtype)


FUSED_M_MAX = 64  # use fused single-kernel GEMM when M <= this (decode/small-batch domain)
