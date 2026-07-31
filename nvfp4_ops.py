#!/usr/bin/env python3
"""NVFP4 + FP8 GEMM operations for RWKV-7 v3a inference.

Provides:
- is_nvfp4_weight: detect NVFP4 quantized weights (has .nf4_b_scale sibling)
- is_fp8_weight: detect FP8 quantized weights (has .fp8_scale sibling)
- load_nvfp4_weight: load + swizzle block scale during model init
- load_fp8_weight: load FP8 weight + per-tensor scale
- linear_nvfp4: NVFP4 GEMM (FP4×FP4→BF16) with FUSED activation quantization
- linear_fp8: FP8 GEMM (FP8×FP8→BF16) with online activation quantization
- linear_quantized: dispatcher that picks the right GEMM based on weight_info
"""
import torch
import os
import importlib.util
import triton
import triton.language as tl

_mx = None

def _get_mx():
    """Load mx_utils from torch._vendor.quack (bypass __init__ which needs cutlass)."""
    global _mx
    if _mx is None:
        quack_dir = os.path.join(os.path.dirname(torch.__file__), '_vendor', 'quack')
        mx_path = os.path.join(quack_dir, 'mx_utils.py')
        spec = importlib.util.spec_from_file_location('_mx_utils', mx_path)
        _mx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mx)
    return _mx


# ============================================================================
# Fused NVFP4 Activation Quantization Triton Kernel
# Replaces mx.to_nvfp4 + mx.to_blocked with a single pass:
#   bf16 input -> packed FP4 uint8 + swizzled fp8 block scales
# ============================================================================

@triton.jit
def _fused_nvfp4_quant_kernel(
    x_ptr,              # [M, K] bf16 input
    out_packed_ptr,     # [M, K//2] uint8 output (packed FP4)
    out_bs_ptr,         # 1D fp8_e4m3fn output (swizzled block scales)
    pts_ptr,            # scalar fp32 (per-tensor scale)
    M, K,
    stride_xm, stride_xk,
    stride_pm, stride_pk,
    n_col_blocks,       # ceil(K//16 / 4), number of 4-block columns in swizzle
    BLOCK_K: tl.constexpr,  # elements per program along K (multiple of 16, power of 2)
):
    """Fused NVFP4 quantization: quantize + pack + swizzle in one kernel."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    pts = tl.load(pts_ptr)
    inv_pts = 1.0 / pts

    k_start = pid_k * BLOCK_K
    k_block_start = k_start // 16
    N_LOCAL_BLOCKS: tl.constexpr = BLOCK_K // 16

    # 2D indices: [N_LOCAL_BLOCKS, 16]
    block_row = tl.arange(0, N_LOCAL_BLOCKS)[:, None]
    elem_col = tl.arange(0, 16)[None, :]
    k_idx = k_start + block_row * 16 + elem_col

    mask_2d = (k_idx < K) & (pid_m < M)

    x = tl.load(
        x_ptr + pid_m * stride_xm + k_idx * stride_xk,
        mask=mask_2d, other=0.0
    ).to(tl.float32)

    # Per-block max_abs
    max_abs = tl.max(tl.abs(x), axis=1)

    # Block scale = max_abs / 6.0 (match reference: max_abs / F4_E2M1_MAX)
    block_scale = max_abs / 6.0
    # Use division (not mul by reciprocal) to match reference float32 rounding
    scaled_bs = block_scale / pts

    # Clamp to FP8 E4M3 range and cast (E4M3_EPS = 0.015625, F8E4M3_MAX = 448.0)
    scaled_bs = tl.maximum(scaled_bs, 0.015625)
    scaled_bs = tl.minimum(scaled_bs, 448.0)
    bs_fp8 = scaled_bs.to(tl.float8e4nv)

    # Data scaling: recip = (1/pts) / bs_fp8  (match reference)
    bs_f32 = bs_fp8.to(tl.float32)
    recip = inv_pts / bs_f32

    x_scaled = x * recip[:, None]
    x_scaled = tl.maximum(x_scaled, -6.0)
    x_scaled = tl.minimum(x_scaled, 6.0)

    # FP4 E2M1 conversion (round-to-nearest-even)
    sign = tl.where(x_scaled < 0, 1, 0).to(tl.uint8)
    a = tl.abs(x_scaled)

    code = tl.where(a <= 0.25, 0,
           tl.where(a < 0.75, 1,
           tl.where(a <= 1.25, 2,
           tl.where(a < 1.75, 3,
           tl.where(a <= 2.5, 4,
           tl.where(a < 3.5, 5,
           tl.where(a <= 5.0, 6, 7))))))).to(tl.uint8)

    fp4 = sign * 8 + code

    # Pack FP4 pairs into uint8
    fp4_3d = tl.reshape(fp4, (N_LOCAL_BLOCKS, 8, 2))
    lo, hi = tl.split(fp4_3d)
    packed_2d = hi * 16 + lo
    packed = tl.reshape(packed_2d, (N_LOCAL_BLOCKS * 8,))

    # Write packed output
    packed_idx = k_start // 2 + tl.arange(0, N_LOCAL_BLOCKS * 8)
    packed_mask = (packed_idx < K // 2) & (pid_m < M)
    tl.store(
        out_packed_ptr + pid_m * stride_pm + packed_idx * stride_pk,
        packed, mask=packed_mask
    )

    # Write block scales to swizzled (128x4) layout
    block_indices = k_block_start + tl.arange(0, N_LOCAL_BLOCKS)

    block_row_idx = pid_m // 128
    block_col_idx = block_indices // 4
    block_id = block_row_idx * n_col_blocks + block_col_idx

    m_in_block = pid_m % 128
    group = m_in_block // 32
    row_in_group = m_in_block % 32
    col_in_block = block_indices % 4

    within_block = row_in_group * 16 + group * 4 + col_in_block
    bs_output_idx = block_id * 512 + within_block

    bs_mask = (pid_m < M) & (block_indices < (K // 16))
    tl.store(out_bs_ptr + bs_output_idx, bs_fp8, mask=bs_mask)


def fused_nvfp4_quant(x, per_tensor_scale=None):
    """Fused NVFP4 quantization: bf16 -> packed FP4 + swizzled block scales.

    Args:
        x: [M, K] bf16/fp16 input (K must be multiple of 16)
        per_tensor_scale: scalar fp32, or None (auto-compute from amax)

    Returns:
        packed: [M, K//2] uint8 (packed FP4 E2M1)
        bs_swizzled: 1D fp8_e4m3fn (128x4 swizzled block scales)
        per_tensor_scale: scalar fp32
    """
    M, K = x.shape

    if x.dtype == torch.float16:
        x = x.to(torch.bfloat16)

    if per_tensor_scale is None:
        amax = x.abs().max()
        if amax > 0:
            per_tensor_scale = amax.float() / 2688.0  # amax / (448 * 6)
        else:
            per_tensor_scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)

    packed = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)

    n_col_blocks = (K // 16 + 3) // 4
    n_row_blocks = (M + 127) // 128
    bs_size = n_row_blocks * n_col_blocks * 512
    bs_swizzled = torch.zeros(bs_size, dtype=torch.float8_e4m3fn, device=x.device)

    BLOCK_K = 256
    if K < BLOCK_K:
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_K = max(BLOCK_K, 16)

    grid = (M, triton.cdiv(K, BLOCK_K))

    _fused_nvfp4_quant_kernel[grid](
        x, packed, bs_swizzled, per_tensor_scale,
        M, K,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        n_col_blocks,
        BLOCK_K=BLOCK_K,
    )

    return packed, bs_swizzled, per_tensor_scale


# ============================================================================
# Detection
# ============================================================================

def is_nvfp4_weight(z, key):
    """Check if a weight key has NVFP4 quantization (has .nf4_b_scale sibling)."""
    return (key + ".nf4_b_scale") in z

def is_fp8_weight(z, key):
    """Check if a weight key has FP8 quantization (has .fp8_scale sibling)."""
    return (key + ".fp8_scale") in z


# ============================================================================
# Loading
# ============================================================================

def load_nvfp4_weight(z, key, dev):
    """Load NVFP4 weight: packed uint8 + pre-swizzled block scale + tensor scale.

    Removes the .nf4_b_scale and .nvfp4_t_scale keys from z.
    Returns a dict with weight, block_scale (swizzled 1D), tensor_scale (scalar), qtype="nvfp4".
    """
    mx = _get_mx()
    w = z[key].to(device=dev).contiguous()           # [N, K//2] uint8
    bs = z[key + ".nf4_b_scale"].to(device=dev).contiguous()  # [N, K//16] float8_e4m3fn
    ts = z[key + ".nvfp4_t_scale"].to(device=dev)    # scalar float32
    bs_swizzled = mx.to_blocked(bs)                   # 1D flat, swizzled for cuBLAS
    del z[key + ".nf4_b_scale"]
    del z[key + ".nvfp4_t_scale"]
    return {
        "weight": w,
        "block_scale": bs_swizzled,
        "tensor_scale": ts,
        "qtype": "nvfp4",
    }

def load_fp8_weight(z, key, dev):
    """Load FP8 weight: float8_e4m3fn + per-tensor scale.

    Removes the .fp8_scale key from z.
    Returns a dict with weight, tensor_scale (scalar), qtype="fp8".
    """
    w = z[key].to(device=dev).contiguous()            # [N, K] float8_e4m3fn
    scale = z[key + ".fp8_scale"].to(device=dev)      # scalar float32
    del z[key + ".fp8_scale"]
    return {
        "weight": w,
        "tensor_scale": scale,
        "qtype": "fp8",
    }


# ============================================================================
# GEMM
# ============================================================================

FP8_E4M3_MAX = 448.0

def linear_nvfp4(x, weight_info, out_dtype=torch.float16):
    """NVFP4 GEMM: quantize input on-the-fly using FUSED Triton kernel, then _scaled_mm.

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_nvfp4_weight
        out_dtype: output dtype (default fp16 for v3a compatibility)

    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w = weight_info["weight"]              # [N, K//2] uint8
    w_bs = weight_info["block_scale"]      # 1D swizzled float8_e4m3fn
    w_ts = weight_info["tensor_scale"]     # scalar float32

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    # Fused quantization: quantize + pack + swizzle in one kernel
    x_packed, x_bs_swizzled, x_ts = fused_nvfp4_quant(x_2d)

    # FP4×FP4→BF16 GEMM
    a_fp4 = x_packed.view(torch.float4_e2m1fn_x2)
    b_fp4 = w.view(torch.float4_e2m1fn_x2)

    out = torch._scaled_mm(a_fp4, b_fp4.t(),
        scale_a=x_bs_swizzled, scale_b=w_bs,
        out_dtype=torch.bfloat16)

    # Fold per-tensor scales
    out = out * x_ts * w_ts

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_fp8(x, weight_info, out_dtype=torch.float16):
    """FP8 GEMM: quantize input on-the-fly to FP8, use torch._scaled_mm.

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_fp8_weight
        out_dtype: output dtype (default fp16 for v3a compatibility)

    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w = weight_info["weight"]              # [N, K] float8_e4m3fn
    w_scale = weight_info["tensor_scale"]  # scalar float32

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    # Quantize input to FP8 E4M3 with per-tensor scale
    amax_x = x_2d.abs().max()
    if amax_x > 0:
        x_scale = (amax_x / FP8_E4M3_MAX).float()
    else:
        x_scale = torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)

    x_fp8 = (x_2d.float() / x_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    out = torch._scaled_mm(x_fp8, w.t(),
        scale_a=x_scale.reshape(1),
        scale_b=w_scale.reshape(1),
        out_dtype=torch.bfloat16)

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized(x, weight_info, out_dtype=torch.float16):
    """Dispatcher: pick the right GEMM based on qtype in weight_info."""
    qtype = weight_info.get("qtype", "nvfp4")
    if qtype == "fp8":
        return linear_fp8(x, weight_info, out_dtype)
    return linear_nvfp4(x, weight_info, out_dtype)
