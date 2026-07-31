#!/usr/bin/env python3
"""NVFP4 + FP8 GEMM operations for RWKV-7 v3a inference.

Provides:
- is_nvfp4_weight: detect NVFP4 quantized weights (has .nf4_b_scale sibling)
- is_fp8_weight: detect FP8 quantized weights (has .fp8_scale sibling)
- load_nvfp4_weight: load + optionally swizzle block scale during model init
- load_fp8_weight: load FP8 weight + per-tensor scale
- linear_nvfp4: NVFP4 GEMM (FP4×FP4→BF16) with FUSED activation quantization (W4A4)
- linear_nvfp4_w4a16: NVFP4 W4A16 GEMM (dequant weight→BF16, FP16 activation)
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
# NVFP4 Dequantization (W4A16 path)
# ============================================================================

# FP4 E2M1 lookup table: index → float value
# Encoding: bit 3 = sign, bits 0-2 = magnitude code
# 0:+0.0  1:+0.5  2:+1.0  3:+1.5  4:+2.0  5:+3.0  6:+4.0  7:+6.0
# 8:-0.0  9:-0.5 10:-1.0 11:-1.5 12:-2.0 13:-3.0 14:-4.0 15:-6.0
_FP4_TABLE = None

def _get_fp4_table(device):
    global _FP4_TABLE
    if _FP4_TABLE is None or _FP4_TABLE.device != device:
        _FP4_TABLE = torch.tensor([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=torch.float32, device=device)
    return _FP4_TABLE


def dequantize_nvfp4(packed, block_scale, tensor_scale):
    """Dequantize NVFP4 packed weight to BF16 (pure PyTorch, W4A16 path).

    Args:
        packed: [N, K//2] uint8 (packed FP4 pairs, hi*16+lo convention)
        block_scale: [N, K//16] float8_e4m3fn (NOT swizzled, normal layout)
        tensor_scale: scalar float32 (per-tensor) or [N] float32 (per-channel)

    Returns:
        [N, K] bfloat16 tensor
    """
    N, K_half = packed.shape
    K = K_half * 2

    # Unpack: [N, K//2] uint8 → [N, K] uint8 (FP4 nibble indices)
    lo = (packed & 0x0F)                                   # [N, K//2] - even indices
    hi = ((packed >> 4) & 0x0F)                            # [N, K//2] - odd indices
    fp4_idx = torch.empty(N, K, dtype=torch.uint8, device=packed.device)
    fp4_idx[:, 0::2] = lo
    fp4_idx[:, 1::2] = hi

    # Lookup FP4 float values
    table = _get_fp4_table(packed.device)
    values = table[fp4_idx.long()]                         # [N, K] float32

    # Apply block scales via reshape + broadcast (auto-detect block_size)
    n_blocks = block_scale.shape[1]                        # K // block_size
    block_size = K // n_blocks
    values = values.view(N, n_blocks, block_size)          # [N, n_blocks, block_size]
    bs = block_scale.to(torch.float32).unsqueeze(-1)       # [N, n_blocks, 1]
    out = (values * bs).view(N, K)                         # [N, K] float32

    # Apply tensor scale (scalar for per-tensor, [N] for per-channel)
    if tensor_scale.dim() == 0:
        out = out * tensor_scale                           # scalar
    else:
        out = out * tensor_scale.unsqueeze(-1)             # [N, 1] broadcast
    return out.to(torch.bfloat16)


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

def load_nvfp4_weight(z, key, dev, swizzle=True, fused=False):
    """Load NVFP4 weight: packed uint8 + block scale + tensor scale.

    Args:
        z: weight dict
        key: weight key
        dev: target device
        swizzle: if True, swizzle block scales for _scaled_mm (W4A4 path).
                 if False, keep normal [N, K//16] layout.
        fused: if True, mark qtype for fused single-kernel GEMM (uses [N,K//16] layout).

    Removes the .nf4_b_scale and .nvfp4_t_scale keys from z.
    Returns a dict with weight, block_scale, tensor_scale, qtype.
    """
    w = z[key].to(device=dev).contiguous()           # [N, K//2] uint8
    bs = z[key + ".nf4_b_scale"].to(device=dev).contiguous()  # [N, K//16] float8_e4m3fn
    ts = z[key + ".nvfp4_t_scale"].to(device=dev)    # scalar float32
    del z[key + ".nf4_b_scale"]
    del z[key + ".nvfp4_t_scale"]

    # Load AWQ scale if present
    awq_scale = None
    if (key + ".awq_scale") in z:
        awq_scale = z[key + ".awq_scale"].to(device=dev)  # [K] float32
        del z[key + ".awq_scale"]

    if swizzle:
        mx = _get_mx()
        bs_out = mx.to_blocked(bs)                    # 1D flat, swizzled for cuBLAS
        qtype = "nvfp4"
    else:
        bs_out = bs                                   # [N, K//16] normal layout
        qtype = "nvfp4_fused" if fused else "nvfp4_w4a16"

    result = {
        "weight": w,
        "block_scale": bs_out,
        "tensor_scale": ts,
        "qtype": qtype,
    }
    if fused:
        # Keep swizzled copy for _scaled_mm fallback (prefill routing)
        result["block_scale_sw"] = _get_mx().to_blocked(bs).contiguous()
    if awq_scale is not None:
        result["awq_scale"] = awq_scale
    # Load FP8 residual if present (NVFP4+FP8 residual scheme)
    if (key + ".res_fp8") in z:
        result["res_fp8"] = z[key + ".res_fp8"].to(device=dev).contiguous()
        if (key + ".res_bs") in z:
            # Task 3: per-block residual scales [N, K//16] fp8 (ratio)
            rbs = z[key + ".res_bs"].to(device=dev).contiguous()
            result["res_block_scale"] = rbs                       # [N, K//16]
            result["res_block_scale_sw"] = _get_mx().to_blocked(rbs).contiguous()
            del z[key + ".res_bs"]
            if (key + ".res_fp8_scale") in z:
                result["res_fp8_scale"] = z[key + ".res_fp8_scale"].to(device=dev)
                del z[key + ".res_fp8_scale"]
        else:
            # legacy per-tensor residual scale
            result["res_fp8_scale"] = z[key + ".res_fp8_scale"].to(device=dev)
            del z[key + ".res_fp8_scale"]
        del z[key + ".res_fp8"]
        if qtype == "nvfp4":
            result["qtype"] = "nvfp4_res"
        elif qtype == "nvfp4_fused":
            result["qtype"] = "nvfp4_res_fused"
        else:
            result["qtype"] = "nvfp4_res_w4a16"
    return result

def load_fp8_weight(z, key, dev, w8a16=False):
    """Load FP8 weight: float8_e4m3fn + per-tensor scale.

    Args:
        z: weight dict
        key: weight key
        dev: target device
        w8a16: if True, use W8A16 path (weight-only, FP16 activation).
               if False, use W8A8 path (both weight and activation quantized).

    Removes the .fp8_scale key from z.
    Returns a dict with weight, tensor_scale (scalar), qtype.
    """
    w = z[key].to(device=dev).contiguous()            # [N, K] float8_e4m3fn
    scale = z[key + ".fp8_scale"].to(device=dev)      # scalar float32
    del z[key + ".fp8_scale"]
    return {
        "weight": w,
        "tensor_scale": scale,
        "qtype": "fp8_w8a16" if w8a16 else "fp8",
    }


# ============================================================================
# GEMM
# ============================================================================

FP8_E4M3_MAX = 448.0

def linear_nvfp4(x, weight_info, out_dtype=torch.float16):
    """NVFP4 GEMM (W4A4): quantize input on-the-fly, then FP4×FP4 _scaled_mm.
    No dequantization — true quantized GEMM.

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_nvfp4_weight (swizzle=True)
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

    # Apply AWQ inverse scaling BEFORE quantization: x' = x / s
    awq_scale = weight_info.get("awq_scale", None)
    if awq_scale is not None:
        x_2d = x_2d / awq_scale.to(x_2d.dtype)

    # Fused quantization: quantize + pack + swizzle in one kernel
    x_packed, x_bs_swizzled, x_ts = fused_nvfp4_quant(x_2d)

    # FP4×FP4→BF16 GEMM (no dequantization)
    a_fp4 = x_packed.view(torch.float4_e2m1fn_x2)
    b_fp4 = w.view(torch.float4_e2m1fn_x2)

    out = torch._scaled_mm(a_fp4, b_fp4.t(),
        scale_a=x_bs_swizzled, scale_b=w_bs,
        out_dtype=torch.bfloat16)

    # Fold per-tensor scales
    out = out * x_ts * w_ts

    # Add FP8 residual if present (NVFP4+FP8 residual scheme)
    # Residual is in AWQ-scaled space, same x_2d already has AWQ applied
    if "res_fp8" in weight_info:
        res_w = weight_info["res_fp8"]           # [N, K] float8_e4m3fn
        if "res_block_scale" in weight_info:
            # Task 3: per-block residual — pure FP8xFP8 GEMM in quantized domain.
            # scale_b = swizzled per-block fp8 scales, scale_a = tensor (res_ts).
            amax_x = x_2d.abs().max()
            if amax_x > 0:
                x_scale = (amax_x / FP8_E4M3_MAX).float()
            else:
                x_scale = torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)
            x_fp8 = (x_2d.float() / x_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            res_bs_sw = weight_info["res_block_scale_sw"]  # swizzled [N*K/16] fp8
            res_ts = weight_info["res_fp8_scale"]
            # scale_b = fp8 block ratios; tensor scale folded after GEMM (NVFP4-style)
            res_out = torch._scaled_mm(x_fp8, res_w.t(),
                scale_a=x_scale.reshape(1), scale_b=res_bs_sw,
                out_dtype=torch.bfloat16)
            out = out + res_out * res_ts.float()
        else:
            res_scale = weight_info["res_fp8_scale"] # scalar fp32
            # FP8×FP8→BF16 GEMM for residual (W8A8, no dequantization)
            amax_x = x_2d.abs().max()
            if amax_x > 0:
                x_scale = (amax_x / FP8_E4M3_MAX).float()
            else:
                x_scale = torch.tensor(1.0, dtype=torch.float32, device=x_2d.device)
            x_fp8 = (x_2d.float() / x_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            res_out = torch._scaled_mm(x_fp8, res_w.t(),
                scale_a=x_scale.reshape(1), scale_b=res_scale.reshape(1),
                out_dtype=torch.bfloat16)
            out = out + res_out

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_nvfp4_w4a16(x, weight_info, out_dtype=torch.float16):
    """NVFP4 W4A16 GEMM: dequantize weight to BF16, then FP16 GEMM.

    Weight-only quantization: activations stay at FP16 precision.
    Eliminates activation quantization error (main bottleneck of W4A4).

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_nvfp4_weight (swizzle=False)
        out_dtype: output dtype (default fp16 for v3a compatibility)

    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w_packed = weight_info["weight"]        # [N, K//2] uint8
    w_bs = weight_info["block_scale"]       # [N, K//16] float8_e4m3fn (normal layout)
    w_ts = weight_info["tensor_scale"]      # scalar float32

    # Dequantize weight to BF16
    w_bf16 = dequantize_nvfp4(w_packed, w_bs, w_ts)  # [N, K] bfloat16

    # Add FP8 residual if present (NVFP4+FP8 residual scheme)
    if "res_fp8" in weight_info:
        if "res_block_scale" in weight_info:
            res_bs = weight_info["res_block_scale"]
            n_b = res_bs.shape[1]
            rbs_exp = res_bs.float().unsqueeze(-1).repeat(1, 1, 16).reshape(weight_info["res_fp8"].shape)
            ts_r = weight_info["res_fp8_scale"].float()
            res_deq = (weight_info["res_fp8"].float() * rbs_exp * ts_r).to(torch.bfloat16)
            w_bf16 = w_bf16 + res_deq
        else:
            w_bf16 = w_bf16 + weight_info["res_fp8"].to(torch.bfloat16) * weight_info["res_fp8_scale"].to(torch.bfloat16)

    # Reshape input
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()

    # Apply AWQ inverse scaling: x' = x / s
    awq_scale = weight_info.get("awq_scale", None)
    if awq_scale is not None:
        x_2d = x_2d / awq_scale.to(x_2d.dtype)

    # FP16 GEMM (input FP16, weight BF16→FP16)
    if x_2d.dtype == torch.bfloat16:
        x_2d = x_2d.to(torch.float16)
    w_fp16 = w_bf16.to(torch.float16)

    out = torch.mm(x_2d, w_fp16.t())  # [M, N]

    N = w_packed.size(0)
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


def linear_fp8_w8a16(x, weight_info, out_dtype=torch.float16):
    """FP8 W8A16 GEMM: dequantize weight to FP16, then FP16 GEMM.

    Weight-only quantization: activations stay at FP16 precision.
    Eliminates FP8 activation quantization error.

    Args:
        x: [B, T, K] or [M, K] fp16/bf16 input
        weight_info: dict from load_fp8_weight (w8a16=True)
        out_dtype: output dtype

    Returns:
        [B, T, N] or [M, N] in out_dtype
    """
    w = weight_info["weight"]              # [N, K] float8_e4m3fn
    w_scale = weight_info["tensor_scale"]  # scalar float32

    # Dequantize weight: FP8 → FP16, then multiply by scale
    w_fp16 = w.to(torch.float16) * w_scale.to(torch.float16)

    # Reshape input
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.bfloat16:
        x_2d = x_2d.to(torch.float16)

    # FP16 GEMM
    out = torch.mm(x_2d, w_fp16.t())  # [M, N]

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
    if qtype == "fp8_w8a16":
        return linear_fp8_w8a16(x, weight_info, out_dtype)
    if qtype in ("nvfp4_w4a16", "nvfp4_res_w4a16"):
        return linear_nvfp4_w4a16(x, weight_info, out_dtype)
    # nvfp4 and nvfp4_res both use W4A4 quantized GEMM (no dequantization)
    return linear_nvfp4(x, weight_info, out_dtype)


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Hybrid dispatcher for quantized GEMMs.

    - M <= FUSED_M_MAX (decode/small batch): single-kernel fused GEMM
      (nvfp4_fused / nvfp4_res_fused / fp8) — 2 launches per linear (prep_x + GEMM).
    - M > FUSED_M_MAX (prefill/large batch): _scaled_mm path
      (swizzled scales from block_scale_sw) — cuBLAS wins for large M.

    Requires block scales in [N, K//16] unswizzled layout (load with fused=True).
    """
    from fused_nvfp4_gemm import linear_quantized_fused as _dispatch
    return _dispatch(x, weight_info, out_dtype)


FUSED_M_MAX = 64  # use fused single-kernel GEMM when M <= this (decode/small-batch domain)
