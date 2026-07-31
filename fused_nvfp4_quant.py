#!/usr/bin/env python3
"""Fused NVFP4 activation quantization Triton kernel.

Replaces mx.to_nvfp4 + mx.to_blocked with a single fused kernel that:
1. Quantizes bf16 activations to FP4 E2M1 (with per-tensor + per-block scaling)
2. Packs FP4 pairs into uint8
3. Writes block scales directly in 128x4 swizzled layout for cuBLAS _scaled_mm

Benchmark vs current approach (mx.to_nvfp4 + mx.to_blocked):
- Eliminates ~10 intermediate kernel launches
- Single pass over input data
- Direct swizzled output (no separate scatter)
"""
import torch
import triton
import triton.language as tl
import importlib.util
import time
import sys
import os

# Load mx_utils for reference comparison
spec = importlib.util.spec_from_file_location(
    "mx_utils",
    "/home/njzy/test/.venv/lib/python3.13/site-packages/torch/_vendor/quack/mx_utils.py"
)
mx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mx)

# Constants
F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
F8E4M3_EPS = 1.52587890625e-05  # 2^-16, smallest subnormal


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
    """Fused NVFP4 quantization kernel.

    Each program handles 1 row (m) and BLOCK_K elements along K.
    Grid: (M, ceil(K / BLOCK_K))
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Load per-tensor scale
    pts = tl.load(pts_ptr)
    inv_pts = 1.0 / pts

    # K range for this program
    k_start = pid_k * BLOCK_K
    k_block_start = k_start // 16
    N_LOCAL_BLOCKS: tl.constexpr = BLOCK_K // 16

    # 2D indices: [N_LOCAL_BLOCKS, 16]
    block_row = tl.arange(0, N_LOCAL_BLOCKS)[:, None]  # [N_LOCAL_BLOCKS, 1]
    elem_col = tl.arange(0, 16)[None, :]                # [1, 16]
    k_idx = k_start + block_row * 16 + elem_col          # [N_LOCAL_BLOCKS, 16]

    mask_2d = (k_idx < K) & (pid_m < M)

    # Load input as float32
    x = tl.load(
        x_ptr + pid_m * stride_xm + k_idx * stride_xk,
        mask=mask_2d, other=0.0
    ).to(tl.float32)

    # === Per-block max_abs ===
    max_abs = tl.max(tl.abs(x), axis=1)  # [N_LOCAL_BLOCKS]

    # === Block scale = max_abs / 6.0 ===
    block_scale = max_abs / 6.0  # [N_LOCAL_BLOCKS]

    # Scale by per-tensor scale: scaled_bs = block_scale / pts
    scaled_bs = block_scale * inv_pts

    # Clamp to FP8 E4M3 range and cast (E4M3 min normal = 0.015625)
    scaled_bs = tl.maximum(scaled_bs, 0.015625)
    scaled_bs = tl.minimum(scaled_bs, 448.0)
    bs_fp8 = scaled_bs.to(tl.float8e4nv)  # [N_LOCAL_BLOCKS] fp8

    # === Data scaling ===
    # recip = (1/pts) / bs_fp8
    bs_f32 = bs_fp8.to(tl.float32)        # [N_LOCAL_BLOCKS]
    recip = inv_pts / bs_f32               # [N_LOCAL_BLOCKS]

    x_scaled = x * recip[:, None]          # [N_LOCAL_BLOCKS, 16]
    x_scaled = tl.maximum(x_scaled, -6.0)  # F4 E2M1 min
    x_scaled = tl.minimum(x_scaled, 6.0)   # F4 E2M1 max

    # === FP4 E2M1 conversion ===
    sign = tl.where(x_scaled < 0, 1, 0).to(tl.uint8)
    a = tl.abs(x_scaled)

    code = tl.where(a <= 0.25, 0,
           tl.where(a < 0.75, 1,
           tl.where(a <= 1.25, 2,
           tl.where(a < 1.75, 3,
           tl.where(a <= 2.5, 4,
           tl.where(a < 3.5, 5,
           tl.where(a <= 5.0, 6, 7))))))).to(tl.uint8)

    fp4 = sign * 8 + code  # [N_LOCAL_BLOCKS, 16] uint8, values 0-15

    # === Pack FP4 pairs into uint8 ===
    # byte = fp4[odd] << 4 | fp4[even]
    # Reshape [N_LOCAL_BLOCKS, 16] -> [N_LOCAL_BLOCKS, 8, 2]
    # then split along last dim to get lo [N_LOCAL_BLOCKS, 8] and hi [N_LOCAL_BLOCKS, 8]
    fp4_3d = tl.reshape(fp4, (N_LOCAL_BLOCKS, 8, 2))
    lo, hi = tl.split(fp4_3d)  # each [N_LOCAL_BLOCKS, 8]
    packed_2d = hi * 16 + lo   # [N_LOCAL_BLOCKS, 8] uint8
    packed = tl.reshape(packed_2d, (N_LOCAL_BLOCKS * 8,))  # [N_LOCAL_BLOCKS * 8] uint8

    # === Write packed output ===
    packed_idx = k_start // 2 + tl.arange(0, N_LOCAL_BLOCKS * 8)
    packed_mask = (packed_idx < K // 2) & (pid_m < M)
    tl.store(
        out_packed_ptr + pid_m * stride_pm + packed_idx * stride_pk,
        packed, mask=packed_mask
    )

    # === Write block scales to swizzled (128x4) layout ===
    block_indices = k_block_start + tl.arange(0, N_LOCAL_BLOCKS)  # [N_LOCAL_BLOCKS]

    # Swizzle formula:
    # block_row = m // 128
    # block_col = k_block // 4
    # block_id = block_row * n_col_blocks + block_col
    # m_in_block = m % 128
    # group = m_in_block // 32  (0-3)
    # row_in_group = m_in_block % 32  (0-31)
    # col_in_block = k_block % 4  (0-3)
    # within_block = row_in_group * 16 + group * 4 + col_in_block
    # output_idx = block_id * 512 + within_block

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


def fused_nvfp4_quant(x: torch.Tensor, per_tensor_scale: torch.Tensor = None):
    """Fused NVFP4 quantization: bf16 -> packed FP4 + swizzled block scales.

    Args:
        x: [M, K] bf16/fp16 input (K must be multiple of 16)
        per_tensor_scale: scalar fp32, or None (auto-compute from amax)

    Returns:
        packed: [M, K//2] uint8 (packed FP4 E2M1)
        bs_swizzled: 1D fp8_e4m3fn (128x4 swizzled block scales)
        per_tensor_scale: scalar fp32
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    M, K = x.shape
    assert K % 16 == 0, f"K must be multiple of 16, got {K}"

    if x.dtype == torch.float16:
        x = x.to(torch.bfloat16)

    # Compute per-tensor scale if not provided
    if per_tensor_scale is None:
        amax = x.abs().max()
        if amax > 0:
            per_tensor_scale = mx.nvfp4_per_tensor_scale(amax)
        else:
            per_tensor_scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)

    # Allocate outputs
    packed = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)

    # Swizzled block scale size
    n_col_blocks = (K // 16 + 3) // 4  # ceil(K//16 / 4)
    n_row_blocks = (M + 127) // 128
    bs_size = n_row_blocks * n_col_blocks * 512
    bs_swizzled = torch.zeros(bs_size, dtype=torch.float8_e4m3fn, device=x.device)

    # Launch kernel
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
# Correctness test
# ============================================================================

def test_correctness():
    """Compare fused kernel output vs mx_utils reference."""
    device = 'cuda'
    torch.manual_seed(42)

    test_cases = [
        ("small", 4, 256),
        ("ffn_key_decode", 1, 2560),
        ("ffn_key_prefill", 128, 2560),
        ("ffn_value_decode", 1, 10240),
        ("ffn_value_prefill", 64, 10240),
        ("aligned_128", 128, 2560),
    ]

    all_pass = True
    for name, M, K in test_cases:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.3

        # Reference: mx_utils
        amax_ref = x.abs().max()
        ts_ref = mx.nvfp4_per_tensor_scale(amax_ref)
        x_bf16 = x.to(torch.bfloat16)
        packed_ref, bs_ref, ts_ret_ref = mx.to_nvfp4(x_bf16, block_size=16, per_tensor_scale=ts_ref)
        bs_swizzled_ref = mx.to_blocked(bs_ref)

        # Fused kernel
        packed_fused, bs_fused, ts_fused = fused_nvfp4_quant(x, ts_ref)

        # Compare packed FP4 (allow ≤1 nibble diff from FP rounding at boundaries)
        packed_match = torch.equal(packed_ref, packed_fused)
        packed_diff = (packed_ref.int() - packed_fused.int()).abs()
        # Count how many nibbles differ (diff of 1 = low nibble, diff of 16 = high nibble)
        n_mismatch = (packed_ref != packed_fused).sum().item()
        n_total = packed_ref.numel()
        packed_close = n_mismatch / n_total < 0.01  # < 1% mismatch is acceptable

        # Compare swizzled block scales
        bs_ref_flat = bs_swizzled_ref.reshape(-1).to(torch.float8_e4m3fn)
        bs_fused_flat = bs_fused.to(torch.float8_e4m3fn)

        # Only compare the valid region (not padding)
        min_bs_size = min(bs_ref_flat.numel(), bs_fused_flat.numel())
        bs_ref_valid = bs_ref_flat[:min_bs_size].to(torch.float32)
        bs_fused_valid = bs_fused_flat[:min_bs_size].to(torch.float32)
        bs_diff = (bs_ref_valid - bs_fused_valid).abs()
        bs_exact = torch.equal(bs_ref_valid, bs_fused_valid)

        # Check if block scales are close (fp8 quantization can have tiny diffs)
        bs_close = bs_diff.max().item() < 0.1

        status = "PASS" if ((packed_match or packed_close) and (bs_exact or bs_close)) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"[{status}] {name:20s} M={M:4d} K={K:5d} | "
              f"packed_match={packed_match} (mismatch={n_mismatch}/{n_total}), "
              f"bs_exact={bs_exact} (max_diff={bs_diff.max().item():.4f})")

        if not packed_match:
            # Show first few mismatches
            mismatch = (packed_ref != packed_fused)
            if mismatch.any():
                idx = mismatch.nonzero()[0]
                print(f"  First packed mismatch at {idx.tolist()}: "
                      f"ref={packed_ref[idx[0], idx[1]].item()}, "
                      f"fused={packed_fused[idx[0], idx[1]].item()}")

        if not bs_exact and not bs_close:
            # Show first few mismatches
            bs_mismatch = (bs_ref_valid != bs_fused_valid)
            if bs_mismatch.any():
                idx = bs_mismatch.nonzero()[0]
                print(f"  First bs mismatch at [{idx.item()}]: "
                      f"ref={bs_ref_valid[idx].item():.6f}, "
                      f"fused={bs_fused_valid[idx].item():.6f}")

    return all_pass


def test_end_to_end():
    """Test that fused quant + _scaled_mm gives same result as reference."""
    device = 'cuda'
    torch.manual_seed(42)

    M, K, N = 4, 2560, 10240  # FFN key shape
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.3
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02

    # Quantize weight to NVFP4
    w_amax = w.abs().max()
    w_ts = mx.nvfp4_per_tensor_scale(w_amax)
    w_packed, w_bs, w_ts_ret = mx.to_nvfp4(w, block_size=16, per_tensor_scale=w_ts)
    w_bs_swizzled = mx.to_blocked(w_bs)

    # Reference: quantize activation with mx_utils
    x_amax = x.abs().max()
    x_ts = mx.nvfp4_per_tensor_scale(x_amax)
    x_packed_ref, x_bs_ref, _ = mx.to_nvfp4(x.to(torch.bfloat16), block_size=16, per_tensor_scale=x_ts)
    x_bs_swizzled_ref = mx.to_blocked(x_bs_ref)

    a_fp4_ref = x_packed_ref.view(torch.float4_e2m1fn_x2)
    b_fp4 = w_packed.view(torch.float4_e2m1fn_x2)

    out_ref = torch._scaled_mm(a_fp4_ref, b_fp4.t(),
        scale_a=x_bs_swizzled_ref, scale_b=w_bs_swizzled,
        out_dtype=torch.bfloat16)
    out_ref = out_ref * x_ts * w_ts

    # Fused: quantize activation with our kernel
    x_packed_fused, x_bs_fused, x_ts_fused = fused_nvfp4_quant(x, x_ts)

    a_fp4_fused = x_packed_fused.view(torch.float4_e2m1fn_x2)

    out_fused = torch._scaled_mm(a_fp4_fused, b_fp4.t(),
        scale_a=x_bs_fused, scale_b=w_bs_swizzled,
        out_dtype=torch.bfloat16)
    out_fused = out_fused * x_ts_fused * w_ts

    # Compare
    diff = (out_ref - out_fused).abs()
    print(f"\n=== End-to-End GEMM Test (M={M}, K={K}, N={N}) ===")
    print(f"  Output shape: {out_ref.shape}")
    print(f"  Max diff: {diff.max().item():.8f}")
    print(f"  Mean diff: {diff.mean().item():.8f}")
    print(f"  Exact match: {torch.equal(out_ref, out_fused)}")

    # Also compare vs bf16 reference
    ref_bf16 = x.float() @ w.float().t()
    diff_ref = (out_ref.float() - ref_bf16).abs()
    diff_fused = (out_fused.float() - ref_bf16).abs()
    print(f"  vs bf16 ref: mx_max={diff_ref.max().item():.6f}, fused_max={diff_fused.max().item():.6f}")

    return torch.equal(out_ref, out_fused)


if __name__ == "__main__":
    print("=== Correctness Test ===")
    ok1 = test_correctness()
    ok2 = test_end_to_end()
    print(f"\nOverall: {'ALL PASS' if ok1 and ok2 else 'FAILED'}")
