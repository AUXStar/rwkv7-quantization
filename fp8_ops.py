#!/usr/bin/env python3
"""FP8 quantized GEMM operations for RWKV-7 v3a inference.

Provides:
- is_fp8_weight: detect FP8 quantized weights (has .fp8_scale sibling)
- load_fp8_weight: load FP8 weight + per-tensor scale
- linear_fp8: FP8 GEMM (FP8×FP8→BF16) with online activation quantization
- (已移除) 反量化路径已删除：FP8 权重永远保持 FP8，只走张量核
- linear_quantized: dispatcher that picks the right GEMM based on weight_info
"""
import torch

# ============================================================================
# Detection
# ============================================================================

def is_fp8_weight(z, key):
    """Check if a weight key has FP8 quantization (has .fp8_scale sibling)."""
    return (key + ".fp8_scale") in z


# ============================================================================
# Loading
# ============================================================================

def load_fp8_weight(z, key, dev):
    """Load quantized weight: FP8 or INT8 + per-tensor scale.

    FP8 权重保持 FP8 域，由 FP8 张量核（_scaled_mm / tl.dot(fp8,fp8)）计算，
    绝不反量化回 fp16/bf16 再 GEMM。

    INT8 权重无硬件张量核加速，加载时一次性反量化缓存为 fp16，
    后续 forward 直接用缓存做普通 GEMM，避免每次 forward 重复反量化。

    Removes the .fp8_scale key from z.
    Returns a dict with weight, tensor_scale (scalar), qtype.
    """
    w = z[key].to(device=dev).contiguous()
    scale = z[key + ".fp8_scale"].to(device=dev)      # scalar float32
    del z[key + ".fp8_scale"]
    qtype = "fp8" if w.dtype == torch.float8_e4m3fn else "int8"
    info = {
        "weight": w,
        "tensor_scale": scale,
        "qtype": qtype,
    }
    # INT8: 一次性反量化缓存，避免每次 forward 重复反量化（从 18.9→~44 tok/s）
    # 反量化后释放原始 INT8 权重，节省 VRAM（否则同时存 int8+fp16 两份）
    # 转置为 [K, N] 布局，与非量化权重加载时的 .t() 一致，
    # 使 linear_f16_m1_splitk / linear_f16 CUDA 内核可直接使用
    if qtype == "int8":
        w_dequant = (w.to(torch.float32) * scale).to(torch.float16)
        info["dequantized_weight"] = w_dequant.t().contiguous()  # [N,K] → [K,N]
        info["weight"] = None  # 释放原始 INT8 权重
    return info


# ============================================================================
# GEMM
# ============================================================================

FP8_E4M3_MAX = 448.0

def linear_fp8(x, weight_info, out_dtype=torch.float16):
    """FP8 GEMM: quantize input on-the-fly to FP8, use torch._scaled_mm.

    全程 GPU 计算，无 D2H 同步。激活量化用 prep_x_fp8（2 个 Triton kernel），
    然后 _scaled_mm（cuBLAS FP8 张量核）。
    """
    from fused_fp8_gemm import prep_x_fp8

    w = weight_info["weight"]              # [N, K] float8_e4m3fn
    w_scale = weight_info["tensor_scale"]  # scalar float32

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype == torch.float16:
        x_2d = x_2d.to(torch.bfloat16)

    # GPU-only activation quantization (no D2H sync)
    x_fp8, x_scale = prep_x_fp8(x_2d)

    out = torch._scaled_mm(x_fp8, w.t(),
        scale_a=x_scale.reshape(1),
        scale_b=w_scale.reshape(1),
        out_dtype=torch.bfloat16)

    N = w.size(0)
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out



def linear_int8(x, weight_info, out_dtype=torch.float16):
    """INT8 GEMM: use cached dequantized weight [K,N] with CUDA kernel.

    INT8 无硬件张量核加速，反量化是预期行为。
    权重在 load_fp8_weight 时已反量化并转置为 [K,N] 布局缓存。
    M=1 时使用 split-K CUDA 内核（与 baseline 相同路径），大幅加速 decode。
    """
    w_fp16 = weight_info.get("dequantized_weight")
    if w_fp16 is None:  # fallback
        w = weight_info["weight"]
        w_scale = weight_info["tensor_scale"]
        w_fp16 = (w.to(torch.float32) * w_scale).to(out_dtype).t().contiguous()
    if w_fp16.dtype != out_dtype:
        w_fp16 = w_fp16.to(out_dtype)

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if x_2d.dtype != out_dtype:
        x_2d = x_2d.to(out_dtype)

    M = x_2d.size(0)
    N = w_fp16.size(1)  # [K, N] layout

    # M=1 decode: 使用 split-K CUDA 内核（与 baseline 相同路径）
    if M == 1 and N % 64 == 0:
        try:
            out = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x_2d, w_fp16)
            out = out.reshape(*orig_shape[:-1], N)
            if out_dtype != out.dtype:
                out = out.to(out_dtype)
            return out
        except (AttributeError, RuntimeError):
            pass

    # M>1: 使用 linear_f16 CUDA 内核（如有），否则 PyTorch matmul
    try:
        out = torch.ops.rwkv7_v3a_ops.linear_f16(x_2d, w_fp16)
        out = out.reshape(*orig_shape[:-1], N)
        if out_dtype != out.dtype:
            out = out.to(out_dtype)
        return out
    except (AttributeError, RuntimeError):
        pass

    # Fallback: PyTorch matmul (w is [K,N], x@w = [M,K]@[K,N] = [M,N])
    out = x_2d @ w_fp16
    out = out.reshape(*orig_shape[:-1], N)
    if out_dtype != out.dtype:
        out = out.to(out_dtype)
    return out


def linear_quantized(x, weight_info, out_dtype=torch.float16):
    """Dispatcher: pick the right GEMM based on qtype in weight_info."""
    qtype = weight_info.get("qtype", "fp8")
    if qtype == "int8":
        return linear_int8(x, weight_info, out_dtype)
    # FP8 权重永远走 FP8 张量核（_scaled_mm），保持 FP8 域，禁止反量化
    return linear_fp8(x, weight_info, out_dtype)


def linear_quantized_fused(x, weight_info, out_dtype=torch.float16):
    """Hybrid dispatcher for quantized GEMMs.

    Dispatches to fused single-kernel GEMM in fused_fp8_gemm module.
    """
    from fused_fp8_gemm import linear_quantized_fused as _dispatch
    return _dispatch(x, weight_info, out_dtype)


FUSED_M_MAX = 512  # decode+prefill: Triton fused (2 launches) beats _scaled_mm (3 launches)
