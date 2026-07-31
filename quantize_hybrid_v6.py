#!/usr/bin/env python3
"""Hybrid v6 quantization: NVFP4 with per-block optimal clip ratio.

Iteration 6 — minimize per-block quantization error by searching for optimal clip ratio:
- FFN key.weight:   NVFP4 with per-block optimal clip ratio (block_size=16)
- FFN value.weight: BF16 (unchanged)
- Attention r/k/v/output.weight: FP8 E4M3 (W8A16)

Problem: Default NVFP4 uses block_amax/6.0 as block scale. When a block has outliers,
this inflates the scale and reduces precision for smaller values. By searching over
clip ratios [0.6-1.0], we find the optimal scale that minimizes per-block MSE,
allowing controlled clipping of outliers to preserve majority precision.

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v6.pth
"""
import torch
import os
import time
import argparse
import sys

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.015625
FP4_E2M1_MAX = 6.0
NVFP4_TS_DIVISOR = 448.0 * 6.0  # 2688.0

# FP4 E2M1 values for lookup: index -> float value
_FP4_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def _round_to_fp4(x_scaled):
    """Round scaled values to FP4 E2M1 indices (round-to-nearest-even).

    Returns uint8 indices [0, 15] where bit 3 = sign, bits 0-2 = magnitude code.
    """
    sign = torch.where(x_scaled < 0, 1, 0).to(torch.uint8)
    a = x_scaled.abs()

    code = torch.where(a <= 0.25, 0,
           torch.where(a < 0.75, 1,
           torch.where(a <= 1.25, 2,
           torch.where(a < 1.75, 3,
           torch.where(a <= 2.5, 4,
           torch.where(a < 3.5, 5,
           torch.where(a <= 5.0, 6, 7))))))).to(torch.uint8)

    return sign * 8 + code


def quantize_nvfp4_optimal_clip(w, block_size=16, clip_ratios=None, device='cuda'):
    """NVFP4 quantization with per-block optimal clip ratio.

    For each block, tries multiple clip ratios and picks the one with lowest MSE.
    A clip ratio of 1.0 means no clipping (scale = amax/6.0).
    A clip ratio of 0.8 means scale = 0.8 * amax/6.0 (some values clipped to +/-6.0).

    Args:
        w: [N, K] bfloat16 weight
        block_size: number of elements per block (16)
        clip_ratios: list of clip ratios to search
        device: device for computation ('cuda' for speed)

    Returns:
        packed: [N, K//2] uint8 (packed FP4 pairs)
        block_scale: [N, K//block_size] float8_e4m3fn
        tensor_scale: scalar float32
    """
    if clip_ratios is None:
        clip_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    w_orig = w.to(device=device)
    N, K = w.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    assert K % 2 == 0, f"K={K} not even (needed for FP4 packing)"
    w_f32 = w_orig.float()
    n_blocks = K // block_size

    # Per-tensor scale
    amax = w_f32.abs().max()
    ts = amax / NVFP4_TS_DIVISOR
    if ts.item() == 0:
        ts = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Reshape to blocks: [N, n_blocks, block_size]
    w_blocks = w_f32.view(N, n_blocks, block_size)

    # Block amax: [N, n_blocks]
    block_amax = w_blocks.abs().amax(dim=2)

    # Try each clip ratio and find the best per block
    best_mse = torch.full((N, n_blocks), float('inf'), device=device)
    best_fp4_idx = None   # [N, n_blocks, block_size] uint8
    best_bs_fp8 = None    # [N, n_blocks] float8_e4m3fn

    fp4_table = _FP4_VALUES.to(device)

    for ratio in clip_ratios:
        # Compute block scale with this clip ratio
        bs_scaled = block_amax * ratio / FP4_E2M1_MAX / ts  # [N, n_blocks]
        bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
        bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

        # Effective scale: ts * bs_fp8
        bs_f32 = bs_fp8.to(torch.float32)  # [N, n_blocks]
        eff_scale = ts * bs_f32  # [N, n_blocks]

        # Scale and quantize
        w_scaled = w_blocks / eff_scale.unsqueeze(-1)  # [N, n_blocks, block_size]
        w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

        fp4_idx = _round_to_fp4(w_scaled)  # [N, n_blocks, block_size] uint8

        # Dequantize to compute MSE
        fp4_val = fp4_table[fp4_idx.long()]  # [N, n_blocks, block_size]
        w_deq = fp4_val * eff_scale.unsqueeze(-1)  # [N, n_blocks, block_size]

        # Per-block MSE: [N, n_blocks]
        block_mse = ((w_blocks - w_deq) ** 2).mean(dim=2)

        # Update best where this ratio is better
        improved = block_mse < best_mse
        best_mse = torch.where(improved, block_mse, best_mse)

        if best_fp4_idx is None:
            best_fp4_idx = fp4_idx.clone()
            best_bs_fp8 = bs_fp8.clone()
        else:
            mask = improved.unsqueeze(-1)  # [N, n_blocks, 1]
            best_fp4_idx = torch.where(mask, fp4_idx, best_fp4_idx)
            best_bs_fp8 = torch.where(improved, bs_fp8, best_bs_fp8)

    # Pack FP4 pairs into uint8: [N, K//2, 2] -> hi*16 + lo
    fp4_flat = best_fp4_idx.view(N, K // 2, 2)
    lo = fp4_flat[:, :, 0]
    hi = fp4_flat[:, :, 1]
    packed = (hi * 16 + lo).to(torch.uint8)

    # Move back to CPU
    packed = packed.cpu()
    best_bs_fp8 = best_bs_fp8.cpu()
    ts_cpu = ts.cpu()

    return packed, best_bs_fp8, ts_cpu


def quantize_nvfp4_default(w, block_size=16, device='cuda'):
    """Default NVFP4 quantization (clip ratio = 1.0, no clipping). For comparison."""
    w_orig = w.to(device=device)
    N, K = w.shape
    w_f32 = w_orig.float()
    n_blocks = K // block_size

    amax = w_f32.abs().max()
    ts = amax / NVFP4_TS_DIVISOR
    if ts.item() == 0:
        ts = torch.tensor(1.0, dtype=torch.float32, device=device)

    w_blocks = w_f32.view(N, n_blocks, block_size)
    block_amax = w_blocks.abs().amax(dim=2)

    bs_scaled = block_amax / FP4_E2M1_MAX / ts
    bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
    bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

    bs_f32 = bs_fp8.to(torch.float32)
    eff_scale = ts * bs_f32
    w_scaled = w_blocks / eff_scale.unsqueeze(-1)
    w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    fp4_idx = _round_to_fp4(w_scaled)

    fp4_flat = fp4_idx.view(N, K // 2, 2)
    lo = fp4_flat[:, :, 0]
    hi = fp4_flat[:, :, 1]
    packed = (hi * 16 + lo).to(torch.uint8)

    return packed.cpu(), bs_fp8.cpu(), ts.cpu()


def quantize_to_fp8(w):
    """Quantize bf16/fp16 weight to FP8 E4M3 with per-tensor scale."""
    amax = w.abs().max()
    if amax > 0:
        scale = (amax / FP8_E4M3_MAX).float()
    else:
        scale = torch.tensor(1.0, dtype=torch.float32)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def verify_quantization(w_orig, packed, bs, ts):
    """Verify NVFP4 quantization by computing reconstruction error."""
    sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
    from nvfp4_ops import dequantize_nvfp4

    w_deq = dequantize_nvfp4(packed, bs, ts)
    w_orig_f16 = w_orig.to(torch.float16)
    w_deq_f16 = w_deq.to(torch.float16)

    mse = ((w_orig_f16 - w_deq_f16) ** 2).mean().item()
    max_err = (w_orig_f16 - w_deq_f16).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        w_orig_f16.flatten().unsqueeze(0),
        w_deq_f16.flatten().unsqueeze(0)
    ).item()
    return mse, max_err, cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth")
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v6.pth")
    parser.add_argument("--block-size", type=int, default=16, choices=[16])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading original model from {args.input} ...")
    z = torch.load(args.input, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")
    print(f"  Block size: {args.block_size}")
    print(f"  Clip ratios: [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]")

    nvfp4_count = 0
    att_fp8_count = 0
    skip_count = 0
    orig_bytes = 0
    quant_bytes = 0

    ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")
    first_key_orig = None

    for key in list(z.keys()):
        if not torch.is_tensor(z[key]):
            skip_count += 1
            continue

        w = z[key]
        orig_bytes += w.nbytes

        # FFN key -> NVFP4 with optimal clip ratio
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape

            if nvfp4_count == 0:
                first_key_orig = w.clone()

            w_packed, w_bs, w_ts = quantize_nvfp4_optimal_clip(w, block_size=args.block_size)
            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes
            if nvfp4_count <= 2:
                print(f"  NVFP4 clip-opt {key}: [{N},{K}] -> packed[{N},{K//2}] bs[{N},{K//16}]")

        # FFN value -> SKIP (keep BF16)
        elif ".ffn.value.weight" in key:
            quant_bytes += w.nbytes
            skip_count += 1

        # Attention r/k/v/output -> FP8 (W8A16)
        elif (key.startswith("blocks.") and ".att." in key and
              any(key.endswith(s) for s in ATT_SUFFIXES)):
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            w_fp8, scale = quantize_to_fp8(w)
            z[key] = w_fp8.contiguous()
            z[key + ".fp8_scale"] = scale.contiguous()
            att_fp8_count += 1
            quant_bytes += w_fp8.nbytes + scale.nbytes

        else:
            skip_count += 1
            quant_bytes += w.nbytes

    print(f"\n  Quantization summary:")
    print(f"    NVFP4 clip-opt (FFN key): {nvfp4_count} tensors")
    print(f"    FP8     (Attention):  {att_fp8_count} tensors")
    print(f"    BF16    (FFN value):  preserved")
    print(f"    Skipped (unchanged):  {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Verify reconstruction
    if args.verify and first_key_orig is not None:
        print(f"\n=== Verification: NVFP4 optimal clip ratio reconstruction ===")
        packed = z["blocks.0.ffn.key.weight"]
        bs = z["blocks.0.ffn.key.weight.nf4_b_scale"]
        ts = z["blocks.0.ffn.key.weight.nvfp4_t_scale"]

        # Optimal clip ratio
        mse_opt, max_err_opt, cos_sim_opt = verify_quantization(first_key_orig, packed, bs, ts)
        print(f"  clip-opt:  MSE={mse_opt:.10f}, MaxErr={max_err_opt:.6f}, CosSim={cos_sim_opt:.10f}")

        # Default (clip ratio = 1.0)
        packed_def, bs_def, ts_def = quantize_nvfp4_default(first_key_orig, block_size=16)
        mse_def, max_err_def, cos_sim_def = verify_quantization(first_key_orig, packed_def, bs_def, ts_def)
        print(f"  default:   MSE={mse_def:.10f}, MaxErr={max_err_def:.6f}, CosSim={cos_sim_def:.10f}")
        print(f"  Improvement: MSE {(1 - mse_opt/mse_def)*100:.1f}% lower, MaxErr {(1 - max_err_opt/max_err_def)*100:.1f}% lower, CosSim +{cos_sim_opt - cos_sim_def:.8f}")

    # Metadata
    meta = {
        "v": 6,
        "r": [
            [0, L-1, 4, 2],  # ffn_key: nvfp4
            [0, L-1, 1, 1],  # att_receptance: fp8
            [0, L-1, 2, 1],  # att_key: fp8
            [0, L-1, 3, 1],  # att_value: fp8
            [0, L-1, 6, 1],  # att_output: fp8
        ],
        "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "ffn.value.weight",
        ],
        "desc": "Hybrid v6: FFN key NVFP4 optimal-clip + FFN value BF16 + Attention FP8 W8A16",
    }
    z["meta"] = meta

    t1 = time.perf_counter()
    print(f"\nSaving to {args.output} ...")
    torch.save(z, args.output)
    file_size = os.path.getsize(args.output) / (1024**3)
    print(f"  Saved in {time.perf_counter()-t1:.1f}s, file size: {file_size:.2f} GB")
    print(f"Total time: {time.perf_counter()-t0:.1f}s")

    # Verify
    print("\n=== Verification ===")
    z2 = torch.load(args.output, map_location="cpu", mmap=True)
    nvfp4_keys = sum(1 for k in z2 if k.endswith(".nf4_b_scale"))
    fp8_keys = sum(1 for k in z2 if k.endswith(".fp8_scale"))
    print(f"  NVFP4 block scale keys: {nvfp4_keys}")
    print(f"  FP8 scale keys: {fp8_keys}")
    print(f"  Total tensors: {len(z2)}")

    for prefix in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
                   "blocks.0.att.key.weight"]:
        w = z2[prefix]
        print(f"  {prefix}: {list(w.shape)} {w.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
