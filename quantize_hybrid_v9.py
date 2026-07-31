#!/usr/bin/env python3
"""Hybrid v9 quantization: GPTQ + AWQ + per-block clip ratio.

Iteration 9 — add GPTQ error compensation on top of v8:
- FFN key.weight:   NVFP4 W4A16 with GPTQ + AWQ scaling + per-block clip ratio
- FFN value.weight: BF16 (unchanged)
- Attention r/k/v/output.weight: FP8 E4M3 (W8A16)

GPTQ (Generalized Post-Training Quantization) insight:
  v8 minimizes per-block weight reconstruction error, but Top-1 measures
  output (logit) agreement. GPTQ directly minimizes output error by:
  1. Processing weight columns in groups
  2. After quantizing each group, computing the error
  3. Propagating the error to remaining unquantized columns using the Hessian

  This compensates for quantization errors in earlier columns by adjusting
  later columns, preserving the overall output as much as possible.

Fixed v9 key changes (vs broken initial GPTQ):
  1. group_size = block_size (16), not 128 — 16x16 Hessian is far better conditioned
  2. Hessian normalized by mean diagonal before damping
  3. damping_ratio=0.1 (relative to normalized diagonal)
  4. torch.linalg.solve instead of Cholesky for robustness
  5. Weight clamping after updates to prevent runaway values
  6. NaN checks to skip ill-conditioned blocks

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9.pth
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

_FP4_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def _round_to_fp4(x_scaled):
    """Round scaled values to FP4 E2M1 indices (round-to-nearest-even)."""
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


def compute_awq_scale(w, act_stats, alpha=0.5, device='cuda'):
    """Compute AWQ per-channel scaling factor."""
    w_dev = w.to(device=device).float()
    act_dev = act_stats.to(device=device).float()
    w_mean = w_dev.abs().mean(dim=0)
    s = (act_dev.clamp(min=1e-8) ** alpha) / (w_mean.clamp(min=1e-8) ** (1 - alpha))
    s = s / s.mean()
    return s.cpu()


def quantize_group_nvfp4_clip(w_group, ts, block_size=16, clip_ratios=None, device='cuda'):
    """Quantize a group of columns using NVFP4 with per-block clip ratio search.

    Args:
        w_group: [N, group_K] float32 weight (already AWQ-scaled)
        ts: per-tensor scale (scalar)
        block_size: 16
        clip_ratios: list of ratios to search

    Returns:
        w_quant: [N, group_K] float32 dequantized weight
        fp4_idx: [N, group_K] uint8 FP4 indices
        bs_fp8: [N, group_K//block_size] float8_e4m3fn block scales
    """
    if clip_ratios is None:
        clip_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    N, group_K = w_group.shape
    n_blocks = group_K // block_size
    w_blocks = w_group.view(N, n_blocks, block_size)
    block_amax = w_blocks.abs().amax(dim=2)

    best_mse = torch.full((N, n_blocks), float('inf'), device=device)
    best_fp4_idx = None
    best_bs_fp8 = None
    fp4_table = _FP4_VALUES.to(device)

    for ratio in clip_ratios:
        bs_scaled = block_amax * ratio / FP4_E2M1_MAX / ts
        bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
        bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

        bs_f32 = bs_fp8.to(torch.float32)
        eff_scale = ts * bs_f32

        w_scaled = w_blocks / eff_scale.unsqueeze(-1)
        w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

        fp4_idx = _round_to_fp4(w_scaled)
        fp4_val = fp4_table[fp4_idx.long()]
        w_deq = fp4_val * eff_scale.unsqueeze(-1)

        block_mse = ((w_blocks - w_deq) ** 2).mean(dim=2)
        improved = block_mse < best_mse
        best_mse = torch.where(improved, block_mse, best_mse)

        if best_fp4_idx is None:
            best_fp4_idx = fp4_idx.clone()
            best_bs_fp8 = bs_fp8.clone()
        else:
            mask = improved.unsqueeze(-1)
            best_fp4_idx = torch.where(mask, fp4_idx, best_fp4_idx)
            best_bs_fp8 = torch.where(improved, bs_fp8, best_bs_fp8)

    # Reconstruct quantized weight
    w_quant = fp4_table[best_fp4_idx.long()] * (ts * best_bs_fp8.to(torch.float32)).unsqueeze(-1)
    w_quant = w_quant.view(N, group_K)

    return w_quant, best_fp4_idx.view(N, group_K), best_bs_fp8


def gptq_quantize_nvfp4(w, H, awq_scale, block_size=16,
                        clip_ratios=None, damping_ratio=0.1, device='cuda'):
    """GPTQ with NVFP4 + AWQ + clip ratio.

    Processes one NVFP4 block (16 columns) at a time for numerical stability.
    The 16x16 Hessian sub-block is far better conditioned than 128x128.

    Key fixes vs broken v9:
    1. group_size = block_size (16), not 128
    2. Hessian normalized by mean diagonal before damping
    3. damping_ratio=0.1 (relative to normalized diagonal)
    4. torch.linalg.solve instead of Cholesky
    5. Weight clamping after updates to prevent runaway

    Args:
        w: [N, K] bfloat16 weight
        H: [K, K] float32 Hessian (X^T @ X)
        awq_scale: [K] float32 AWQ scaling factor
        block_size: NVFP4 block size (16), also GPTQ group size
        clip_ratios: list of clip ratios to search
        damping_ratio: Hessian damping ratio (relative to mean diagonal)

    Returns:
        packed: [N, K//2] uint8 (packed FP4)
        block_scale: [N, K//block_size] float8_e4m3fn
        tensor_scale: scalar float32
        awq_scale: [K] float32
    """
    N, K = w.shape
    n_blocks = K // block_size

    # Apply AWQ scaling
    w_orig = w.to(device=device).float()
    s = awq_scale.to(device=device).float()
    W = w_orig * s.unsqueeze(0)  # [N, K]

    # Per-tensor scale (fixed for entire matrix)
    ts = W.abs().max() / NVFP4_TS_DIVISOR
    if ts.item() == 0:
        ts = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Prepare Hessian: transform for AWQ, normalize, then add damping
    # CRITICAL: Hessian was collected from original input x, but GPTQ operates
    # on AWQ-scaled weight W' = W * s. The effective input is x' = x / s,
    # so the correct Hessian is H' = diag(1/s) @ H @ diag(1/s).
    H = H.to(device=device).float()
    inv_s = (1.0 / s)  # [K]
    H = H * (inv_s.unsqueeze(0) * inv_s.unsqueeze(1))  # H' = diag(1/s) @ H @ diag(1/s)

    # Normalize by mean diagonal, then add damping
    H_scale = H.diag().mean()
    H = H / H_scale  # Normalize so mean(diag) = 1.0
    H.diagonal().add_(damping_ratio)  # Add damping to normalized Hessian

    # Storage
    all_fp4_idx = torch.zeros(N, K, dtype=torch.uint8, device=device)
    all_bs = torch.zeros(N, n_blocks, dtype=torch.float8_e4m3fn, device=device)

    w_orig_max = W.abs().max().item()

    # Process one block (16 columns) at a time
    for b in range(n_blocks):
        col_start = b * block_size
        col_end = col_start + block_size

        # Quantize this block
        w_block = W[:, col_start:col_end].contiguous()
        w_quant, fp4_idx, bs_fp8 = quantize_group_nvfp4_clip(
            w_block, ts, block_size, clip_ratios, device)

        # Store results
        all_fp4_idx[:, col_start:col_end] = fp4_idx
        all_bs[:, b] = bs_fp8.squeeze(1) if bs_fp8.dim() > 1 else bs_fp8

        # Compute error
        err = W[:, col_start:col_end] - w_quant  # [N, 16]

        # Propagate error to remaining columns
        if col_end < K:
            H_block = H[col_start:col_end, col_start:col_end]  # [16, 16]
            H_cross = H[col_start:col_end, col_end:]            # [16, K-col_end]

            # Solve H_block * x = H_cross using torch.linalg.solve
            update = torch.linalg.solve(H_block, H_cross)  # [16, K-col_end]

            # Check for NaN
            if torch.isnan(update).any():
                if b == 0:
                    print(f"    WARNING: NaN in GPTQ update at block {b}, skipping")
                continue

            W[:, col_end:] -= err @ update  # [N, K-col_end]

            # Clamp weights to prevent runaway values
            max_val = w_orig_max * 2.0
            W[:, col_end:] = W[:, col_end:].clamp(-max_val, max_val)

        if b == 0:
            print(f"    GPTQ block 0: err_norm={err.norm():.4f}, "
                  f"W_max={W.abs().max().item():.6f}")
        elif b == n_blocks - 1:
            print(f"    GPTQ block {b}: err_norm={err.norm():.4f}, "
                  f"W_max={W.abs().max().item():.6f}")

    # Pack FP4 pairs
    fp4_flat = all_fp4_idx.view(N, K // 2, 2)
    lo = fp4_flat[:, :, 0]
    hi = fp4_flat[:, :, 1]
    packed = (hi * 16 + lo).to(torch.uint8)

    return packed.cpu(), all_bs.cpu(), ts.cpu(), awq_scale


def quantize_to_fp8(w):
    """Quantize bf16/fp16 weight to FP8 E4M3 with per-tensor scale."""
    amax = w.abs().max()
    if amax > 0:
        scale = (amax / FP8_E4M3_MAX).float()
    else:
        scale = torch.tensor(1.0, dtype=torch.float32)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth")
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9.pth")
    parser.add_argument("--act-stats", default="/home/njzy/test/eval_tmp/awq_act_stats.pt")
    parser.add_argument("--hessians", default="/home/njzy/test/eval_tmp/gptq_hessians.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--damping-ratio", type=float, default=0.1)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading original model from {args.input} ...")
    z = torch.load(args.input, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")

    # Load activation statistics and Hessians
    print(f"Loading activation statistics from {args.act_stats} ...")
    act_stats = torch.load(args.act_stats, map_location="cpu")
    print(f"  {len(act_stats)} layers, alpha={args.alpha}")

    print(f"Loading Hessians from {args.hessians} ...")
    hessians = torch.load(args.hessians, map_location="cpu")
    print(f"  {len(hessians)} layers")

    nvfp4_count = 0
    att_fp8_count = 0
    skip_count = 0
    orig_bytes = 0
    quant_bytes = 0

    ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")

    for key in list(z.keys()):
        if not torch.is_tensor(z[key]):
            skip_count += 1
            continue

        w = z[key]
        orig_bytes += w.nbytes

        # FFN key -> GPTQ + AWQ + clip NVFP4
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            layer_idx = int(key.split(".")[1])

            # Compute AWQ scaling
            act = act_stats[layer_idx]
            awq_s = compute_awq_scale(w, act, alpha=args.alpha)

            # Get Hessian for this layer
            H = hessians[layer_idx]

            # GPTQ quantization (block_size=16 as group, damping_ratio from args)
            w_packed, w_bs, w_ts, _ = gptq_quantize_nvfp4(
                w, H, awq_s,
                block_size=16,
                damping_ratio=args.damping_ratio
            )

            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            z[key + ".awq_scale"] = awq_s.contiguous()

            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes + awq_s.nbytes
            if nvfp4_count <= 2:
                print(f"  GPTQ+AWQ+clip {key}: [{N},{K}] "
                      f"s_min={awq_s.min():.4f} s_max={awq_s.max():.4f}")

        # FFN value -> SKIP
        elif ".ffn.value.weight" in key:
            quant_bytes += w.nbytes
            skip_count += 1

        # Attention -> FP8
        elif (key.startswith("blocks.") and ".att." in key and
              any(key.endswith(s) for s in ATT_SUFFIXES)):
            w_fp8, scale = quantize_to_fp8(w)
            z[key] = w_fp8.contiguous()
            z[key + ".fp8_scale"] = scale.contiguous()
            att_fp8_count += 1
            quant_bytes += w_fp8.nbytes + scale.nbytes

        else:
            skip_count += 1
            quant_bytes += w.nbytes

    print(f"\n  Quantization summary:")
    print(f"    GPTQ+AWQ+clip+NVFP4 (FFN key):  {nvfp4_count} tensors")
    print(f"    FP8      (Attention):           {att_fp8_count} tensors")
    print(f"    BF16     (FFN value):           preserved")
    print(f"    Skipped  (unchanged):           {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Metadata
    meta = {
        "v": 9,
        "r": [
            [0, L-1, 4, 2],
            [0, L-1, 1, 1],
            [0, L-1, 2, 1],
            [0, L-1, 3, 1],
            [0, L-1, 6, 1],
        ],
        "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32", "awq": True,
              "clip": True, "gptq": True, "alpha": args.alpha,
              "damping_ratio": args.damping_ratio},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "ffn.value.weight",
        ],
        "desc": f"Hybrid v9: GPTQ(bs=16)+AWQ+clip NVFP4 (alpha={args.alpha}, damp={args.damping_ratio}) + BF16 val + FP8 att",
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
    awq_keys = sum(1 for k in z2 if k.endswith(".awq_scale"))
    print(f"  NVFP4 block scale keys: {nvfp4_keys}")
    print(f"  FP8 scale keys: {fp8_keys}")
    print(f"  AWQ scale keys: {awq_keys}")
    print(f"  Total tensors: {len(z2)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
