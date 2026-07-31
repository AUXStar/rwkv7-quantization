#!/usr/bin/env python3
"""Hybrid v9b: GPTQ with configurable group_size + AWQ Hessian transformation.

Key insight from debugging:
- group_size=16 (160 iterations): too much error compounding → Top-1 68% at damp=10
- group_size=128 (20 iterations): less compounding, but 128x128 Hessian less conditioned
- AWQ Hessian transform H' = diag(1/s) @ H @ diag(1/s) is CRITICAL

This version supports both group_size and correct AWQ Hessian.
Also adds: GPTQ update magnitude monitoring + optional skip for ill-conditioned blocks.
"""
import torch
import os
import time
import argparse
import sys

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.015625
FP4_E2M1_MAX = 6.0
NVFP4_TS_DIVISOR = 448.0 * 6.0

_FP4_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def _round_to_fp4(x_scaled):
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
    w_dev = w.to(device=device).float()
    act_dev = act_stats.to(device=device).float()
    w_mean = w_dev.abs().mean(dim=0)
    s = (act_dev.clamp(min=1e-8) ** alpha) / (w_mean.clamp(min=1e-8) ** (1 - alpha))
    s = s / s.mean()
    return s.cpu()


def quantize_group_nvfp4_clip(w_group, ts, block_size=16, clip_ratios=None, device='cuda'):
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

    w_quant = fp4_table[best_fp4_idx.long()] * (ts * best_bs_fp8.to(torch.float32)).unsqueeze(-1)
    w_quant = w_quant.view(N, group_K)

    return w_quant, best_fp4_idx.view(N, group_K), best_bs_fp8


def gptq_quantize_nvfp4(w, H, awq_scale, block_size=16, group_size=128,
                        clip_ratios=None, damping_ratio=5.0, device='cuda'):
    """GPTQ with configurable group_size + AWQ Hessian transformation.

    Args:
        w: [N, K] bfloat16 weight
        H: [K, K] float32 Hessian (X^T @ X, from original input)
        awq_scale: [K] float32 AWQ scaling factor
        block_size: NVFP4 block size (16)
        group_size: GPTQ group size (columns per iteration)
        clip_ratios: list of clip ratios to search
        damping_ratio: Hessian damping (relative to normalized mean diagonal)
    """
    N, K = w.shape
    n_blocks = K // block_size
    n_groups = (K + group_size - 1) // group_size

    # Apply AWQ scaling
    w_orig = w.to(device=device).float()
    s = awq_scale.to(device=device).float()
    W = w_orig * s.unsqueeze(0)

    # Per-tensor scale
    ts = W.abs().max() / NVFP4_TS_DIVISOR
    if ts.item() == 0:
        ts = torch.tensor(1.0, dtype=torch.float32, device=device)

    # CRITICAL: Transform Hessian for AWQ
    # H was collected from original input x, but GPTQ operates on W' = W*s
    # Effective input is x' = x/s, so H' = diag(1/s) @ H @ diag(1/s)
    H = H.to(device=device).float()
    inv_s = (1.0 / s)
    H = H * (inv_s.unsqueeze(0) * inv_s.unsqueeze(1))

    # Normalize + damp
    H_scale = H.diag().mean()
    H = H / H_scale
    H.diagonal().add_(damping_ratio)

    # Storage
    all_fp4_idx = torch.zeros(N, K, dtype=torch.uint8, device=device)
    all_bs = torch.zeros(N, n_blocks, dtype=torch.float8_e4m3fn, device=device)

    w_orig_max = W.abs().max().item()
    total_update_norm = 0.0

    for g_idx in range(n_groups):
        g_start = g_idx * group_size
        g_end = min(g_start + group_size, K)

        # Quantize this group
        w_group = W[:, g_start:g_end].contiguous()
        w_quant, fp4_idx, bs_fp8 = quantize_group_nvfp4_clip(
            w_group, ts, block_size, clip_ratios, device)

        # Store
        all_fp4_idx[:, g_start:g_end] = fp4_idx
        block_start = g_start // block_size
        block_end = g_end // block_size
        all_bs[:, block_start:block_end] = bs_fp8.squeeze(1) if bs_fp8.dim() > 1 else bs_fp8

        # Error
        err = W[:, g_start:g_end] - w_quant

        # GPTQ update
        if g_end < K:
            H_block = H[g_start:g_end, g_start:g_end]
            H_cross = H[g_start:g_end, g_end:]

            try:
                update = torch.linalg.solve(H_block, H_cross)
            except Exception:
                # Fallback: add extra damping
                update = torch.linalg.solve(
                    H_block + 0.1 * torch.eye(H_block.shape[0], device=device),
                    H_cross)

            if torch.isnan(update).any():
                continue

            delta = err @ update
            W[:, g_end:] -= delta
            W[:, g_end:] = W[:, g_end:].clamp(-w_orig_max * 2, w_orig_max * 2)
            total_update_norm += delta.norm().item()

        if g_idx == 0:
            print(f"    GPTQ group 0-{g_end}: err={err.norm():.4f}, "
                  f"W_max={W.abs().max().item():.6f}")

    print(f"    GPTQ total: {n_groups} groups, "
          f"total_update_norm={total_update_norm:.4f}, "
          f"W_max_final={W.abs().max().item():.6f}")

    # Pack
    fp4_flat = all_fp4_idx.view(N, K // 2, 2)
    lo = fp4_flat[:, :, 0]
    hi = fp4_flat[:, :, 1]
    packed = (hi * 16 + lo).to(torch.uint8)

    return packed.cpu(), all_bs.cpu(), ts.cpu(), awq_scale


def quantize_to_fp8(w):
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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9b.pth")
    parser.add_argument("--act-stats", default="/home/njzy/test/eval_tmp/awq_act_stats.pt")
    parser.add_argument("--hessians", default="/home/njzy/test/eval_tmp/gptq_hessians.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--damping-ratio", type=float, default=5.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading original model from {args.input} ...")
    z = torch.load(args.input, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}, group_size={args.group_size}, damp={args.damping_ratio}")

    act_stats = torch.load(args.act_stats, map_location="cpu")
    hessians = torch.load(args.hessians, map_location="cpu")
    print(f"  AWQ stats: {len(act_stats)} layers, Hessians: {len(hessians)} layers")

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

        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16
            N, K = w.shape
            layer_idx = int(key.split(".")[1])

            act = act_stats[layer_idx]
            awq_s = compute_awq_scale(w, act, alpha=args.alpha)
            H = hessians[layer_idx]

            w_packed, w_bs, w_ts, _ = gptq_quantize_nvfp4(
                w, H, awq_s,
                block_size=16, group_size=args.group_size,
                damping_ratio=args.damping_ratio)

            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            z[key + ".awq_scale"] = awq_s.contiguous()

            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes + awq_s.nbytes

        elif ".ffn.value.weight" in key:
            quant_bytes += w.nbytes
            skip_count += 1

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

    print(f"\n  Summary: {nvfp4_count} NVFP4, {att_fp8_count} FP8, {skip_count} skip")
    print(f"  Size: {orig_bytes/2**30:.3f} → {quant_bytes/2**30:.3f} GiB "
          f"({(1-quant_bytes/orig_bytes)*100:.1f}% savings)")

    meta = {
        "v": "9b",
        "r": [[0, L-1, 4, 2], [0, L-1, 1, 1], [0, L-1, 2, 1], [0, L-1, 3, 1], [0, L-1, 6, 1]],
        "s": {"blk": 16, "awq": True, "clip": True, "gptq": True,
              "alpha": args.alpha, "group_size": args.group_size, "damping": args.damping_ratio},
        "n": ["emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
              "x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "w1", "w2",
              "a0", "a1", "a2", "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
              "ffn.value.weight"],
        "desc": f"v9b: GPTQ(gs={args.group_size},damp={args.damping_ratio})+AWQ+clip NVFP4",
    }
    z["meta"] = meta

    t1 = time.perf_counter()
    print(f"Saving to {args.output} ...")
    torch.save(z, args.output)
    print(f"  Saved in {time.perf_counter()-t1:.1f}s, "
          f"size: {os.path.getsize(args.output)/(1024**3):.2f} GB")
    print(f"Total: {time.perf_counter()-t0:.1f}s\nDone!")


if __name__ == "__main__":
    main()
