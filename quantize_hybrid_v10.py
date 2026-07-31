#!/usr/bin/env python3
"""Hybrid v10 quantization: Hessian-weighted AWQ + clip ratio (no GPTQ).

If GPTQ error propagation proves too unstable for NVFP4, this approach
directly minimizes output error by weighting the clip ratio search with
the Hessian diagonal:

  v8 minimizes:  mean(err^2)           (weight MSE)
  v10 minimizes: mean(err^2 * h_weight) (output-weighted MSE)

where h_weight = H_diag[block] / mean(H_diag[block])

This gives more precision to channels with high activation magnitude
(high output contribution), without the risky error propagation of GPTQ.

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v10.pth
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


def quantize_hessian_weighted_clip(w_group, ts, h_weight, block_size=16,
                                    clip_ratios=None, device='cuda'):
    """Quantize using NVFP4 with Hessian-weighted per-block clip ratio search.

    Instead of minimizing weight MSE, minimizes output-weighted MSE:
      loss = mean(err^2 * h_weight)
    where h_weight captures the relative importance of each channel.

    Args:
        w_group: [N, group_K] float32 weight (already AWQ-scaled)
        ts: per-tensor scale (scalar)
        h_weight: [group_K] float32, Hessian diagonal normalized to mean=1
        block_size: 16
        clip_ratios: list of ratios to search

    Returns:
        packed: [N, group_K//2] uint8 (packed FP4)
        block_scale: [N, group_K//block_size] float8_e4m3fn
    """
    if clip_ratios is None:
        clip_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    N, group_K = w_group.shape
    n_blocks = group_K // block_size
    w_blocks = w_group.view(N, n_blocks, block_size)
    block_amax = w_blocks.abs().amax(dim=2)

    # Reshape h_weight to [1, n_blocks, block_size] for broadcasting
    hw_blocks = h_weight.view(1, n_blocks, block_size)

    best_loss = torch.full((N, n_blocks), float('inf'), device=device)
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

        # Hessian-weighted MSE: mean(err^2 * h_weight)
        err_sq = (w_blocks - w_deq) ** 2  # [N, n_blocks, block_size]
        block_loss = (err_sq * hw_blocks).mean(dim=2)  # [N, n_blocks]

        improved = block_loss < best_loss
        best_loss = torch.where(improved, block_loss, best_loss)

        if best_fp4_idx is None:
            best_fp4_idx = fp4_idx.clone()
            best_bs_fp8 = bs_fp8.clone()
        else:
            mask = improved.unsqueeze(-1)
            best_fp4_idx = torch.where(mask, fp4_idx, best_fp4_idx)
            best_bs_fp8 = torch.where(improved, bs_fp8, best_bs_fp8)

    # Pack FP4 pairs
    fp4_flat = best_fp4_idx.view(N, group_K // 2, 2)
    lo = fp4_flat[:, :, 0]
    hi = fp4_flat[:, :, 1]
    packed = (hi * 16 + lo).to(torch.uint8)

    return packed, best_bs_fp8


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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v10.pth")
    parser.add_argument("--act-stats", default="/home/njzy/test/eval_tmp/awq_act_stats.pt")
    parser.add_argument("--hessians", default="/home/njzy/test/eval_tmp/gptq_hessians.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
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

        # FFN key -> Hessian-weighted AWQ + clip NVFP4
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            layer_idx = int(key.split(".")[1])

            # Compute AWQ scaling
            act = act_stats[layer_idx]
            awq_s = compute_awq_scale(w, act, alpha=args.alpha)

            # Get Hessian diagonal for this layer
            H = hessians[layer_idx]
            H_diag = H.diag().float()

            # Transform Hessian diagonal for AWQ: H' = diag(1/s) @ H @ diag(1/s)
            # Diagonal of H' = H_diag / s^2
            awq_s_dev = awq_s.float()
            h_diag_awq = H_diag / (awq_s_dev ** 2)

            # Normalize to mean=1 for weighting
            h_weight = h_diag_awq / h_diag_awq.mean()

            # Move to GPU for quantization
            device = 'cuda'
            w_dev = w.to(device=device).float()
            s_dev = awq_s_dev.to(device=device)
            W = w_dev * s_dev.unsqueeze(0)  # AWQ-scaled weight

            # Per-tensor scale
            ts = W.abs().max() / NVFP4_TS_DIVISOR
            if ts.item() == 0:
                ts = torch.tensor(1.0, dtype=torch.float32, device=device)

            # Hessian-weighted clip ratio quantization
            hw = h_weight.to(device=device)
            w_packed, w_bs = quantize_hessian_weighted_clip(
                W, ts, hw, block_size=16, device=device)

            z[key] = w_packed.contiguous().cpu()
            z[key + ".nf4_b_scale"] = w_bs.contiguous().cpu()
            z[key + ".nvfp4_t_scale"] = ts.cpu()
            z[key + ".awq_scale"] = awq_s.contiguous()

            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + ts.nbytes + awq_s.nbytes
            if nvfp4_count <= 2:
                print(f"  Hessian-AWQ+clip {key}: [{N},{K}] "
                      f"hw_min={h_weight.min():.4f} hw_max={h_weight.max():.4f}")

            del w_dev, s_dev, W, hw, w_packed, w_bs, ts
            torch.cuda.empty_cache()

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
    print(f"    Hessian-AWQ+clip+NVFP4 (FFN key):  {nvfp4_count} tensors")
    print(f"    FP8      (Attention):              {att_fp8_count} tensors")
    print(f"    BF16     (FFN value):              preserved")
    print(f"    Skipped  (unchanged):              {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Metadata
    meta = {
        "v": 10,
        "r": [
            [0, L-1, 4, 2],
            [0, L-1, 1, 1],
            [0, L-1, 2, 1],
            [0, L-1, 3, 1],
            [0, L-1, 6, 1],
        ],
        "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32", "awq": True,
              "clip": True, "hessian_weighted": True, "alpha": args.alpha},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "ffn.value.weight",
        ],
        "desc": f"Hybrid v10: Hessian-weighted AWQ+clip NVFP4 (alpha={args.alpha}) + BF16 val + FP8 att",
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
