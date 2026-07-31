#!/usr/bin/env python3
"""Hybrid v11: Per-channel tensor scale + AWQ + clip ratio.

Key insight: v8 uses per-TENSOR scale (one scalar for entire weight matrix).
This means channels with small weights "waste" FP4 range, while channels with
large weights might clip. Per-CHANNEL scale gives each output row its own scale,
allowing the block scales to be better utilized.

The dequantize_nvfp4 function already supports per-channel ts:
  if tensor_scale.dim() == 0: out *= ts        # scalar
  else:                        out *= ts.unsqueeze(-1)  # [N, 1] broadcast

Changes from v8:
  - ts: scalar → [N] per-channel
  - Block scale: bs = block_amax / (FP4_MAX * ts_n) instead of / (FP4_MAX * ts)
  - Storage: ts is now [N] float32 instead of scalar

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v11.pth
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


def quantize_per_channel_nvfp4_clip(w, awq_scale, block_size=16,
                                     clip_ratios=None, device='cuda'):
    """NVFP4 quantization with per-channel tensor scale + AWQ + clip ratio.

    Args:
        w: [N, K] bfloat16 weight
        awq_scale: [K] float32 AWQ scaling factor
        block_size: 16
        clip_ratios: list of ratios to search

    Returns:
        packed: [N, K//2] uint8 (packed FP4)
        block_scale: [N, K//block_size] float8_e4m3fn
        tensor_scale: [N] float32 (per-channel!)
        awq_scale: [K] float32
    """
    if clip_ratios is None:
        clip_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    N, K = w.shape
    n_blocks = K // block_size

    # Apply AWQ scaling
    w_orig = w.to(device=device).float()
    s = awq_scale.to(device=device).float()
    W = w_orig * s.unsqueeze(0)  # [N, K]

    # Per-CHANNEL tensor scale: one scale per output row
    # ts_n = max(|W[n, :]|) / NVFP4_TS_DIVISOR for each row n
    row_amax = W.abs().amax(dim=1)  # [N]
    ts = row_amax / NVFP4_TS_DIVISOR  # [N]
    ts = ts.clamp(min=1e-10)  # avoid division by zero

    # Reshape for broadcasting: ts [N] → [N, 1] for block scale computation
    ts_col = ts.unsqueeze(1)  # [N, 1]

    w_blocks = W.view(N, n_blocks, block_size)
    block_amax = w_blocks.abs().amax(dim=2)  # [N, n_blocks]

    best_mse = torch.full((N, n_blocks), float('inf'), device=device)
    best_fp4_idx = None
    best_bs_fp8 = None
    fp4_table = _FP4_VALUES.to(device)

    for ratio in clip_ratios:
        # Block scale: bs = block_amax * ratio / (FP4_MAX * ts_n)
        bs_scaled = block_amax * ratio / FP4_E2M1_MAX / ts_col  # [N, n_blocks]
        bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
        bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

        bs_f32 = bs_fp8.to(torch.float32)
        # Effective scale: ts_n * bs_{n,b} → [N, n_blocks]
        eff_scale = ts_col * bs_f32  # [N, n_blocks]

        # Scale weights: w_scaled = W / eff_scale
        w_scaled = w_blocks / eff_scale.unsqueeze(-1)  # [N, n_blocks, block_size]
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

    # Pack FP4 pairs
    fp4_flat = best_fp4_idx.view(N, K // 2, 2)
    lo = fp4_flat[:, :, 0]
    hi = fp4_flat[:, :, 1]
    packed = (hi * 16 + lo).to(torch.uint8)

    return packed.cpu(), best_bs_fp8.cpu(), ts.cpu(), awq_scale


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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v11.pth")
    parser.add_argument("--act-stats", default="/home/njzy/test/eval_tmp/awq_act_stats.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading original model from {args.input} ...")
    z = torch.load(args.input, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")

    act_stats = torch.load(args.act_stats, map_location="cpu")
    print(f"  AWQ stats: {len(act_stats)} layers, alpha={args.alpha}")

    nvfp4_count = 0
    att_fp8_count = 0
    skip_count = 0
    orig_bytes = 0
    quant_bytes = 0

    ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")

    # For verification
    sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")

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

            w_packed, w_bs, w_ts, _ = quantize_per_channel_nvfp4_clip(
                w, awq_s, block_size=16)

            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()  # [N] per-channel!
            z[key + ".awq_scale"] = awq_s.contiguous()

            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes + awq_s.nbytes

            if nvfp4_count <= 2:
                print(f"  per-ch NVFP4 {key}: [{N},{K}] "
                      f"ts=[{N}] min={w_ts.min():.8f} max={w_ts.max():.8f}")

            if nvfp4_count == 1 and args.verify:
                from nvfp4_ops import dequantize_nvfp4
                w_deq = dequantize_nvfp4(w_packed.cuda(), w_bs.cuda(), w_ts.cuda())
                w_deq = (w_deq.float() / awq_s.float().unsqueeze(0).cuda()).cpu()
                w_orig_f16 = w.to(torch.float16)
                w_deq_f16 = w_deq.to(torch.float16)
                mse = ((w_orig_f16 - w_deq_f16) ** 2).mean().item()
                cos = torch.nn.functional.cosine_similarity(
                    w_orig_f16.flatten().unsqueeze(0),
                    w_deq_f16.flatten().unsqueeze(0)).item()

                # Compare with per-tensor (v8 style)
                from quantize_hybrid_v8 import quantize_nvfp4
                w_scaled = (w.float() * awq_s.float().unsqueeze(0)).to(torch.bfloat16)
                p8, b8, t8 = quantize_nvfp4(w_scaled, block_size=16)
                w_deq8 = dequantize_nvfp4(p8.cuda(), b8.cuda(), t8.cuda())
                w_deq8 = (w_deq8.float() / awq_s.float().unsqueeze(0).cuda()).cpu()
                w_deq8_f16 = w_deq8.to(torch.float16)
                mse8 = ((w_orig_f16 - w_deq8_f16) ** 2).mean().item()
                cos8 = torch.nn.functional.cosine_similarity(
                    w_orig_f16.flatten().unsqueeze(0),
                    w_deq8_f16.flatten().unsqueeze(0)).item()

                print(f"  Verify: per-ch MSE={mse:.10f} cos={cos:.10f}")
                print(f"          per-ts  MSE={mse8:.10f} cos={cos8:.10f}")
                print(f"          Improvement: MSE {(1-mse/mse8)*100:.1f}%, cos +{cos-cos8:.8f}")

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

    print(f"\n  Summary: {nvfp4_count} NVFP4(per-ch), {att_fp8_count} FP8, {skip_count} skip")
    print(f"  Size: {orig_bytes/2**30:.3f} → {quant_bytes/2**30:.3f} GiB "
          f"({(1-quant_bytes/orig_bytes)*100:.1f}% savings)")

    meta = {
        "v": 11,
        "r": [[0, L-1, 4, 2], [0, L-1, 1, 1], [0, L-1, 2, 1], [0, L-1, 3, 1], [0, L-1, 6, 1]],
        "s": {"blk": 16, "awq": True, "clip": True, "per_channel_ts": True, "alpha": args.alpha},
        "n": ["emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
              "x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "w1", "w2",
              "a0", "a1", "a2", "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
              "ffn.value.weight"],
        "desc": f"v11: per-channel-ts AWQ+clip NVFP4 (alpha={args.alpha}) + BF16 val + FP8 att",
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
