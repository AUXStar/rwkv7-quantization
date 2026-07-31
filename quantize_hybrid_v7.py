#!/usr/bin/env python3
"""Hybrid v7 quantization: NVFP4 with AWQ activation-aware channel scaling.

Iteration 7 — protect important weight channels using activation statistics:
- FFN key.weight:   NVFP4 W4A16 with AWQ per-channel scaling (block_size=16)
- FFN value.weight: BF16 (unchanged)
- Attention r/k/v/output.weight: FP8 E4M3 (W8A16)

AWQ (Activation-aware Weight Quantization) insight:
  Not all weight channels are equally important. Channels with large activations
  contribute more to the output and should be preserved with higher precision.
  AWQ scales up important channels before quantization, then scales down the
  input to compensate:
    W' = W * s    (scale up important channels)
    x' = x / s    (compensate in activation)
    y = x' @ W'^T = x @ W^T  (mathematically equivalent for full precision)

  Scaling factor: s_k = (mean(|x_k|)^alpha) / (mean(|w_k|)^(1-alpha))
  where alpha=0.5 balances activation and weight importance.

  This gives important channels more FP4 precision (larger values → more levels
  used), while less important channels use fewer levels (acceptable loss).

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v7.pth
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


def quantize_nvfp4(w, block_size=16, device='cuda'):
    """Default NVFP4 quantization (clip ratio = 1.0, no clipping).

    Args:
        w: [N, K] float weight (already AWQ-scaled)
        block_size: 16

    Returns:
        packed: [N, K//2] uint8
        block_scale: [N, K//block_size] float8_e4m3fn
        tensor_scale: scalar float32
    """
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


def compute_awq_scale(w, act_stats, alpha=0.5, device='cuda'):
    """Compute AWQ per-channel scaling factor.

    s_k = (mean(|x_k|)^alpha) / (mean(|w_k|)^(1-alpha))
    Then normalize so mean(s) = 1.0.

    Args:
        w: [N, K] weight tensor
        act_stats: [K] per-channel activation mean(|x_k|)
        alpha: balance factor (0.5 default)

    Returns:
        s: [K] scaling factor
    """
    w_dev = w.to(device=device).float()
    act_dev = act_stats.to(device=device).float()

    # Per-channel weight magnitude: mean(|W[:, k]|) over output channels
    w_mean = w_dev.abs().mean(dim=0)  # [K]

    # AWQ scaling factor
    s = (act_dev.clamp(min=1e-8) ** alpha) / (w_mean.clamp(min=1e-8) ** (1 - alpha))

    # Normalize: mean(s) = 1.0 (preserve overall weight magnitude)
    s = s / s.mean()

    return s.cpu()


def quantize_to_fp8(w):
    """Quantize bf16/fp16 weight to FP8 E4M3 with per-tensor scale."""
    amax = w.abs().max()
    if amax > 0:
        scale = (amax / FP8_E4M3_MAX).float()
    else:
        scale = torch.tensor(1.0, dtype=torch.float32)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def verify_awq_quantization(w_orig, packed, bs, ts, awq_scale, device='cuda'):
    """Verify AWQ NVFP4 quantization by computing reconstruction error."""
    sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
    from nvfp4_ops import dequantize_nvfp4

    # Dequantize: this gives W' = W * s (the AWQ-scaled weight)
    w_deq_scaled = dequantize_nvfp4(packed, bs, ts)

    # Undo AWQ scaling: W_reconstructed = W' / s
    s = awq_scale.to(device=device).float()
    w_deq = (w_deq_scaled.to(device=device).float() / s.unsqueeze(0)).cpu()

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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v7.pth")
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

    # Load activation statistics
    print(f"Loading activation statistics from {args.act_stats} ...")
    act_stats = torch.load(args.act_stats, map_location="cpu")
    print(f"  {len(act_stats)} layers, alpha={args.alpha}")

    nvfp4_count = 0
    att_fp8_count = 0
    skip_count = 0
    orig_bytes = 0
    quant_bytes = 0

    ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")
    first_key_orig = None
    first_key_awq = None

    for key in list(z.keys()):
        if not torch.is_tensor(z[key]):
            skip_count += 1
            continue

        w = z[key]
        orig_bytes += w.nbytes

        # FFN key -> NVFP4 with AWQ scaling
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            layer_idx = int(key.split(".")[1])

            if nvfp4_count == 0:
                first_key_orig = w.clone()

            # Compute AWQ scaling factor
            act = act_stats[layer_idx]  # [K]
            awq_s = compute_awq_scale(w, act, alpha=args.alpha)  # [K]

            if nvfp4_count == 0:
                first_key_awq = awq_s.clone()

            # Scale weights: W' = W * s
            w_scaled = (w.float() * awq_s.float().unsqueeze(0)).to(torch.bfloat16)

            # Quantize scaled weights to NVFP4
            w_packed, w_bs, w_ts = quantize_nvfp4(w_scaled, block_size=16)

            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            z[key + ".awq_scale"] = awq_s.contiguous()  # [K] float32

            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes + awq_s.nbytes
            if nvfp4_count <= 2:
                print(f"  AWQ+NVFP4 {key}: [{N},{K}] s=[{K}] "
                      f"s_min={awq_s.min():.4f} s_max={awq_s.max():.4f}")

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
    print(f"    AWQ+NVFP4 (FFN key):  {nvfp4_count} tensors")
    print(f"    FP8      (Attention): {att_fp8_count} tensors")
    print(f"    BF16     (FFN value): preserved")
    print(f"    Skipped  (unchanged): {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Verify reconstruction
    if args.verify and first_key_orig is not None:
        print(f"\n=== Verification: AWQ NVFP4 reconstruction ===")
        packed = z["blocks.0.ffn.key.weight"]
        bs = z["blocks.0.ffn.key.weight.nf4_b_scale"]
        ts = z["blocks.0.ffn.key.weight.nvfp4_t_scale"]
        awq_s = z["blocks.0.ffn.key.weight.awq_scale"]

        # AWQ + NVFP4
        mse_awq, max_err_awq, cos_awq = verify_awq_quantization(
            first_key_orig, packed, bs, ts, awq_s)
        print(f"  AWQ+NVFP4:  MSE={mse_awq:.10f}, MaxErr={max_err_awq:.6f}, CosSim={cos_awq:.10f}")

        # Default NVFP4 (no AWQ) for comparison
        packed_def, bs_def, ts_def = quantize_nvfp4(first_key_orig, block_size=16)
        sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
        from nvfp4_ops import dequantize_nvfp4
        w_deq_def = dequantize_nvfp4(packed_def, bs_def, ts_def)
        w_orig_f16 = first_key_orig.to(torch.float16)
        w_deq_f16 = w_deq_def.to(torch.float16)
        mse_def = ((w_orig_f16 - w_deq_f16) ** 2).mean().item()
        max_err_def = (w_orig_f16 - w_deq_f16).abs().max().item()
        cos_def = torch.nn.functional.cosine_similarity(
            w_orig_f16.flatten().unsqueeze(0), w_deq_f16.flatten().unsqueeze(0)).item()
        print(f"  default:    MSE={mse_def:.10f}, MaxErr={max_err_def:.6f}, CosSim={cos_def:.10f}")
        print(f"  Improvement: MSE {(1 - mse_awq/mse_def)*100:.1f}%, "
              f"MaxErr {(1 - max_err_awq/max_err_def)*100:.1f}%, "
              f"CosSim +{cos_awq - cos_def:.8f}")

    # Metadata
    meta = {
        "v": 7,
        "r": [
            [0, L-1, 4, 2],  # ffn_key: nvfp4 + awq
            [0, L-1, 1, 1],  # att_receptance: fp8
            [0, L-1, 2, 1],  # att_key: fp8
            [0, L-1, 3, 1],  # att_value: fp8
            [0, L-1, 6, 1],  # att_output: fp8
        ],
        "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32", "awq": True, "alpha": args.alpha},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "ffn.value.weight",
        ],
        "desc": f"Hybrid v7: FFN key AWQ+NVFP4 (alpha={args.alpha}) + FFN value BF16 + Attention FP8 W8A16",
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

    awq_s = z2["blocks.0.ffn.key.weight.awq_scale"]
    print(f"  blocks.0.awq_scale: shape={list(awq_s.shape)}, "
          f"min={awq_s.min():.4f}, max={awq_s.max():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
