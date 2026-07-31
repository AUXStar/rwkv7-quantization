#!/usr/bin/env python3
"""Hybrid v4 quantization: Per-channel NVFP4 FFN key + BF16 value + FP8 Attention.

Iteration 4 — per-channel tensor scale to break the 98.43% Top-1 barrier:
- FFN key.weight:   NVFP4 with PER-CHANNEL tensor scale (one scale per output row)
- FFN value.weight: BF16 (unchanged)
- Attention r/k/v/output.weight: FP8 E4M3 (W8A16)

Problem with per-tensor scale (v3): all 10240 output channels share one scale.
Channels with small weights lose precision because the scale is dominated by
large-weight channels. Per-channel scaling gives each row its own dynamic range.

Storage change: nvfp4_t_scale becomes [N] float32 instead of scalar float32.
The dequantize_nvfp4 function auto-detects via tensor_scale.dim().

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v4.pth
"""
import torch
import os
import time
import argparse

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.015625  # 2^(-6)
FP4_E2M1_MAX = 6.0
NVFP4_TS_DIVISOR = 448.0 * 6.0  # = 2688.0


def quantize_nvfp4_perchannel(w, block_size=16):
    """Quantize weight to NVFP4 with per-output-channel tensor scale.

    Args:
        w: [N, K] bfloat16 weight
        block_size: 16 for NVFP4

    Returns:
        packed: [N, K//2] uint8 (packed FP4 pairs)
        block_scale: [N, K//16] float8_e4m3fn
        tensor_scale: [N] float32 (per-channel)
    """
    N, K = w.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    w_f32 = w.float()
    n_blocks = K // block_size

    # Per-channel amax and tensor scale
    amax = w_f32.abs().amax(dim=1)  # [N]
    ts = amax / NVFP4_TS_DIVISOR    # [N]
    ts = torch.where(ts > 0, ts, torch.ones_like(ts))

    # Reshape to blocks: [N, K//16, 16]
    w_blocks = w_f32.view(N, n_blocks, block_size)

    # Block amax: [N, K//16]
    block_amax = w_blocks.abs().amax(dim=2)

    # Scaled block scale = block_amax / 6.0 / ts (per-channel)
    # After FP8 cast, dequant = fp4 * ts * bs_fp8 ≈ fp4 * block_amax / 6.0
    bs_scaled = block_amax / FP4_E2M1_MAX / ts.unsqueeze(-1)  # [N, K//16]

    # Clamp to FP8 E4M3 range and cast
    bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
    bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

    # Effective scale for quantization: ts * bs_fp8
    bs_f32 = bs_fp8.to(torch.float32)  # [N, K//16]
    eff_scale = ts.unsqueeze(-1) * bs_f32  # [N, K//16]

    # Scale weights: [N, K//16, 16] / [N, K//16, 1]
    w_scaled = w_blocks / eff_scale.unsqueeze(-1)

    # Clamp to FP4 range [-6, 6]
    w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    # Round to FP4 E2M1 (round-to-nearest-even)
    sign = torch.where(w_scaled < 0, 1, 0).to(torch.uint8)
    a = w_scaled.abs()

    code = torch.where(a <= 0.25, 0,
           torch.where(a < 0.75, 1,
           torch.where(a <= 1.25, 2,
           torch.where(a < 1.75, 3,
           torch.where(a <= 2.5, 4,
           torch.where(a < 3.5, 5,
           torch.where(a <= 5.0, 6, 7))))))).to(torch.uint8)

    fp4 = sign * 8 + code  # [N, K//16, 16]

    # Pack FP4 pairs into uint8: [N, K//2, 2] → hi*16 + lo
    fp4_pairs = fp4.view(N, K // 2, 2)
    lo = fp4_pairs[:, :, 0]  # even indices
    hi = fp4_pairs[:, :, 1]  # odd indices
    packed = (hi * 16 + lo).to(torch.uint8)  # [N, K//2]

    return packed, bs_fp8, ts


def quantize_to_fp8(w):
    """Quantize bf16/fp16 weight to FP8 E4M3 with per-tensor scale."""
    amax = w.abs().max()
    if amax > 0:
        scale = (amax / FP8_E4M3_MAX).float()
    else:
        scale = torch.tensor(1.0, dtype=torch.float32)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def verify_quantization(w_orig, packed, bs, ts, block_size=16):
    """Verify per-channel NVFP4 quantization by computing reconstruction error."""
    import sys
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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v4.pth")
    parser.add_argument("--verify", action="store_true", help="Verify reconstruction error on first layer")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading original model from {args.input} ...")
    z = torch.load(args.input, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")

    nvfp4_count = 0
    att_fp8_count = 0
    skip_count = 0
    orig_bytes = 0
    quant_bytes = 0

    ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")

    # Save first layer's original key for verification
    first_key_orig = None

    for key in list(z.keys()):
        if not torch.is_tensor(z[key]):
            skip_count += 1
            continue

        w = z[key]
        orig_bytes += w.nbytes

        # FFN key -> NVFP4 with per-channel tensor scale
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape

            if nvfp4_count == 0:
                first_key_orig = w.clone()

            w_packed, w_bs, w_ts = quantize_nvfp4_perchannel(w, block_size=16)
            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()  # [N] per-channel!
            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes
            if nvfp4_count <= 2:
                print(f"  NVFP4-PC {key}: [{N},{K}] -> packed[{N},{K//2}] bs[{N},{K//16}] ts[{N}]")
                print(f"    ts range: [{w_ts.min().item():.6f}, {w_ts.max().item():.6f}]")

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
    print(f"    NVFP4-PC (FFN key):   {nvfp4_count} tensors (per-channel tensor scale)")
    print(f"    FP8     (Attention):  {att_fp8_count} tensors")
    print(f"    BF16    (FFN value):  preserved")
    print(f"    Skipped (unchanged):  {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Verify reconstruction on first layer
    if args.verify and first_key_orig is not None:
        print("\n=== Verification: Per-channel NVFP4 reconstruction ===")
        packed = z["blocks.0.ffn.key.weight"]
        bs = z["blocks.0.ffn.key.weight.nf4_b_scale"]
        ts = z["blocks.0.ffn.key.weight.nvfp4_t_scale"]
        mse, max_err, cos_sim = verify_quantization(first_key_orig, packed, bs, ts)
        print(f"  Layer 0 FFN key: MSE={mse:.8f}, MaxErr={max_err:.6f}, CosSim={cos_sim:.8f}")

        # Compare with per-tensor (old method)
        import importlib.util
        quack_dir = os.path.join(os.path.dirname(torch.__file__), '_vendor', 'quack')
        mx_path = os.path.join(quack_dir, 'mx_utils.py')
        spec = importlib.util.spec_from_file_location('_mx_utils', mx_path)
        mx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mx)

        amax = first_key_orig.abs().max()
        ts_scalar = mx.nvfp4_per_tensor_scale(amax)
        w_packed_pt, w_bs_pt, w_ts_pt = mx.to_nvfp4(first_key_orig, block_size=16, per_tensor_scale=ts_scalar)
        mse_pt, max_err_pt, cos_sim_pt = verify_quantization(first_key_orig, w_packed_pt, w_bs_pt, w_ts_pt)
        print(f"  Per-tensor:      MSE={mse_pt:.8f}, MaxErr={max_err_pt:.6f}, CosSim={cos_sim_pt:.8f}")
        print(f"  Improvement:     MSE {(1 - mse/mse_pt)*100:.1f}% lower, CosSim +{cos_sim - cos_sim_pt:.6f}")

    # Metadata
    meta = {
        "v": 5,
        "r": [
            [0, L-1, 4, 2],  # ffn_key: nvfp4 (per-channel)
            [0, L-1, 1, 1],  # att_receptance: fp8 (W8A16)
            [0, L-1, 2, 1],  # att_key: fp8 (W8A16)
            [0, L-1, 3, 1],  # att_value: fp8 (W8A16)
            [0, L-1, 6, 1],  # att_output: fp8 (W8A16)
        ],
        "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "ffn.value.weight",
        ],
        "desc": "Hybrid v4: FFN key NVFP4 per-channel + FFN value BF16 + Attention FP8 W8A16",
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

    # Check per-channel tensor scale
    ts = z2["blocks.0.ffn.key.weight.nvfp4_t_scale"]
    print(f"  blocks.0.ffn.key.weight.nvfp4_t_scale: shape={list(ts.shape)}, dtype={ts.dtype}")
    if ts.dim() == 0:
        print(f"    WARNING: scalar tensor scale (per-tensor, not per-channel!)")
    else:
        print(f"    Per-channel: [{ts.shape[0]}] channels, range=[{ts.min().item():.6f}, {ts.max().item():.6f}]")

    # Spot check
    for prefix in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
                   "blocks.0.att.key.weight"]:
        w = z2[prefix]
        print(f"  {prefix}: {list(w.shape)} {w.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
