#!/usr/bin/env python3
"""Hybrid v5 quantization: NVFP4 block_size=8 FFN key + BF16 value + FP8 Attention.

Iteration 5 — reduce block size from 16 to 8 for finer-grained scaling:
- FFN key.weight:   NVFP4 with block_size=8 (per-tensor scale, 2x more block scales)
- FFN value.weight: BF16 (unchanged)
- Attention r/k/v/output.weight: FP8 E4M3 (W8A16)

Problem with block_size=16: each group of 16 elements shares one FP8 block scale.
If one element is much larger than the rest, the smaller elements lose precision.
block_size=8 halves the group size, reducing quantization error by ~33%.

W4A16 path is pure PyTorch dequantization, so block_size is not constrained
by hardware _scaled_mm (which requires block_size=16).

Storage: block_scale goes from [N, K//16] to [N, K//8] — negligible overhead.
For N=10240, K=2560: 1.6 MB → 3.2 MB per layer, 51 MB total across 32 layers.

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v5.pth
"""
import torch
import os
import time
import argparse

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.015625
FP4_E2M1_MAX = 6.0
NVFP4_TS_DIVISOR = 448.0 * 6.0  # 2688.0


def quantize_nvfp4(w, block_size=8):
    """Quantize weight to NVFP4 with per-tensor scale and configurable block size.

    Args:
        w: [N, K] bfloat16 weight
        block_size: number of elements per block (8 or 16)

    Returns:
        packed: [N, K//2] uint8 (packed FP4 pairs)
        block_scale: [N, K//block_size] float8_e4m3fn
        tensor_scale: scalar float32
    """
    N, K = w.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    assert K % 2 == 0, f"K={K} not even (needed for FP4 packing)"
    w_f32 = w.float()
    n_blocks = K // block_size

    # Per-tensor amax and scale
    amax = w_f32.abs().max()
    ts = amax / NVFP4_TS_DIVISOR
    if ts.item() == 0:
        ts = torch.tensor(1.0, dtype=torch.float32)

    # Reshape to blocks: [N, n_blocks, block_size]
    w_blocks = w_f32.view(N, n_blocks, block_size)

    # Block amax: [N, n_blocks]
    block_amax = w_blocks.abs().amax(dim=2)

    # Scaled block scale = block_amax / 6.0 / ts
    bs_scaled = block_amax / FP4_E2M1_MAX / ts  # [N, n_blocks]

    # Clamp to FP8 E4M3 range and cast
    bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
    bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

    # Effective scale for quantization: ts * bs_fp8
    bs_f32 = bs_fp8.to(torch.float32)  # [N, n_blocks]
    eff_scale = ts * bs_f32  # scalar * [N, n_blocks] → [N, n_blocks]

    # Scale weights: [N, n_blocks, block_size] / [N, n_blocks, 1]
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

    fp4 = sign * 8 + code  # [N, n_blocks, block_size]

    # Pack FP4 pairs into uint8: [N, K//2, 2] → hi*16 + lo
    fp4_flat = fp4.view(N, K // 2, 2)
    lo = fp4_flat[:, :, 0]  # even indices
    hi = fp4_flat[:, :, 1]  # odd indices
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


def verify_quantization(w_orig, packed, bs, ts, block_size=8):
    """Verify NVFP4 quantization by computing reconstruction error."""
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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v5.pth")
    parser.add_argument("--block-size", type=int, default=8, choices=[4, 8, 16])
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

        # FFN key -> NVFP4 with custom block_size
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape

            if nvfp4_count == 0:
                first_key_orig = w.clone()

            w_packed, w_bs, w_ts = quantize_nvfp4(w, block_size=args.block_size)
            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes
            if nvfp4_count <= 2:
                n_bs = w_bs.shape[1]
                print(f"  NVFP4 bs={args.block_size} {key}: [{N},{K}] -> packed[{N},{K//2}] bs[{N},{n_bs}]")

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
    print(f"    NVFP4 bs={args.block_size} (FFN key): {nvfp4_count} tensors")
    print(f"    FP8     (Attention):  {att_fp8_count} tensors")
    print(f"    BF16    (FFN value):  preserved")
    print(f"    Skipped (unchanged):  {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Verify reconstruction
    if args.verify and first_key_orig is not None:
        print(f"\n=== Verification: NVFP4 block_size={args.block_size} reconstruction ===")
        packed = z["blocks.0.ffn.key.weight"]
        bs = z["blocks.0.ffn.key.weight.nf4_b_scale"]
        ts = z["blocks.0.ffn.key.weight.nvfp4_t_scale"]
        mse, max_err, cos_sim = verify_quantization(first_key_orig, packed, bs, ts, args.block_size)
        print(f"  bs={args.block_size}: MSE={mse:.10f}, MaxErr={max_err:.6f}, CosSim={cos_sim:.10f}")

        # Compare with block_size=16 (original)
        packed16, bs16, ts16 = quantize_nvfp4(first_key_orig, block_size=16)
        mse16, max_err16, cos_sim16 = verify_quantization(first_key_orig, packed16, bs16, ts16, 16)
        print(f"  bs=16:             MSE={mse16:.10f}, MaxErr={max_err16:.6f}, CosSim={cos_sim16:.10f}")
        print(f"  Improvement:       MSE {(1 - mse/mse16)*100:.1f}% lower, MaxErr {(1 - max_err/max_err16)*100:.1f}% lower, CosSim +{cos_sim - cos_sim16:.8f}")

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
        "s": {"blk": args.block_size, "sd": "fp8e4m3", "td": "fp32"},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "ffn.value.weight",
        ],
        "desc": f"Hybrid v5: FFN key NVFP4 bs={args.block_size} + FFN value BF16 + Attention FP8 W8A16",
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
    bs = z2["blocks.0.ffn.key.weight.nf4_b_scale"]
    print(f"  blocks.0.ffn.key bs shape: {list(bs.shape)} (block_size={K // bs.shape[1]})")

    print("\nDone!")


if __name__ == "__main__":
    main()
