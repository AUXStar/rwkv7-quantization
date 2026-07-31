#!/usr/bin/env python3
"""Quantize attention key/value/receptance/output weights to FP8 E4M3.

Scheme: W8A16 (weight 8-bit FP8, activation 16-bit FP16/BF16)
- Per-tensor symmetric quantization
- Dequantization on-the-fly during inference (no _scaled_mm needed for attention)
- Targets: blocks.{i}.att.{receptance,key,value,output}.weight (64 tensors total)

Usage:
    python quantize_att_fp8.py [--input INPUT] [--output OUTPUT]

Defaults:
    INPUT  = /home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth
    OUTPUT = /home/njzy/model/rwkv7-g1h-2.9b-att-fp8.pth
"""
import torch
import argparse
import os
import time

FP8_E4M3_MAX = 448.0
FP8_E4M3_EPS = 0.015625  # min normal positive


def quantize_fp8(w: torch.Tensor):
    """Quantize FP16/BF16 weight to FP8 E4M3 with per-tensor scale.

    Returns:
        w_fp8: float8_e4m3fn tensor (same shape)
        scale: scalar float32 tensor
    """
    amax = w.abs().max()
    if amax > 0:
        scale = (amax / FP8_E4M3_MAX).float()
    else:
        scale = torch.tensor(1.0, dtype=torch.float32)

    # Quantize: divide by scale, clamp, cast
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth")
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-att-fp8.pth")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    t0 = time.time()
    z = torch.load(args.input, map_location="cpu", mmap=True)
    print(f"  Loaded {len(z)} tensors in {time.time()-t0:.1f}s")

    # Identify attention weights to quantize
    att_keys = []
    for k in z.keys():
        if k.startswith("blocks.") and ".att." in k and k.endswith(".weight"):
            # Only quantize the 4 main attention projections
            if any(k.endswith(suffix) for suffix in [
                "receptance.weight", "key.weight", "value.weight", "output.weight"
            ]):
                att_keys.append(k)

    print(f"\nFound {len(att_keys)} attention weights to quantize:")
    for k in att_keys[:4]:
        print(f"  {k}: {z[k].shape} {z[k].dtype}")
    print(f"  ... ({len(att_keys)} total)")

    # Calculate original size
    orig_bytes = sum(z[k].nbytes for k in att_keys)
    print(f"\nOriginal attention weights: {orig_bytes / 1e6:.1f} MB ({orig_bytes / 2**30:.3f} GiB)")

    # Quantize each weight
    quant_bytes = 0
    scale_bytes = 0
    for k in att_keys:
        w = z[k]
        if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            print(f"  SKIP {k}: dtype={w.dtype}")
            continue

        w_fp8, scale = quantize_fp8(w)
        z[k] = w_fp8
        z[k + ".fp8_scale"] = scale

        quant_bytes += w_fp8.nbytes
        scale_bytes += scale.nbytes

    print(f"\nQuantized attention weights: {quant_bytes / 1e6:.1f} MB ({quant_bytes / 2**30:.3f} GiB)")
    print(f"Scale tensors: {scale_bytes / 1e6:.1f} MB")
    print(f"VRAM savings: {(orig_bytes - quant_bytes - scale_bytes) / 1e6:.1f} MB ({(orig_bytes - quant_bytes - scale_bytes) / 2**30:.3f} GiB)")
    print(f"Compression ratio: {orig_bytes / (quant_bytes + scale_bytes):.2f}x")

    # Save
    print(f"\nSaving to: {args.output}")
    t0 = time.time()
    torch.save(z, args.output)
    print(f"  Saved in {time.time()-t0:.1f}s")
    print(f"  File size: {os.path.getsize(args.output) / 2**30:.3f} GiB")

    # Verify
    print("\nVerification:")
    z2 = torch.load(args.output, map_location="cpu", mmap=True)
    fp8_count = sum(1 for k in z2.keys() if k.endswith(".fp8_scale"))
    fp8_tensors = sum(1 for k in z2.keys() if z2[k].dtype == torch.float8_e4m3fn and ".att." in k)
    print(f"  FP8 scale keys: {fp8_count}")
    print(f"  FP8 attention tensors: {fp8_tensors}")
    print(f"  Total tensors: {len(z2)}")


if __name__ == "__main__":
    main()
