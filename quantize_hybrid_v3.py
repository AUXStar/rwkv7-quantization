#!/usr/bin/env python3
"""Hybrid v3 quantization: FFN key NVFP4 + FFN value BF16 + Attention FP8 W8A16.

Iteration 3 strategy — maximize NVFP4 precision by keeping FFN value in BF16:
- FFN key.weight:   NVFP4 (4-bit, block scaled) — largest tensor, maximum compression
- FFN value.weight: BF16 (unchanged) — eliminate value quantization error
- Attention r/k/v/output.weight: FP8 E4M3 (W8A16) — high precision, low overhead

Rationale: W4A16+W8A16 gave Top-1 98.43%, PPL delta 0.0094.
The remaining error splits between NVFP4 key (16 levels) and FP8 value (256 levels).
By keeping value in BF16, we isolate pure NVFP4 key error + attention FP8 error.
Attention FP8 W8A16 alone gave 99.71% Top-1, so the combined should be ~98.5-99%.

Output: /home/njzy/model/rwkv7-g1h-2.9b-hybrid-v3.pth
"""
import torch
import importlib.util
import os
import time
import argparse

FP8_E4M3_MAX = 448.0


def load_mx_utils():
    quack_dir = os.path.join(os.path.dirname(torch.__file__), '_vendor', 'quack')
    mx_path = os.path.join(quack_dir, 'mx_utils.py')
    spec = importlib.util.spec_from_file_location('_mx_utils', mx_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    parser.add_argument("--output", default="/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v3.pth")
    args = parser.parse_args()

    mx = load_mx_utils()
    to_nvfp4 = mx.to_nvfp4
    nvfp4_per_tensor_scale = mx.nvfp4_per_tensor_scale

    t0 = time.perf_counter()
    print(f"Loading original model from {args.input} ...")
    z = torch.load(args.input, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")

    nvfp4_count = 0    # FFN key -> NVFP4
    att_fp8_count = 0  # Attention r/k/v/output -> FP8
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

        # FFN key -> NVFP4
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            amax = w.abs().max()
            ts = nvfp4_per_tensor_scale(amax)
            w_packed, w_bs, w_ts = to_nvfp4(w, block_size=16, per_tensor_scale=ts)
            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            nvfp4_count += 1
            quant_bytes += w_packed.nbytes + w_bs.nbytes + w_ts.nbytes
            if nvfp4_count <= 2:
                print(f"  NVFP4 {key}: [{N},{K}] -> packed[{N},{K//2}] bs[{N},{K//16}]")

        # FFN value -> SKIP (keep BF16)
        elif ".ffn.value.weight" in key:
            quant_bytes += w.nbytes
            skip_count += 1
            if skip_count <= 2:
                print(f"  SKIP  {key}: [{w.shape[0]},{w.shape[1]}] {w.dtype} (BF16 preserved)")

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
            if att_fp8_count <= 4:
                print(f"  FP8   {key}: [{N},{K}] -> fp8[{N},{K}] scale={scale.item():.6f}")

        else:
            skip_count += 1
            quant_bytes += w.nbytes

    print(f"\n  Quantization summary:")
    print(f"    NVFP4 (FFN key):      {nvfp4_count} tensors")
    print(f"    FP8   (Attention):    {att_fp8_count} tensors")
    print(f"    BF16  (FFN value):    preserved (no quantization)")
    print(f"    Skipped (unchanged):  {skip_count} tensors")
    print(f"    Original size:  {orig_bytes / 2**30:.3f} GiB")
    print(f"    Quantized size: {quant_bytes / 2**30:.3f} GiB")
    print(f"    VRAM savings:   {(orig_bytes - quant_bytes) / 2**30:.3f} GiB ({(1 - quant_bytes/orig_bytes)*100:.1f}%)")

    # Metadata
    meta = {
        "v": 4,
        "r": [
            [0, L-1, 4, 2],  # ffn_key: nvfp4
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
        "desc": "Hybrid v3: FFN key NVFP4 + FFN value BF16 + Attention r/k/v/output FP8 W8A16",
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

    # Spot check
    for prefix in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
                   "blocks.0.att.key.weight", "blocks.0.att.receptance.weight"]:
        w = z2[prefix]
        print(f"  {prefix}: {list(w.shape)} {w.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
