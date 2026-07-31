#!/usr/bin/env python3
"""Quantize FFN: key -> NVFP4, value -> FP8 E4M3.

Mixed approach for better precision:
- ffn.key.weight [10240, 2560]: NVFP4 (4-bit, block scaled)
- ffn.value.weight [2560, 10240]: FP8 E4M3 (8-bit, per-tensor scaled)

Output: /home/njzy/model/rwkv7-g1h-2.9b-mixed-nvfp4-fp8.pth
"""
import torch
import importlib.util
import os
import time

def load_mx_utils():
    quack_dir = os.path.join(os.path.dirname(torch.__file__), '_vendor', 'quack')
    mx_path = os.path.join(quack_dir, 'mx_utils.py')
    spec = importlib.util.spec_from_file_location('_mx_utils', mx_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mx = load_mx_utils()
to_nvfp4 = mx.to_nvfp4
nvfp4_per_tensor_scale = mx.nvfp4_per_tensor_scale

ORIG_PATH = "/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth"
OUT_PATH  = "/home/njzy/model/rwkv7-g1h-2.9b-mixed-nvfp4-fp8.pth"

FP8_E4M3_MAX = 448.0

def quantize_to_fp8(w):
    """Quantize bf16 weight to FP8 E4M3 with per-tensor scale.
    
    Returns: (fp8_weight, per_tensor_scale)
    """
    amax = w.abs().max()
    scale = amax / FP8_E4M3_MAX
    if scale.item() == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    w_scaled = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    return w_fp8, scale.float()

def main():
    t0 = time.perf_counter()
    print(f"Loading original model from {ORIG_PATH} ...")
    z = torch.load(ORIG_PATH, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")

    nvfp4_count = 0
    fp8_count = 0
    skip_count = 0

    for key in list(z.keys()):
        if ".ffn.key.weight" in key and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key:
            # NVFP4 quantization for key
            w = z[key]
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            amax = w.abs().max()
            ts = nvfp4_per_tensor_scale(amax)
            w_packed, w_bs, w_ts = to_nvfp4(w, block_size=16, per_tensor_scale=ts)
            z[key] = w_packed.contiguous()
            z[key + ".nf4_b_scale"] = w_bs.contiguous()
            z[key + ".nvfp4_t_scale"] = w_ts.contiguous()
            nvfp4_count += 1
            if nvfp4_count <= 2:
                print(f"  NVFP4 {key}: [{N},{K}] -> packed[{N},{K//2}] bs[{N},{K//16}]")

        elif ".ffn.value.weight" in key and ".fp8_scale" not in key:
            # FP8 quantization for value
            w = z[key]
            assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
            N, K = w.shape
            w_fp8, scale = quantize_to_fp8(w)
            z[key] = w_fp8.contiguous()
            z[key + ".fp8_scale"] = scale.contiguous()
            fp8_count += 1
            if fp8_count <= 2:
                print(f"  FP8   {key}: [{N},{K}] -> fp8[{N},{K}] scale={scale.item():.6f}")

        else:
            skip_count += 1

    print(f"  NVFP4: {nvfp4_count}, FP8: {fp8_count}, skipped: {skip_count}")

    meta = {
        "v": 2,
        "r": [
            [0, L-1, 4, 2],  # ffn_key: nvfp4
            [0, L-1, 5, 1],  # ffn_value: fp8
        ],
        "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},
        "n": [
            "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
            "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
            "w0", "w1", "w2", "a0", "a1", "a2",
            "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k",
            "att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight",
        ],
    }
    z["meta"] = meta

    t1 = time.perf_counter()
    print(f"Saving to {OUT_PATH} ...")
    torch.save(z, OUT_PATH)
    file_size = os.path.getsize(OUT_PATH) / (1024**3)
    print(f"  Saved in {time.perf_counter()-t1:.1f}s, size: {file_size:.2f} GB")
    print(f"Total time: {time.perf_counter()-t0:.1f}s")

    # Verify
    print("\n=== Verification ===")
    z2 = torch.load(OUT_PATH, map_location="cpu", mmap=True)
    for prefix in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight"]:
        w = z2[prefix]
        print(f"  {prefix}: shape={list(w.shape)}, dtype={w.dtype}")
        if prefix.endswith("key.weight"):
            bs = z2[prefix + ".nf4_b_scale"]
            ts = z2[prefix + ".nvfp4_t_scale"]
            print(f"    nf4_b_scale: {list(bs.shape)}, nvfp4_t_scale: {ts.item():.6f}")
        elif prefix.endswith("value.weight"):
            sc = z2[prefix + ".fp8_scale"]
            print(f"    fp8_scale: {sc.item():.6f}")

    # Check non-quantized
    att_w = z2["blocks.0.att.key.weight"]
    print(f"  blocks.0.att.key.weight: {list(att_w.shape)} {att_w.dtype} (should be bf16)")

    # Estimate storage
    total_bytes = 0
    for key, tensor in z2.items():
        if key == "meta":
            continue
        total_bytes += tensor.numel() * tensor.element_size()
    print(f"\n  Estimated tensor storage: {total_bytes / (1024**3):.2f} GB")

    print("\nDone!")

if __name__ == "__main__":
    main()
