#!/usr/bin/env python3
"""Quantize FFN key/value weights of RWKV-7 2.9B to NVFP4.

#2 experiment: FFN-only NVFP4 baseline.
- Quantize: blocks.*.ffn.key.weight [10240,2560] + ffn.value.weight [2560,10240]
- Keep: everything else in bf16
- Output: /home/njzy/model/rwkv7-g1h-2.9b-nvfp4-ffn-only.pth
"""
import torch
import importlib.util
import os
import time
import sys

# Load mx_utils directly (bypass quack.__init__ which needs cutlass)
def load_mx_utils():
    quack_dir = os.path.join(os.path.dirname(torch.__file__), '_vendor', 'quack')
    mx_path = os.path.join(quack_dir, 'mx_utils.py')
    if not os.path.exists(mx_path):
        raise FileNotFoundError(f"mx_utils.py not found at {mx_path}")
    spec = importlib.util.spec_from_file_location('_mx_utils', mx_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mx = load_mx_utils()
to_nvfp4 = mx.to_nvfp4
nvfp4_per_tensor_scale = mx.nvfp4_per_tensor_scale

ORIG_PATH = "/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth"
OUT_PATH  = "/home/njzy/model/rwkv7-g1h-2.9b-nvfp4-ffn-only.pth"

QUANTIZE_PATTERNS = [
    ".ffn.key.weight",
    ".ffn.value.weight",
]

def is_ffn_quant_target(key):
    return any(p in key for p in QUANTIZE_PATTERNS) and ".nf4_b_scale" not in key and ".nvfp4_t_scale" not in key

def main():
    t0 = time.perf_counter()
    print(f"Loading original model from {ORIG_PATH} ...")
    z = torch.load(ORIG_PATH, map_location="cpu")
    print(f"  Loaded {len(z)} tensors in {time.perf_counter()-t0:.1f}s")

    C = z["blocks.0.ln1.weight"].shape[0]
    L = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks.")) + 1
    print(f"  Model: C={C}, L={L}")

    quant_count = 0
    skip_count = 0
    for key in list(z.keys()):
        if not is_ffn_quant_target(key):
            skip_count += 1
            continue

        w = z[key]
        assert w.dtype == torch.bfloat16, f"{key}: expected bf16, got {w.dtype}"
        N, K = w.shape
        assert K % 16 == 0, f"{key}: K={K} not divisible by 16"

        amax = w.abs().max()
        ts = nvfp4_per_tensor_scale(amax)
        w_packed, w_bs, w_ts = to_nvfp4(w, block_size=16, per_tensor_scale=ts)

        assert w_packed.shape == (N, K // 2), f"{key}: packed shape {w_packed.shape} != ({N}, {K//2})"
        assert w_bs.shape == (N, K // 16), f"{key}: bs shape {w_bs.shape} != ({N}, {K//16})"

        z[key] = w_packed.contiguous()
        z[key + ".nf4_b_scale"] = w_bs.contiguous()
        z[key + ".nvfp4_t_scale"] = w_ts.contiguous()

        quant_count += 1
        if quant_count <= 4:
            print(f"  Quantized {key}: [{N},{K}] -> packed[{N},{K//2}] bs[{N},{K//16}] ts={w_ts.item():.6f}")

    print(f"  Quantized {quant_count} tensors, skipped {skip_count}")

    meta = {
        "v": 1,
        "r": [
            [0, L-1, 4, 2],
            [0, L-1, 5, 2],
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

    print("\n=== Verification ===")
    z2 = torch.load(OUT_PATH, map_location="cpu", mmap=True)
    z_orig = torch.load(ORIG_PATH, map_location="cpu", mmap=True)

    for prefix in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight"]:
        w_q = z2[prefix]
        w_bs = z2[prefix + ".nf4_b_scale"]
        w_ts = z2[prefix + ".nvfp4_t_scale"]
        w_o = z_orig[prefix]
        print(f"  {prefix}: quant={list(w_q.shape)} bs={list(w_bs.shape)} ts={w_ts.item():.6f} orig={list(w_o.shape)}")

    att_w = z2["blocks.0.att.key.weight"]
    print(f"  blocks.0.att.key.weight: {list(att_w.shape)} {att_w.dtype} (should be bf16)")

    print("\nDone!")

if __name__ == "__main__":
    main()
