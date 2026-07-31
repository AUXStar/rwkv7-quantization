#!/usr/bin/env python3
"""Calibrate AWQ scaling factors from activation statistics.

Collects per-channel activation magnitudes for FFN key inputs by running
the v3 quantized model on calibration tokens. The statistics are used to
compute AWQ scaling factors that protect important weight channels.

Output: /home/njzy/test/eval_tmp/awq_act_stats.pt
  Dict: {layer_idx: tensor[C]} where C is the model dimension (2560)
  Each tensor contains mean(|x_k|) for input channel k of that layer's FFN key.
"""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")

import torch
import json
import os
import nvfp4_ops

# ============================================================================
# Monkey-patch linear_nvfp4_w4a16 to collect input activations
# ============================================================================
_original_w4a16 = nvfp4_ops.linear_nvfp4_w4a16
_act_stats = {}  # layer_idx -> [running_sum, count]
_call_counter = [0]  # increments each call, resets per forward pass

def _patched_w4a16(x, weight_info, out_dtype=torch.float16):
    # Compute per-channel mean(|x_k|) incrementally
    # x shape: [B, T, K] or [M, K]
    x_2d = x.reshape(-1, x.shape[-1])  # [M, K]
    channel_abs_mean = x_2d.abs().mean(dim=0)  # [K]

    layer_idx = _call_counter[0]
    if layer_idx not in _act_stats:
        _act_stats[layer_idx] = channel_abs_mean.clone()
    else:
        # Running average (weighted by number of tokens)
        old = _act_stats[layer_idx]
        _act_stats[layer_idx] = (old + channel_abs_mean) / 2.0

    _call_counter[0] += 1
    return _original_w4a16(x, weight_info, out_dtype)

nvfp4_ops.linear_nvfp4_w4a16 = _patched_w4a16

# Also patch linear_quantized dispatcher since it calls linear_nvfp4_w4a16
# Actually, linear_quantized calls linear_nvfp4_w4a16 directly, so the patch
# above should work. But let me also patch it to reset the counter.

_original_dispatch = nvfp4_ops.linear_quantized

import rwkv7_fast_v3a as engine
from rwkv7_fast_v3a import RWKV7, load_extensions

HYBRID_MODEL = "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v3.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"

engine.NVFP4_W4A16 = True
engine.FP8_W8A16 = False
engine.WKV_MODE = "fp16"
engine.EMB_DEVICE = "cpu"
engine.RKV_MODE = "off"
engine.CMIX_SPARSE = "no-fc"
engine.LOWRANK_WEIGHT = "both"
engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}

print("=" * 80)
print("  AWQ Calibration: Collecting FFN key activation statistics")
print("=" * 80)

load_extensions(engine.WKV_MODE)
engine.MODEL_PATH = HYBRID_MODEL
model = RWKV7()
print(f"  Model loaded, VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

# Load calibration tokens
with open(f"{EVAL_DIR}/test_2100.json") as f:
    tokens_2100 = json.load(f)["tokens"]
with open(f"{EVAL_DIR}/test_446.json") as f:
    tokens_446 = json.load(f)["tokens"]

# Run forward passes to collect activations
# Use multiple texts for better statistics
print("\n  Collecting activations from calibration data...")

for tokens, label in [(tokens_446, "446"), (tokens_2100, "2100")]:
    _call_counter[0] = 0  # Reset counter for each forward pass
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = model.zero_state(1)
    model.forward(tok_tensor, state)
    print(f"    [{label}] Collected activations for {_call_counter[0]} layers")

# The counter goes: layer0_key, layer0_value(?), layer1_key, ...
# Wait - with NVFP4_W4A16=True, FFN value is BF16 (not quantized), so
# linear_nvfp4_w4a16 is only called for FFN key, not value.
# But attention uses FP8 W8A16, which calls linear_fp8_w8a16, not linear_nvfp4_w4a16.
# So the counter should be 32 (one per layer).

print(f"\n  Total layers collected: {len(_act_stats)}")
for i in sorted(_act_stats.keys())[:5]:
    s = _act_stats[i]
    print(f"    Layer {i}: shape={list(s.shape)}, min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}")

# Save activation statistics
# Convert to dict of layer_idx -> tensor
act_stats_clean = {}
for k, v in _act_stats.items():
    act_stats_clean[k] = v.cpu()

torch.save(act_stats_clean, f"{EVAL_DIR}/awq_act_stats.pt")
print(f"\n  Saved activation statistics to {EVAL_DIR}/awq_act_stats.pt")
print(f"  {len(act_stats_clean)} layers, each with {act_stats_clean[0].shape[0]} channels")

# Compute AWQ scaling factors preview
print("\n  AWQ scaling factor preview (alpha=0.5):")
# Load original weights for weight statistics
z_orig = torch.load("/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth",
                     map_location="cpu", mmap=True)

for layer_idx in [0, 1, 31]:
    w_key = f"blocks.{layer_idx}.ffn.key.weight"
    w = z_orig[w_key].float()  # [N, K]
    act = act_stats_clean[layer_idx].float()  # [K]

    # Weight per-channel mean: mean(|W[:, k]|) over output channels
    w_mean = w.abs().mean(dim=0)  # [K]

    # AWQ scale: s = (act^alpha) / (w_mean^(1-alpha))
    alpha = 0.5
    s = (act.clamp(min=1e-8) ** alpha) / (w_mean.clamp(min=1e-8) ** (1 - alpha))

    # Normalize: mean(s) = 1.0
    s = s / s.mean()

    print(f"    Layer {layer_idx}: s min={s.min():.4f}, max={s.max():.4f}, "
          f"std={s.std():.4f}, ratio max/min={s.max()/s.min():.2f}")

print("\nDone!")
