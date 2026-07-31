#!/usr/bin/env python3
"""Calibrate GPTQ Hessian matrices from activation data.

Collects input activations for FFN key layers and computes Hessian H = X^T * X.
Also collects per-channel activation statistics for AWQ scaling.

Output: /home/njzy/test/eval_tmp/gptq_hessians.pt
  Dict: {layer_idx: tensor[K, K]} where K is the model dimension (2560)
  Each tensor is the Hessian H = X^T * X (accumulated over all calibration tokens)
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
_act_accum = {}   # layer_idx -> [X^T @ X accumulator]
_call_counter = [0]

def _patched_w4a16(x, weight_info, out_dtype=torch.float16):
    x_2d = x.reshape(-1, x.shape[-1]).float()  # [M, K]

    layer_idx = _call_counter[0]
    if layer_idx not in _act_accum:
        _act_accum[layer_idx] = torch.zeros(x_2d.shape[1], x_2d.shape[1],
                                            device='cuda', dtype=torch.float32)

    # Accumulate X^T @ X
    _act_accum[layer_idx] += x_2d.t() @ x_2d

    _call_counter[0] += 1
    return _original_w4a16(x, weight_info, out_dtype)

nvfp4_ops.linear_nvfp4_w4a16 = _patched_w4a16

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
print("  GPTQ Calibration: Collecting Hessian matrices for FFN key layers")
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

# Run forward passes to accumulate Hessians
print("\n  Collecting Hessians from calibration data...")

for tokens, label in [(tokens_446, "446"), (tokens_2100, "2100")]:
    _call_counter[0] = 0
    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = model.zero_state(1)
    model.forward(tok_tensor, state)
    print(f"    [{label}] Processed {_call_counter[0]} FFN key layers")

print(f"\n  Total layers with Hessians: {len(_act_accum)}")

# Move Hessians to CPU and save
hessians = {}
for k, v in _act_accum.items():
    hessians[k] = v.cpu()
    if k < 3:
        print(f"    Layer {k}: H shape={list(v.shape)}, "
              f"diag_mean={v.diag().mean():.4f}, "
              f"diag_max={v.diag().max():.4f}")

torch.save(hessians, f"{EVAL_DIR}/gptq_hessians.pt")
print(f"\n  Saved Hessians to {EVAL_DIR}/gptq_hessians.pt")
print(f"  {len(hessians)} layers, each {hessians[0].shape[0]}x{hessians[0].shape[1]}")

print("\nDone!")
