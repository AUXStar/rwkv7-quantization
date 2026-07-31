#!/usr/bin/env python3
"""Quick quality benchmark: only PPL + Top-1, no speed benchmarks."""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")

import torch
import json
import math
import time

import rwkv7_fast_v3a as engine
from rwkv7_fast_v3a import RWKV7, load_extensions

EVAL_DIR = "/home/njzy/test/eval_tmp"

# Model path from command line
model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v8.pth"
label = sys.argv[2] if len(sys.argv) > 2 else "model"

engine.NVFP4_W4A16 = True
engine.FP8_W8A16 = False
engine.WKV_MODE = "fp16"
engine.EMB_DEVICE = "cpu"
engine.RKV_MODE = "off"
engine.CMIX_SPARSE = "no-fc"
engine.LOWRANK_WEIGHT = "both"
engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
engine.MODEL_PATH = model_path

print(f"  Loading {label}: {model_path}")
load_extensions(engine.WKV_MODE)
model = RWKV7()
vram = torch.cuda.memory_allocated() / 2**30

with open(f"{EVAL_DIR}/test_2100.json") as f:
    tokens_2100 = json.load(f)["tokens"]
logits_orig = torch.load(f"{EVAL_DIR}/logits_orig_2100.pt", map_location="cpu").float()

# Generate logits
tok_tensor = torch.tensor([tokens_2100], dtype=torch.long)
state = model.zero_state(1)
out = model.forward_all_logits(tok_tensor, state)
logits_q = out[0].float().cpu()

# Compare
n = min(logits_orig.shape[0], logits_q.shape[0])
lo, lq = logits_orig[:n], logits_q[:n]
targets = torch.tensor(tokens_2100[1:n+1], dtype=torch.long)
top1_agree = (lo.argmax(dim=-1) == lq.argmax(dim=-1)).float().mean().item()
ce_o = torch.nn.functional.cross_entropy(lo, targets, reduction="mean").item()
ce_q = torch.nn.functional.cross_entropy(lq, targets, reduction="mean").item()
ppl_o = math.exp(ce_o)
ppl_q = math.exp(ce_q)
ppl_delta = ppl_q - ppl_o

print(f"\n  {label}:")
print(f"    PPL: {ppl_q:.4f} (orig {ppl_o:.4f}, delta {ppl_delta:+.4f})")
print(f"    Top-1: {top1_agree*100:.2f}%")
print(f"    CE delta: {ce_q - ce_o:+.6f}")
print(f"    VRAM: {vram:.2f} GiB")
