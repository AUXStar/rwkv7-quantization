#!/usr/bin/env python3
"""Check ffn_key residual scale type in X5 model."""
import sys
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
import torch
import rwkv7_fast_v3a as engine
from rwkv7_fast_v3a import RWKV7, load_extensions
engine.NVFP4_W4A16 = False; engine.FP8_W8A16 = False; engine.FUSED_GEMM = True
engine.WKV_MODE = "fp16"; engine.EMB_DEVICE = "cpu"; engine.RKV_MODE = "off"
engine.CMIX_SPARSE = "no-fc"; engine.LOWRANK_WEIGHT = "both"
engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
engine.MODEL_PATH = "/home/njzy/model/rwkv7-7.2b-X5.pth"
load_extensions(engine.WKV_MODE)
m = RWKV7()
wfk = m.z["blocks.1.ffn.key.weight"]
print("ffn_key keys:", list(wfk.keys()))
print("has res_block_scale:", "res_block_scale" in wfk)
print("has res_fp8_scale:", "res_fp8_scale" in wfk)
if "res_fp8_scale" in wfk:
    print("res_fp8_scale:", wfk["res_fp8_scale"])
if "res_block_scale" in wfk:
    print("res_block_scale shape:", wfk["res_block_scale"].shape)
print("res_fp8 dtype:", wfk["res_fp8"].dtype)
print("res_fp8 shape:", wfk["res_fp8"].shape)
