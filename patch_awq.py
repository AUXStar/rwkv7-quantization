#!/usr/bin/env python3
"""Patch nvfp4_ops.py and rwkv7_fast_v3a.py to support AWQ scaling."""

NVFP4_OPS = "/home/njzy/test/Albatross/faster3a_2605/nvfp4_ops.py"
V3A = "/home/njzy/test/Albatross/faster3a_2605/rwkv7_fast_v3a.py"

# ============================================================================
# Patch 1: nvfp4_ops.py
# ============================================================================
with open(NVFP4_OPS, "r") as f:
    content = f.read()

# 1a. load_nvfp4_weight: add AWQ scale loading
old1 = '    del z[key + ".nf4_b_scale"]\n    del z[key + ".nvfp4_t_scale"]\n\n    if swizzle:'
new1 = '    del z[key + ".nf4_b_scale"]\n    del z[key + ".nvfp4_t_scale"]\n\n    # Load AWQ scale if present\n    awq_scale = None\n    if (key + ".awq_scale") in z:\n        awq_scale = z[key + ".awq_scale"].to(device=dev)  # [K] float32\n        del z[key + ".awq_scale"]\n\n    if swizzle:'

if old1 in content:
    content = content.replace(old1, new1)
    print("  Patched load_nvfp4_weight: AWQ scale loading")
else:
    print("  WARNING: load_nvfp4_weight pattern not found!")

# 1b. load_nvfp4_weight: return AWQ scale in result dict
old2 = '    return {\n        "weight": w,\n        "block_scale": bs_out,\n        "tensor_scale": ts,\n        "qtype": qtype,\n    }'
new2 = '    result = {\n        "weight": w,\n        "block_scale": bs_out,\n        "tensor_scale": ts,\n        "qtype": qtype,\n    }\n    if awq_scale is not None:\n        result["awq_scale"] = awq_scale\n    return result'

if old2 in content:
    content = content.replace(old2, new2)
    print("  Patched load_nvfp4_weight: return AWQ scale")
else:
    print("  WARNING: return pattern not found!")

# 1c. linear_nvfp4_w4a16: add AWQ inverse scaling
old3 = '    # Reshape input\n    orig_shape = x.shape\n    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()\n\n    # FP16 GEMM'
new3 = '    # Reshape input\n    orig_shape = x.shape\n    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()\n\n    # Apply AWQ inverse scaling: x\' = x / s\n    awq_scale = weight_info.get("awq_scale", None)\n    if awq_scale is not None:\n        x_2d = x_2d / awq_scale.to(x_2d.dtype)\n\n    # FP16 GEMM'

if old3 in content:
    content = content.replace(old3, new3)
    print("  Patched linear_nvfp4_w4a16: AWQ inverse scaling")
else:
    print("  WARNING: linear_nvfp4_w4a16 pattern not found!")

with open(NVFP4_OPS, "w") as f:
    f.write(content)

# ============================================================================
# Patch 2: rwkv7_fast_v3a.py - skip .awq_scale keys
# ============================================================================
with open(V3A, "r") as f:
    content = f.read()

old4 = 'or key.endswith(".fp8_scale"):'
new4 = 'or key.endswith(".fp8_scale") or key.endswith(".awq_scale"):'

if old4 in content:
    content = content.replace(old4, new4)
    print("  Patched rwkv7_fast_v3a.py: skip .awq_scale keys")
else:
    print("  WARNING: skip pattern not found in rwkv7_fast_v3a.py!")

with open(V3A, "w") as f:
    f.write(content)

print("\nAll patches applied successfully!")
