#!/usr/bin/env python3
"""Debug GPTQ: test on a single layer to find the bug."""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")

import torch
import math

# Load original weight and Hessian
z = torch.load("/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth",
               map_location="cpu", mmap=True)
w = z["blocks.0.ffn.key.weight"].float()  # [10240, 2560]
print(f"Weight: shape={w.shape}, min={w.min():.6f}, max={w.max():.6f}, mean={w.mean():.6f}")

# Load Hessian
H = torch.load("/home/njzy/test/eval_tmp/gptq_hessians.pt", map_location="cpu")[0]  # [2560, 2560]
print(f"Hessian: shape={H.shape}")
print(f"  diag: min={H.diag().min():.4f}, max={H.diag().max():.4f}, mean={H.diag().mean():.4f}")
print(f"  cond: {torch.linalg.cond(H):.2f}")

# Load AWQ stats
act_stats = torch.load("/home/njzy/test/eval_tmp/awq_act_stats.pt", map_location="cpu")
act = act_stats[0].float()
w_mean = w.abs().mean(dim=0)
alpha = 0.5
s = (act.clamp(min=1e-8) ** alpha) / (w_mean.clamp(min=1e-8) ** (1 - alpha))
s = s / s.mean()
print(f"AWQ scale: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")

# Apply AWQ
W = w * s.unsqueeze(0)
print(f"AWQ weight: min={W.min():.6f}, max={W.max():.6f}")

# Per-tensor scale
NVFP4_TS_DIVISOR = 2688.0
ts = W.abs().max() / NVFP4_TS_DIVISOR
print(f"Tensor scale: {ts:.8f}")
print(f"  Max representable: {6.0 * ts * 448:.4f} (weight max: {W.abs().max():.4f})")

# Simple GPTQ test: process one group
group_size = 128
damping = 0.01

# Add damping
H_damped = H.clone()
H_damped.diagonal().add_(damping * H.diag().mean())

g = 0
g_end = group_size

# Quantize first group (simple, no clip ratio)
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.015625

w_group = W[:, g:g_end]  # [N, 128]
n_blocks = group_size // 16
w_blocks = w_group.view(-1, n_blocks, 16)
block_amax = w_blocks.abs().amax(dim=2)

bs_scaled = block_amax / FP4_E2M1_MAX / ts
bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)

bs_f32 = bs_fp8.to(torch.float32)
eff_scale = ts * bs_f32
w_scaled = w_blocks / eff_scale.unsqueeze(-1)
w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

# Round to FP4
sign = torch.where(w_scaled < 0, 1, 0).to(torch.uint8)
a = w_scaled.abs()
code = torch.where(a <= 0.25, 0,
       torch.where(a < 0.75, 1,
       torch.where(a <= 1.25, 2,
       torch.where(a < 1.75, 3,
       torch.where(a <= 2.5, 4,
       torch.where(a < 3.5, 5,
       torch.where(a <= 5.0, 6, 7))))))).to(torch.uint8)
fp4_idx = sign * 8 + code

fp4_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32)
fp4_val = fp4_table[fp4_idx.long()]
w_deq = fp4_val * eff_scale.unsqueeze(-1)
w_quant = w_deq.view(-1, group_size)

# Error
err = W[:, g:g_end] - w_quant
print(f"\nGroup 0-128:")
print(f"  err: min={err.min():.6f}, max={err.max():.6f}, norm={err.norm():.4f}")
print(f"  w_quant: min={w_quant.min():.6f}, max={w_quant.max():.6f}")

# GPTQ update
H_block = H_damped[g:g_end, g:g_end]  # [128, 128]
H_cross = H_damped[g:g_end, g_end:]    # [128, 2432]

print(f"  H_block: min={H_block.min():.4f}, max={H_block.max():.4f}")
print(f"  H_block diag: min={H_block.diag().min():.4f}, max={H_block.diag().max():.4f}")

# Check if H_block is positive definite
try:
    L = torch.linalg.cholesky(H_block)
    print(f"  Cholesky: OK, L min={L.min():.6f}, max={L.max():.6f}")

    # Solve
    update = torch.cholesky_solve(H_cross.unsqueeze(0), L.unsqueeze(0)).squeeze(0)
    print(f"  update (cholesky): min={update.min():.6f}, max={update.max():.6f}, nan={torch.isnan(update).any()}")
except Exception as e:
    print(f"  Cholesky FAILED: {e}")
    update = torch.linalg.solve(H_block, H_cross)
    print(f"  update (solve): min={update.min():.6f}, max={update.max():.6f}, nan={torch.isnan(update).any()}")

# Apply update
W_update = W[:, g_end:] - err @ update
print(f"\n  W[:, g_end:] before: min={W[:, g_end:].min():.6f}, max={W[:, g_end:].max():.6f}")
print(f"  W[:, g_end:] after:  min={W_update.min():.6f}, max={W_update.max():.6f}")
print(f"  Change: min={(W_update - W[:, g_end:]).min():.6f}, max={(W_update - W[:, g_end:]).max():.6f}")
print(f"  Change norm: {(W_update - W[:, g_end:]).norm():.4f}")
print(f"  Original norm: {W[:, g_end:].norm():.4f}")

# Check ratio
ratio = (W_update - W[:, g_end:]).norm() / W[:, g_end:].norm()
print(f"  Change/Original ratio: {ratio:.4f}")

# Now test with larger damping
for damping in [0.1, 1.0, 10.0]:
    H_damped2 = H.clone()
    H_damped2.diagonal().add_(damping * H.diag().mean())
    H_block2 = H_damped2[g:g_end, g:g_end]
    H_cross2 = H_damped2[g:g_end, g_end:]

    try:
        update2 = torch.linalg.solve(H_block2, H_cross2)
        W_update2 = W[:, g_end:] - err @ update2
        ratio2 = (W_update2 - W[:, g_end:]).norm() / W[:, g_end:].norm()
        print(f"\n  damping={damping}: change/original={ratio2:.4f}, "
              f"update_range=[{update2.min():.4f}, {update2.max():.4f}]")
    except Exception as e:
        print(f"\n  damping={damping}: FAILED: {e}")
