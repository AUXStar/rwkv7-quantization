#!/usr/bin/env python3
"""Debug GPTQ v2: diagnose why GPTQ destroys the model.

Key hypothesis: The Hessian is collected from original input x,
but GPTQ operates on AWQ-scaled weight W' = W * s.
The correct Hessian should be H' = diag(1/s) @ H @ diag(1/s),
because the effective input for AWQ is x' = x / s.

This script tests:
1. Original Hessian (wrong) vs AWQ-transformed Hessian (correct)
2. Different damping ratios
3. Weight update magnitudes
"""
import sys
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")

import torch
import math

# Load original weight and Hessian
print("Loading data...")
z = torch.load("/home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth",
               map_location="cpu", mmap=True)
w = z["blocks.0.ffn.key.weight"].float().cuda()  # [10240, 2560]
print(f"Weight: shape={w.shape}, min={w.min():.6f}, max={w.max():.6f}")

H = torch.load("/home/njzy/test/eval_tmp/gptq_hessians.pt", map_location="cpu")[0].float().cuda()
print(f"Hessian: shape={H.shape}, diag_mean={H.diag().mean():.4f}")

# Load AWQ stats
act_stats = torch.load("/home/njzy/test/eval_tmp/awq_act_stats.pt", map_location="cpu")
act = act_stats[0].float().cuda()
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

# ============================================================================
# Test 1: Compare original Hessian vs AWQ-transformed Hessian
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: Hessian transformation comparison")
print("=" * 80)

# Original Hessian (WRONG for AWQ)
H_orig = H.clone()

# AWQ-transformed Hessian (CORRECT)
inv_s = (1.0 / s)
H_awq = H * (inv_s.unsqueeze(0) * inv_s.unsqueeze(1))

print(f"  Original H:  diag_mean={H_orig.diag().mean():.4f}, diag_min={H_orig.diag().min():.4f}")
print(f"  AWQ H:       diag_mean={H_awq.diag().mean():.4f}, diag_min={H_awq.diag().min():.4f}")
print(f"  Ratio AWQ/Orig diag_mean: {H_awq.diag().mean() / H_orig.diag().mean():.4f}")

# ============================================================================
# Test 2: GPTQ update magnitude with different Hessians and damping
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: GPTQ update magnitude (first 5 blocks)")
print("=" * 80)

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = 0.015625

fp4_table = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32, device='cuda')

def quantize_block(w_block, ts, clip_ratio=0.85):
    """Simple NVFP4 quantization with clip ratio."""
    N, K = w_block.shape
    bs = K // 16
    w_blocks = w_block.view(N, bs, 16)
    block_amax = w_blocks.abs().amax(dim=2)
    
    bs_scaled = block_amax * clip_ratio / FP4_E2M1_MAX / ts
    bs_scaled = bs_scaled.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)
    
    bs_f32 = bs_fp8.to(torch.float32)
    eff_scale = ts * bs_f32
    w_scaled = w_blocks / eff_scale.unsqueeze(-1)
    w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    
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
    fp4_val = fp4_table[fp4_idx.long()]
    w_deq = fp4_val * eff_scale.unsqueeze(-1)
    w_quant = w_deq.view(N, K)
    return w_quant

W_test = W.clone()
w_orig_max = W_test.abs().max().item()

for hessian_label, H_test in [("Original H", H_orig), ("AWQ H", H_awq)]:
    print(f"\n  --- {hessian_label} ---")
    # Normalize and damp
    H_norm = H_test.clone()
    H_scale = H_norm.diag().mean()
    H_norm = H_norm / H_scale
    H_norm.diagonal().add_(0.1)
    
    W_copy = W_test.clone()
    
    for b in range(5):
        col_start = b * 16
        col_end = col_start + 16
        
        # Quantize
        w_block = W_copy[:, col_start:col_end].contiguous()
        w_quant = quantize_block(w_block, ts, clip_ratio=0.85)
        err = W_copy[:, col_start:col_end] - w_quant
        
        if col_end < 2560:
            H_block = H_norm[col_start:col_end, col_start:col_end]
            H_cross = H_norm[col_start:col_end, col_end:]
            
            update = torch.linalg.solve(H_block, H_cross)
            delta = err @ update
            W_copy[:, col_end:] -= delta
            W_copy[:, col_end:] = W_copy[:, col_end:].clamp(-w_orig_max * 2, w_orig_max * 2)
            
            change_ratio = delta.norm().item() / W_copy[:, col_end:].norm().item()
            print(f"    Block {b}: err_norm={err.norm():.4f}, "
                  f"update_range=[{update.min():.4f}, {update.max():.4f}], "
                  f"change_ratio={change_ratio:.4f}")

# ============================================================================
# Test 3: Full layer GPTQ with AWQ Hessian, check weight drift
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Full layer GPTQ weight drift comparison")
print("=" * 80)

for hessian_label, H_test in [("Original H (WRONG)", H_orig), ("AWQ H (CORRECT)", H_awq)]:
    print(f"\n  --- {hessian_label} ---")
    H_norm = H_test.clone()
    H_scale = H_norm.diag().mean()
    H_norm = H_norm / H_scale
    H_norm.diagonal().add_(0.1)
    
    W_copy = W.clone()
    
    for b in range(160):  # 2560/16 = 160 blocks
        col_start = b * 16
        col_end = col_start + 16
        
        w_block = W_copy[:, col_start:col_end].contiguous()
        w_quant = quantize_block(w_block, ts, clip_ratio=0.85)
        err = W_copy[:, col_start:col_end] - w_quant
        
        if col_end < 2560:
            H_block = H_norm[col_start:col_end, col_start:col_end]
            H_cross = H_norm[col_start:col_end, col_end:]
            update = torch.linalg.solve(H_block, H_cross)
            W_copy[:, col_end:] -= err @ update
            W_copy[:, col_end:] = W_copy[:, col_end:].clamp(-w_orig_max * 2, w_orig_max * 2)
    
    # Compare final W_copy with original W
    drift = (W_copy - W).abs()
    print(f"    W drift: mean={drift.mean():.6f}, max={drift.max():.6f}")
    print(f"    W_copy range: [{W_copy.min():.6f}, {W_copy.max():.6f}]")
    print(f"    W original range: [{W.min():.6f}, {W.max():.6f}]")
    print(f"    Cosine sim: {torch.nn.functional.cosine_similarity(W.flatten().unsqueeze(0), W_copy.flatten().unsqueeze(0)).item():.8f}")

# ============================================================================
# Test 4: Try very high damping with AWQ Hessian
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: High damping with AWQ Hessian")
print("=" * 80)

for damp in [0.1, 1.0, 5.0, 10.0]:
    H_norm = H_awq.clone()
    H_scale = H_norm.diag().mean()
    H_norm = H_norm / H_scale
    H_norm.diagonal().add_(damp)
    
    W_copy = W.clone()
    
    for b in range(160):
        col_start = b * 16
        col_end = col_start + 16
        
        w_block = W_copy[:, col_start:col_end].contiguous()
        w_quant = quantize_block(w_block, ts, clip_ratio=0.85)
        err = W_copy[:, col_start:col_end] - w_quant
        
        if col_end < 2560:
            H_block = H_norm[col_start:col_end, col_start:col_end]
            H_cross = H_norm[col_start:col_end, col_end:]
            update = torch.linalg.solve(H_block, H_cross)
            W_copy[:, col_end:] -= err @ update
            W_copy[:, col_end:] = W_copy[:, col_end:].clamp(-w_orig_max * 2, w_orig_max * 2)
    
    drift = (W_copy - W).abs()
    cos_sim = torch.nn.functional.cosine_similarity(W.flatten().unsqueeze(0), W_copy.flatten().unsqueeze(0)).item()
    print(f"  damp={damp:5.1f}: drift_mean={drift.mean():.6f}, drift_max={drift.max():.6f}, cos_sim={cos_sim:.8f}")

print("\nDone!")
