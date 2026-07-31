#!/usr/bin/env python3
"""Unified quantization toolchain for RWKV-7 models.

Usage:
    python quantize_model.py --model /path/to/model.pth --output /path/to/quantized.pth
    python quantize_model.py --model ... --scheme experimental  # all NVFP4

Produces a .pth file with:
- NVFP4 packed weights (uint8) + block scales (fp8) + tensor scales (fp32) + AWQ scales (fp32)
- FP8 weights (float8_e4m3fn) + per-tensor scales (fp32)
- NVFP4+FP8 residual (packed + scales + residual_fp8 + residual_scale)
- meta dict with quantization rules
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

# ============================================================================
# Constants
# ============================================================================
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.015625
FP4_E2M1_MAX = 6.0
NVFP4_TS_DIVISOR = 448.0 * 6.0
BLOCK_SIZE = 16
ALPHA = 0.5  # AWQ alpha

CLIP_RATIOS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

_FP4_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

# dtype codes
BF16 = 0
FP8 = 1
NVFP4 = 2
NVFP4_RES = 3  # NVFP4 + FP8 residual

DTYPE_NAMES = {0: "bf16", 1: "fp8", 2: "nvfp4", 3: "nvfp4+res"}

# component codes
COMP_MAP = {
    "receptance": 0, "key": 1, "value": 2, "output": 3,
    "ffn_key": 4, "ffn_value": 5
}

ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")


# ============================================================================
# Quantization functions
# ============================================================================

def _round_to_fp4(x_scaled):
    sign = torch.where(x_scaled < 0, 1, 0).to(torch.uint8)
    a = x_scaled.abs()
    code = torch.where(a <= 0.25, 0,
           torch.where(a < 0.75, 1,
           torch.where(a <= 1.25, 2,
           torch.where(a < 1.75, 3,
           torch.where(a <= 2.5, 4,
           torch.where(a < 3.5, 5,
           torch.where(a <= 5.0, 6, 7))))))).to(torch.uint8)
    return sign * 8 + code


def compute_awq_scale(w, act_stats=None, alpha=ALPHA, device='cuda'):
    """Compute AWQ channel scale. Uses weight-based heuristic if no act_stats."""
    w_dev = w.to(device=device).float()
    if act_stats is not None:
        act_dev = act_stats.to(device=device).float()
    else:
        # Weight-based heuristic: column abs mean as activation proxy
        act_dev = w_dev.abs().mean(dim=0)
    w_mean = w_dev.abs().mean(dim=0)
    s = (act_dev.clamp(min=1e-8) ** alpha) / (w_mean.clamp(min=1e-8) ** (1 - alpha))
    s = s / s.mean()
    return s.cpu()


def quantize_nvfp4(w, awq_scale, per_channel_ts=False, device='cuda'):
    """NVFP4 quantization with AWQ + clip ratio search.
    Returns: packed, block_scale, tensor_scale, dequant_weight
    """
    N, K = w.shape
    n_blocks = K // BLOCK_SIZE
    w_orig = w.to(device=device).float()
    s = awq_scale.to(device=device).float()
    W = w_orig * s.unsqueeze(0)

    if per_channel_ts:
        ts = (W.abs().amax(dim=1) / NVFP4_TS_DIVISOR).clamp(min=1e-10)
    else:
        ts = W.abs().max() / NVFP4_TS_DIVISOR
        if ts.item() == 0:
            ts = torch.tensor(1.0, dtype=torch.float32, device=device)

    ts_col = ts.unsqueeze(1) if per_channel_ts else ts
    w_blocks = W.view(N, n_blocks, BLOCK_SIZE)
    block_amax = w_blocks.abs().amax(dim=2)

    best_mse = torch.full((N, n_blocks), float('inf'), device=device)
    best_fp4_idx = None
    best_bs_fp8 = None
    fp4_table = _FP4_VALUES.to(device)

    for ratio in CLIP_RATIOS:
        bs_scaled = (block_amax * ratio / FP4_E2M1_MAX / ts_col).clamp(FP8_E4M3_MIN_NORMAL, FP8_E4M3_MAX)
        bs_fp8 = bs_scaled.to(torch.float8_e4m3fn)
        bs_f32 = bs_fp8.to(torch.float32)
        eff_scale = ts_col * bs_f32
        w_scaled = (w_blocks / eff_scale.unsqueeze(-1)).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
        fp4_idx = _round_to_fp4(w_scaled)
        fp4_val = fp4_table[fp4_idx.long()]
        w_deq = fp4_val * eff_scale.unsqueeze(-1)
        block_mse = ((w_blocks - w_deq) ** 2).mean(dim=2)
        improved = block_mse < best_mse
        best_mse = torch.where(improved, block_mse, best_mse)
        if best_fp4_idx is None:
            best_fp4_idx = fp4_idx.clone()
            best_bs_fp8 = bs_fp8.clone()
        else:
            mask = improved.unsqueeze(-1)
            best_fp4_idx = torch.where(mask, fp4_idx, best_fp4_idx)
            best_bs_fp8 = torch.where(improved, bs_fp8, best_bs_fp8)

    w_quant = (fp4_table[best_fp4_idx.long()] * (ts_col * best_bs_fp8.to(torch.float32)).unsqueeze(-1)).view(N, K)
    fp4_flat = best_fp4_idx.view(N, K // 2, 2)
    packed = (fp4_flat[:, :, 1] * 16 + fp4_flat[:, :, 0]).to(torch.uint8)
    return packed.cpu(), best_bs_fp8.cpu(), ts.cpu(), w_quant.cpu()


def quantize_to_fp8(w):
    """Per-tensor FP8 E4M3 quantization."""
    amax = w.abs().max()
    scale = (amax / FP8_E4M3_MAX).float() if amax > 0 else torch.tensor(1.0, dtype=torch.float32)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def quantize_nvfp4_with_residual(w, awq_scale, device='cuda'):
    """NVFP4 + FP8 residual quantization (v12 scheme).
    Returns: packed, bs, ts, awq_scale, res_fp8, res_scale
    """
    packed, bs, ts, w_nvfp4_deq = quantize_nvfp4(w, awq_scale, per_channel_ts=False, device=device)
    # Residual in AWQ-scaled space
    W_awq = (w.to(device).float() * awq_scale.to(device).float().unsqueeze(0)).cpu()
    residual = W_awq - w_nvfp4_deq
    res_fp8, res_scale = quantize_to_fp8(residual)
    return packed, bs, ts, awq_scale, res_fp8, res_scale


# ============================================================================
# Scheme definitions
# ============================================================================

def get_scheme_1_5b():
    """Updated scheme for 1.5B (24 layers) based on #2-#6 experiments.
    
    Findings:
    - #4: att.key NVFP4 OK (99.67%), att.value more sensitive → keep FP8
    - #5: L0 value FP8 near-lossless (99.95%), no BF16 needed
    - #2 v12: FFN key NVFP4+FP8 residual → 99.05% Top-1
    - #6: no state divergence over 8192 tokens
    """
    return [
        # [layer_start, layer_end, comp, dtype]
        # Attention key: FP8 at edges, NVFP4 in middle
        [0,  3,  1, FP8],      # L0-3 key FP8
        [4,  19, 1, NVFP4],    # L4-19 key NVFP4
        [20, 23, 1, FP8],      # L20-23 key FP8
        # Attention value: FP8 everywhere (more sensitive than key)
        [0,  23, 2, FP8],
        # Receptance + output: NVFP4 everywhere (low sensitivity)
        [0,  23, 0, NVFP4],
        [0,  23, 3, NVFP4],
        # FFN key: NVFP4+FP8 residual everywhere (v12)
        [0,  23, 4, NVFP4_RES],
        # FFN value: FP8 everywhere
        [0,  23, 5, FP8],
    ]


def get_scheme_2_9b():
    """Scheme for 2.9B (32 layers), scaled from 1.5B findings."""
    return [
        # Attention key: FP8 at edges, NVFP4 in middle
        [0,  3,  1, FP8],      # L0-3 key FP8
        [4,  27, 1, NVFP4],    # L4-27 key NVFP4
        [28, 31, 1, FP8],      # L28-31 key FP8
        # Attention value: FP8 everywhere
        [0,  31, 2, FP8],
        # Receptance + output: NVFP4 everywhere
        [0,  31, 0, NVFP4],
        [0,  31, 3, NVFP4],
        # FFN key: NVFP4+FP8 residual
        [0,  31, 4, NVFP4_RES],
        # FFN value: FP8
        [0,  31, 5, FP8],
    ]


def get_scheme_experimental():
    """All NVFP4 scheme for testing worst case."""
    return [
        [0, 999, 0, NVFP4],
        [0, 999, 1, NVFP4],
        [0, 999, 2, NVFP4],
        [0, 999, 3, NVFP4],
        [0, 999, 4, NVFP4],
        [0, 999, 5, NVFP4],
    ]


def get_scheme_fp8():
    """All FP8 scheme for testing."""
    return [
        [0, 999, 0, FP8],
        [0, 999, 1, FP8],
        [0, 999, 2, FP8],
        [0, 999, 3, FP8],
        [0, 999, 4, FP8],
        [0, 999, 5, FP8],
    ]


SCHEMES = {
    "1.5b": get_scheme_1_5b,
    "2.9b": get_scheme_2_9b,
    "experimental": get_scheme_experimental,
    "fp8": get_scheme_fp8,
}


# ============================================================================
# Weight classification
# ============================================================================

def classify_weight(key, num_layers):
    """Classify a weight key into (layer, component) or None if not quantizable.
    
    Returns: (layer_idx, comp_code) or None
    """
    if not key.startswith("blocks."):
        return None
    parts = key.split(".")
    layer = int(parts[1])
    if layer >= num_layers:
        return None
    
    # Check if it's an attention weight
    if len(parts) >= 4 and parts[2] == "att":
        comp_name = parts[3]
        if comp_name in COMP_MAP and key.endswith(".weight"):
            return (layer, COMP_MAP[comp_name])
    
    # Check if it's an FFN weight
    if len(parts) >= 4 and parts[2] == "ffn":
        comp_name = parts[3]
        if comp_name == "key" and key.endswith(".weight"):
            return (layer, COMP_MAP["ffn_key"])
        if comp_name == "value" and key.endswith(".weight"):
            return (layer, COMP_MAP["ffn_value"])
    
    return None


def get_dtype_for(rules, layer, comp):
    """Look up dtype from rules for a given (layer, component)."""
    for ls, le, c, dt in rules:
        if ls <= layer <= le and c == comp:
            return dt
    return BF16  # default: no quantization


# ============================================================================
# Main quantization pipeline
# ============================================================================

def quantize_model(model_path, output_path, scheme_name="1.5b", device='cuda'):
    """Quantize a model according to the specified scheme."""
    print(f"Loading model: {model_path}", flush=True)
    z = torch.load(model_path, map_location="cpu", mmap=True)
    
    # Detect number of layers
    layer_keys = [k for k in z.keys() if k.startswith("blocks.")]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    print(f"Model: {num_layers} layers", flush=True)
    
    # Get scheme
    scheme_fn = SCHEMES.get(scheme_name)
    if scheme_fn is None:
        print(f"Unknown scheme: {scheme_name}. Available: {list(SCHEMES.keys())}")
        return
    rules = scheme_fn()
    
    # Print scheme
    print(f"\nQuantization scheme: {scheme_name}", flush=True)
    print(f"{'Layer':<8} {'key':<12} {'value':<12} {'rec':<12} {'out':<12} {'ffn_k':<14} {'ffn_v':<12}", flush=True)
    for layer in range(num_layers):
        dtypes = {}
        for comp in range(6):
            dtypes[comp] = DTYPE_NAMES[get_dtype_for(rules, layer, comp)]
        print(f"L{layer:<7} {dtypes[1]:<12} {dtypes[2]:<12} {dtypes[0]:<12} {dtypes[3]:<12} {dtypes[4]:<14} {dtypes[5]:<12}", flush=True)
    
    # Quantize
    stats = {"bf16": 0, "fp8": 0, "nvfp4": 0, "nvfp4_res": 0}
    total_orig = 0
    total_quant = 0
    t0 = time.perf_counter()
    
    for key in list(z.keys()):
        if not torch.is_tensor(z[key]):
            continue
        w = z[key]
        if w.dim() != 2:
            continue
        
        result = classify_weight(key, num_layers)
        if result is None:
            continue
        
        layer, comp = result
        dtype = get_dtype_for(rules, layer, comp)
        
        orig_size = w.numel() * w.element_size()
        total_orig += orig_size
        
        if dtype == BF16:
            stats["bf16"] += 1
            total_quant += orig_size
            continue
        
        # Compute AWQ scale (weight-based heuristic)
        awq_s = compute_awq_scale(w, act_stats=None, alpha=ALPHA, device=device)
        
        if dtype == FP8:
            w_fp8, scale = quantize_to_fp8(w)
            z[key] = w_fp8.contiguous()
            z[key + ".fp8_scale"] = scale.contiguous()
            stats["fp8"] += 1
            total_quant += w_fp8.numel() * 1 + scale.numel() * 4  # 1 byte + scale
            
        elif dtype == NVFP4:
            packed, bs, ts, _ = quantize_nvfp4(w, awq_s, per_channel_ts=False, device=device)
            z[key] = packed.contiguous()
            z[key + ".nf4_b_scale"] = bs.contiguous()
            z[key + ".nvfp4_t_scale"] = ts.contiguous()
            z[key + ".awq_scale"] = awq_s.contiguous()
            stats["nvfp4"] += 1
            # packed: 0.5 byte/elem, bs: 1 byte/16 elems, ts: 4 bytes, awq: 4 bytes/elem
            total_quant += packed.numel() * 0.5 + bs.numel() * 1 + 4 + awq_s.numel() * 4
            
        elif dtype == NVFP4_RES:
            packed, bs, ts, awq_s, res_fp8, res_scale = quantize_nvfp4_with_residual(w, awq_s, device=device)
            z[key] = packed.contiguous()
            z[key + ".nf4_b_scale"] = bs.contiguous()
            z[key + ".nvfp4_t_scale"] = ts.contiguous()
            z[key + ".awq_scale"] = awq_s.contiguous()
            z[key + ".res_fp8"] = res_fp8.contiguous()
            z[key + ".res_fp8_scale"] = res_scale.contiguous()
            stats["nvfp4_res"] += 1
            total_quant += packed.numel() * 0.5 + bs.numel() * 1 + 4 + awq_s.numel() * 4
            total_quant += res_fp8.numel() * 1 + 4  # residual FP8
    
    elapsed = time.perf_counter() - t0
    
    # Generate meta
    non_quant_prefixes = [
        "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
        "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
        "w0", "w1", "w2", "a0", "a1", "a2",
        "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k"
    ]
    
    meta = {
        "v": 1,
        "scheme": scheme_name,
        "layers": num_layers,
        "r": rules,
        "s": {"blk": BLOCK_SIZE, "sd": "fp8e4m3", "td": "fp32"},
        "n": non_quant_prefixes,
        "stats": stats,
        "orig_size_gb": total_orig / 2**30,
        "quant_size_gb": total_quant / 2**30,
        "compression": total_orig / max(total_quant, 1),
    }
    z["meta"] = meta
    
    # Save
    print(f"\nQuantization complete in {elapsed:.1f}s", flush=True)
    print(f"  BF16: {stats['bf16']}, FP8: {stats['fp8']}, NVFP4: {stats['nvfp4']}, NVFP4+res: {stats['nvfp4_res']}", flush=True)
    print(f"  Original: {total_orig/2**30:.2f} GB → Quantized: {total_quant/2**30:.2f} GB ({total_orig/max(total_quant,1):.1f}x compression)", flush=True)
    
    # Clone all tensors to detach from mmap (prevents file bloat)
    for _k in list(z.keys()):
        if torch.is_tensor(z[_k]):
            z[_k] = z[_k].clone()
    torch.save(z, output_path)
    print(f"  Saved to: {output_path}", flush=True)
    
    return meta


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RWKV-7 unified quantization tool")
    parser.add_argument("--model", required=True, help="Input model path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--scheme", default="1.5b", choices=list(SCHEMES.keys()),
                        help="Quantization scheme (default: 1.5b)")
    parser.add_argument("--device", default="cuda", help="Device for quantization")
    args = parser.parse_args()
    
    quantize_model(args.model, args.output, args.scheme, args.device)


if __name__ == "__main__":
    main()
