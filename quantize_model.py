#!/usr/bin/env python3
"""Unified quantization toolchain for RWKV-7 models.

Usage:
    python quantize_model.py --model /path/to/model.pth --output /path/to/quantized.pth
    python quantize_model.py --model ... --scheme fp8

Produces a .pth file with:
- FP8 weights (float8_e4m3fn) + per-tensor scales (fp32)
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

# dtype codes
BF16 = 0
FP8 = 1

DTYPE_NAMES = {0: "bf16", 1: "fp8"}

# component codes
COMP_MAP = {
    "receptance": 0, "key": 1, "value": 2, "output": 3,
    "ffn_key": 4, "ffn_value": 5
}

ATT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")


# ============================================================================
# Quantization functions
# ============================================================================

def quantize_to_fp8(w, per_channel=False):
    """FP8 E4M3 quantization. per_channel=True -> per-column scale [N]."""
    if per_channel:
        amax = w.abs().amax(dim=0)
        scale = (amax / FP8_E4M3_MAX).float()
        scale = scale.clamp(min=1e-10)
        w_fp8 = (w.float() / scale.unsqueeze(0)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return w_fp8, scale
    amax = w.abs().max()
    scale = (amax / FP8_E4M3_MAX).float() if amax > 0 else torch.tensor(1.0, dtype=torch.float32)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


# ============================================================================
# Scheme definitions
# ============================================================================

def get_scheme_fp8():
    """All FP8 scheme."""
    return [
        [0, 999, 0, FP8],
        [0, 999, 1, FP8],
        [0, 999, 2, FP8],
        [0, 999, 3, FP8],
        [0, 999, 4, FP8],
        [0, 999, 5, FP8],
    ]


SCHEMES = {
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

    if len(parts) >= 4 and parts[2] == "att":
        comp_name = parts[3]
        if comp_name in COMP_MAP and key.endswith(".weight"):
            return (layer, COMP_MAP[comp_name])

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
    return BF16


# ============================================================================
# Main quantization pipeline
# ============================================================================

def quantize_model(model_path, output_path, scheme_name="fp8", device='cuda', _scheme_override=None):
    """Quantize a model according to the specified scheme."""
    print(f"Loading model: {model_path}", flush=True)
    z = torch.load(model_path, map_location="cpu", mmap=True)

    layer_keys = [k for k in z.keys() if k.startswith("blocks.")]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    print(f"Model: {num_layers} layers", flush=True)

    if _scheme_override is not None:
        rules = _scheme_override
        scheme_name = "custom"
    else:
        scheme_fn = SCHEMES.get(scheme_name)
        if scheme_fn is None:
            print(f"Unknown scheme: {scheme_name}. Available: {list(SCHEMES.keys())}")
            return
        rules = scheme_fn()

    print(f"\nQuantization scheme: {scheme_name}", flush=True)
    print(f"{'Layer':<8} {'key':<12} {'value':<12} {'rec':<12} {'out':<12} {'ffn_k':<14} {'ffn_v':<12}", flush=True)
    for layer in range(num_layers):
        dtypes = {}
        for comp in range(6):
            dtypes[comp] = DTYPE_NAMES[get_dtype_for(rules, layer, comp)]
        print(f"L{layer:<7} {dtypes[1]:<12} {dtypes[2]:<12} {dtypes[0]:<12} {dtypes[3]:<12} {dtypes[4]:<14} {dtypes[5]:<12}", flush=True)

    stats = {"bf16": 0, "fp8": 0}
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

        if dtype == FP8:
            w_fp8, scale = quantize_to_fp8(w)
            z[key] = w_fp8.contiguous()
            z[key + ".fp8_scale"] = scale.contiguous()
            stats["fp8"] += 1
            total_quant += w_fp8.numel() * 1 + scale.numel() * 4

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
        "s": {"sd": "fp8e4m3", "td": "fp32"},
        "n": non_quant_prefixes,
        "stats": stats,
        "orig_size_gb": total_orig / 2**30,
        "quant_size_gb": total_quant / 2**30,
        "compression": total_orig / max(total_quant, 1),
    }
    z["meta"] = meta

    # Save
    print(f"\nQuantization complete in {elapsed:.1f}s", flush=True)
    print(f"  BF16: {stats['bf16']}, FP8: {stats['fp8']}", flush=True)
    print(f"  Original: {total_orig/2**30:.2f} GB -> Quantized: {total_quant/2**30:.2f} GB ({total_orig/max(total_quant,1):.1f}x compression)", flush=True)

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
    parser.add_argument("--scheme", default="fp8", choices=list(SCHEMES.keys()),
                        help="Quantization scheme (default: fp8)")
    parser.add_argument("--device", default="cuda", help="Device for quantization")
    args = parser.parse_args()

    quantize_model(args.model, args.output, args.scheme, args.device)


if __name__ == "__main__":
    main()
