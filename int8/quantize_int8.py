#!/usr/bin/env python3
# coding=utf-8
"""INT8 quantization tool for RWKV-7 models.

Two schemes:
  1. Per-tensor symmetric (W8A8): simple, hardware-friendly
  2. Affine (MM8-style): per-row + per-column, higher precision

Usage:
  python quantize_int8.py --model /path/to/model.pth --output /path/to/quantized.pth --scheme per_tensor
  python quantize_int8.py --model /path/to/model.pth --output /path/to/quantized.pth --scheme affine
"""
from __future__ import annotations
import argparse
import os
import sys
import torch

# Same component classification as FP8 quantizer
QUANTIZED_COMPONENTS = {"att_r", "att_k", "att_v", "att_o", "ffn_k", "ffn_v"}


def classify_weight(key: str, num_layers: int):
    """Classify a weight key into (layer, component) or None."""
    parts = key.split(".")
    if len(parts) >= 4 and parts[0] == "blocks":
        layer = int(parts[1])
        if layer < 0 or layer >= num_layers:
            return None
        group = parts[2]
        name = parts[3]
        if group == "att":
            mapping = {"receptance": "att_r", "key": "att_k", "value": "att_v", "output": "att_o"}
            comp = mapping.get(name)
        elif group == "ffn":
            mapping = {"key": "ffn_k", "value": "ffn_v"}
            comp = mapping.get(name)
        else:
            comp = None
        return (layer, comp) if comp in QUANTIZED_COMPONENTS else None
    return None


# ---------------------------------------------------------------------------
# Scheme 1: Per-Tensor Symmetric INT8
# ---------------------------------------------------------------------------

INT8_MAX = 127.0


def quantize_to_int8_per_tensor(w: torch.Tensor):
    """Quantize weight to per-tensor symmetric int8.

    Returns (w_int8, scale) where:
      w_int8: int8 tensor, same shape as w
      scale: scalar float32, W_approx = w_int8 * scale
    """
    w = w.float()
    amax = w.abs().max()
    scale = (amax / INT8_MAX).clamp(min=1e-12)
    w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_q, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Scheme 2: Affine INT8 (MM8-style, per-row + per-column)
# ---------------------------------------------------------------------------

def quantize_to_int8_affine(w: torch.Tensor):
    """Quantize weight to affine int8 (MM8-style).

    Uses per-row + per-column dual affine:
      W_approx = (u8 + 0.5) * ry * rx + my + mx

    Returns dict with:
      w_u8: uint8 [N, M] (quantized weight)
      mx:   [M]    per-column offset
      rx:   [M]    per-column scale (stored /16)
      my:   [N, 1] per-row offset
      ry:   [N, 1] per-row scale (stored /16)
    """
    w = w.float()  # [N, M]
    N, M = w.shape

    # Per-row offset (dim=1 -> along M)
    my = w.amin(dim=1, keepdim=True)  # [N, 1]
    w = w - my

    # Per-column offset (dim=0 -> along N)
    mx = w.amin(dim=0, keepdim=True)  # [1, M]
    w = w - mx

    # Per-column scale
    rx = w.amax(dim=0, keepdim=True).clamp(min=1e-12)  # [1, M]
    w = w / rx

    # Per-row scale
    ry = w.amax(dim=1, keepdim=True).clamp(min=1e-12)  # [N, 1]
    w = w / ry

    # Quantize to uint8 (256 levels, 0-255)
    w_u8 = (w * 256.0).floor().clamp(0, 255).to(torch.uint8)

    # Absorb 256-level factor into scales (stored /16 for historical compatibility)
    rx_stored = (rx / 16.0).squeeze(0).to(torch.float32)  # [M]
    ry_stored = (ry / 16.0).squeeze(1).to(torch.float32)  # [N]

    return {
        "w_u8": w_u8,
        "mx": mx.squeeze(0).to(torch.float32),    # [M]
        "rx": rx_stored,                            # [M]
        "my": my.squeeze(1).to(torch.float32),    # [N]
        "ry": ry_stored,                            # [N]
    }


# ---------------------------------------------------------------------------
# Main quantization flow
# ---------------------------------------------------------------------------

def quantize_model(model_path: str, output_path: str, scheme: str = "per_tensor"):
    """Load model, quantize 6 linear components to INT8, save."""
    print(f"Loading model from {model_path}...")
    z = torch.load(model_path, map_location="cpu", weights_only=True)

    # Detect number of layers
    num_layers = 0
    for key in z:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "blocks":
            layer = int(parts[1])
            num_layers = max(num_layers, layer + 1)
    print(f"Detected {num_layers} layers")

    # Determine non-quantized prefixes
    non_quant_prefixes = set()
    for key in z:
        if classify_weight(key, num_layers) is None:
            non_quant_prefixes.add(key)

    # Quantize
    quant_count = 0
    keys_to_delete = []

    for key in list(z.keys()):
        info = classify_weight(key, num_layers)
        if info is None:
            continue
        layer, comp = info
        w = z[key].float()

        if scheme == "per_tensor":
            w_int8, scale = quantize_to_int8_per_tensor(w)
            z[key + ".int8_weight"] = w_int8
            z[key + ".int8_scale"] = scale
        elif scheme == "affine":
            affine = quantize_to_int8_affine(w)
            for suffix, tensor in affine.items():
                z[key + ".int8_" + suffix] = tensor
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        keys_to_delete.append(key)
        quant_count += 1
        if quant_count % 24 == 0:
            print(f"  Quantized {quant_count} weights...")

    # Delete original weights
    for key in keys_to_delete:
        del z[key]

    # Add meta
    z["meta"] = {
        "quantization": f"int8_{scheme}",
        "scheme": scheme,
        "components": list(QUANTIZED_COMPONENTS),
        "non_quant_prefixes": list(non_quant_prefixes),
        "num_layers": num_layers,
        "dtype": "int8",
    }

    print(f"\nQuantized {quant_count} weights to INT8 ({scheme})")
    print(f"Saving to {output_path}...")
    torch.save(z, output_path)

    # Report sizes
    orig_size = os.path.getsize(model_path) / 1e9
    quant_size = os.path.getsize(output_path) / 1e9
    print(f"\nOriginal:  {orig_size:.2f} GB")
    print(f"Quantized: {quant_size:.2f} GB")
    print(f"Ratio:     {quant_size / orig_size:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize RWKV-7 model to INT8")
    parser.add_argument("--model", required=True, help="Path to original model .pth")
    parser.add_argument("--output", required=True, help="Path to quantized output .pth")
    parser.add_argument("--scheme", default="per_tensor",
                        choices=["per_tensor", "affine"],
                        help="Quantization scheme (default: per_tensor)")
    args = parser.parse_args()
    quantize_model(args.model, args.output, args.scheme)
