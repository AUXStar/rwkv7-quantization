#!/usr/bin/env python3
# coding=utf-8
"""INT4 quantization tool for RWKV-7 models.

Three schemes:
  1. Per-tensor symmetric (W4A16): simplest, weight-only
  2. Affine (MM4-style): per-row + per-column, paired nibble packing
  3. Group-wise: per-group quantization with configurable group size

Usage:
  python quantize_int4.py --model /path/to/model.pth --output /path/to/quantized.pth --scheme per_tensor
  python quantize_int4.py --model /path/to/model.pth --output /path/to/quantized.pth --scheme affine
  python quantize_int4.py --model /path/to/model.pth --output /path/to/quantized.pth --scheme groupwise --group-size 128
"""
from __future__ import annotations
import argparse
import os
import torch

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
# Scheme 1: Per-Tensor Symmetric INT4 (W4A16)
# ---------------------------------------------------------------------------

INT4_MAX = 7.0  # symmetric: range [-8, 7]


def quantize_to_int4_per_tensor(w: torch.Tensor):
    """Quantize weight to per-tensor symmetric int4.

    Returns (packed, scale) where:
      packed: uint8 [N, M//2] (two int4 values packed per byte)
      scale: scalar float32, W_approx = unpack(packed) * scale
    """
    w = w.float()  # [N, M]
    N, M = w.shape
    assert M % 2 == 0, f"Output dim must be even, got {M}"

    amax = w.abs().max()
    scale = (amax / INT4_MAX).clamp(min=1e-12)
    w_q = (w / scale).round().clamp(-8, 7).to(torch.int8)  # [N, M]

    # Pack two int4 into one uint8 along M (output dim)
    lo = (w_q[:, 0::2] & 0xF).to(torch.uint8)   # [N, M//2]
    hi = (w_q[:, 1::2] & 0xF).to(torch.uint8)   # [N, M//2]
    packed = lo | (hi << 4)  # [N, M//2]

    return packed, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Scheme 2: Affine INT4 (MM4-style, per-row + per-column)
# ---------------------------------------------------------------------------

def quantize_to_int4_affine(w: torch.Tensor):
    """Quantize weight to affine int4 (MM4-style).

    Paired nibble packing along output dimension:
      packed[n, b] = u4[n, 2b] | (u4[n, 2b+1] << 4)

    Returns dict with:
      packed:  uint8 [N, M_pad//2]
      mx:      [M_pad]     per-col offset
      rx:      [M_pad]     per-col scale (stored /4)
      my:      [N]         per-row offset
      ry:      [N]         per-row scale (stored /4)
      m_orig:  int         original M (before padding)
    """
    w = w.float()  # [N, M]
    N, M = w.shape

    # Pad M to even
    if M % 2 != 0:
        w = torch.nn.functional.pad(w, (0, 1))
        M_pad = M + 1
    else:
        M_pad = M

    # Per-row offset
    my = w.amin(dim=1, keepdim=True)  # [N, 1]
    w = w - my

    # Per-column offset
    mx = w.amin(dim=0, keepdim=True)  # [1, M_pad]
    w = w - mx

    # Per-column scale
    rx = w.amax(dim=0, keepdim=True).clamp(min=1e-12)  # [1, M_pad]
    w = w / rx

    # Per-row scale
    ry = w.amax(dim=1, keepdim=True).clamp(min=1e-12)  # [N, 1]
    w = w / ry

    # Quantize to uint4 (16 levels, 0-15)
    w_u4 = (w * 16.0).floor().clamp(0, 15).to(torch.uint8)  # [N, M_pad]

    # Pack paired nibbles
    packed = w_u4[:, 0::2] | (w_u4[:, 1::2] << 4)  # [N, M_pad//2]

    # Absorb 16-level factor into scales (stored /4)
    rx_stored = (rx / 4.0).squeeze(0).to(torch.float32)  # [M_pad]
    ry_stored = (ry / 4.0).squeeze(1).to(torch.float32)  # [N]

    return {
        "packed": packed,
        "mx": mx.squeeze(0).to(torch.float32),
        "rx": rx_stored,
        "my": my.squeeze(1).to(torch.float32),
        "ry": ry_stored,
        "m_orig": M,
    }


# ---------------------------------------------------------------------------
# Scheme 3: Group-wise INT4
# ---------------------------------------------------------------------------

def quantize_to_int4_groupwise(w: torch.Tensor, group_size: int = 128):
    """Quantize weight to group-wise int4.

    Each group of `group_size` elements along the input dim (K) gets
    its own (scale, zero_point) pair.

    Returns dict with:
      packed:    uint8 [N, M//2]  (paired nibble along M)
      scales:    [N, K//group_size]  per-group scale
      zeros:     [N, K//group_size]  per-group zero point
      group_size: int
    """
    w = w.float()  # [N, M] = [out, in]
    N, M = w.shape
    # Group along input dim (M for weight [out, in])
    K = M  # K = input dimension
    n_groups = K // group_size
    assert K % group_size == 0, f"Input dim {K} not divisible by group_size {group_size}"

    # Reshape to groups: [N, n_groups, group_size]
    w_grouped = w.reshape(N, n_groups, group_size)

    # Per-group min/max
    w_min = w_grouped.amin(dim=2, keepdim=True)  # [N, n_groups, 1]
    w_max = w_grouped.amax(dim=2, keepdim=True)  # [N, n_groups, 1]

    # Scale and zero point
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-12)  # [N, n_groups, 1]
    zero = w_min  # [N, n_groups, 1]

    # Quantize to uint4 (0-15)
    w_q = ((w_grouped - zero) / scale).round().clamp(0, 15).to(torch.uint8)  # [N, n_groups, group_size]
    w_q = w_q.reshape(N, M)  # back to [N, M]

    # Pack paired nibbles along M
    assert M % 2 == 0, f"Output dim must be even, got {M}"
    packed = w_q[:, 0::2] | (w_q[:, 1::2] << 4)  # [N, M//2]

    return {
        "packed": packed,
        "scales": scale.squeeze(2).to(torch.float32),   # [N, n_groups]
        "zeros": zero.squeeze(2).to(torch.float32),     # [N, n_groups]
        "group_size": group_size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def quantize_model(model_path: str, output_path: str, scheme: str = "per_tensor",
                   group_size: int = 128):
    """Load model, quantize 6 linear components to INT4, save."""
    print(f"Loading model from {model_path}...")
    z = torch.load(model_path, map_location="cpu", weights_only=True)

    num_layers = 0
    for key in z:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "blocks":
            layer = int(parts[1])
            num_layers = max(num_layers, layer + 1)
    print(f"Detected {num_layers} layers, scheme={scheme}")

    non_quant_prefixes = set()
    for key in z:
        if classify_weight(key, num_layers) is None:
            non_quant_prefixes.add(key)

    quant_count = 0
    keys_to_delete = []

    for key in list(z.keys()):
        info = classify_weight(key, num_layers)
        if info is None:
            continue
        w = z[key].float()

        if scheme == "per_tensor":
            packed, scale = quantize_to_int4_per_tensor(w)
            z[key + ".int4_packed"] = packed
            z[key + ".int4_scale"] = scale
        elif scheme == "affine":
            affine = quantize_to_int4_affine(w)
            for suffix, tensor in affine.items():
                if isinstance(tensor, int):
                    z[key + ".int4_" + suffix] = tensor
                else:
                    z[key + ".int4_" + suffix] = tensor
        elif scheme == "groupwise":
            gw = quantize_to_int4_groupwise(w, group_size)
            for suffix, tensor in gw.items():
                if isinstance(tensor, int):
                    z[key + ".int4_" + suffix] = tensor
                else:
                    z[key + ".int4_" + suffix] = tensor
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        keys_to_delete.append(key)
        quant_count += 1
        if quant_count % 24 == 0:
            print(f"  Quantized {quant_count} weights...")

    for key in keys_to_delete:
        del z[key]

    z["meta"] = {
        "quantization": f"int4_{scheme}",
        "scheme": scheme,
        "group_size": group_size if scheme == "groupwise" else 0,
        "components": list(QUANTIZED_COMPONENTS),
        "non_quant_prefixes": list(non_quant_prefixes),
        "num_layers": num_layers,
        "dtype": "int4",
    }

    print(f"\nQuantized {quant_count} weights to INT4 ({scheme})")
    print(f"Saving to {output_path}...")
    torch.save(z, output_path)

    orig_size = os.path.getsize(model_path) / 1e9
    quant_size = os.path.getsize(output_path) / 1e9
    print(f"\nOriginal:  {orig_size:.2f} GB")
    print(f"Quantized: {quant_size:.2f} GB")
    print(f"Ratio:     {quant_size / orig_size:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize RWKV-7 model to INT4")
    parser.add_argument("--model", required=True, help="Path to original model .pth")
    parser.add_argument("--output", required=True, help="Path to quantized output .pth")
    parser.add_argument("--scheme", default="per_tensor",
                        choices=["per_tensor", "affine", "groupwise"],
                        help="Quantization scheme (default: per_tensor)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for groupwise scheme (default: 128)")
    args = parser.parse_args()
    quantize_model(args.model, args.output, args.scheme, args.group_size)
