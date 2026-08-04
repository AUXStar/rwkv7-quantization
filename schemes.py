#!/usr/bin/env python3
# coding=utf-8
"""Scheme registry — single source of truth for all quantization formats.

Usage:
    from schemes import get_scheme, list_schemes, SCHEMES
    scheme = get_scheme("fp8")
    w_q, meta = scheme["quantize"](weight, **scheme.get("quantize_kwargs", {}))
"""
from __future__ import annotations
from typing import Callable, Any
import torch

# ---------------------------------------------------------------------------
# Quantization functions (import lazily to avoid circular deps)
# ---------------------------------------------------------------------------

def _q_fp8(w: torch.Tensor, **kw):
    """Per-tensor FP8 E4M3 quantization."""
    MAX = 448.0
    amax = w.abs().max().clamp(min=1e-12)
    scale = (amax / MAX).float()
    w_q = (w.float() / scale).clamp(-MAX, MAX).round().to(torch.float8_e4m3fn)
    return w_q, scale

def _q_fp8_perchannel(w: torch.Tensor, **kw):
    """Per-output-channel FP8 E4M3 quantization."""
    MAX = 448.0
    if w.dim() == 1:
        return _q_fp8(w)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / MAX).float()
    w_q = (w.float() / scale).clamp(-MAX, MAX).round().to(torch.float8_e4m3fn)
    return w_q, scale.squeeze(1)

def _q_int8_sym(w: torch.Tensor, **kw):
    """Per-tensor symmetric INT8."""
    MAX = 127.0
    amax = w.abs().max().clamp(min=1e-12)
    scale = (amax / MAX).float()
    w_q = (w.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return w_q, scale

def _q_int8_affine(w: torch.Tensor, **kw):
    """INT8 affine (MM8-style): per-row + per-column dual offset/scale."""
    wf = w.float()
    if wf.dim() != 2:
        wf = wf.reshape(1, -1)
    N, M = wf.shape
    my = wf.amin(dim=1, keepdim=True)
    w2 = wf - my
    mx = w2.amin(dim=0, keepdim=True)
    w2 = w2 - mx
    rx = w2.amax(dim=0, keepdim=True).clamp(min=1e-12)
    w2 = w2 / rx
    ry = w2.amax(dim=1, keepdim=True).clamp(min=1e-12)
    w2 = w2 / ry
    wq = (w2 * 256.0).floor().clamp(0, 255)
    w_approx = ((wq + 0.5) * (ry * 16.0) * (rx * 16.0) + my + mx)
    if w.dim() != 2:
        w_approx = w_approx.reshape(w.shape)
    return w_approx.to(w.dtype)

def _q_int4_sym(w: torch.Tensor, **kw):
    """Per-tensor symmetric INT4 (W4A16)."""
    MAX = 7.0
    amax = w.abs().max().clamp(min=1e-12)
    scale = (amax / MAX).float()
    w_q = (w.float() / scale).round().clamp(-8, 7)
    return (w_q * scale).to(w.dtype)

def _q_int4_gw(w: torch.Tensor, group_size: int = 128, **kw):
    """INT4 group-wise quantization (W4A16)."""
    import torch.nn.functional as F
    wf = w.float()
    shape = wf.shape
    if wf.dim() != 2:
        wf = wf.reshape(1, -1)
    N, M = wf.shape
    pad = (group_size - M % group_size) % group_size
    if pad:
        wf = F.pad(wf, (0, pad))
        M = wf.shape[1]
    n_groups = M // group_size
    wg = wf.reshape(N, n_groups, group_size)
    wmin = wg.amin(dim=2, keepdim=True)
    wmax = wg.amax(dim=2, keepdim=True)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-12)
    zero = wmin
    wq = ((wg - zero) / scale).round().clamp(0, 15)
    w_approx = (wq * scale + zero).reshape(N, M)[:, :shape[-1]]
    if w_approx.numel() == torch.Size(shape).numel():
        w_approx = w_approx.reshape(shape)
    return w_approx.to(w.dtype)


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------

SCHEMES: dict[str, dict[str, Any]] = {
    "fp8": {
        "name": "FP8 E4M3 (W8A8)",
        "format": "float8_e4m3fn",
        "bits_per_weight": 8,
        "activation_bits": 8,
        "compression": 2.0,
        "quantize": _q_fp8,
        "hardware_req": "SM >= 8.9 (Ada/Hopper/Blackwell)",
        "has_hardware_accel": True,
    },
    "fp8_perchannel": {
        "name": "FP8 E4M3 per-channel (W8A8)",
        "format": "float8_e4m3fn",
        "bits_per_weight": 8,
        "activation_bits": 8,
        "compression": 2.0,
        "quantize": _q_fp8_perchannel,
        "hardware_req": "SM >= 8.9",
        "has_hardware_accel": True,
    },
    "int8_symmetric": {
        "name": "INT8 Per-tensor Symmetric (W8A8)",
        "format": "int8",
        "bits_per_weight": 8,
        "activation_bits": 8,
        "compression": 2.0,
        "quantize": _q_int8_sym,
        "hardware_req": "Any CUDA GPU",
        "has_hardware_accel": False,
    },
    "int8_affine": {
        "name": "INT8 Affine MM8-style (W8A8)",
        "format": "uint8 + affine",
        "bits_per_weight": 8,
        "activation_bits": 8,
        "compression": 2.0,
        "quantize": _q_int8_affine,
        "hardware_req": "Any CUDA GPU",
        "has_hardware_accel": False,
    },
    "int4_symmetric": {
        "name": "INT4 Per-tensor Symmetric (W4A16)",
        "format": "int4 packed",
        "bits_per_weight": 4,
        "activation_bits": 16,
        "compression": 4.0,
        "quantize": _q_int4_sym,
        "hardware_req": "Any CUDA GPU",
        "has_hardware_accel": False,
    },
    "int4_groupwise_128": {
        "name": "INT4 Group-wise (group=128, W4A16)",
        "format": "int4 packed + per-group scale",
        "bits_per_weight": 4,
        "activation_bits": 16,
        "compression": 3.5,
        "quantize": _q_int4_gw,
        "quantize_kwargs": {"group_size": 128},
        "hardware_req": "Any CUDA GPU",
        "has_hardware_accel": False,
    },
    "int4_groupwise_256": {
        "name": "INT4 Group-wise (group=256, W4A16)",
        "format": "int4 packed + per-group scale",
        "bits_per_weight": 4,
        "activation_bits": 16,
        "compression": 3.7,
        "quantize": _q_int4_gw,
        "quantize_kwargs": {"group_size": 256},
        "hardware_req": "Any CUDA GPU",
        "has_hardware_accel": False,
    },
}


def get_scheme(name: str) -> dict:
    """Get a quantization scheme by name."""
    if name not in SCHEMES:
        raise ValueError(f"Unknown scheme: {name!r}. Available: {list_schemes()}")
    return SCHEMES[name]


def list_schemes() -> list[str]:
    """List all available scheme names."""
    return list(SCHEMES.keys())


def print_scheme_table():
    """Print a formatted table of all schemes."""
    print(f"{'Name':20s} {'Bits':>4s} {'Act':>3s} {'Comp':>4s} {'HW Accel':>8s}  Description")
    print("-" * 72)
    for name, s in SCHEMES.items():
        hw = "Yes" if s["has_hardware_accel"] else "No"
        print(f"{name:20s} {s['bits_per_weight']:4d} {s['activation_bits']:3d} {s['compression']:4.1f}x {hw:>8s}  {s['name']}")
