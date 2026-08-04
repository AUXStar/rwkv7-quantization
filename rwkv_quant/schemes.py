# coding=utf-8
"""量化方案注册表 + 权重分类 + 模型状态加载。"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .utils import ROOT

# ── 量化方案注册表 ────────────────────────────────────────────
# b: 权重位宽  a: 激活位宽  c: 理论压缩比  hw: 硬件要求  col: 表格颜色
SCHEMES = {
    "fp8":                dict(n="FP8 E4M3",             b=8, a=8,  c=2.0, hw="SM>=8.9",  col="green"),
    "fp8_perchannel":     dict(n="FP8 Per-Channel",      b=8, a=8,  c=2.0, hw="SM>=8.9",  col="green"),
    "int8_symmetric":     dict(n="INT8 Symmetric",       b=8, a=8,  c=2.0, hw="Any CUDA", col="cyan"),
    "int8_affine":        dict(n="INT8 Affine (MM8)",    b=8, a=8,  c=2.0, hw="Any CUDA", col="cyan"),
    "int4_symmetric":     dict(n="INT4 Symmetric",       b=4, a=16, c=4.0, hw="Any CUDA", col="red"),
    "int4_groupwise_128": dict(n="INT4 Group g=128",     b=4, a=16, c=3.5, hw="Any CUDA", col="red"),
    "int4_groupwise_256": dict(n="INT4 Group g=256",     b=4, a=16, c=3.7, hw="Any CUDA", col="red"),
}


# ── 权重分类 ──────────────────────────────────────────────────
# 可量化的 6 个组件/层（att_r/att_k/att_v/att_o/ffn_k/ffn_v）
QUANT = {"att_r", "att_k", "att_v", "att_o", "ffn_k", "ffn_v"}
_CM = {"receptance": "att_r", "key": "att_k", "value": "att_v", "output": "att_o"}
_FM = {"key": "ffn_k", "value": "ffn_v"}


def classify(key: str, num_layers: int):
    """把权重 key 分类为 (layer, component)；不可量化返回 None。

    例如 "blocks.5.att.key.weight" -> (5, "att_k")。
    """
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    if layer < 0 or layer >= num_layers:
        return None
    component = _CM.get(parts[3]) if parts[2] == "att" else (_FM.get(parts[3]) if parts[2] == "ffn" else None)
    return (layer, component) if component in QUANT else None


def load_state(model_path: str):
    """加载模型状态字典，返回 (state_dict, num_layers)。"""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    num_layers = max(
        (int(k.split(".")[1]) for k in state if k.startswith("blocks.") and len(k.split(".")) >= 2),
        default=0,
    ) + 1
    return state, num_layers


def num_model_layers(model_path: str) -> int:
    """只统计层数（轻量）。"""
    _, num_layers = load_state(model_path)
    return num_layers
