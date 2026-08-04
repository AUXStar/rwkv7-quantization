# coding=utf-8
"""评估指标：EAR 与 Top-1 一致性。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 评估用 prompt（多领域、长度不一，覆盖常见 token 分布）
PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "def fibonacci(n):",
    "To solve 2x+5=11",
    "Binary search is O(",
    "Largest planet is",
    "Water boils at 100",
    "Quick brown fox",
    "Python list comprehension",
    "Derivative of x^2",
    "Sort a list in Python",
    "Binary search function",
    "Pythagorean theorem",
    "import torch",
    "AI in 2025",
    "Neural network 3 layers",
    "Chemical symbol gold",
    "Photosynthesis converts",
    "Square root of 144",
    "def reverse_linked_list(head):",
]


def compute_metrics(baseline_logits, quantized_logits):
    """计算 EAR（期望重合率）与 Top-1 一致率。

    EAR: 逐位置 softmax 后取 min 再求和，衡量分布相似度。
    Top-1: argmax 完全一致的 token 占比。
    返回 (ear, top1)。
    """
    total_ear = 0.0
    total_match = 0
    total_tokens = 0

    for base, quant in zip(baseline_logits, quantized_logits):
        length = min(base.size(0), quant.size(0))
        base, quant = base[:length], quant[:length]

        p_base = F.softmax(base, dim=-1)
        p_quant = F.softmax(quant, dim=-1)
        total_ear += torch.minimum(p_base, p_quant).sum().item()
        total_match += (base.argmax(-1) == quant.argmax(-1)).float().sum().item()
        total_tokens += length

    return total_ear / total_tokens, total_match / total_tokens
