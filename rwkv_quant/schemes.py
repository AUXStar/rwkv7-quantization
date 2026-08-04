# coding=utf-8
"""量化方案注册表 + 权重分类 + 模型状态加载。

注意：torch 只在 load_state 内导入（懒加载），
让 SCHEMES / classify 可以被 list 等轻量命令直接使用。
"""

from __future__ import annotations

# ── 量化方案注册表 ────────────────────────────────────────────
# 字段说明：
#   n     显示名          b      权重位宽          a      激活位宽
#   c     理论压缩比      hw     硬件要求          col    表格颜色
#   fmt   存储格式        group  真量化"true" / 模拟量化"sim"
#   use   推荐场景（list 展示用）
#   score 综合评分 0-5（压缩35% + 精度30% + 兼容20% + 速度15%）
#   desc  方案说明（list 展示用）
#
# 两种实现方式：
#   真量化 (true)：quantize 返回 (w_q, scale)，权重真正以 fp8/int8 低精度
#                 存储，文件大小随之缩小，推理端可反量化回原精度。
#   模拟量化 (sim)：quantize 先量化成整数、再立刻反量化回浮点，返回的
#                   是"近似权重"（仍存为 bf16）。文件大小不变，用于在
#                   实现真正的压缩打包+推理加速之前，先评估该量化算法
#                   对精度的损失（即"模拟"真实低比特推理的数值效果）。
SCHEMES = {
    # ── 真量化：权重以低精度存储，文件真正变小 ──
    "fp8": dict(
        n="FP8 E4M3", b=8, a=8, c=2.0, hw="SM>=8.9", col="green",
        fmt="float8+scale", group="true",
        use="新卡(RTX 40/50系·H100·SM≥8.9)省显存/省磁盘：首选",
        score=4.5,
        desc="Per-tensor FP8：权重直接存为 float8（1字节）+1个全局scale，推理走FP8张量核，禁止反量化。文件约0.57x，精度高。",
    ),
    "fp8_perchannel": dict(
        n="FP8 Per-Channel", b=8, a=8, c=2.0, hw="SM>=8.9", col="green",
        fmt="float8+[N]scale", group="true",
        use="新卡·对精度敏感的任务（代码/数学）",
        score=4.5,
        desc="每输出通道独立scale，对离群通道更鲁棒，精度略优于per-tensor。推理走FP8张量核。文件约0.57x。",
    ),
    "int8_symmetric": dict(
        n="INT8 Symmetric", b=8, a=8, c=2.0, hw="Any CUDA", col="cyan",
        fmt="int8+scale", group="true",
        use="旧卡(A100/RTX 20/30系·无FP8)省空间：首选",
        score=4.2,
        desc="权重直接存为 int8（1字节）+1个scale。任意CUDA GPU可跑，文件约0.57x。",
    ),

    # ── 模拟量化：量化-反量化后仍存 bf16，文件不变，用于精度评估 ──
    "int8_affine": dict(
        n="INT8 Affine (MM8)", b=8, a=8, c=2.0, hw="Any CUDA", col="cyan",
        fmt="bf16近似", group="sim",
        use="研究评估：非对称INT8算法精度",
        score=2.8,
        desc="行/列双偏移+双scale的非对称INT8算法。模拟量化：反量化成近似bf16权重，评估该算法精度。",
    ),
    "int4_symmetric": dict(
        n="INT4 Symmetric", b=4, a=16, c=4.0, hw="Any CUDA", col="red",
        fmt="bf16近似", group="sim",
        use="研究评估：INT4精度损失上限",
        score=2.0,
        desc="对称INT4（理论4x压缩）。模拟量化：近似bf16权重。4bit打包+解码未实现。",
    ),
    "int4_groupwise_128": dict(
        n="INT4 Group g=128", b=4, a=16, c=3.5, hw="Any CUDA", col="red",
        fmt="bf16近似", group="sim",
        use="研究评估：组量化 vs per-tensor",
        score=2.3,
        desc="每128通道一组，组内独立scale/zero，精度优于per-tensor。模拟量化。",
    ),
    "int4_groupwise_256": dict(
        n="INT4 Group g=256", b=4, a=16, c=3.7, hw="Any CUDA", col="red",
        fmt="bf16近似", group="sim",
        use="研究评估：压缩比/精度权衡",
        score=2.1,
        desc="每256通道一组，压缩比略高，精度稍低。模拟量化。",
    ),
}

# 分组标签（list 表格的分隔标题）
GROUP_LABELS = {
    "true": "真量化 · 权重以低精度存储，文件变小",
    "sim": "模拟量化 · 量化后仍存 bf16，文件不变",
}

# 量化类型短标签（list 表格 Type 列）
GROUP_TYPE = {
    "true": "真量化",
    "sim": "模拟量化",
}

# 表格底部脚注：解释"模拟量化"与评分依据
LIST_FOOTNOTE = (
    "说明：\"模拟量化\" = 先按低比特规则量化、再立即反量化回浮点权重（仍存bf16，文件不变），"
    "在实现真正的压缩打包+推理加速前先估算精度损失。\n"
    "评分 = 加权平均（压缩35% / 精度30% / 兼容20% / 速度15%），满分5.0。"
    "FP8 推理走 FP8 张量核（_scaled_mm / Triton tl.dot），禁止反量化，速度与精度兼优；"
    "INT8 无硬件张量核，推理时反量化回 fp16 再算，速度略慢但兼容所有 CUDA GPU；"
    "int4_* 因未实现压缩存储，综合分低。"
)


def score_stars(score: float) -> str:
    """把 0-5 评分转成 ★ 显示，如 '★★★★☆ 4.0'。"""
    stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))
    return f"{stars} {score:.1f}"


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
    """加载模型状态字典，返回 (state_dict, num_layers)。

    使用 mmap 懒加载：张量按需从磁盘读入，内存占用小，
    可处理超过物理内存的大模型（compare 对 2.9B 模型尤其重要）。
    """
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    num_layers = max(
        (int(k.split(".")[1]) for k in state if k.startswith("blocks.") and len(k.split(".")) >= 2),
        default=0,
    ) + 1
    return state, num_layers


def compact_state(state: dict) -> dict:
    """打破张量间的共享 storage，返回新的独立存储 dict。

    原因：RWKV 的 .pth 把所有张量打包进少数几个连续 storage（共享偏移）。
    量化替换部分权重后，原 storage 整块仍被其余张量引用，torch.save
    必须完整保留原数据，再追加新存储，导致文件膨胀。
    clone 后每个张量独占 storage，文件大小 = 实际数据量。
    """
    return {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state.items()}
