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
#   fmt   存储格式        group  真压缩/"真量化"  或 "近似权重"
#   desc  方案说明（list 展示用，可用 \n 换行成多行）
SCHEMES = {
    # ── 真量化：保存 fp8/int8 权重 + scale，文件真正变小 ──
    "fp8": dict(
        n="FP8 E4M3", b=8, a=8, c=2.0, hw="SM>=8.9", col="green",
        fmt="float8+scale", group="true",
        desc="Per-tensor FP8：权重1字节+1个scale，精度高，文件约0.57x，需Ada/Hopper/Blackwell加速",
    ),
    "fp8_perchannel": dict(
        n="FP8 Per-Channel", b=8, a=8, c=2.0, hw="SM>=8.9", col="green",
        fmt="float8+[N]scale", group="true",
        desc="逐输出通道独立scale，对离群通道更鲁棒，精度略优于per-tensor",
    ),
    "int8_symmetric": dict(
        n="INT8 Symmetric", b=8, a=8, c=2.0, hw="Any CUDA", col="cyan",
        fmt="int8+scale", group="true",
        desc="对称INT8：权重1字节+1个scale，通用GPU可跑，文件约0.57x，精度良好",
    ),

    # ── 近似权重：输出反量化后的 bf16 权重，文件大小不变，用于精度对比 ──
    "int8_affine": dict(
        n="INT8 Affine (MM8)", b=8, a=8, c=2.0, hw="Any CUDA", col="cyan",
        fmt="bf16近似", group="approx",
        desc="MM8风格非对称INT8：行/列双偏移+双scale，精度最高，输出为近似权重",
    ),
    "int4_symmetric": dict(
        n="INT4 Symmetric", b=4, a=16, c=4.0, hw="Any CUDA", col="red",
        fmt="bf16近似", group="approx",
        desc="对称INT4：理论4x压缩，精度损失大，输出为近似权重",
    ),
    "int4_groupwise_128": dict(
        n="INT4 Group g=128", b=4, a=16, c=3.5, hw="Any CUDA", col="red",
        fmt="bf16近似", group="approx",
        desc="每128通道一组量化，组内独立scale/zero，精度优于per-tensor",
    ),
    "int4_groupwise_256": dict(
        n="INT4 Group g=256", b=4, a=16, c=3.7, hw="Any CUDA", col="red",
        fmt="bf16近似", group="approx",
        desc="每256通道一组量化，压缩比略高，精度稍低",
    ),
}

# 分组标签（list 表格的分隔标题）
GROUP_LABELS = {
    "true": "真量化 · 文件真正变小",
    "approx": "近似权重 · 文件大小不变（精度对比用）",
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
