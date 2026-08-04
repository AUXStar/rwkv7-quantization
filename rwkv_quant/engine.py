# coding=utf-8
"""推理引擎封装：内置 rwkv7_fast_v3a（+ cuda/ 扩展），无需外部路径。"""

from __future__ import annotations

import gc
import os
import sys
import time

import torch

from .utils import ROOT, VOCAB, err_exit

ENGINE_FILE = os.path.join(ROOT, "rwkv7_fast_v3a.py")


def _engine_module():
    """加载内置推理引擎模块；失败返回 None。"""
    if not os.path.exists(ENGINE_FILE):
        return None
    sys.path.insert(0, ROOT)
    try:
        import rwkv7_fast_v3a as v3a
        return v3a
    except Exception:
        return None


def check_inference() -> None:
    """检查推理能力；不可用时给出可操作的错误提示。"""
    if not torch.cuda.is_available():
        err_exit("CUDA GPU 不可用：eval/speed 需要 NVIDIA GPU")
    if VOCAB is None:
        err_exit("找不到 vocab 文件（rwkv_vocab_v20230424.txt）。请设置环境变量 RWKV_VOCAB 指向它")
    if not os.path.exists(ENGINE_FILE):
        err_exit("内置推理引擎 rwkv7_fast_v3a.py 缺失")


def _load_engine():
    """加载引擎并校验能力，失败直接退出。"""
    check_inference()
    v3a = _engine_module()
    if v3a is None:
        err_exit("内置引擎加载失败：请检查 CUDA 工具链与编译环境")
    return v3a


def _cleanup_modules() -> None:
    """卸载引擎相关模块，释放显存。"""
    for mod in list(sys.modules.keys()):
        if any(x in mod for x in ("fp8_ops", "fused_fp8", "rwkv7_fast")):
            del sys.modules[mod]
    gc.collect()
    torch.cuda.empty_cache()


def _configure(v3a, model_path: str) -> None:
    """配置引擎参数。"""
    v3a.MODEL_PATH = model_path
    v3a.WKV_MODE = "fp16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "off"
    v3a.LOWRANK_WEIGHT = "transpose"
    v3a.ORIG_LINEAR_GROUPS = {"head"}
    v3a.load_extensions(v3a.WKV_MODE)


def load_logits(model_path: str, prompts):
    """用内置引擎跑一批 prompt，返回各 prompt 的 logits 列表。"""
    import torch
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    v3a = _load_engine()
    tokenizer = TRIE_TOKENIZER(VOCAB)
    _configure(v3a, model_path)
    model = v3a.RWKV7()

    logits = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)[:512]
        inp = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
        state = model.zero_state(1)
        with torch.no_grad():
            lg = model.forward_all_logits(inp, state)
        if lg.dim() == 3:
            lg = lg[0]
        logits.append(lg.cpu().float())

    del model
    _cleanup_modules()
    return logits


def measure_speed(model_path: str, warmup: int = 3, steps: int = 20) -> float:
    """测量单 token 生成速度（tok/s）。"""
    import torch
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    v3a = _load_engine()
    tokenizer = TRIE_TOKENIZER(VOCAB)
    _configure(v3a, model_path)
    model = v3a.RWKV7()

    tokens = tokenizer.encode("Write a Python function")[:128]
    inp = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)

    for _ in range(warmup):
        state = model.zero_state(1)
        model.forward(inp, state)

    torch.cuda.synchronize()
    t0 = time.time()
    state = model.zero_state(1)
    with torch.no_grad():
        logits = model.forward(inp, state)
        for _ in range(steps):
            nxt = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
            logits = model.forward(nxt, state)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    del model
    _cleanup_modules()
    return steps / elapsed
