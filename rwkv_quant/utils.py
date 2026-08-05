# coding=utf-8
"""通用工具：终端样式、格式化、错误处理、vocab 定位。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 项目根目录（含推理引擎 rwkv7_fast_v3a.py + cuda/）
ROOT = str(Path(__file__).resolve().parent.parent)

# ── 终端样式（ANSI，无依赖） ──────────────────────────────────
R = "\033[0m"; BD = "\033[1m"; CY = "\033[36m"; GN = "\033[32m"
YL = "\033[33m"; RD = "\033[31m"; DM = "\033[2m"; MG = "\033[35m"

_TTY = sys.stdout.isatty()


def C(text, code) -> str:
    """上色；非 TTY 时返回纯文本。"""
    return f"{code}{text}{R}" if _TTY else str(text)


def hd(text): return C(text, BD + CY)
def ok(text): return C(text, GN)
def wr(text): return C(text, YL)
def er(text): return C(text, RD)
def dm(text): return C(text, DM)
def mg(text): return C(text, MG)


def err_exit(msg: str, code: int = 1) -> None:
    """红色错误输出到 stderr 并退出。"""
    sys.stderr.write(f"{er('Error:')} {msg}\n")
    sys.stderr.flush()
    sys.exit(code)


def hs(num: float) -> str:
    """人类可读的文件大小。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


# ── vocab 定位 ────────────────────────────────────────────────
_VOCAB_CANDIDATES = (
    "rwkv_vocab_v20230424.txt",                       # 项目内
    "~/.local/lib/python3.13/site-packages/rwkv/rwkv_vocab_v20230424.txt",
    os.path.expanduser("~/.venv/lib/python3.13/site-packages/rwkv/rwkv_vocab_v20230424.txt"),
)


def find_vocab() -> str | None:
    """定位 vocab 文件：环境变量优先，其次候选路径。找不到返回 None。"""
    env = os.environ.get("RWKV_VOCAB")
    if env and os.path.exists(env):
        return env
    for cand in _VOCAB_CANDIDATES:
        path = os.path.join(ROOT, cand) if not cand.startswith("~") else os.path.expanduser(cand)
        if os.path.exists(path):
            return path
    return None


VOCAB = find_vocab()
