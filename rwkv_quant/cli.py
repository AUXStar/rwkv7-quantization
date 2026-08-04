# coding=utf-8
"""CLI 入口：参数解析、子命令缩写、分发。"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .commands import COMMANDS
from .utils import err_exit

# 子命令缩写 → 全名
ABBREVIATIONS = {
    "l": "list", "li": "list",
    "i": "info",
    "q": "quantize", "quant": "quantize",
    "c": "compare", "cmp": "compare",
    "e": "eval",
    "s": "sensitivity", "sens": "sensitivity",
}

EPILOG = """\
Examples:
  rwkv-quant info -m model.pth
  rwkv-quant list
  rwkv-quant quantize -m model.pth -o ./out -s fp8
  rwkv-quant compare -m model.pth
  rwkv-quant eval -b orig.pth -q quantized.pth
  rwkv-quant sensitivity -m model.pth
Aliases: i / l / q / c / e / s
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rwkv-quant",
        description=f"RWKV-7 Quantization Toolkit v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    sub = parser.add_subparsers(dest="cmd", help="command")

    p = sub.add_parser("info", help="Show model info [i]")
    p.add_argument("-m", "--model", default=None)

    sub.add_parser("list", help="List quantization schemes [l]")

    p = sub.add_parser("quantize", help="Quantize a model [q]")
    p.add_argument("-m", "--model", default=None)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("-s", "--scheme", default="fp8")

    p = sub.add_parser("compare", help="Compare all schemes [c]")
    p.add_argument("-m", "--model", default=None)
    p.add_argument("-C", "--outdir", default=None)

    p = sub.add_parser("eval", help="Evaluate quantized model [e]")
    p.add_argument("-b", "--baseline", default=None)
    p.add_argument("-q", "--quantized", default=None)

    p = sub.add_parser("sensitivity", help="Per-tensor stats [s]")
    p.add_argument("-m", "--model", default=None)

    return parser


def _expand_abbreviation() -> None:
    """把第一个位置参数（子命令）的缩写展开为全名。"""
    for index, arg in enumerate(sys.argv[1:], start=1):
        if not arg.startswith("-"):
            if arg in ABBREVIATIONS:
                sys.argv[index] = ABBREVIATIONS[arg]
            return


def main(argv=None) -> int:
    _expand_abbreviation()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return 0
    if args.cmd not in COMMANDS:
        err_exit(f"unknown command '{args.cmd}'")

    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
