# coding=utf-8
"""rwkv-quant 的 6 个子命令实现。"""

from __future__ import annotations

import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from . import engine as eng
from .evaluate import PROMPTS, compute_metrics
from .schemes import SCHEMES, classify, load_state
from .utils import ROOT, hs, err_exit, C, hd, ok, wr, er, dm, mg, BD

try:  # rich 可选：提供更漂亮的表格/进度条
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import box
    _r = Console()
    RICH = True
except ImportError:
    RICH = False


def _scheme_quantizer(name: str):
    """加载指定方案的量化函数。"""
    sys.path.insert(0, ROOT)
    from schemes import get_scheme  # 根目录 schemes.py（已有量化实现）
    return get_scheme(name)


# ─────────────────────────────────────────────────────────────
#  list —— 列出所有量化方案
# ─────────────────────────────────────────────────────────────
def cmd_list(args) -> int:
    if RICH:
        table = Table(title="✦ Quantization Schemes ✦", box=box.ROUNDED,
                      header_style="bold magenta")
        table.add_column("Scheme", style="bold")
        table.add_column("Bits", justify="center")
        table.add_column("Act", justify="center")
        table.add_column("Compression", justify="center")
        table.add_column("HW Requirement", justify="center")
        for name, s in SCHEMES.items():
            table.add_row(f"[{s['col']}]{name}[/]", f"{s['b']}W", f"{s['a']}A",
                          f"[{s['col']}]{s['c']:.1f}x[/]", f"[yellow]{s['hw']}[/]")
        _r.print(table)
    else:
        print(f"\n{hd('✦ Quantization Schemes ✦')}")
        print(f"  {dm('─' * 60)}")
        print(f"  {'Scheme':20s} {'Bits':>5s} {'Act':>4s} {'Comp':>7s} {'HW Req':>12s}")
        for name, s in SCHEMES.items():
            print(f"  {ok(name):20s} {s['b']:>3d}W {s['a']:>3d}A {s['c']:>6.1f}x {s['hw']:>12s}")
        print(f"  {dm('─' * 60)}")
    return 0


# ─────────────────────────────────────────────────────────────
#  info —— 模型信息
# ─────────────────────────────────────────────────────────────
def cmd_info(args) -> int:
    path = args.model
    if not path or not os.path.exists(path):
        err_exit(f"model file not found: {path}")

    print(f"  {dm('Loading model...')}")
    t0 = time.time()
    state, num_layers = load_state(path)
    load_time = time.time() - t0

    w0 = state.get("blocks.0.att.receptance.weight")
    hidden = w0.shape[1] if w0 is not None else "?"
    emb = state.get("emb.weight")
    vocab = emb.shape[0] if emb is not None else "?"
    params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    meta = state.get("quant_meta") or state.get("meta")
    quantized = meta.get("scheme") if meta else "No"

    rows = [
        ("File", str(path)),
        ("Size", hs(os.path.getsize(path))),
        ("Layers", str(num_layers)),
        ("Hidden", str(hidden)),
        ("Vocab", str(vocab)),
        ("Params", f"{params:,}"),
        ("Quantized", quantized),
        ("Load time", f"{load_time:.1f}s"),
    ]

    if RICH:
        from rich.panel import Panel
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan")
        grid.add_column()
        for label, value in rows:
            colored = value if not (label == "Quantized" and value == "No") else f"[red]{value}[/]"
            grid.add_row(label, colored)
        _r.print(Panel(grid, title="✦ Model Info ✦", border_style="cyan"))
    else:
        print(f"\n{hd('✦ Model Info ✦')}")
        print(f"  {dm('─' * 44)}")
        for label, value in rows:
            print(f"  {C(label + ':', BD):12s} {value}")
        print(f"  {dm('─' * 44)}")
    return 0


# ─────────────────────────────────────────────────────────────
#  quantize —— 量化模型
# ─────────────────────────────────────────────────────────────
def cmd_quantize(args) -> int:
    model_path, output, scheme = args.model, args.output, args.scheme
    if not model_path:
        err_exit("-m/--model is required (quantize what?)")
    if not output:
        err_exit("-o/--output is required (where to save?)")
    if scheme not in SCHEMES:
        err_exit(f"unknown scheme '{scheme}'. Run 'rwkv-quant l' to see all schemes")

    # 输出为目录时自动生成文件名：{模型名}_{方案}_{时间戳}.pth
    if os.path.isdir(output) or output.endswith("/") or output.endswith(os.sep):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = os.path.join(output, f"{Path(model_path).stem}_{scheme}_{timestamp}.pth")
        print(f"  {dm('Output dir detected, auto name:')} {ok(output)}")

    quantizer = _scheme_quantizer(scheme)
    qfn = quantizer["quantize"]
    qkw = dict(quantizer.get("quantize_kwargs", {}))

    print(f"\n  {hd('Quantizing')}  {ok(SCHEMES[scheme]['n'])}")
    print(f"  {dm('Loading')} {model_path} ...")
    t0 = time.time()

    state, num_layers = load_state(model_path)
    targets = [(k, classify(k, num_layers)) for k in state if classify(k, num_layers) is not None]
    total = len(targets)
    print(f"  {num_layers} layers · {total} weights to quantize · {len(state)} total keys\n")

    if RICH:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(bar_width=30),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TextColumn("({task.completed}/{task.total})"),
                      TimeElapsedColumn(), console=_r) as prog:
            task = prog.add_task(f"[{SCHEMES[scheme]['col']}]Quantizing {scheme}", total=total)
            done = 0
            for key, _info in targets:
                weight = state[key].float()
                state[key] = qfn(weight, **qkw)
                done += 1
                prog.advance(task)
    else:
        done = 0
        for i, (key, _info) in enumerate(targets):
            weight = state[key].float()
            state[key] = qfn(weight, **qkw)
            done += 1
            if (i + 1) % 24 == 0 or i + 1 == total:
                pct = (i + 1) / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r  {bar} {ok(f'{pct:5.1f}%')} ({i + 1}/{total})", end="", flush=True)
        print()

    # 保存
    state["quant_meta"] = {
        "scheme": scheme,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    print(f"  {dm('Saving')} → {output} ...", end=" ", flush=True)
    torch.save(state, output)
    print(ok("done"))

    elapsed = time.time() - t0
    orig_size = os.path.getsize(model_path)
    new_size = os.path.getsize(output)
    ratio = orig_size / max(new_size, 1)
    saved_pct = (1 - new_size / max(orig_size, 1)) * 100

    print(f"\n{ok('✅ Quantization Complete')}")
    print(f"  {dm('─' * 44)}")
    print(f"  {C('Scheme:', BD)}   {SCHEMES[scheme]['n']}")
    print(f"  {C('Weights:', BD)}  {done:,} / {total}")
    print(f"  {C('Layers:', BD)}   {num_layers}")
    print(f"  {C('Time:', BD)}     {elapsed:.1f}s ({done / elapsed:.0f} w/s)")
    size_note = ok(f"{hs(new_size)} ({ratio:.2f}x, save {saved_pct:.0f}%)") if ratio > 1 \
        else wr(f"{hs(new_size)} ({ratio:.2f}x)")
    print(f"  {C('Size:', BD)}     {hs(orig_size)} → {size_note}")
    print(f"  {dm('─' * 44)}")
    return 0


# ─────────────────────────────────────────────────────────────
#  compare —— 对比所有方案的文件大小
# ─────────────────────────────────────────────────────────────
def cmd_compare(args) -> int:
    model_path = args.model
    if not model_path:
        err_exit("-m/--model is required")

    outdir = args.outdir or "./compare_out"
    os.makedirs(outdir, exist_ok=True)

    orig_size = os.path.getsize(model_path)
    print(f"  {dm('Source:')} {model_path} ({hs(orig_size)})")
    print(f"  {dm('Output:')} {outdir}\n")

    state, num_layers = load_state(model_path)
    results = []

    for name in SCHEMES:
        quantizer = _scheme_quantizer(name)
        qfn = quantizer["quantize"]
        qkw = dict(quantizer.get("quantize_kwargs", {}))

        work = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state.items()}
        for key in list(work.keys()):
            if classify(key, num_layers) is None:
                continue
            work[key] = qfn(work[key].float(), **qkw)

        out = os.path.join(outdir, f"model_{name}.pth")
        torch.save(work, out)
        del work
        gc.collect()

        size = os.path.getsize(out)
        results.append((name, SCHEMES[name]["n"], size, size / orig_size, SCHEMES[name]["col"]))
        print(f"  {ok(name):26s} → {ok(hs(size)):>9s} ({ok(f'{size / orig_size:.2f}x')})")

    if RICH:
        table = Table(title="✦ Comparison ✦", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Scheme", style="bold")
        table.add_column("Name")
        table.add_column("Size", justify="right")
        table.add_column("Ratio", justify="right")
        for name, label, size, ratio, col in sorted(results, key=lambda r: r[3]):
            table.add_row(f"[{col}]{name}[/]", label, f"[{col}]{hs(size)}[/]",
                          f"[{'green' if ratio < 1 else 'yellow'}]{ratio:.2f}x[/]")
        _r.print(table)
    else:
        print(f"\n{hd('✦ Comparison ✦')}")
        for name, label, size, ratio, col in sorted(results, key=lambda r: r[3]):
            print(f"  {ok(name):26s} {hs(size):>9s} {ratio:.2f}x")
    print(f"\n  {dm('Saved to')} {ok(outdir)}/")
    return 0


# ─────────────────────────────────────────────────────────────
#  eval —— EAR / Top-1 / 速度评估
# ─────────────────────────────────────────────────────────────
def cmd_eval(args) -> int:
    baseline, quantized = args.baseline, args.quantized
    if not baseline or not quantized:
        err_exit("-b/--baseline and -q/--quantized are required")

    print(f"  {dm('Loading baseline:')} {baseline}")
    base_logits = eng.load_logits(baseline, PROMPTS)
    base_speed = eng.measure_speed(baseline)
    print(f"  {ok('Baseline:')} {ok(f'{base_speed:.1f} tok/s')}\n")

    for qpath in quantized.split(",") if "," in quantized else [quantized]:
        name = Path(qpath).stem
        print(f"  {hd(f'Evaluating: {name}')}")
        q_logits = eng.load_logits(qpath, PROMPTS)
        ear, top1 = compute_metrics(base_logits, q_logits)
        speed = eng.measure_speed(qpath)

        speed_note = ok(f"{speed:.1f} tok/s ({speed / base_speed:.2f}x)") if speed >= base_speed \
            else wr(f"{speed:.1f} tok/s ({speed / base_speed:.2f}x)")
        print(f"  {C('EAR:', BD)}    {ear:.6f}")
        print(f"  {C('Top-1:', BD)}  {top1 * 100:.2f}%")
        print(f"  {C('Speed:', BD)}  {speed_note}")
        print(f"  {C('Size:', BD)}   {hs(os.path.getsize(qpath))}\n")
    return 0


# ─────────────────────────────────────────────────────────────
#  sensitivity —— 逐组件统计
# ─────────────────────────────────────────────────────────────
def cmd_sensitivity(args) -> int:
    model_path = args.model
    if not model_path:
        err_exit("-m/--model is required")

    print(f"  {dm('Analyzing:')} {model_path}")
    state, num_layers = load_state(model_path)

    groups = defaultdict(list)
    for key in sorted(state.keys()):
        tensor = state[key]
        info = classify(key, num_layers)
        component = info[1] if info else "other"
        if not (isinstance(tensor, torch.Tensor) and tensor.dim() == 2 and tensor.numel() >= 10000):
            continue
        wf = tensor.float()
        std = float(wf.std())
        skew = float(((wf - wf.mean()) ** 3).mean() / (std ** 3 + 1e-20))
        kurt = float(((wf - wf.mean()) ** 4).mean() / (std ** 4 + 1e-20) - 3)
        groups[component].append((std, skew, kurt))

    if RICH:
        table = Table(title="✦ Tensor Statistics ✦", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Component", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Avg Std", justify="right")
        table.add_column("Avg Skew", justify="right")
        table.add_column("Avg Kurt", justify="right")
        for component in sorted(groups):
            items = groups[component]
            if not items:
                continue
            n = len(items)
            table.add_row(component, str(n),
                          f"{sum(x[0] for x in items) / n:.4f}",
                          f"{sum(x[1] for x in items) / n:+.3f}",
                          f"{sum(x[2] for x in items) / n:+.3f}")
        _r.print(table)
    else:
        print(f"\n{hd('✦ Tensor Statistics ✦')}")
        for component in sorted(groups):
            items = groups[component]
            if not items:
                continue
            n = len(items)
            print(f"  {ok(component):18s} n={n:3d}  std={sum(x[0] for x in items) / n:.4f}"
                  f"  skew={sum(x[1] for x in items) / n:+.3f}  kurt={sum(x[2] for x in items) / n:+.3f}")
    return 0


COMMANDS = {
    "info": cmd_info,
    "list": cmd_list,
    "quantize": cmd_quantize,
    "compare": cmd_compare,
    "eval": cmd_eval,
    "sensitivity": cmd_sensitivity,
}
