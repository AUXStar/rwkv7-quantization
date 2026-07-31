#!/usr/bin/env python3
"""#11 MATH500 greedy — rwkv库decode + 生成质量诊断

用 rwkv 库 PIPELINE（官方 tokenizer）decode 生成文本，
并诊断生成退化：乱码率 / 长度 / 停止原因 / 答案可提取性。
"""
import sys, os, json, re, math, time, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")

import torch
from rwkv.utils import PIPELINE

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
QUANT_MODEL = "/tmp/rwkv7-1.5b-math2.pth"
DATASET = "/home/njzy/test/Albatross/faster3a_2605/dataset/MATH500.jsonl"
N_LIMIT = 500
MAX_NEW = 256

tok = PIPELINE(None, "rwkv_vocab_v20230424")


def build_model(model_path):
    import rwkv7_fast_v3a as engine
    from rwkv7_fast_v3a import RWKV7, load_extensions
    engine.NVFP4_W4A16 = False
    engine.FP8_W8A16 = False
    engine.FUSED_GEMM = True
    engine.WKV_MODE = "fp16"
    engine.EMB_DEVICE = "cpu"
    engine.RKV_MODE = "off"
    engine.CMIX_SPARSE = "no-fc"
    engine.LOWRANK_WEIGHT = "both"
    engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    engine.MODEL_PATH = model_path
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def normalize_number(s):
    s = s.strip()
    if not s:
        return None
    m = re.fullmatch(r"\\frac\{([^}]*)\}\{([^}]*)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def extract_answer(text):
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    for b in reversed(boxes):
        v = normalize_number(b)
        if v is not None:
            return v, "boxed"
    lines = [l for l in reversed(text.splitlines()) if l.strip()]
    for line in lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\\frac\{[^}]*\}\{[^}]*\}", line)
        if nums:
            v = normalize_number(nums[-1])
            if v is not None:
                return v, "number"
    return None, "none"


def golden_value(item):
    a = item["answer"]
    m = re.search(r"\\boxed\{([^}]*)\}", a)
    return normalize_number(m.group(1)) if m else normalize_number(a)


def greedy_generate(model, prompt_ids):
    ids = [0] + prompt_ids
    state = model.zero_state(1)
    model.forward(torch.tensor([ids], dtype=torch.long), state)
    gen = []
    for _ in range(MAX_NEW):
        out = model.forward(torch.tensor([[ids[-1]]], dtype=torch.long), state)
        nxt = out[0].argmax(dim=-1).item()
        if nxt == 0:
            break
        gen.append(nxt)
        ids.append(nxt)
    text = tok.decode(gen)
    return text, len(gen)


def load_tasks():
    rows = []
    with open(DATASET) as f:
        for i, line in enumerate(f):
            if i >= N_LIMIT:
                break
            rows.append(json.loads(line))
    return rows


def evaluate(model, tasks, name):
    correct = 0
    stats = {"n": 0, "boxed": 0, "number": 0, "none": 0,
             "garbled": 0, "short": 0, "tot_len": 0}
    results = []
    t0 = time.perf_counter()
    for i, item in enumerate(tasks):
        problem = item["problem"].strip().replace("\r\n", "\n")
        prompt = f"User: {problem}\n\nAssistant:"
        prompt_ids = tok.encode(prompt)
        if len(prompt_ids) + MAX_NEW > 8192:
            prompt_ids = prompt_ids[: 8192 - MAX_NEW]
        text, n_gen = greedy_generate(model, prompt_ids)
        pred, how = extract_answer(text)
        gold = golden_value(item)
        ok = False
        if pred is not None and gold is not None:
            ok = abs(pred - gold) < max(1e-3, abs(gold) * 1e-3)
        correct += int(ok)
        stats["n"] += 1
        stats[how] += 1
        stats["tot_len"] += n_gen
        if "\ufffd" in text:
            stats["garbled"] += 1
        if n_gen < 30:
            stats["short"] += 1
        results.append({"ok": ok, "pred": pred, "gold": gold, "how": how, "len": n_gen})
        if (i + 1) % 100 == 0:
            print(f"  [{name}] {i+1}/{len(tasks)} acc={correct/stats['n']:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    stats["acc"] = correct / stats["n"]
    stats["avg_len"] = stats["tot_len"] / stats["n"]
    print(f"[{name}] acc={stats['acc']:.3f} ({correct}/{stats['n']}) "
          f"avg_len={stats['avg_len']:.0f} boxed={stats['boxed']} number={stats['number']} "
          f"none={stats['none']} garbled={stats['garbled']} short={stats['short']}", flush=True)
    return stats, results


def main():
    print("=" * 60, flush=True)
    print(f"#11 MATH500 greedy (N={N_LIMIT}, rwkv lib decode + quality diag)", flush=True)
    print("=" * 60, flush=True)

    tasks = load_tasks()
    print(f"loaded {len(tasks)} tasks", flush=True)

    from quantize_model import quantize_model
    if not os.path.exists(QUANT_MODEL):
        quantize_model(MODEL_1_5B, QUANT_MODEL, scheme_name="1.5b")

    print("loading original...", flush=True)
    m_orig = build_model(MODEL_1_5B)
    print("loading quantized...", flush=True)
    m_quant = build_model(QUANT_MODEL)

    s_orig, r_orig = evaluate(m_orig, tasks, "orig")
    s_quant, r_quant = evaluate(m_quant, tasks, "quant")

    delta = s_orig["acc"] - s_quant["acc"]
    print(f"\n{'='*60}", flush=True)
    print(f"MATH500 greedy (N={len(tasks)}, rwkv decode)", flush=True)
    print(f"  orig:  acc={s_orig['acc']:.3f}  avg_len={s_orig['avg_len']:.0f} "
          f"none={s_orig['none']} garbled={s_orig['garbled']}", flush=True)
    print(f"  quant: acc={s_quant['acc']:.3f}  avg_len={s_quant['avg_len']:.0f} "
          f"none={s_quant['none']} garbled={s_quant['garbled']}", flush=True)
    print(f"  delta: {delta:+.3f}  (验收 ≤0.02)", flush=True)
    print(f"  {'PASS' if delta <= 0.02 else 'FAIL'}", flush=True)
    print(f"\n生成质量诊断:", flush=True)
    print(f"  无法提取答案(none): orig={s_orig['none']} ({s_orig['none']/s_orig['n']*100:.1f}%) "
          f"quant={s_quant['none']} ({s_quant['none']/s_quant['n']*100:.1f}%)", flush=True)
    print(f"  乱码(garbled):       orig={s_orig['garbled']} quant={s_quant['garbled']}", flush=True)
    print(f"  过早停止(<30tok):    orig={s_orig['short']} quant={s_quant['short']}", flush=True)

    with open("/home/njzy/test/eval_tmp/math500_greedy_v2.json", "w") as f:
        json.dump({"acc_orig": s_orig, "acc_quant": s_quant, "delta": delta,
                   "orig": r_orig, "quant": r_quant}, f, indent=1)

    del m_orig, m_quant
    gc.collect()
    torch.cuda.empty_cache()
    os.remove(QUANT_MODEL)


if __name__ == "__main__":
    main()
