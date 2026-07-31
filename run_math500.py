#!/usr/bin/env python3
"""#11 MATH500 greedy evaluation (hard constraint: generation-quality acceptance).

Compares original bf16 vs quantized (final 1.5B scheme, fused kernel).
Greedy decode (temperature 0) to amplify tiny logit differences.
Answer verification: extract \boxed{...} / last number, normalize, compare with golden.
"""
import sys, os, json, re, math, time, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
sys.path.insert(0, "/home/njzy/test/Albatross/faster2_251201")
os.chdir("/home/njzy/test/rwkv7-quantization")

import torch

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
QUANT_MODEL = "/tmp/rwkv7-1.5b-math.pth"
DATASET = "/home/njzy/test/Albatross/faster3a_2605/dataset/MATH500.jsonl"
VOCAB = "/home/njzy/test/Albatross/faster2_251201/reference/rwkv_vocab_v20230424.txt"
N_LIMIT = 500      # full MATH500
MAX_NEW = 256

from reference.utils import TRIE_TOKENIZER


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
    """Normalize a math answer string to a comparable float, or None."""
    s = s.strip()
    if not s:
        return None
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\{([^}]*)\}\{([^}]*)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            return None
    # percent
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    # commas
    s = s.replace(",", "")
    # scientific notation
    try:
        v = float(s)
        return v
    except Exception:
        return None


def extract_answer(text):
    """Extract answer from generated text."""
    # last \boxed{...}
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    for b in reversed(boxes):
        v = normalize_number(b)
        if v is not None:
            return v, "boxed"
    # last line with a number
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
    # answer like "\\boxed{42}" or "42" or "\\frac{1}{2}"
    m = re.search(r"\\boxed\{([^}]*)\}", a)
    if m:
        return normalize_number(m.group(1))
    return normalize_number(a)


def greedy_generate(model, tok, prompt_ids, max_new):
    ids = [0] + prompt_ids
    tok_ids = torch.tensor([ids], dtype=torch.long)
    state = model.zero_state(1)
    # prefill
    model.forward(tok_ids, state)
    gen = []
    for _ in range(max_new):
        out = model.forward(torch.tensor([[ids[-1]]], dtype=torch.long), state)
        nxt = out[0].argmax(dim=-1).item()
        if nxt == 0:  # EOD
            break
        gen.append(nxt)
        ids.append(nxt)
    return tok.decode(gen)


def load_tasks():
    rows = []
    with open(DATASET) as f:
        for i, line in enumerate(f):
            if i >= N_LIMIT:
                break
            rows.append(json.loads(line))
    return rows


def evaluate(model, tok, tasks, name):
    correct = 0
    total = 0
    t0 = time.perf_counter()
    results = []
    for i, item in enumerate(tasks):
        problem = item["problem"].strip().replace("\r\n", "\n")
        prompt = f"User: {problem}\n\nAssistant:"
        prompt_ids = tok.encode(prompt)
        if len(prompt_ids) + MAX_NEW > 8192:
            prompt_ids = prompt_ids[: 8192 - MAX_NEW]
        text = greedy_generate(model, tok, prompt_ids, MAX_NEW)
        pred, how = extract_answer(text)
        gold = golden_value(item)
        ok = False
        if pred is not None and gold is not None:
            ok = abs(pred - gold) < max(1e-3, abs(gold) * 1e-3)
        correct += int(ok)
        total += 1
        results.append({"ok": ok, "pred": pred, "gold": gold, "how": how})
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(tasks)} acc={correct/total:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    acc = correct / total
    print(f"[{name}] accuracy={acc:.3f} ({correct}/{total}) elapsed={time.perf_counter()-t0:.0f}s", flush=True)
    return acc, results


def main():
    print("=" * 60, flush=True)
    print(f"#11 MATH500 greedy (subset {N_LIMIT}, max_new={MAX_NEW})", flush=True)
    print("=" * 60, flush=True)

    tok = TRIE_TOKENIZER(VOCAB)
    tasks = load_tasks()
    print(f"loaded {len(tasks)} tasks", flush=True)

    from quantize_model import quantize_model
    if not os.path.exists(QUANT_MODEL):
        quantize_model(MODEL_1_5B, QUANT_MODEL, scheme_name="1.5b")

    print("loading original...", flush=True)
    m_orig = build_model(MODEL_1_5B)
    print("loading quantized...", flush=True)
    m_quant = build_model(QUANT_MODEL)

    acc_orig, res_orig = evaluate(m_orig, tok, tasks, "orig")
    acc_quant, res_quant = evaluate(m_quant, tok, tasks, "quant")

    delta = acc_orig - acc_quant
    print(f"\n{'='*60}", flush=True)
    print(f"MATH500 greedy (N={len(tasks)})", flush=True)
    print(f"  orig:  {acc_orig:.3f}", flush=True)
    print(f"  quant: {acc_quant:.3f}", flush=True)
    print(f"  delta: {delta:+.3f}  (验收: ≤0.02)", flush=True)
    print(f"  {'PASS' if delta <= 0.02 else 'FAIL'}", flush=True)

    with open("/home/njzy/test/eval_tmp/math500_greedy.json", "w") as f:
        json.dump({"acc_orig": acc_orig, "acc_quant": acc_quant, "delta": delta,
                   "orig": res_orig, "quant": res_quant}, f, indent=1)

    del m_orig, m_quant
    gc.collect()
    torch.cuda.empty_cache()
    os.remove(QUANT_MODEL)


if __name__ == "__main__":
    main()
