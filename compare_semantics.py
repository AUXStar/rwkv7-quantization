#!/usr/bin/env python3
"""对比语义检查：量化答错但orig答对的题，逐token对比两者生成轨迹，找偏离点。

用法: python3 compare_semantics.py <quant_model.pth> <tag> <q_index>
输出: orig/quant 逐token生成 + 首个分歧token位置 + 分歧前后文本
"""
import sys, os, json, re, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch
from rwkv.utils import PIPELINE

DATASET = "/home/njzy/test/Albatross/faster3a_2605/dataset/MATH500.jsonl"
MAX_NEW = 120
tok = PIPELINE(None, "rwkv_vocab_v20230424")


def build_model(path):
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
    engine.MODEL_PATH = path
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def gen_tokens(model, prompt_ids, max_new=MAX_NEW):
    ids = [0] + prompt_ids
    state = model.zero_state(1)
    model.forward(torch.tensor([ids], dtype=torch.long), state)
    gen = []
    for _ in range(max_new):
        out = model.forward(torch.tensor([[ids[-1]]], dtype=torch.long), state)
        nxt = out[0].argmax(dim=-1).item()
        if nxt == 0:
            break
        gen.append(nxt)
        ids.append(nxt)
    return gen, tok.decode(gen)


def main():
    model_path = sys.argv[1]
    tag = sys.argv[2]
    qi = int(sys.argv[3])

    tasks = []
    with open(DATASET) as f:
        for line in f:
            tasks.append(json.loads(line))
    item = tasks[qi]
    problem = item["problem"].strip().replace("\r\n", "\n")
    prompt = f"User: {problem}\n\nAssistant:"
    pid = tok.encode(prompt)

    m_orig = build_model("/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth")
    g_o, t_o = gen_tokens(m_orig, pid)
    del m_orig
    torch.cuda.empty_cache()

    m_q = build_model(model_path)
    g_q, t_q = gen_tokens(m_q, pid)

    print("=" * 70)
    print(f"Q{qi} [{tag}] gold={item['answer'][:80]}")
    print(f"PROBLEM: {problem[:150]}")
    print(f"\n--- ORIG 生成 ({len(g_o)} tok) ---")
    print(t_o[:600])
    print(f"\n--- QUANT 生成 ({len(g_q)} tok) ---")
    print(t_q[:600])

    # 找首个分歧token
    nd = min(len(g_o), len(g_q))
    div = next((i for i in range(nd) if g_o[i] != g_q[i]), None)
    print(f"\n=== 首个分歧token位置: {div} (of min {nd}) ===")
    if div is not None:
        print(f"orig  前后: ...{tok.decode(g_o[max(0,div-3):div+3])!r}")
        print(f"quant 前后: ...{tok.decode(g_q[max(0,div-3):div+3])!r}")

    del m_q
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
