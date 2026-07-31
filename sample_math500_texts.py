#!/usr/bin/env python3
"""样本语义检查：量化模型MATH500生成文本，逐题看语义正确性。

用法: python3 sample_math500_texts.py <quant_model.pth> <tag> [n_questions]
在指定题号集合上生成完整文本，打印出来供语义检查。
"""
import sys, os, json, re, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch
from rwkv.utils import PIPELINE

DATASET = "/home/njzy/test/Albatross/faster3a_2605/dataset/MATH500.jsonl"
MAX_NEW = 200
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


def gen(model, prompt_ids):
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
    return tok.decode(gen)


def main():
    model_path = sys.argv[1]
    tag = sys.argv[2]
    q_indices = [int(x) for x in sys.argv[3].split(",")]

    tasks = []
    with open(DATASET) as f:
        for i, line in enumerate(f):
            tasks.append(json.loads(line))

    m = build_model(model_path)
    for qi in q_indices:
        item = tasks[qi]
        problem = item["problem"].strip().replace("\r\n", "\n")
        prompt = f"User: {problem}\n\nAssistant:"
        text = gen(m, tok.encode(prompt))
        print("=" * 70)
        print(f"### Q{qi} [{tag}]  gold={item['answer'][:80]}")
        print(f"PROBLEM: {problem[:200]}")
        print("--- 生成文本 ---")
        print(text)
        print()

    del m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
