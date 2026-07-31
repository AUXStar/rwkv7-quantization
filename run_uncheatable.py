#!/usr/bin/env python3
"""#11 Uncheatable Eval (compression rate) — fresh-corpus, anti-memorization.

Ports the official methodology:
  tokenize doc -> chunks of 4000 tokens -> prepend token 0 -> CE(logits[:-1], chunk[1:])
  neg_log_prob_sum = mean total NLL (nats)
  compression_rate = (neg_log_prob_sum / mean_bytes) * (1/ln2) * 0.125 * 100

Compares original bf16 vs quantized (final 1.5B scheme).
"""
import sys, os, json, math, time, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
sys.path.insert(0, "/home/njzy/test/Albatross/faster2_251201")
os.chdir("/home/njzy/test/rwkv7-quantization")

import torch

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
QUANT_MODEL = "/tmp/rwkv7-1.5b-uncheat.pth"
UNCH_DIR = "/home/njzy/rwkv-sglang/bench/data/uncheatable"
VOCAB = "/home/njzy/test/Albatross/faster2_251201/reference/rwkv_vocab_v20230424.txt"
CHUNK = 4000
MAX_DOCS = 6
DOCS = ["arxiv_math.json", "bbc_news.json", "github_cpp.json", "wikipedia_english.json"]

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


def load_docs():
    docs = []
    for fn in DOCS:
        p = os.path.join(UNCH_DIR, fn)
        if not os.path.exists(p):
            print(f"  skip missing {fn}", flush=True)
            continue
        with open(p) as f:
            data = json.load(f)
        # format: list of strings or list of {"content": ...}
        for d in data[:MAX_DOCS]:
            if isinstance(d, str):
                docs.append(d)
            elif isinstance(d, dict) and "content" in d:
                docs.append(d["content"])
    return docs


def eval_compression(model, tok, docs):
    total_nll = 0.0
    total_bytes = 0.0
    n_chunks = 0
    for di, doc in enumerate(docs):
        raw_bytes = len(doc.encode("utf-8"))
        total_bytes += raw_bytes
        ids = tok.encode(doc)
        if not ids:
            continue
        nll_doc = 0.0
        for s in range(0, len(ids), CHUNK):
            chunk = [0] + ids[s:s + CHUNK]
            t = torch.tensor([chunk], dtype=torch.long)
            state = model.zero_state(1)
            out = model.forward_all_logits(t, state)
            logits = out[0].float()
            # CE(logits[:-1], chunk[1:])
            tgt = torch.tensor(chunk[1:], dtype=torch.long, device=logits.device)
            ce = torch.nn.functional.cross_entropy(logits[:-1], tgt)
            nll_doc += ce.item() * (len(chunk) - 1)
            n_chunks += 1
        total_nll += nll_doc
        print(f"  doc {di}: nll={nll_doc:.1f} nats, bytes={raw_bytes}", flush=True)
    neg_log_prob = total_nll / len(docs)
    avg_bytes = total_bytes / len(docs)
    bpb = (neg_log_prob / avg_bytes) * (1.0 / math.log(2.0))
    comp = bpb * 0.125 * 100
    return {"neg_log_prob": neg_log_prob, "avg_bytes": avg_bytes, "bpb": bpb, "compression_rate": comp}


def main():
    print("=" * 60, flush=True)
    print("#11 Uncheatable Eval (compression rate, fresh corpus)", flush=True)
    print("=" * 60, flush=True)

    tok = TRIE_TOKENIZER(VOCAB)
    docs = load_docs()
    print(f"loaded {len(docs)} docs from {DOCS}", flush=True)

    from quantize_model import quantize_model
    if not os.path.exists(QUANT_MODEL):
        quantize_model(MODEL_1_5B, QUANT_MODEL, scheme_name="1.5b")

    print("loading original...", flush=True)
    m_orig = build_model(MODEL_1_5B)
    r_orig = eval_compression(m_orig, tok, docs)
    print(f"  orig: bpb={r_orig['bpb']:.4f} compression={r_orig['compression_rate']:.2f}%", flush=True)
    del m_orig
    gc.collect()
    torch.cuda.empty_cache()

    print("loading quantized...", flush=True)
    m_quant = build_model(QUANT_MODEL)
    r_quant = eval_compression(m_quant, tok, docs)
    print(f"  quant: bpb={r_quant['bpb']:.4f} compression={r_quant['compression_rate']:.2f}%", flush=True)

    ratio = r_quant["compression_rate"] / r_orig["compression_rate"]
    print(f"\n{'='*60}", flush=True)
    print(f"Uncheatable Eval (docs={len(docs)}, chunks={CHUNK})", flush=True)
    print(f"  orig  compression: {r_orig['compression_rate']:.2f}%", flush=True)
    print(f"  quant compression: {r_quant['compression_rate']:.2f}%", flush=True)
    print(f"  ratio: {ratio:.4f} (验收: quant ≥ 99% of orig)", flush=True)
    print(f"  {'PASS' if ratio >= 0.99 else 'FAIL'}", flush=True)

    with open("/home/njzy/test/eval_tmp/uncheatable_eval.json", "w") as f:
        json.dump({"orig": r_orig, "quant": r_quant, "ratio": ratio, "docs": DOCS}, f, indent=1)

    del m_quant
    gc.collect()
    torch.cuda.empty_cache()
    os.remove(QUANT_MODEL)


if __name__ == "__main__":
    main()
