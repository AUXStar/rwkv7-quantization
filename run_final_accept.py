#!/usr/bin/env python3
"""Final acceptance for T4 scheme (key FP8 + lowrank bf16 + residual per-block + alpha=0.3)."""
import sys, os, json, math, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
QUANT = "/tmp/rwkv7-1.5b-final.pth"
EVAL = "/home/njzy/test/eval_tmp"
LENGTHS = [1024, 2048, 4096, 8192]

import quantize_model as qm
qm.ALPHA = 0.3


def build_model(path, fused=True):
    import rwkv7_fast_v3a as engine
    from rwkv7_fast_v3a import RWKV7, load_extensions
    engine.NVFP4_W4A16 = False
    engine.FP8_W8A16 = False
    engine.FUSED_GEMM = fused
    engine.WKV_MODE = "fp16"
    engine.EMB_DEVICE = "cpu"
    engine.RKV_MODE = "off"
    engine.CMIX_SPARSE = "no-fc"
    engine.LOWRANK_WEIGHT = "both"
    engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    engine.MODEL_PATH = path
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def ppl_on_len(model, tokens, LEN):
    t = torch.tensor([tokens[:LEN]], dtype=torch.long)
    s = model.zero_state(1)
    out = model.forward_all_logits(t, s)
    logits = out[0].float()
    tgt = torch.tensor(tokens[1:LEN], dtype=torch.long, device=logits.device)
    return math.exp(torch.nn.functional.cross_entropy(logits[:-1], tgt).item())


def decode_speed(model, start_token, n=100):
    tok = torch.tensor([[start_token]], dtype=torch.long)
    s = model.zero_state(1)
    for _ in range(5):
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time_perf()
    for _ in range(n):
        out = model.forward(tok, s)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return n / (time_perf() - t0)


def time_perf():
    import time
    return time.perf_counter()


def state_mse_8192(model_orig, model_q):
    """8192-token state MSE at L23 (deepest)."""
    import json as _j
    with open(f"{EVAL}/gen_8192.json") as f:
        toks = _j.load(f)["tokens"]
    so, sq = model_orig.zero_state(1), model_q.zero_state(1)
    to = torch.tensor([toks], dtype=torch.long)
    model_orig.forward(to, so)
    model_q.forward(to, sq)
    mse = (so[1][-1].float() - sq[1][-1].float()).pow(2).mean().item()
    return mse


def main():
    qm.quantize_model(MODEL_1_5B, QUANT, scheme_name="1.5b")

    with open(f"{EVAL}/gen_8192.json") as f:
        tokens = json.load(f)["tokens"]

    m_orig = build_model(MODEL_1_5B, fused=False)
    m_q = build_model(QUANT, fused=True)

    res = {"scheme": "T4 (key FP8, lowrank bf16, res per-block, alpha 0.3)"}
    for LEN in LENGTHS:
        po = ppl_on_len(m_orig, tokens, LEN)
        pq = ppl_on_len(m_q, tokens, LEN)
        res[f"ppl_{LEN}"] = {"orig": po, "quant": pq, "delta": pq - po}
        print(f"PPL {LEN}: orig={po:.4f} quant={pq:.4f} delta={pq-po:+.4f}", flush=True)

    do = decode_speed(m_orig, tokens[0])
    dq = decode_speed(m_q, tokens[0])
    res["decode_tps"] = {"orig": do, "quant": dq}
    print(f"decode: orig={do:.1f} quant={dq:.1f}", flush=True)

    sm = state_mse_8192(m_orig, m_q)
    res["state_mse_8192_L23"] = sm
    print(f"8192 state MSE L23: {sm:.4e}", flush=True)

    del m_orig
    gc.collect()
    torch.cuda.empty_cache()
    vq = torch.cuda.memory_allocated() / 2**30
    res["vram_gb_quant"] = vq
    print(f"VRAM quant: {vq:.2f} GiB", flush=True)

    with open(f"{EVAL}/final_acceptance.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved final_acceptance.json", flush=True)


if __name__ == "__main__":
    main()
