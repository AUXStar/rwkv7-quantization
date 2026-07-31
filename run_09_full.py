#!/usr/bin/env python3
"""#9 完整 benchmark 补齐（1.5B）：多长度PPL / decode / throughput / JSON

注: wikitext-2 下载失败(404)，标准 PPL 用本地 8192-token 长文(gen_8192.json)代替，
诚实标注非 wikitext-2。
"""
import sys, os, json, math, time, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")

import torch

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
QUANT_MODEL = "/tmp/rwkv7-1.5b-bench.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"
LOGITS_ORIG = f"{EVAL_DIR}/logits_orig_1_5b_2100.pt"
LENGTHS = [1024, 2048, 4096, 8192]


def build_model(model_path, fused=True):
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
    engine.MODEL_PATH = model_path
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def ppl_on_len(model, tokens, LEN):
    tok_tensor = torch.tensor([tokens[:LEN]], dtype=torch.long)
    state = model.zero_state(1)
    out = model.forward_all_logits(tok_tensor, state)
    logits = out[0].float()
    tgt = torch.tensor(tokens[1:LEN], dtype=torch.long, device=logits.device)
    ce = torch.nn.functional.cross_entropy(logits[:-1], tgt)
    return math.exp(ce.item())


def decode_speed(model, start_token, n=100):
    tok = torch.tensor([[start_token]], dtype=torch.long)
    state = model.zero_state(1)
    for _ in range(5):
        out = model.forward(tok, state)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = model.forward(tok, state)
        tok = out[0].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def throughput(model, tokens, batch_sizes=(1,), seq_len=2048, n_gen=20):
    """Prefill seq_len then decode n_gen tokens, tokens/sec overall.
    NOTE: engine decode path (T=1) supports B=1; B>1 hits shift_state shape
    assert in tmix_mix6 — engine limitation, recorded as-is.
    """
    results = {}
    for B in batch_sizes:
        tok = torch.tensor([tokens[:seq_len]] * B, dtype=torch.long)
        state = model.zero_state(B)
        model.forward(tok, state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = tok[:, -1:]
        for _ in range(n_gen):
            out = model.forward(last, state)
            last = out[0].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        results[B] = B * n_gen / dt
    return results


def main():
    print("=" * 60, flush=True)
    print("#9 full benchmark (1.5B, wikitext-2 download failed -> local 8192 doc)", flush=True)
    print("=" * 60, flush=True)

    from quantize_model import quantize_model
    if not os.path.exists(QUANT_MODEL):
        quantize_model(MODEL_1_5B, QUANT_MODEL, scheme_name="1.5b")

    with open(f"{EVAL_DIR}/gen_8192.json") as f:
        tokens = json.load(f)["tokens"]

    print("loading original...", flush=True)
    m_orig = build_model(MODEL_1_5B, fused=False)
    print("loading quantized...", flush=True)
    m_quant = build_model(QUANT_MODEL, fused=True)

    results = {"model": "rwkv7-g1h-1.5b", "device": "12GB-GPU",
               "dataset_note": "wikitext-2 download 404; used local 8192-token doc"}

    # 1) multi-length PPL
    ppl_o, ppl_q = {}, {}
    for LEN in LENGTHS:
        ppl_o[LEN] = ppl_on_len(m_orig, tokens, LEN)
        ppl_q[LEN] = ppl_on_len(m_quant, tokens, LEN)
        print(f"PPL {LEN}: orig={ppl_o[LEN]:.4f} quant={ppl_q[LEN]:.4f} delta={ppl_q[LEN]-ppl_o[LEN]:+.4f}", flush=True)
    results["ppl"] = {str(l): {"orig": ppl_o[l], "quant": ppl_q[l],
                               "delta": ppl_q[l] - ppl_o[l]} for l in LENGTHS}

    # 2) decode speed
    spd_o = decode_speed(m_orig, tokens[0], 100)
    spd_q = decode_speed(m_quant, tokens[0], 100)
    print(f"decode: orig={spd_o:.1f} quant={spd_q:.1f} tok/s", flush=True)
    results["decode_tps"] = {"orig": spd_o, "quant": spd_q}

    # 3) throughput
    th_o = throughput(m_orig, tokens)
    th_q = throughput(m_quant, tokens)
    for B in th_o:
        print(f"throughput B={B}: orig={th_o[B]:.1f} quant={th_q[B]:.1f} tok/s", flush=True)
    results["throughput"] = {"orig": th_o, "quant": th_q}

    # 4) VRAM
    torch.cuda.empty_cache()
    vram_o = torch.cuda.memory_allocated() / 2**30
    print(f"VRAM orig: {vram_o:.2f} GiB", flush=True)
    del m_orig
    gc.collect()
    torch.cuda.empty_cache()
    vram_q = torch.cuda.memory_allocated() / 2**30
    print(f"VRAM quant: {vram_q:.2f} GiB", flush=True)
    results["vram_gb"] = {"orig": vram_o, "quant": vram_q}

    with open("/home/njzy/test/eval_tmp/benchmark_full.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved benchmark_full.json", flush=True)

    del m_quant
    gc.collect()
    torch.cuda.empty_cache()
    os.remove(QUANT_MODEL)


if __name__ == "__main__":
    main()
