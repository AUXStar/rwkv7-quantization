#!/usr/bin/env python3
"""PPL validation for 7.2B optimized kernels.

Runs PPL on gen_8192 corpus at multiple lengths to verify
the FP8 hwdot optimization doesn't change model quality.
"""
import sys, os, json, math, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

MODEL = "/home/njzy/model/rwkv7-7.2b-X5.pth"
EVAL = "/home/njzy/test/eval_tmp"
LENGTHS = [1024, 2048, 4096, 8192]


def build_model():
    import rwkv7_fast_v3a as engine
    from rwkv7_fast_v3a import RWKV7, load_extensions
    engine.NVFP4_W4A16 = False; engine.FP8_W8A16 = False; engine.FUSED_GEMM = True
    engine.WKV_MODE = "fp16"; engine.EMB_DEVICE = "cpu"; engine.RKV_MODE = "off"
    engine.CMIX_SPARSE = "no-fc"; engine.LOWRANK_WEIGHT = "both"
    engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    engine.MODEL_PATH = MODEL
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def ppl_on_len(model, tokens, length):
    """Compute PPL on first `length` tokens."""
    t = torch.tensor([tokens[:length]], dtype=torch.long)
    s = model.zero_state(1)
    out = model.forward_all_logits(t, s)
    logits = out[0].float()
    tgt = torch.tensor(tokens[1:length], dtype=torch.long, device=logits.device)
    return math.exp(torch.nn.functional.cross_entropy(logits[:-1], tgt).item())


def main():
    print("=" * 70)
    print("7.2B PPL Validation (optimized FP8 hwdot kernels)")
    print("=" * 70)

    # Load corpus
    with open(f"{EVAL}/gen_8192.json") as f:
        tokens = json.load(f)["tokens"]
    print(f"Corpus: {len(tokens)} tokens", flush=True)

    m = build_model()

    results = {}
    for length in LENGTHS:
        if length > len(tokens):
            continue
        ppl = ppl_on_len(m, tokens, length)
        results[f"ppl_{length}"] = ppl
        print(f"PPL@{length}: {ppl:.4f}", flush=True)

    # VRAM check
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    results["vram_used_gib"] = used / 2**30
    results["vram_allocated_gib"] = allocated / 2**30
    print(f"\nVRAM: used={used/2**30:.2f}GiB allocated={allocated/2**30:.2f}GiB", flush=True)

    # Previous baseline PPL values (from reports)
    # 7.2B X5 PPL delta was +0.0012 (report 12)
    # Original 7.2B PPL@8192 should be around 1.0x (very low for this corpus)
    print("\n[Comparison]", flush=True)
    print(f"  PPL@8192 = {results.get('ppl_8192', 'N/A')}", flush=True)
    print(f"  Previous 7.2B X5 PPL delta = +0.0012 (from report 12_x5_multi_model.md)", flush=True)
    print(f"  Acceptance: PPL delta <= 0.02 for 7.2B", flush=True)
    print(f"  Note: FP8 hwdot max_diff=0.0, so PPL should be identical to pre-optimization", flush=True)

    with open(f"{EVAL}/ppl_optimized_7b.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved ppl_optimized_7b.json", flush=True)


if __name__ == "__main__":
    main()
