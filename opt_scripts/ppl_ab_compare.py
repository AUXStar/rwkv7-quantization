#!/usr/bin/env python3
"""A/B PPL comparison: original kernels vs optimized kernels.

1. Run PPL with original fused_nvfp4_gemm.py (backup)
2. Run PPL with optimized fused_nvfp4_gemm.py (patched)
3. Compare results — should be identical (FP8 hwdot max_diff=0.0)
"""
import sys, os, json, math, subprocess, shutil
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch

FUSED_FILE = "/home/njzy/test/Albatross/faster3a_2605/fused_nvfp4_gemm.py"
FUSED_BAK = FUSED_FILE + ".bak"
EVAL = "/home/njzy/test/eval_tmp"
MODEL = "/home/njzy/model/rwkv7-7.2b-X5.pth"
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
    t = torch.tensor([tokens[:length]], dtype=torch.long)
    s = model.zero_state(1)
    out = model.forward_all_logits(t, s)
    logits = out[0].float()
    tgt = torch.tensor(tokens[1:length], dtype=torch.long, device=logits.device)
    return math.exp(torch.nn.functional.cross_entropy(logits[:-1], tgt).item())


def run_ppl(label):
    """Run PPL with current fused_nvfp4_gemm.py."""
    # Force reimport
    for mod_name in list(sys.modules.keys()):
        if "fused_nvfp4" in mod_name or "rwkv7_fast" in mod_name or "nvfp4_ops" in mod_name:
            del sys.modules[mod_name]

    print(f"\n{'='*60}")
    print(f"PPL with {label}")
    print(f"{'='*60}")

    m = build_model()
    with open(f"{EVAL}/gen_8192.json") as f:
        tokens = json.load(f)["tokens"]

    results = {}
    for length in LENGTHS:
        if length > len(tokens):
            continue
        ppl = ppl_on_len(m, tokens, length)
        results[f"ppl_{length}"] = ppl
        print(f"PPL@{length}: {ppl:.6f}", flush=True)

    del m
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 70)
    print("A/B PPL Comparison: Original vs Optimized Kernels")
    print("=" * 70)

    # Check backup exists
    if not os.path.exists(FUSED_BAK):
        print(f"ERROR: backup file {FUSED_BAK} not found!")
        return

    import shutil as _sh

    # Set CC for triton backend compilation
    os.environ["CC"] = "/usr/bin/cc"

    # --- A: Original kernels ---
    _sh.copy2(FUSED_BAK, FUSED_FILE)
    ppl_orig = run_ppl("ORIGINAL kernels (FP16 dot)")

    # --- B: Optimized kernels ---
    # Re-apply patch
    subprocess.run(["/usr/bin/python3", "/mnt/c/Users/njzy/.trae-cn/work/6a6b7147083abdc8623c651a/patch_fused_gemm.py"],
                   capture_output=True)
    ppl_opt = run_ppl("OPTIMIZED kernels (FP8 hwdot)")

    # --- Compare ---
    print(f"\n{'='*60}")
    print("A/B Comparison")
    print(f"{'='*60}")
    print(f"{'Length':<10} {'Original':<12} {'Optimized':<12} {'Delta':<12}")
    for length in LENGTHS:
        key = f"ppl_{length}"
        if key in ppl_orig and key in ppl_opt:
            delta = ppl_opt[key] - ppl_orig[key]
            print(f"{length:<10} {ppl_orig[key]:<12.6f} {ppl_opt[key]:<12.6f} {delta:<+12.6f}")

    results = {"original": ppl_orig, "optimized": ppl_opt}
    with open(f"{EVAL}/ppl_ab_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved ppl_ab_comparison.json")


if __name__ == "__main__":
    main()
