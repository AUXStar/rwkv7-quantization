#!/usr/bin/env python3
"""Per-tensor sensitivity analysis - Part 2: EAR-based per-component + per-tensor.
Reuses existing ear_attribution.json for component-level, runs inference for per-tensor.
"""
import json, os, sys, gc, time, copy
import torch, torch.nn.functional as F

MODEL = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
ENGINE = "/home/njzy/test/Albatross/faster3a_2607"
QUANT = "/home/njzy/test/rwkv7-quantization"
OUT_JSON = os.path.join(QUANT, "experiments/tensor_sensitivity.json")

FP8_MAX = 448.0

PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "To solve the equation 2x + 5 = 11, we first",
    "The time complexity of binary search is O(",
    "The largest planet in our solar system is",
    "Water boils at 100 degrees Celsius because",
    "The quick brown fox jumps over the lazy dog",
    "In Python, a list comprehension is written as",
    "The derivative of x^2 is",
]

def quantize_fp8_inplace(z, keys):
    """Quantize specific keys to FP8 in-place."""
    for key in keys:
        if key not in z or not torch.is_tensor(z[key]) or z[key].dim() != 2:
            continue
        w = z[key].float()
        amax = w.abs().max()
        scale = (amax / FP8_MAX).float() if amax > 0 else torch.tensor(1.0)
        z[key] = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()
        z[key + ".fp8_scale"] = scale.contiguous()

def load_model_logits(model_path, temp_z=None):
    """Load engine and get logits for all prompts."""
    sys.path.insert(0, ENGINE)
    sys.path.insert(0, QUANT)
    import rwkv7_fast_v3a as v3a
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import rwkv
    vocab = os.path.join(os.path.dirname(rwkv.__file__), "rwkv_vocab_v20230424.txt")
    tokenizer = TRIE_TOKENIZER(vocab)
    v3a.MODEL_PATH = model_path
    v3a.WKV_MODE = "fp16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "off"
    v3a.LOWRANK_WEIGHT = "transpose"
    v3a.ORIG_LINEAR_GROUPS = {"head"}
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    all_logits = []
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)[:512]
        inp = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
        state = model.zero_state(1)
        with torch.no_grad():
            lg = model.forward_all_logits(inp, state)
            if lg.dim() == 3: lg = lg[0]
        all_logits.append(lg.cpu().float())
    # Cleanup
    del model
    del sys.modules['rwkv7_fast_v3a']
    for m in list(sys.modules.keys()):
        if any(x in m for x in ['fp8_ops','fused_fp8','rwkv7_fast']):
            del sys.modules[m]
    gc.collect(); torch.cuda.empty_cache()
    return all_logits

def compute_ear(logs_o, logs_q):
    total_ear = total_t = total_top1 = 0.0
    for lo, lq in zip(logs_o, logs_q):
        mn = min(lo.size(0), lq.size(0))
        lo, lq = lo[:mn], lq[:mn]
        p_o, p_q = F.softmax(lo,-1), F.softmax(lq,-1)
        total_ear += torch.minimum(p_o, p_q).sum().item()
        total_top1 += (lo.argmax(-1)==lq.argmax(-1)).float().sum().item()
        total_t += mn
    return total_ear/total_t, total_top1/total_t

def main():
    # Load existing pass1 data
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f: out_data = json.load(f)
    else:
        out_data = {}

    # Load existing EAR attribution
    ear_path = os.path.join(QUANT, "experiments/ear_attribution.json")
    if os.path.exists(ear_path):
        with open(ear_path) as f: existing_ear = json.load(f)
    else:
        existing_ear = {}

    # Component-level EAR from existing data
    comp_ear = {}
    if "per_component" in existing_ear:
        for r in existing_ear["per_component"]:
            comp_ear[r["component"]] = {"ear": r["ear"], "top1": r["top1"]}
    print("Existing component EAR:", json.dumps(comp_ear, indent=2))

    # Load original model
    print("\nLoading baseline model...")
    logits_baseline = load_model_logits(MODEL)
    print(f"Baseline: {len(logits_baseline)} prompts loaded")

    # Per-component EAR (re-measure with consistent prompts)
    print("\n--- Per-Component EAR (6 main linear weights) ---")
    comp_names = ["att.receptance","att.key","att.value","att.output","ffn.key","ffn.value"]
    comp_short = ["att_rec","att_key","att_value","att_output","ffn_key","ffn_value"]
    comp_ear_remeasured = {}

    for cname, cshort in zip(comp_names, comp_short):
        z = torch.load(MODEL, map_location="cpu", mmap=True)
        keys = [f"blocks.{l}.{cname}.weight" for l in range(24)]
        quantize_fp8_inplace(z, keys)
        tmp = "/tmp/tsa_comp.pth"
        for k in list(z.keys()):
            if torch.is_tensor(z[k]): z[k] = z[k].clone()
        torch.save(z, tmp); del z
        lg = load_model_logits(tmp)
        ear, top1 = compute_ear(logits_baseline, lg)
        comp_ear_remeasured[cshort] = {"ear": ear, "top1": top1}
        print(f"  {cshort:12s}: EAR={ear:.6f}, Top-1={top1*100:.2f}%")
        os.remove(tmp)

    # Per-layer EAR (re-measure with consistent prompts)
    print("\n--- Per-Layer EAR (all 6 components in layer i) ---")
    layer_ear = []
    for layer in range(24):
        z = torch.load(MODEL, map_location="cpu", mmap=True)
        for cname in comp_names:
            quantize_fp8_inplace(z, [f"blocks.{layer}.{cname}.weight"])
        tmp = f"/tmp/tsa_layer_{layer}.pth"
        for k in list(z.keys()):
            if torch.is_tensor(z[k]): z[k] = z[k].clone()
        torch.save(z, tmp); del z
        lg = load_model_logits(tmp)
        ear, top1 = compute_ear(logits_baseline, lg)
        layer_ear.append({"layer":layer,"ear":ear,"top1":top1})
        print(f"  Layer {layer:2d}: EAR={ear:.6f}, Top-1={top1*100:.2f}%")
        os.remove(tmp)

    # Per-tensor EAR for representative tensors (Layer 0 + Layer 12 + Layer 23)
    print("\n--- Per-Tensor EAR (individual weight quantization) ---")
    target_tensors = []
    for layer in [0, 12, 23]:
        for cname in comp_names:
            target_tensors.append(f"blocks.{layer}.{cname}.weight")
    target_tensors.append("head.weight")

    per_tensor_ear = {}
    for tkey in target_tensors:
        z = torch.load(MODEL, map_location="cpu", mmap=True)
        quantize_fp8_inplace(z, [tkey])
        tmp = "/tmp/tsa_tensor.pth"
        for k in list(z.keys()):
            if torch.is_tensor(z[k]): z[k] = z[k].clone()
        torch.save(z, tmp); del z
        lg = load_model_logits(tmp)
        ear, top1 = compute_ear(logits_baseline, lg)
        per_tensor_ear[tkey] = {"ear": ear, "top1": top1}
        print(f"  {tkey:45s}: EAR={ear:.6f}, Top-1={top1*100:.2f}%")
        os.remove(tmp)

    # Save all results
    out_data["pass2_results"] = {
        "comp_ear_remeasured": comp_ear_remeasured,
        "layer_ear": layer_ear,
        "per_tensor_ear": per_tensor_ear,
        "num_prompts": len(PROMPTS),
    }

    # Compute layer sensitivity ranking
    ear_mean = sum(r["ear"] for r in layer_ear) / len(layer_ear)
    ear_std = (sum((r["ear"]-ear_mean)**2 for r in layer_ear)/len(layer_ear))**0.5
    out_data["pass2_results"]["layer_stats"] = {
        "mean": round(ear_mean,6), "std": round(ear_std,6),
        "cv": round(ear_std/ear_mean,6),
        "worst_layer": min(layer_ear, key=lambda x:x["ear"])["layer"],
        "best_layer": max(layer_ear, key=lambda x:x["ear"])["layer"],
    }

    # Compute per-component sensitivity ranking
    sorted_comps = sorted(comp_ear_remeasured.items(), key=lambda x: x[1]["ear"])
    out_data["pass2_results"]["comp_sensitivity_ranking"] = [
        {"component": c, "ear": v["ear"], "top1": v["top1"],
         "ear_loss": round(1-v["ear"],6)} for c, v in sorted_comps
    ]

    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved pass2 to {OUT_JSON}")

    # Print summary
    print("\n" + "="*70)
    print("SENSITIVITY RANKING (most sensitive -> least)")
    print("="*70)
    print("\nBy Component (FP8, all layers):")
    for r in out_data["pass2_results"]["comp_sensitivity_ranking"]:
        bar = "█" * int(r["ear_loss"] * 500)
        print(f"  {r['component']:12s}: EAR={r['ear']:.6f} loss={r['ear_loss']:.4f} {bar}")

    print(f"\nBy Layer (CV={out_data['pass2_results']['layer_stats']['cv']:.4f}):")
    sorted_layers = sorted(layer_ear, key=lambda x: x["ear"])
    for r in sorted_layers[:5]:
        print(f"  Layer {r['layer']:2d}: EAR={r['ear']:.6f} (worst)")
    print("  ...")
    for r in sorted_layers[-3:]:
        print(f"  Layer {r['layer']:2d}: EAR={r['ear']:.6f} (best)")

if __name__ == "__main__":
    main()
