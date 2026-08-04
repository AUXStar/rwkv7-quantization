#!/usr/bin/env python3
"""Per-layer EAR attribution: quantify each layer's contribution to EAR loss.

For each layer i (0..23):
  - Quantize ONLY layer i to FP8, keep all other layers BF16
  - Measure EAR vs fully-BF16 baseline
  => "single-layer EAR" = how much EAR drops from quantizing just this one layer

Also do per-component attribution:
  - Quantize only one component type across ALL layers
  => "per-component EAR" = att.key vs ffn.key vs ...

If all layers contribute equally (user's hypothesis), the bottleneck is
quantization COUNT, not layer-specific sensitivity.
"""
import sys, os, gc, json, time, copy
import torch
import torch.nn.functional as F

ENGINE_DIR = "/home/njzy/test/Albatross/faster3a_2607"
QUANT_DIR = "/home/njzy/test/rwkv7-quantization"
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, QUANT_DIR)

FP8_E4M3_MAX = 448.0

TEST_PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "数学中，质数的定义是",
    "Once upon a time in a distant galaxy,",
    "The chemical formula for water is H2O because",
    "To solve the equation 2x + 5 = 11, we first",
    "The time complexity of binary search is O(",
    "在深度学习中，反向传播算法的核心思想是",
    "The largest planet in our solar system is",
]

COMP_MAP = {
    "receptance": 0, "key": 1, "value": 2, "output": 3,
    "ffn_key": 4, "ffn_value": 5
}

def quantize_layer_inplace(z, layer, comps=None):
    """Quantize specific layer's weights to FP8 in-place. comps=None => all 6."""
    if comps is None:
        comps = range(6)
    for comp_idx in comps:
        if comp_idx < 4:
            comp_name = ["receptance", "key", "value", "output"][comp_idx]
            key = f"blocks.{layer}.att.{comp_name}.weight"
        else:
            comp_name = ["key", "value"][comp_idx - 4]
            key = f"blocks.{layer}.ffn.{comp_name}.weight"
        if key not in z or not torch.is_tensor(z[key]) or z[key].dim() != 2:
            continue
        w = z[key].float()
        amax = w.abs().max()
        scale = (amax / FP8_E4M3_MAX).float() if amax > 0 else torch.tensor(1.0)
        w_fp8 = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        z[key] = w_fp8.contiguous()
        z[key + ".fp8_scale"] = scale.contiguous()

def quantize_component_all_layers(z, comp_idx):
    """Quantize one component type across ALL layers."""
    comp_names = ["receptance", "key", "value", "output"]
    ffn_names = ["key", "value"]
    layer_keys = [k for k in z.keys() if k.startswith("blocks.")]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    for layer in range(num_layers):
        if comp_idx < 4:
            key = f"blocks.{layer}.att.{comp_names[comp_idx]}.weight"
        else:
            key = f"blocks.{layer}.ffn.{ffn_names[comp_idx - 4]}.weight"
        if key not in z or not torch.is_tensor(z[key]) or z[key].dim() != 2:
            continue
        w = z[key].float()
        amax = w.abs().max()
        scale = (amax / FP8_E4M3_MAX).float() if amax > 0 else torch.tensor(1.0)
        w_fp8 = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        z[key] = w_fp8.contiguous()
        z[key + ".fp8_scale"] = scale.contiguous()

def load_model_get_logits(model_path, prompts, temp_z=None):
    """Load model and get logits. If temp_z provided, use it as weight dict."""
    import rwkv7_fast_v3a as v3a
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import rwkv
    vocab_path = os.path.join(os.path.dirname(rwkv.__file__), "rwkv_vocab_v20230424.txt")
    tokenizer = TRIE_TOKENIZER(vocab_path)
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
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 512:
            tokens = tokens[:512]
        token_tensor = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
        state = model.zero_state(1)
        with torch.no_grad():
            logits = model.forward_all_logits(token_tensor, state)
            if logits.dim() == 3:
                logits = logits[0]
        all_logits.append(logits.cpu().float())
    del model
    del v3a
    gc.collect()
    torch.cuda.empty_cache()
    for mod_name in list(sys.modules.keys()):
        if 'rwkv7_fast_v3a' in mod_name or 'fp8_ops' in mod_name or 'fused_fp8' in mod_name:
            del sys.modules[mod_name]
    return all_logits

def compute_ear_summary(logits_orig_list, logits_quant_list):
    """Compute weighted-average EAR across all prompts."""
    total_ear = 0.0
    total_tokens = 0
    total_top1 = 0.0
    for lo, lq in zip(logits_orig_list, logits_quant_list):
        min_len = min(lo.size(0), lq.size(0))
        lo, lq = lo[:min_len], lq[:min_len]
        p_o = F.softmax(lo, dim=-1)
        p_q = F.softmax(lq, dim=-1)
        ears = torch.minimum(p_o, p_q).sum(dim=-1)
        top1 = (lo.argmax(-1) == lq.argmax(-1)).float()
        total_ear += ears.sum().item()
        total_top1 += top1.sum().item()
        total_tokens += min_len
    return total_ear / total_tokens, total_top1 / total_tokens

def main():
    MODEL_PATH = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
    print("=" * 70, flush=True)
    print("Per-Layer EAR Attribution (1.5B model)", flush=True)
    print("=" * 70, flush=True)

    # Baseline: original model
    print("\n[0] Loading original model (BF16 baseline)...", flush=True)
    logits_baseline = load_model_get_logits(MODEL_PATH, TEST_PROMPTS)
    print(f"  Got {len(logits_baseline)} prompts\n", flush=True)

    # Full FP8 for reference
    print("[1] Full FP8 quantization (reference)...", flush=True)
    z = torch.load(MODEL_PATH, map_location="cpu", mmap=True)
    layer_keys = [k for k in z.keys() if k.startswith("blocks.")]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    for layer in range(num_layers):
        quantize_layer_inplace(z, layer)
    # Save temp model
    temp_path = "/tmp/attr_full_fp8.pth"
    for _k in list(z.keys()):
        if torch.is_tensor(z[_k]):
            z[_k] = z[_k].clone()
    torch.save(z, temp_path)
    del z
    logits_full_fp8 = load_model_get_logits(temp_path, TEST_PROMPTS)
    ear_full, top1_full = compute_ear_summary(logits_baseline, logits_full_fp8)
    print(f"  Full FP8: EAR={ear_full:.6f}, Top-1={top1_full*100:.2f}%\n", flush=True)
    os.remove(temp_path)

    # Per-layer attribution
    print("[2] Per-layer attribution (quantize ONLY layer i)...", flush=True)
    layer_results = []
    for layer in range(num_layers):
        z = torch.load(MODEL_PATH, map_location="cpu", mmap=True)
        quantize_layer_inplace(z, layer)
        temp_path = f"/tmp/attr_layer_{layer}.pth"
        for _k in list(z.keys()):
            if torch.is_tensor(z[_k]):
                z[_k] = z[_k].clone()
        torch.save(z, temp_path)
        del z
        logits_q = load_model_get_logits(temp_path, TEST_PROMPTS)
        ear, top1 = compute_ear_summary(logits_baseline, logits_q)
        layer_results.append({"layer": layer, "ear": ear, "top1": top1})
        print(f"  Layer {layer:2d}: EAR={ear:.6f}, Top-1={top1*100:.2f}%", flush=True)
        os.remove(temp_path)

    # Per-component attribution
    print(f"\n[3] Per-component attribution (quantize ONE component across ALL layers)...", flush=True)
    comp_results = []
    for comp_idx in range(6):
        comp_name = ["att.rec", "att.key", "att.val", "att.out", "ffn.key", "ffn.val"][comp_idx]
        z = torch.load(MODEL_PATH, map_location="cpu", mmap=True)
        quantize_component_all_layers(z, comp_idx)
        temp_path = f"/tmp/attr_comp_{comp_idx}.pth"
        for _k in list(z.keys()):
            if torch.is_tensor(z[_k]):
                z[_k] = z[_k].clone()
        torch.save(z, temp_path)
        del z
        logits_q = load_model_get_logits(temp_path, TEST_PROMPTS)
        ear, top1 = compute_ear_summary(logits_baseline, logits_q)
        comp_results.append({"component": comp_name, "ear": ear, "top1": top1})
        print(f"  {comp_name:10s}: EAR={ear:.6f}, Top-1={top1*100:.2f}%", flush=True)
        os.remove(temp_path)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\nBaseline (BF16): EAR=1.000000, Top-1=100.00%", flush=True)
    print(f"Full FP8:        EAR={ear_full:.6f}, Top-1={top1_full*100:.2f}%", flush=True)

    # Layer analysis
    ears_layer = [r["ear"] for r in layer_results]
    ear_mean = sum(ears_layer) / len(ears_layer)
    ear_min = min(ears_layer)
    ear_max = max(ears_layer)
    ear_std = (sum((e - ear_mean)**2 for e in ears_layer) / len(ears_layer)) ** 0.5

    print(f"\n--- Per-Layer (single layer quantized) ---", flush=True)
    print(f"  EAR range: [{ear_min:.6f}, {ear_max:.6f}]", flush=True)
    print(f"  EAR mean:  {ear_mean:.6f}, std: {ear_std:.6f}", flush=True)
    print(f"  CV (std/mean): {ear_std/ear_mean:.4f}", flush=True)
    if ear_std / ear_mean < 0.05:
        print(f"  >>> Layers are UNIFORM (CV < 5%) — sensitivity is evenly distributed", flush=True)
    else:
        worst = min(layer_results, key=lambda x: x["ear"])
        print(f"  >>> Layers VARY — worst: layer {worst['layer']} (EAR={worst['ear']:.6f})", flush=True)

    # Component analysis
    print(f"\n--- Per-Component (one component across all layers) ---", flush=True)
    for r in comp_results:
        print(f"  {r['component']:10s}: EAR={r['ear']:.6f}, Top-1={r['top1']*100:.2f}%", flush=True)
    ears_comp = [r["ear"] for r in comp_results]
    comp_mean = sum(ears_comp) / len(ears_comp)
    comp_std = (sum((e - comp_mean)**2 for e in ears_comp) / len(ears_comp)) ** 0.5
    print(f"  CV (std/mean): {comp_std/comp_mean:.4f}", flush=True)

    # Additivity check: does sum of single-layer losses ≈ full loss?
    single_layer_loss_sum = sum(1 - r["ear"] for r in layer_results)
    full_loss = 1 - ear_full
    print(f"\n--- Additivity Check ---", flush=True)
    print(f"  Sum of single-layer EAR losses: {single_layer_loss_sum:.6f}", flush=True)
    print(f"  Full FP8 EAR loss:              {full_loss:.6f}", flush=True)
    print(f"  Ratio (sum/full):               {single_layer_loss_sum/full_loss:.2f}x", flush=True)
    if single_layer_loss_sum / full_loss > 0.8:
        print(f"  >>> Losses are ADDITIVE — layers contribute independently", flush=True)
    else:
        print(f"  >>> Losses are SUB-ADDITIVE — interactions between layers matter", flush=True)

    # Save
    results = {
        "baseline_ear": 1.0,
        "full_fp8": {"ear": ear_full, "top1": top1_full},
        "per_layer": layer_results,
        "per_component": comp_results,
        "layer_stats": {
            "mean": ear_mean, "std": ear_std, "min": ear_min, "max": ear_max,
            "cv": ear_std / ear_mean,
        },
        "additivity": {
            "sum_single_layer_loss": single_layer_loss_sum,
            "full_loss": full_loss,
            "ratio": single_layer_loss_sum / full_loss,
        }
    }
    out_path = "/home/njzy/test/rwkv7-quantization/experiments/ear_attribution.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
