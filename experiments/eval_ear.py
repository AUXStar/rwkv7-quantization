#!/usr/bin/env python3
"""EAR (Expected Acceptance Rate) evaluation for FP8 quantization.

Based on SLQ paper (arXiv:2605.02404): measures distribution consistency
between original and quantized models.

EAR = E_x[ sum_v min(p_orig(v|x), p_quant(v|x)) ]
  - EAR >= 0.99  => "distribution-lossless"
  - EAR >= 0.95  => "near-lossless"
  - EAR <  0.95  => significant distribution shift

Also computes:
  - KL divergence (orig || quant) and (quant || orig)
  - Top-1 agreement (greedy argmax match)
  - Top-5 agreement
  - Max logit difference

Usage:
    python eval_ear.py --model-orig <orig.pth> --model-quant <fp8.pth> [--num-prompts 10]
"""
import sys, os, gc, json, argparse, time

# Add engine and quantization paths
ENGINE_DIR = "/home/njzy/test/Albatross/faster3a_2607"
QUANT_DIR = "/home/njzy/test/rwkv7-quantization"
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, QUANT_DIR)

import torch
import torch.nn.functional as F

# ============================================================================
# Test prompts (diverse domains for coverage)
# ============================================================================
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

# ============================================================================
# Model loading via Albatross v3a engine
# ============================================================================

def load_model_and_get_logits(model_path, prompts, device='cuda'):
    """Load a model through the v3a engine and collect logits for each prompt."""
    import rwkv7_fast_v3a as v3a
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import rwkv

    vocab_path = os.path.join(os.path.dirname(rwkv.__file__), "rwkv_vocab_v20230424.txt")
    tokenizer = TRIE_TOKENIZER(vocab_path)

    # Configure engine for evaluation
    v3a.MODEL_PATH = model_path
    v3a.WKV_MODE = "fp16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "off"  # disable sparse for quantized models
    v3a.LOWRANK_WEIGHT = "transpose"
    # For quantized models, fp8 ops handle the dispatch
    v3a.ORIG_LINEAR_GROUPS = {"head"}  # head never quantized
    v3a.load_extensions(v3a.WKV_MODE)

    model = v3a.RWKV7()

    # Get logits for each prompt
    all_logits = []
    for i, prompt in enumerate(prompts):
        # Tokenize prompt using RWKV tokenizer
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 512:
            tokens = tokens[:512]
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

        # Create fresh zero state
        state = model.zero_state(1)

        # Run forward to get all logits
        with torch.no_grad():
            logits = model.forward_all_logits(token_tensor, state)
            # logits: [1, T, vocab_size] or [T, vocab_size]
            if logits.dim() == 3:
                logits = logits[0]  # [T, V]
            # logits is now [T, V]

        # Move to CPU to save GPU memory
        all_logits.append(logits.cpu().float())
        print(f"  Prompt {i+1}/{len(prompts)}: {len(tokens)} tokens, logits shape={logits.shape}", flush=True)

    # Cleanup
    del model
    del v3a
    gc.collect()
    torch.cuda.empty_cache()

    # Remove from sys.modules to allow reload
    for mod_name in list(sys.modules.keys()):
        if 'rwkv7_fast_v3a' in mod_name or 'fp8_ops' in mod_name or 'fused_fp8' in mod_name:
            del sys.modules[mod_name]

    return all_logits


# ============================================================================
# Metrics
# ============================================================================

def compute_ear(logits_orig, logits_quant):
    """Compute EAR (Expected Acceptance Rate) for a single sequence.

    Args:
        logits_orig: [T, V] tensor
        logits_quant: [T, V] tensor

    Returns:
        ears: [T] per-position EAR
        mean_ear: scalar
    """
    p_orig = F.softmax(logits_orig, dim=-1)
    p_quant = F.softmax(logits_quant, dim=-1)

    ears = torch.minimum(p_orig, p_quant).sum(dim=-1)  # [T]
    return ears, ears.mean().item()


def compute_kl(p, q, eps=1e-10):
    """KL divergence KL(p || q)."""
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    return (p_safe * (p_safe.log() - q_safe.log())).sum(dim=-1)


def compute_metrics(logits_orig, logits_quant):
    """Compute all metrics for a single sequence pair.

    Args:
        logits_orig: [T, V]
        logits_quant: [T, V]

    Returns dict with per-sequence metrics.
    """
    p_orig = F.softmax(logits_orig, dim=-1)
    p_quant = F.softmax(logits_quant, dim=-1)

    # EAR
    ears = torch.minimum(p_orig, p_quant).sum(dim=-1)

    # KL divergences
    kl_oq = compute_kl(p_orig, p_quant)
    kl_qo = compute_kl(p_quant, p_orig)

    # JS divergence (symmetric)
    m = 0.5 * (p_orig + p_quant)
    js = 0.5 * compute_kl(p_orig, m) + 0.5 * compute_kl(p_quant, m)

    # Top-1 agreement
    top1_orig = logits_orig.argmax(dim=-1)
    top1_quant = logits_quant.argmax(dim=-1)
    top1_match = (top1_orig == top1_quant).float()

    # Top-5 agreement
    top5_orig = logits_orig.topk(5, dim=-1).indices
    top5_quant = logits_quant.topk(5, dim=-1).indices
    top5_match = torch.zeros(logits_orig.size(0), dtype=torch.float)
    for t in range(logits_orig.size(0)):
        top5_match[t] = len(set(top5_orig[t].tolist()) & set(top5_quant[t].tolist())) / 5.0

    # Logit difference
    logit_diff = (logits_orig - logits_quant).abs()
    max_logit_diff = logit_diff.max(dim=-1).values
    mean_logit_diff = logit_diff.mean(dim=-1)

    # Probability mass difference (L1)
    prob_l1 = (p_orig - p_quant).abs().sum(dim=-1)

    return {
        'ear': ears.mean().item(),
        'ear_min': ears.min().item(),
        'kl_orig_quant': kl_oq.mean().item(),
        'kl_quant_orig': kl_qo.mean().item(),
        'js_divergence': js.mean().item(),
        'top1_agreement': top1_match.mean().item(),
        'top5_agreement': top5_match.mean().item(),
        'max_logit_diff': max_logit_diff.mean().item(),
        'mean_logit_diff': mean_logit_diff.mean().item(),
        'prob_l1': prob_l1.mean().item(),
        'num_tokens': logits_orig.size(0),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EAR evaluation: original vs quantized")
    parser.add_argument("--model-orig", required=True, help="Original model path")
    parser.add_argument("--model-quant", required=True, help="Quantized model path")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts")
    parser.add_argument("--output", default="ear_results.json", help="Output JSON file")
    args = parser.parse_args()

    prompts = TEST_PROMPTS[:args.num_prompts]

    print("=" * 70, flush=True)
    print("EAR Evaluation: Original vs FP8 Quantized", flush=True)
    print("=" * 70, flush=True)
    print(f"Original: {args.model_orig}", flush=True)
    print(f"Quantized: {args.model_quant}", flush=True)
    print(f"Prompts: {len(prompts)}", flush=True)
    print(flush=True)

    # Load original model and get logits
    print("--- Loading Original Model ---", flush=True)
    t0 = time.time()
    logits_orig = load_model_and_get_logits(args.model_orig, prompts)
    print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

    # Load quantized model and get logits
    print("--- Loading Quantized Model ---", flush=True)
    t0 = time.time()
    logits_quant = load_model_and_get_logits(args.model_quant, prompts)
    print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

    # Compute metrics
    print("--- Computing Metrics ---", flush=True)
    all_metrics = []
    for i, (lo, lq) in enumerate(zip(logits_orig, logits_quant)):
        # Align lengths (should be same, but just in case)
        min_len = min(lo.size(0), lq.size(0))
        lo = lo[:min_len]
        lq = lq[:min_len]

        m = compute_metrics(lo, lq)
        all_metrics.append(m)

        print(f"\nPrompt {i+1}: \"{prompts[i][:50]}...\"", flush=True)
        print(f"  EAR:           {m['ear']:.6f} (min={m['ear_min']:.6f})", flush=True)
        print(f"  KL(o||q):      {m['kl_orig_quant']:.6f}", flush=True)
        print(f"  KL(q||o):      {m['kl_quant_orig']:.6f}", flush=True)
        print(f"  JS divergence: {m['js_divergence']:.6f}", flush=True)
        print(f"  Top-1 agree:   {m['top1_agreement']*100:.2f}%", flush=True)
        print(f"  Top-5 agree:   {m['top5_agreement']*100:.2f}%", flush=True)
        print(f"  Max logit diff:{m['max_logit_diff']:.4f}", flush=True)
        print(f"  Prob L1 dist:  {m['prob_l1']:.6f}", flush=True)

    # Aggregate
    total_tokens = sum(m['num_tokens'] for m in all_metrics)
    avg_ear = sum(m['ear'] * m['num_tokens'] for m in all_metrics) / total_tokens
    avg_kl = sum(m['kl_orig_quant'] * m['num_tokens'] for m in all_metrics) / total_tokens
    avg_js = sum(m['js_divergence'] * m['num_tokens'] for m in all_metrics) / total_tokens
    avg_top1 = sum(m['top1_agreement'] * m['num_tokens'] for m in all_metrics) / total_tokens
    avg_top5 = sum(m['top5_agreement'] * m['num_tokens'] for m in all_metrics) / total_tokens
    avg_max_diff = sum(m['max_logit_diff'] * m['num_tokens'] for m in all_metrics) / total_tokens
    avg_prob_l1 = sum(m['prob_l1'] * m['num_tokens'] for m in all_metrics) / total_tokens

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY (weighted by tokens)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total tokens:       {total_tokens}", flush=True)
    print(f"  EAR (mean):         {avg_ear:.6f}", flush=True)
    print(f"  KL(o||q):           {avg_kl:.6f}", flush=True)
    print(f"  JS divergence:      {avg_js:.6f}", flush=True)
    print(f"  Top-1 agreement:    {avg_top1*100:.2f}%", flush=True)
    print(f"  Top-5 agreement:    {avg_top5*100:.2f}%", flush=True)
    print(f"  Max logit diff:     {avg_max_diff:.4f}", flush=True)
    print(f"  Prob L1 distance:   {avg_prob_l1:.6f}", flush=True)

    # Classification
    if avg_ear >= 0.99:
        classification = "distribution-lossless (EAR >= 0.99)"
    elif avg_ear >= 0.95:
        classification = "near-lossless (0.95 <= EAR < 0.99)"
    else:
        classification = "significant distribution shift (EAR < 0.95)"
    print(f"\n  Classification:     {classification}", flush=True)

    # Save results
    results = {
        'model_orig': args.model_orig,
        'model_quant': args.model_quant,
        'num_prompts': len(prompts),
        'total_tokens': total_tokens,
        'summary': {
            'ear': avg_ear,
            'kl_orig_quant': avg_kl,
            'js_divergence': avg_js,
            'top1_agreement': avg_top1,
            'top5_agreement': avg_top5,
            'max_logit_diff': avg_max_diff,
            'prob_l1': avg_prob_l1,
            'classification': classification,
        },
        'per_prompt': all_metrics,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
