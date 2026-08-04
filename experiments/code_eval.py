#!/usr/bin/env python3
"""Code generation comparison: original vs FP8 quantized.

Tests greedy code completion on HumanEval-style prompts.
Measures: token-level agreement, exact match, first-divergence position.
"""
import sys, os, gc, json, time
import torch
import torch.nn.functional as F

ENGINE_DIR = "/home/njzy/test/Albatross/faster3a_2607"
QUANT_DIR = "/home/njzy/test/rwkv7-quantization"
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, QUANT_DIR)

# Diverse code prompts (function signatures + docstrings)
CODE_PROMPTS = [
    # Python: algorithms
    "def bubble_sort(arr):\n    \"\"\"Sort array using bubble sort.\"\"\"\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return",
    # Python: math
    "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if",
    # Python: string manipulation
    "def reverse_words(s):\n    \"\"\"Reverse order of words in string.\"\"\"\n    words = s.split()\n    return",
    # Python: data structure
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        if not self.is_empty():\n            return",
    # Python: recursion
    "def factorial(n):\n    \"\"\"Compute factorial recursively.\"\"\"\n    if n <= 1:\n        return 1\n    return",
    # Python: list comprehension
    "def flatten(nested):\n    \"\"\"Flatten a nested list.\"\"\"\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(",
    # Python: dictionary
    "def word_count(text):\n    \"\"\"Count word frequencies.\"\"\"\n    words = text.split()\n    freq = {}\n    for w in words:\n        if w in freq:\n            freq[w] += 1\n        else:\n            freq[w] =",
    # Python: binary search
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo =",
    # Python: file I/O
    "def read_csv(filename):\n    \"\"\"Read CSV file and return list of dicts.\"\"\"\n    import csv\n    with open(filename) as f:\n        reader = csv.DictReader(f)\n        return",
    # Python: class inheritance
    "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        pass\n\nclass Dog(Animal):\n    def speak(self):\n        return",
    # C-style: linked list
    "struct Node {\n    int val;\n    struct Node *next;\n};\n\nstruct Node* reverse_list(struct Node* head) {\n    struct Node *prev = NULL, *curr = head, *next;\n    while (curr) {\n        next = curr->next;\n        curr->next = prev;\n        prev = curr;\n        curr =",
    # Python: decorators
    "def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] =",
    # Python: generator
    "def fib_gen():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b =",
    # Python: exception handling
    "def safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return",
    # Python: sorting with key
    "def sort_by_length(strings):\n    \"\"\"Sort strings by length, then alphabetically.\"\"\"\n    return sorted(strings, key=",
]

def load_model_and_generate(model_path, prompts, gen_len=128):
    """Load model, generate code completions for each prompt."""
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

    results = []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 256:
            tokens = tokens[:256]

        token_tensor = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
        state = model.zero_state(1)

        # Prefill
        with torch.no_grad():
            logits = model.forward(token_tensor, state)
            # logits: [1, V] (last token only for forward)

        # Greedy generate
        generated_tokens = []
        current_token = logits[0].argmax().item() if logits.dim() == 2 else logits[0, -1].argmax().item()

        for step in range(gen_len):
            generated_tokens.append(current_token)
            token_tensor = torch.tensor([[current_token]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                logits = model.forward(token_tensor, state)
            current_token = logits[0].argmax().item() if logits.dim() == 2 else logits[0, -1].argmax().item()
            # Stop on newline-newline (end of function) or EOF
            if current_token == 0:  # EOF token
                break

        generated_text = tokenizer.decode(generated_tokens)
        results.append({
            "prompt": prompt,
            "prompt_tokens": len(tokens),
            "generated_tokens": generated_tokens,
            "generated_text": generated_text,
        })
        print(f"  [{i+1}/{len(prompts)}] prompt={len(tokens)}tok, generated={len(generated_tokens)}tok", flush=True)

    # Cleanup
    del model
    del v3a
    gc.collect()
    torch.cuda.empty_cache()
    for mod_name in list(sys.modules.keys()):
        if 'rwkv7_fast_v3a' in mod_name or 'fp8_ops' in mod_name or 'fused_fp8' in mod_name:
            del sys.modules[mod_name]

    return results


def compare_results(orig_results, quant_results):
    """Compare original vs quantized generation."""
    comparisons = []
    for i, (o, q) in enumerate(zip(orig_results, quant_results)):
        o_tokens = o["generated_tokens"]
        q_tokens = q["generated_tokens"]

        # Token-level comparison
        min_len = min(len(o_tokens), len(q_tokens))
        if min_len == 0:
            match_rate = 0.0
            first_div = 0
        else:
            matches = sum(1 for a, b in zip(o_tokens, q_tokens) if a == b)
            match_rate = matches / min_len
            first_div = min_len
            for j in range(min_len):
                if o_tokens[j] != q_tokens[j]:
                    first_div = j
                    break

        # Exact match (same length + same tokens)
        exact_match = (o_tokens == q_tokens)

        # Same length
        same_length = (len(o_tokens) == len(q_tokens))

        comparisons.append({
            "prompt_idx": i,
            "prompt_preview": o["prompt"][:60] + "...",
            "orig_len": len(o_tokens),
            "quant_len": len(q_tokens),
            "same_length": same_length,
            "exact_match": exact_match,
            "token_match_rate": match_rate,
            "first_divergence": first_div,
            "orig_text": o["generated_text"],
            "quant_text": q["generated_text"],
        })

    return comparisons


def main():
    MODEL_ORIG = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
    MODEL_FP8 = "/home/njzy/model/rwkv7-1.5b-fp8-ffn-only.pth"
    GEN_LEN = 128

    print("=" * 70, flush=True)
    print(f"Code Generation Comparison: Original vs FFN-only FP8 (1.5B)", flush=True)
    print(f"Prompts: {len(CODE_PROMPTS)}, Gen length: {GEN_LEN} tokens", flush=True)
    print("=" * 70, flush=True)

    # Generate with original
    print("\n--- Original Model ---", flush=True)
    t0 = time.time()
    orig_results = load_model_and_generate(MODEL_ORIG, CODE_PROMPTS, GEN_LEN)
    print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

    # Generate with FP8
    print("--- FP8 Quantized Model ---", flush=True)
    t0 = time.time()
    quant_results = load_model_and_generate(MODEL_FP8, CODE_PROMPTS, GEN_LEN)
    print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

    # Compare
    print("--- Comparison ---", flush=True)
    comparisons = compare_results(orig_results, quant_results)

    exact_matches = sum(1 for c in comparisons if c["exact_match"])
    same_lengths = sum(1 for c in comparisons if c["same_length"])
    avg_match = sum(c["token_match_rate"] for c in comparisons) / len(comparisons)
    avg_first_div = sum(c["first_divergence"] for c in comparisons) / len(comparisons)

    print(f"\n{'#':<4} {'Prompt':<55} {'Orig':>5} {'FP8':>5} {'Match':>6} {'1stDiv':>7} {'Exact'}", flush=True)
    print("-" * 90, flush=True)
    for c in comparisons:
        preview = c["prompt_preview"][:53]
        print(f"{c['prompt_idx']:<4} {preview:<55} {c['orig_len']:>5} {c['quant_len']:>5} {c['token_match_rate']*100:>5.1f}% {c['first_divergence']:>7} {'YES' if c['exact_match'] else 'no'}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Prompts:           {len(comparisons)}", flush=True)
    print(f"  Exact matches:     {exact_matches}/{len(comparisons)} ({exact_matches/len(comparisons)*100:.1f}%)", flush=True)
    print(f"  Same length:       {same_lengths}/{len(comparisons)} ({same_lengths/len(comparisons)*100:.1f}%)", flush=True)
    print(f"  Avg token match:   {avg_match*100:.1f}%", flush=True)
    print(f"  Avg 1st diverge:   {avg_first_div:.1f} tokens", flush=True)

    # Show side-by-side for divergent cases
    print(f"\n{'='*70}", flush=True)
    print("DIVERGENT CASES (first 3)", flush=True)
    print(f"{'='*70}", flush=True)
    shown = 0
    for c in comparisons:
        if c["exact_match"]:
            continue
        if shown >= 3:
            break
        shown += 1
        print(f"\n--- Prompt {c['prompt_idx']} (1st divergence at token {c['first_divergence']}) ---", flush=True)
        print(f"Prompt: {c['prompt_preview']}", flush=True)
        print(f"\nOriginal output:", flush=True)
        print(c['orig_text'][:300], flush=True)
        print(f"\nFP8 output:", flush=True)
        print(c['quant_text'][:300], flush=True)
        print(flush=True)

    # Save
    results = {
        "model_orig": MODEL_ORIG,
        "model_quant": MODEL_FP8,
        "gen_len": GEN_LEN,
        "num_prompts": len(CODE_PROMPTS),
        "summary": {
            "exact_matches": exact_matches,
            "same_length": same_lengths,
            "avg_token_match": avg_match,
            "avg_first_divergence": avg_first_div,
        },
        "comparisons": comparisons,
    }
    out_path = "/home/njzy/test/rwkv7-quantization/experiments/code_eval_ffn_only.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
