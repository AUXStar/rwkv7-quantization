#!/usr/bin/env python3
"""#6 长序列 state 累积误差测量（state 张量级，补齐验收）

对最终 1.5B 量化方案（W4A4/W8A8 混合 + fused kernel），
在 128/512/2048/4096/8192 长度上，逐步捕获每层 wkv state，
对比原始 bf16 vs 量化模型：

  mse / rel_err / cosine  (逐层、逐步)
  输出 state_mse_vs_step.png / heatmap / cosine / vs_len 图
"""
import sys, os, json, math, time, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")

import torch
import numpy as np

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
QUANT_MODEL = "/tmp/rwkv7-1.5b-state.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"
OUT_DIR = "/home/njzy/test/eval_tmp/state_analysis"
LENGTHS = [128, 512, 2048, 4096, 8192]
CHECK_LAYERS = [0, 4, 11, 15, 19, 23]

from plot_utils import line_chart, heatmap


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


def run_seq(model_orig, model_quant, seg):
    """Run both models token-by-token, compute state metrics per step (rolling, O(1) mem)."""
    s_orig = model_orig.zero_state(1)
    s_quant = model_quant.zero_state(1)
    res = {layer: {"mse": [], "rel": [], "cos": []} for layer in CHECK_LAYERS}
    for t in seg:
        tok = torch.tensor([[t]], dtype=torch.long)
        model_orig.forward(tok, s_orig)
        model_quant.forward(tok, s_quant)
        for layer in CHECK_LAYERS:
            so = s_orig[1][layer].float()
            sq = s_quant[1][layer].float()
            mse = (so - sq).pow(2).mean().item()
            rel = math.sqrt(mse) / (so.abs().mean().item() + 1e-8)
            cos = torch.nn.functional.cosine_similarity(so.flatten(), sq.flatten(), dim=0).item()
            res[layer]["mse"].append(mse)
            res[layer]["rel"].append(rel)
            res[layer]["cos"].append(cos)
    return res


def main():
    print("=" * 60, flush=True)
    print("#6 state MSE: 1.5B final scheme (fused) vs original", flush=True)
    print("=" * 60, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    from quantize_model import quantize_model
    if not os.path.exists(QUANT_MODEL):
        quantize_model(MODEL_1_5B, QUANT_MODEL, scheme_name="1.5b")

    with open(f"{EVAL_DIR}/gen_8192.json") as f:
        tokens = json.load(f)["tokens"]

    print("loading original model...", flush=True)
    model_orig = build_model(MODEL_1_5B)
    print("loading quantized model...", flush=True)
    model_quant = build_model(QUANT_MODEL)

    results = {}   # length -> {layer -> {"mse":[...], "rel":[...], "cos":[...]}}
    for LEN in LENGTHS:
        seg = tokens[:LEN]
        print(f"\n--- seq_len={LEN} ---", flush=True)
        t0 = time.perf_counter()

        res = run_seq(model_orig, model_quant, seg)
        results[LEN] = res
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
        for layer in CHECK_LAYERS:
            print(f"  L{layer:>2}: final MSE={res[layer]['mse'][-1]:.3e} "
                  f"rel={res[layer]['rel'][-1]*100:.2f}% cos={res[layer]['cos'][-1]:.6f}", flush=True)

    # ---- plots ----
    xs = list(range(8192))
    ys = [results[8192][layer]["mse"] for layer in CHECK_LAYERS]
    labels = [f"L{l}" for l in CHECK_LAYERS]
    line_chart([xs]*len(labels), ys, labels, "State MSE vs step (8192 tokens, log y)",
               "step", "state MSE", f"{OUT_DIR}/state_mse_vs_step.png", log_y=True)

    ys = [results[8192][layer]["cos"] for layer in CHECK_LAYERS]
    line_chart([xs]*len(labels), ys, labels, "State cosine vs step (8192 tokens)",
               "step", "cosine", f"{OUT_DIR}/state_cosine_vs_step.png")

    ys = [results[8192][layer]["rel"] for layer in CHECK_LAYERS]
    line_chart([xs]*len(labels), ys, labels, "State rel_err vs step (8192 tokens, log y)",
               "step", "rel_err", f"{OUT_DIR}/state_rel_vs_step.png", log_y=True)

    mat = np.zeros((len(LENGTHS), len(CHECK_LAYERS)))
    for i, LEN in enumerate(LENGTHS):
        for j, layer in enumerate(CHECK_LAYERS):
            mat[i, j] = results[LEN][layer]["mse"][-1]
    heatmap(mat, [str(l) for l in LENGTHS], [f"L{l}" for l in CHECK_LAYERS],
            "Final state MSE (log10)", f"{OUT_DIR}/state_mse_heatmap.png")

    ys = [[results[LEN][layer]["mse"][-1] for LEN in LENGTHS] for layer in CHECK_LAYERS]
    xs_lens = [list(LENGTHS)] * len(CHECK_LAYERS)
    line_chart(xs_lens, ys, labels, "Final state MSE vs seq_len (log-log)",
               "seq_len", "final MSE", f"{OUT_DIR}/state_mse_vs_len.png",
               log_y=True, log_x=True)

    # ---- summary + acceptance ----
    print(f"\n{'='*60}", flush=True)
    print("#6 State MSE 汇总 (最终 step, L23 为最深层)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'len':>6} {'L0 mse':>10} {'L23 mse':>10} {'L23 cos':>10} {'L23 rel':>8}", flush=True)
    for LEN in LENGTHS:
        m0 = results[LEN][0]["mse"][-1]
        m23 = results[LEN][23]["mse"][-1]
        c23 = results[LEN][23]["cos"][-1]
        r23 = results[LEN][23]["rel"][-1]
        print(f"{LEN:>6} {m0:>10.2e} {m23:>10.2e} {c23:>10.6f} {r23*100:>7.2f}%", flush=True)

    print(f"\n验收核对 (L23 为最深层):", flush=True)
    checks = [
        (128, 1e-4, "128: MSE ≤ 1e-4"),
        (512, 5e-3, "512: MSE ≤ 5e-3"),
        (2048, 1e-2, "2048: MSE ≤ 1e-2"),
        (4096, 5e-2, "4096: MSE ≤ 5e-2"),
        (8192, 1e-1, "8192: MSE ≤ 1e-1"),
    ]
    all_ok = True
    for LEN, thresh, label in checks:
        m23 = results[LEN][23]["mse"][-1]
        ok = m23 <= thresh
        all_ok &= ok
        print(f"  {'✓' if ok else '✗'} {label}: {m23:.2e}", flush=True)

    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)

    # save raw data
    with open(f"{OUT_DIR}/state_mse_results.json", "w") as f:
        json.dump({str(L): {str(l): {k: res[l][k] for k in ("mse", "rel", "cos")}
                             for l in res} for L, res in results.items()}, f)

    del model_orig, model_quant
    gc.collect()
    torch.cuda.empty_cache()
    os.remove(QUANT_MODEL)


if __name__ == "__main__":
    main()
