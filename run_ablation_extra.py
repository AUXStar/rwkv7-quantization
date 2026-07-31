#!/usr/bin/env python3
"""#4C / #3-W8A16 / #5-nvfp4 消融补齐（1.5B）

一次性跑多个消融组，输出 PPL delta / Top-1 / logits MSE / 逐层 state MSE：
  - C: L4-19 key+value 都 NVFP4（#4 消融组C）
  - w8a16: 全 FP8 但激活不量化（W8A16，#3 issue 原始要求）
  - l0nvfp4: L0 value NVFP4（#5 组3）
"""
import sys, os, json, math, time, gc
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")

import torch

MODEL_1_5B = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
EVAL_DIR = "/home/njzy/test/eval_tmp"
LOGITS_ORIG = f"{EVAL_DIR}/logits_orig_1_5b_2100.pt"

FP8, NVFP4, NVFP4_RES = 1, 2, 3

GROUPS = {
    "KVFP8_W8A16": {  # #3 独立验证: 仅 key/value FP8 (W8A16), 其他全 bf16
        "key":   [(0, 23, FP8)],
        "value": [(0, 23, FP8)],
        "desc": "仅key/value FP8 W8A16, 其他bf16 (#3独立)",
        "only_kv": True,
    },
    "C": {   # #4 组C: L4-19 key+value NVFP4
        "key":   [(0, 3, FP8), (4, 19, NVFP4), (20, 23, FP8)],
        "value": [(0, 3, FP8), (4, 19, NVFP4), (20, 23, FP8)],
        "desc": "L4-19 key+value NVFP4 (#4组C)",
    },
    "W8A16": {  # #3: 全 FP8 权重, 激活不量化 (完整方案背景下)
        "key":   [(0, 23, FP8)],
        "value": [(0, 23, FP8)],
        "desc": "完整方案+全FP8 W8A16 (#3补充)",
    },
    "L0NVFP4": {  # #5 组3: L0 value NVFP4
        "key":   [(0, 3, FP8), (4, 19, NVFP4), (20, 23, FP8)],
        "value": [(0, 0, NVFP4), (1, 23, FP8)],
        "desc": "L0 value NVFP4 (#5组3)",
    },
}


def build_scheme(kv_cfg):
    """Build quantize scheme list from kv_cfg (overrides key/value, keeps rest)."""
    scheme = []
    if not kv_cfg.get("only_kv"):
        # rec/out NVFP4 everywhere
        scheme += [
            [0, 23, 0, NVFP4],
            [0, 23, 3, NVFP4],
            # FFN key nvfp4+res, FFN value fp8
            [0, 23, 4, NVFP4_RES],
            [0, 23, 5, FP8],
        ]
    for (s, e, dt) in kv_cfg["key"]:
        scheme.append([s, e, 1, dt])
    for (s, e, dt) in kv_cfg["value"]:
        scheme.append([s, e, 2, dt])
    return scheme


def build_model(model_path, w8a16=False):
    import rwkv7_fast_v3a as engine
    from rwkv7_fast_v3a import RWKV7, load_extensions
    engine.NVFP4_W4A16 = False
    engine.FP8_W8A16 = w8a16
    engine.FUSED_GEMM = not w8a16
    engine.WKV_MODE = "fp16"
    engine.EMB_DEVICE = "cpu"
    engine.RKV_MODE = "off"
    engine.CMIX_SPARSE = "no-fc"
    engine.LOWRANK_WEIGHT = "both"
    engine.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    engine.MODEL_PATH = model_path
    load_extensions(engine.WKV_MODE)
    return RWKV7()


def eval_group(name, cfg):
    print(f"\n{'='*60}", flush=True)
    print(f"Group {name}: {cfg['desc']}", flush=True)
    print(f"{'='*60}", flush=True)

    from quantize_model import quantize_model, SCHEMES
    qpath = f"/tmp/rwkv7-1.5b-{name}.pth"
    scheme = build_scheme(cfg)
    quantize_model(MODEL_1_5B, qpath, _scheme_override=scheme)

    w8a16 = (name == "W8A16")
    model = build_model(qpath, w8a16=w8a16)

    with open(f"{EVAL_DIR}/test_2100.json") as f:
        tokens = json.load(f)["tokens"]
    logits_orig = torch.load(LOGITS_ORIG, map_location="cpu").float()

    tok_tensor = torch.tensor([tokens], dtype=torch.long)
    state = model.zero_state(1)
    out = model.forward_all_logits(tok_tensor, state)
    logits_q = out[0].float().cpu()

    n = min(logits_orig.shape[0], logits_q.shape[0]) - 1
    lo, lq = logits_orig[:n], logits_q[:n]
    targets = torch.tensor(tokens[1:n+1], dtype=torch.long)

    top1 = (lo.argmax(dim=-1) == lq.argmax(dim=-1)).float().mean().item()
    ce_o = torch.nn.functional.cross_entropy(lo, targets).item()
    ce_q = torch.nn.functional.cross_entropy(lq, targets).item()
    ppl_delta = math.exp(ce_q) - math.exp(ce_o)

    # logits MSE (分段: 短128 / 中512 / 长2048)
    mse_short = (lo[:128] - lq[:128]).pow(2).mean().item()
    mse_med = (lo[:512] - lq[:512]).pow(2).mean().item()
    mse_long = (lo[:2048] - lq[:2048]).pow(2).mean().item()

    print(f"  PPL delta: {ppl_delta:+.4f}", flush=True)
    print(f"  Top-1:     {top1*100:.2f}%", flush=True)
    print(f"  logits MSE: 128={mse_short:.3e} 512={mse_med:.3e} 2048={mse_long:.3e}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    os.remove(qpath)
    return {"ppl_delta": ppl_delta, "top1": top1, "mse_128": mse_short, "mse_512": mse_med, "mse_2048": mse_long}


def main():
    import sys as _sys
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    results = {}
    for name, cfg in GROUPS.items():
        if only and name != only:
            continue
        try:
            results[name] = eval_group(name, cfg)
        except Exception as e:
            print(f"Group {name} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
    print("\n=== SUMMARY ===", flush=True)
    for name, r in results.items():
        print(f"  {name}: PPL delta={r['ppl_delta']:+.4f} Top-1={r['top1']*100:.2f}% "
              f"MSE(128/512/2048)={r['mse_128']:.1e}/{r['mse_512']:.1e}/{r['mse_2048']:.1e}", flush=True)


if __name__ == "__main__":
    main()
