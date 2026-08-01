#!/usr/bin/env python3
"""测试拆分 ffn_key_res: NVFP4 主路 + FP8 残差路分开跑 vs 融合。

假设: 融合 kernel 寄存器压力大 (2 个 accumulator), 拆开后每个 kernel
可能用更大 BLOCK_N, 减少 K 循环迭代次数。

同时测试: ffn_value 的 prep_x 能否复用 (省 1 次 launch).
"""
import sys, os, json, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
sys.path.insert(0, "/home/njzy/test/Albatross/faster3a_2605")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch
import triton
import triton.language as tl

MODEL = "/home/njzy/model/rwkv7-7.2b-X5.pth"


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


def bench(fn, n=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n


def main():
    print("=" * 70)
    print("ffn_key_res 拆分测试 (7.2B, C=4096, M=1)")
    print("=" * 70)

    m = build_model()
    import fused_nvfp4_gemm as fused
    z = m.z
    C = 4096
    device = "cuda"
    x = torch.randn(1, C, dtype=torch.float16, device=device)
    wfk = z["blocks.1.ffn.key.weight"]  # NVFP4+res dict

    # 当前: 融合 kernel
    t_fused = bench(lambda: fused.linear_nvfp4_res_fused(x, wfk))
    print(f"\n[1] 融合 (当前):        {t_fused:.4f} ms")

    # 拆分: NVFP4 主路 + FP8 残差路, Python 端合并
    # 主路: linear_nvfp4_fused (只用 NVFP4 weight, 不含 res)
    # 残差: linear_fp8_fused (用 res_fp8 weight)
    w_main = {
        "weight": wfk["weight"],
        "block_scale": wfk["block_scale"],
        "tensor_scale": wfk["tensor_scale"],
        "awq_scale": wfk.get("awq_scale"),
        "qtype": "nvfp4_fused",
    }
    w_res = {
        "weight": wfk["res_fp8"],
        "tensor_scale": wfk.get("res_fp8_scale"),
        "qtype": "fp8",
    }
    # 如果有 per-block residual scale
    if "res_block_scale" in wfk:
        w_res["block_scale"] = wfk["res_block_scale"]

    # 测试拆分 (2 个 prep_x + 2 个 GEMM + 1 个 add)
    def split_fn():
        out_main = fused.linear_nvfp4_fused(x, w_main)
        out_res = fused.linear_fp8_fused(x, w_res)
        # scale and add (scale factors 与融合 kernel 相同)
        # 融合 kernel: out = acc_main * (amax/2688 * w_ts) + acc_res * (amax/448 * res_ts)
        # 但 linear_nvfp4_fused 已经 fold 了 scale, linear_fp8_fused 也 fold 了
        # 所以直接 add 即可
        return out_main + out_res

    t_split = bench(split_fn)
    print(f"[2] 拆分 (nvfp4+fp8+add): {t_split:.4f} ms")
    print(f"    ratio: {t_fused/t_split:.2f}x ({'拆分更快' if t_split < t_fused else '融合更快'})")

    # 更进一步: 拆分 + 复用 prep_x (只做 1 次 prep_x, 2 个 GEMM 复用)
    # 这需要手动调用 prep_x + 2 个 kernel
    from fused_nvfp4_gemm import prep_x, fused_nvfp4_gemm_kernel, fused_fp8_gemm_kernel, _nvfp4_cfg_for

    def split_shared_prep():
        x_awq, amax = prep_x(x, wfk.get("awq_scale"))
        M, K = x_awq.shape
        N = w_main["weight"].size(0)
        bm, bn, bk, nw = _nvfp4_cfg_for(M)

        # NVFP4 GEMM
        out_main = torch.empty(M, N, dtype=torch.float16, device=device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        fused_nvfp4_gemm_kernel[grid](
            x_awq, w_main["weight"], w_main["block_scale"].view(torch.uint8),
            w_main["tensor_scale"], amax, out_main,
            M, N, K, x_awq.stride(0),
            w_main["weight"].stride(0), w_main["block_scale"].stride(0),
            out_main.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
        )

        # FP8 GEMM (复用 amax)
        out_res = torch.empty(M, N, dtype=torch.float16, device=device)
        w_r = w_res["weight"]
        w_r_ts = w_res["tensor_scale"]
        grid2 = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        fused_fp8_gemm_kernel[grid2](
            x_awq, w_r.view(torch.uint8), w_r_ts, amax, out_res,
            M, N, K, x_awq.stride(0), w_r.stride(0), out_res.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=nw,
        )
        return out_main + out_res

    t_shared = bench(split_shared_prep)
    print(f"[3] 拆分+共享prep_x:    {t_shared:.4f} ms")
    print(f"    ratio: {t_fused/t_shared:.2f}x ({'拆分更快' if t_shared < t_fused else '融合更快'})")

    # 数值正确性验证
    out_fused = fused.linear_nvfp4_res_fused(x, wfk)
    out_split = split_fn()
    max_diff = (out_fused - out_split).abs().max().item()
    print(f"\n[4] 数值验证: max_diff = {max_diff:.6f} ({'OK' if max_diff < 0.1 else 'MISMATCH'})")

    # 拆分后单独测各部分
    t_prep = bench(lambda: prep_x(x, wfk.get("awq_scale")))
    t_nvfp4 = bench(lambda: fused.linear_nvfp4_fused(x, w_main))
    t_fp8 = bench(lambda: fused.linear_fp8_fused(x, w_res))
    print(f"\n[5] 拆分各部分耗时:")
    print(f"    prep_x:      {t_prep:.4f} ms")
    print(f"    nvfp4 GEMM:  {t_nvfp4:.4f} ms")
    print(f"    fp8 GEMM:    {t_fp8:.4f} ms")
    print(f"    add:         ~0.005 ms (估计)")
    print(f"    总计:        {t_prep + t_nvfp4 + t_fp8 + 0.005:.4f} ms")

    out = {
        "fused": t_fused, "split": t_split, "split_shared": t_shared,
        "prep_x": t_prep, "nvfp4_only": t_nvfp4, "fp8_only": t_fp8,
        "max_diff": max_diff,
    }
    with open("/home/njzy/test/eval_tmp/ffn_split_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved ffn_split_test.json")


if __name__ == "__main__":
    main()
