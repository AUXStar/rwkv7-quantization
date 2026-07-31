#!/usr/bin/env python3
"""Tune fused kernels for M=1 decode: rkv (N=2048) + res (N=8192)."""
import sys, os, time
sys.path.insert(0, "/home/njzy/test/rwkv7-quantization")
os.chdir("/home/njzy/test/rwkv7-quantization")
import torch
import triton

from quantize_model import quantize_nvfp4, quantize_to_fp8, quantize_nvfp4_with_residual, compute_awq_scale, ALPHA
import fused_nvfp4_gemm as fused

torch.manual_seed(0)


def bench(fn, n_iters=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6  # us


def launch_rkv(xr, xk, xv, wr, wk, wv, amax_r, amax_k, amax_v, M, N, K, bm, bn, bk, nw, k_fp4):
    or_, ok_, ov_ = (torch.empty(M, N, dtype=torch.float16, device="cuda") for _ in range(3))
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused.fused_rkv_gemm_kernel[grid](
        xr, xk, xv,
        wr[0], wr[1].view(torch.uint8), wr[2],
        wk[0], wk[1].view(torch.uint8) if k_fp4 else wk[0].view(torch.uint8), wk[2],
        wv[0].view(torch.uint8), wv[1],
        amax_r, amax_k, amax_v,
        or_, ok_, ov_,
        M, N, K,
        xr.stride(0),
        wr[0].stride(0), wr[1].stride(0),
        wk[0].stride(0) if not k_fp4 else wr[0].stride(0),
        wv[0].stride(0),
        or_.stride(0),
        K_IS_FP4=k_fp4,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )
    return or_, ok_, ov_


def launch_res(x, w, amax, M, N, K, bm, bn, bk, nw):
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused.fused_nvfp4_res_gemm_kernel[grid](
        x, w[0], w[1].view(torch.uint8), w[2], w[3].view(torch.uint8), w[4], amax, out,
        M, N, K,
        x.stride(0),
        w[0].stride(0), w[1].stride(0), w[3].stride(0),
        out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=nw,
    )
    return out


def main():
    import triton
    K, C = 2048, 2048
    # rkv weights
    w_r = torch.randn(C, K, dtype=torch.bfloat16) * 0.02
    awq_r = compute_awq_scale(w_r, act_stats=None, alpha=ALPHA)
    pr, br, tr, _ = quantize_nvfp4(w_r, awq_r, per_channel_ts=False)
    w_k = torch.randn(C, K, dtype=torch.bfloat16) * 0.02
    awq_k = compute_awq_scale(w_k, act_stats=None, alpha=ALPHA)
    pk, bk_, tk, _ = quantize_nvfp4(w_k, awq_k, per_channel_ts=False)
    w_v = torch.randn(C, K, dtype=torch.bfloat16) * 0.02
    wvf, tv = quantize_to_fp8(w_v)
    wr = (pr.to("cuda"), br.to("cuda"), tr.to("cuda"))
    wk = (pk.to("cuda"), bk_.to("cuda"), tk.to("cuda"))
    wv = (wvf.to("cuda"), tv.to("cuda"))

    xr = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.05
    xk = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.04
    xv = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.03
    amax = lambda v: torch.tensor([v], dtype=torch.float32, device="cuda")

    print("=== rkv (M=1, K=2048, N=2048) ===")
    best = None
    for bm, bn, bk, nw in [(16,64,64,4), (16,128,64,4), (16,128,64,8), (16,256,64,8),
                           (32,64,64,4), (32,128,64,8), (16,64,128,4), (16,128,128,8), (8,128,64,4)]:
        try:
            t = bench(lambda: launch_rkv(xr, xk, xv, wr, wk, wv, amax(0.1), amax(0.08), amax(0.06),
                                         1, C, K, bm, bn, bk, nw, True))
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} w={nw}: {t:7.1f}us")
            if best is None or t < best[0]:
                best = (t, bm, bn, bk, nw)
        except Exception as e:
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} w={nw}: FAIL {type(e).__name__}")
    print(f"best rkv: {best[1:]} @ {best[0]:.1f}us")

    # res weights (N=8192)
    N2 = 8192
    w_r2 = torch.randn(N2, K, dtype=torch.bfloat16) * 0.02
    awq2 = compute_awq_scale(w_r2, act_stats=None, alpha=ALPHA)
    pr2, br2, tr2, _, rf2, rt2 = quantize_nvfp4_with_residual(w_r2, awq2)
    wr2 = (pr2.to("cuda"), br2.to("cuda"), tr2.to("cuda"), rf2.to("cuda"), rt2.to("cuda"))
    x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.05

    print("=== res (M=1, K=2048, N=8192) ===")
    best2 = None
    for bm, bn, bk, nw in [(16,64,64,4), (16,128,64,4), (16,128,64,8), (16,256,64,8),
                           (32,128,64,8), (16,64,128,4), (16,128,128,8), (8,128,64,4), (8,256,64,8)]:
        try:
            t = bench(lambda: launch_res(x, wr2, amax(0.1), 1, N2, K, bm, bn, bk, nw))
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} w={nw}: {t:7.1f}us")
            if best2 is None or t < best2[0]:
                best2 = (t, bm, bn, bk, nw)
        except Exception as e:
            print(f"BM={bm:>3} BN={bn:>3} BK={bk:>3} w={nw}: FAIL {type(e).__name__}")
    print(f"best res: {best2[1:]} @ {best2[0]:.1f}us")


if __name__ == "__main__":
    main()
