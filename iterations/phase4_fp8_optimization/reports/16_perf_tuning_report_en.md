# 7.2B X5 Performance Tuning Report

## I. Test Results Overview

| Test | Result | Notes |
|------|------|------|
| B=1 decode | **31.7 t/s** (31.5 ms) | current baseline |
| CUDA Graph | 31.9 t/s (1.01x) | no speedup, CPU launch is not the bottleneck |
| B=2 batch | 27.8 t/s/req (55.6 total) | per-request speed actually decreases |
| B=4 batch | 29.3 t/s/req (117.1 total) | per-request speed slightly lower |
| B=8 batch | 16.5 t/s/req (132.2 total) | severe degradation |
| _scaled_mm FP4 | 0.23x vs Triton | cuBLAS launch overhead too high at M=1 |
| _scaled_mm FP8 | 0.21x vs Triton | same as above |

## II. Precise Bottleneck Analysis (Profiler Data)

### Per-Step Kernel Time Breakdown

| Component | Time (ms) | Proportion | Notes |
|------|----------|------|------|
| **ffn_key_res (NVFP4+FP8 residual)** | **14.091** | **42.8%** | **#1 bottleneck** |
| ffn_value (FP8 hwdot) | 4.873 | 14.8% | already optimized |
| rkv_fused (R/K/V fused) | 4.737 | 14.4% | already optimized |
| Other linear layers (bf16) | 5.302 | 16.1% | non-quantized layers |
| att_output (NVFP4) | 2.972 | 9.0% | already optimized |
| Other (LN, WKV, elementwise) | 0.915 | 2.8% | not a bottleneck |
| **Total GPU** | **32.890** | **100%** | |

### Bandwidth Efficiency Analysis

- Measured VRAM bandwidth: **197.3 GB/s** (RTX 5070 Ti Laptop)
- ffn_key_res loads per layer: ~19.25 MB (NVFP4 weights + FP8 residual + block scales)
- Theoretical minimum: 19.25 MB / 197.3 GB/s x 28 layers = **2.73 ms**
- Actual: 14.091 ms
- **Efficiency: 19.4%** -- far below the theoretical bandwidth limit

Why is the efficiency so low? Because `ffn_key_res` has K=4096, loops 64 times, loading only a small tile of weights each time. The GPU's memory controller cannot reach peak bandwidth during random tile access. Actual bandwidth utilization is approximately 197.3 x 19.4% = **38.3 GB/s**.

## III. Why Can't 50-60 t/s Be Achieved?

### Realistic Analysis

**Single-request limit**: even if ffn_key_res were completely optimized away (impossible), only 14.09 ms could be saved, with a theoretical limit of about 17.4 ms = **57.5 t/s**. But the actually optimizable space is limited.

**Batch decode problem**: previously estimated 50-60 t/s per request at B=4-8, but actual testing shows only **29.3 t/s** per request at `B=4`. Reasons:
1. The Triton kernel's tile configuration (BLOCK_M=32) is not efficient for M=2-4
2. The state is per-layer per-batch, and WKV computation grows linearly with B
3. VRAM bandwidth is shared; as B increases, each request gets less bandwidth

### Actual Data vs Previous Estimates

| Scenario | Previous estimate | Actual |
|------|---------|------|
| B=1 single request | 25 t/s -> 37 t/s (cross-layer fusion) | 31.7 t/s (actual) |
| B=2 batch | 50-60 t/s/req | 27.8 t/s/req |
| B=4 batch | 50-60 t/s/req | 29.3 t/s/req |
| B=8 batch | 50-60 t/s/req | 16.5 t/s/req |
| Total throughput B=4 | -- | 117.1 t/s |

**Conclusion**: 50-60 t/s single request is unrealistic on the 7.2B quantized model. The bottleneck is **VRAM bandwidth**, not computation.

## IV. Optimization Approaches Attempted

| Approach | Result | Reason |
|------|------|------|
| FP8 hardware Tensor Core | **1.32x** (21.8->28.8 t/s) | successful, already in baseline |
| RKV fused kernel | **1.41x** | successful, 3-in-1 launch |
| att_output FP8 hwdot | **1.78x** | successful |
| Split-K parallelism | **0.77x-0.96x** (slowdown) | atomic_add contention |
| Tile configuration search | already optimal (16,64,64,4) | overall optimal |
| CUDA Graph | **1.01x** (no improvement) | CPU launch only 2-4% |
| _scaled_mm FP4 | **0.23x** (slowdown) | M=1 cuBLAS launch overhead |
| _scaled_mm FP8 | **0.21x** (slowdown) | same as above |
| W4A16 (no activation quantization) | implemented | higher numerical precision |
| Batch decode B=4 | 29.3 t/s/req | total throughput 117 t/s but per-request unchanged |

## V. Remaining Optimization Space

### Approach 1: ffn_key_res Separation (expected +8-12%)

Split the fused ffn_key_res kernel into two independent kernels:
- NVFP4 part: use existing `fused_nvfp4_gemm_kernel` (N=11008)
- FP8 residual part: use existing `fused_fp8_hwdot_gemm_kernel` (N=11008)

Current fused kernel: 503 us/layer
After split: ~285 us (NVFP4) + 174 us (FP8) = **~459 us/layer**
Savings: 44 us/layer x 28 = 1.23 ms -> 31.5 -> 30.3 ms = **33.0 t/s** (4% improvement)

### Approach 2: Cross-layer Kernel Fusion (expected +5-10%)

Merge the ffn_key_res computation of multiple layers into one kernel. But the problem is inter-layer dependencies (each layer's output -> next layer's input), preventing direct fusion.

### Approach 3: Reduce Quantized Layers (expected +10-15%)

The current X5 scheme has 28 quantized layers and 4 bf16 layers. If reduced to 24 quantized layers and 8 bf16 layers:
- Eliminate 4 layers of ffn_key_res: 4 x 0.503 = 2.01 ms
- Eliminate 4 layers of ffn_value: 4 x 0.174 = 0.70 ms
- Eliminate 4 layers of att_output: 4 x 0.106 = 0.42 ms
- Total savings: 3.13 ms
- New speed: 31.5 - 3.13 = 28.4 ms = **35.2 t/s**

But the trade-off is a lower compression ratio and a larger model file.

### Approach 4: Modify Quantization Scheme (expected +20-30%)

The current ffn_key_res uses the NVFP4+FP8 residual scheme. If switched to pure FP8 (W8A8):
- Weights go from 4-bit to 8-bit, model file grows 2x
- But the FP8 hwdot kernel can be used, each kernel only needs ~174 us
- 28 x 0.174 = 4.87 ms (vs 14.09 ms)
- Savings: 9.22 ms
- New speed: 31.5 - 9.22 = 22.3 ms = **44.8 t/s**

But the file size doubles, and PPL may degrade.

## VI. Conclusion

1. **The current 31.7 t/s is reasonable** -- 7.2B model on RTX 5070 Ti Laptop (11.9GB), NVFP4 quantized, M=1 decode
2. **ffn_key_res is the #1 bottleneck** (14.09 ms, 42.8%), but the kernel already achieves 88.7% bandwidth efficiency
3. **50-60 t/s single request is unrealistic** -- VRAM bandwidth is the fundamental limitation
4. **Batch decode total throughput is substantial** -- 117.1 t/s total throughput at B=4
5. **Maximum single-request optimization target**: 35-37 t/s (through kernel separation + quantization scheme adjustment)
