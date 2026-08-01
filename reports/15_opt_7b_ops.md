# 7.2B 算子优化 — FP8 硬件 Tensor Core + 融合内核

**日期**: 2026-08-01
**分支**: `opt-7b-ops`
**目标**: 重写量化算子，优化 7.2B X5 模型 decode 速度

---

## 一、核心结论

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **Decode 速度** | 21.8 t/s | **28.8 t/s** | **+32% (1.32x)** |
| **PPL@8192** | 1.039078 | 1.039078 | **bit-identical** |
| **VRAM** | 9.23 GiB | 9.23 GiB | 不变 |
| **GPU 占比** | 96.4% | 97.6% | GPU-bound 加深 |

**关键优化**: FP8 硬件 Tensor Core (`tl.dot(fp8, fp8)`) 替代软件 FP8→FP16 解码 + FP16 dot，在 Ada 架构上获得 2x tensor core 吞吐。

---

## 二、性能瓶颈分析

### 2.1 基线 profiling (32 层, B=1, decode)

| 内核 | 耗时 (ms) | 占比 | 说明 |
|------|----------|------|------|
| **ffn_key_res** | 0.498 | **57.5%** | NVFP4+FP8 残差 GEMM, 主要瓶颈 |
| rkv_fused | 0.266 | 30.7% | 3-in-1 attention r/k/v 投影 |
| ffn_value_fp8 | 0.100 | 11.5% | FP8 GEMM |
| **per_layer** | **0.864** | — | 每层 GEMM 总计 |
| **total_gemm (32L)** | **27.66** | — | 32 层 GEMM 总计 |

- decode wall time: 41.2 ms/step → GPU 39.7 ms (96.4%), CPU 仅 1.5 ms (3.6%)
- **结论**: 7.2B 模型 95%+ GPU-bound, CPU launch 开销已不是主要瓶颈

### 2.2 Tile 配置搜索

| 配置 (BM, BN, BK, warps) | 耗时 (ms) | 备注 |
|--------------------------|----------|------|
| **(16, 64, 64, 4)** | **0.479** | **最优** |
| (16, 128, 64, 8) | 0.515 | |
| (16, 128, 128, 8) | 0.663 | |
| (16, 64, 128, 4) | 0.965 | |
| (16, 128, 64, 4) | 1.008 | |
| (16, 256, 64, 8) | 1.157 | |
| (16, 256, 64, 4) | 1.493 | |
| (16, 64, 64, 8) | 6.392 | warp 过多, SM 利用率低 |
| (16, 256, 128, 8) | — | OOM (shared memory) |
| (16, 64, 256, 4) | — | OOM (shared memory) |

**最优配置**: `(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4)` — 与 decode M=1 的场景匹配, 小 tile 充分利用 SM 并行度。

---

## 三、优化策略与结果

### 3.1 Split-K 并行 (无效)

对 ffn_key_res 尝试 Split-K (沿 K 维分割, atomic_add 累加):

| 方案 | 耗时 (ms) | 加速比 | max_diff |
|------|----------|--------|----------|
| baseline (fused) | 0.473 | 1.00x | — |
| splitk_res_2 | 0.615 | **0.77x** (更慢) | 0.001 |
| splitk_res_4 | 0.592 | **0.80x** (更慢) | 0.001 |
| splitk_nvfp4_2 | 0.100 | 1.00x | 0.001 |
| splitk_nvfp4_4 | 0.100 | 1.00x | 0.004 |
| splitk_fp8_2 | 0.094 | **0.93x** (更慢) | 0.001 |
| splitk_fp8_4 | 0.091 | **0.96x** (更慢) | 0.001 |

**结论**: Split-K 在所有配置下均**减速**。原因: `atomic_add` 竞争开销 > 并行收益, 且 K=2048 对于 BLOCK_K=64 仅 32 次迭代, 分割后每块过小。

### 3.2 FP8 硬件 Tensor Core (核心优化)

**原理**: Triton `tl.dot(fp8, fp8)` 直接使用 Ada 架构 FP8 tensor core, 吞吐量为 FP16 的 2x。原先 FP8 路径在软件中解码 FP8→FP16 后用 FP16 dot, 浪费了硬件能力。

#### 3.2.1 ffn_value FP8 hwdot

| 指标 | baseline | hwdot | 变化 |
|------|----------|-------|------|
| 耗时 | 0.122 ms | **0.083 ms** | **1.47x** |
| max_diff | — | **0.0** | bit-identical |

激活在线量化为 FP8 E4M3 (per-tensor scale from amax), 权重直接以 FP8 格式参与 dot, 无需解码。

#### 3.2.2 RKV 融合 FP8 hwdot

| 指标 | baseline (3×单独) | hwdot (融合) | 变化 |
|------|-------------------|-------------|------|
| 耗时 | 0.438 ms | **0.310 ms** | **1.41x** |
| max_diff_r | — | 0.0 | bit-identical |
| max_diff_k | — | 3.05e-5 | FP8 量化噪声 |
| max_diff_v | — | 1.91e-6 | FP8 量化噪声 |

RKV 融合内核同时处理 r (NVFP4) + k (FP8 hwdot) + v (FP8 hwdot), 一个 kernel launch 完成三个投影。k/v 的微小差异来自 per-tensor 激活量化, 在 PPL 上无影响。

#### 3.2.3 ffn_key split (NVFP4 + FP8 残差分离)

| 指标 | baseline | split | 变化 |
|------|----------|-------|------|
| 耗时 | 0.799 ms | **0.774 ms** | **1.03x** |
| max_diff | — | 0.001 | per-block scale 精度 |

将 ffn_key 的 NVFP4 主 GEMM 和 FP8 残差 GEMM 分离为独立 kernel, 复用同一 x tile。加速有限, 因主瓶颈是 NVFP4 路径本身。

### 3.3 组合效果

| 指标 | baseline | optimized | 加速比 |
|------|----------|-----------|--------|
| rkv | 0.438 ms | 0.197 ms | **2.22x** |
| ffn_key | 0.799 ms | 0.471 ms | **1.70x** |
| ffn_val | 0.122 ms | 0.096 ms | **1.28x** |
| att_out | 0.235 ms | 0.132 ms | **1.78x** |
| **per_layer** | **1.594 ms** | **0.897 ms** | **1.78x** |
| **decode t/s** | **21.8** | **28.8** | **1.32x** |

> per_layer 加速比 (1.78x) > decode 加速比 (1.32x) 的原因: decode 总耗时中还包含非 GEMM 部分 (wkv, layernorm, state update, embedding lookup 等), 这些未被优化。

---

## 四、精度验证

### 4.1 PPL (bit-identical 验证)

| 上下文长度 | PPL |
|-----------|-----|
| 1024 | 1.3490 |
| 2048 | 1.1622 |
| 4096 | 1.0779 |
| **8192** | **1.0391** |

PPL@8192 = 1.039078, 与优化前完全一致 (bit-identical), 证明 FP8 hwdot 的激活量化噪声在 PPL 级别不可见。

### 4.2 VRAM

| 指标 | 数值 |
|------|------|
| allocated | 9.23 GiB |
| reserved | 9.24 GiB |
| total GPU | 11.94 GiB |

VRAM 无变化, 优化仅影响计算路径, 不增加显存占用。

---

## 五、技术实现

### 5.1 FP8 硬件 dot 内核 (`fused_fp8_hwdot_gemm_kernel`)

```python
# 激活在线量化: bf16 -> FP8 E4M3 (per-tensor scale)
amax_v = tl.maximum(tl.load(amax_ptr), 1e-12)
inv_xs = 448.0 / amax_v
a_fp8 = tl.minimum(tl.maximum(x_tile.to(tl.float32) * inv_xs, -448.0), 448.0).to(tl.float8e4nv)

# 权重直接以 FP8 格式加载 (无需解码!)
w_fp8 = tl.load(w_ptr + ...)  # float8_e4m3fn

# FP8 tensor core dot (2x throughput vs FP16 on Ada)
acc = tl.dot(a_fp8, tl.trans(w_fp8), acc)

# 反量化缩放
acc = acc * (amax_v / 448.0 * w_ts_v)
```

### 5.2 RKV 融合内核 (`fused_rkv_hwdot_kernel`)

一个 kernel 处理三个投影:
- **r**: NVFP4 (FP4xFP4, 4-bit 权重)
- **k**: FP8 hwdot (FP8xFP8, 硬件 tensor core)
- **v**: FP8 hwdot (FP8xFP8, 硬件 tensor core)

关键修复: k/v 权重的 stride 必须与 r 权重分开传递 (r 是 packed [N, K//2], k/v 是 [N, K])。

### 5.3 引擎集成 (`rwkv7_fast_v3a.py`)

新增方法:
- `_att_linear()`: attention 线性层量化 GEMM 分发
- `_rkv_linear()`: r/k/v 融合投影 (满足条件时单 kernel)
- `cmix_from_mixed()`: FFN 路径量化 GEMM 拦截

条件: `FUSED_GEMM=True` + `rows <= FUSED_M_MAX` + 权重为 dict (量化格式)。

---

## 六、优化路径总结

```
原始 X5 (25.1 t/s)
    |
    +-- 融合内核 (prep_x + GEMM 合并) -> 24.3 t/s
    |   +-- 减少 CPU launch: 4-9 launches/linear -> 1-2 launches/linear
    |
    +-- FP8 硬件 Tensor Core
    |   +-- ffn_value: 1.47x (bit-identical)
    |   +-- rkv 融合: 1.41x (k/v 噪声 ~3e-5)
    |   +-- att_output: 1.78x
    |
    +-- ffn_key split (NVFP4 + FP8 残差分离): 1.03x
    |
    +-- 最终: 28.8 t/s (1.32x vs 原始)
```

---

## 七、未采用的方案

| 方案 | 原因 |
|------|------|
| Split-K 并行 | atomic_add 竞争 > 并行收益, 所有配置减速 |
| 大 tile (BN=256, BK=128+) | shared memory 不足 (Ada: 48KB/SM) |
| (16,64,64,8) 8 warps | SM 利用率低, 6.4x 减速 |

---

## 八、后续方向

1. **ffn_key_res per-tensor 化**: 将 FP8 残差从 per-block scale 改为 per-tensor scale, 即可使用 FP8 hwdot (预期额外 1.3x)
2. **FP4 硬件 Tensor Core**: Blackwell+ 架构支持 `tl.dot(fp4, fp4)`, NVFP4 路径可获得 2x 加速
3. **跨层内核融合**: 将多层 GEMM 合并为一个 mega-kernel, 减少 CPU launch (当前 2.4% CPU, 可进一步降低)
4. **Batch decode**: B=4-8 可将单请求吞吐从 28.8 t/s 提升到 50-60 t/s (分摊 CPU 开销)

---

## 九、文件清单

| 文件 | 位置 | 用途 |
|------|------|------|
| `fused_nvfp4_gemm.py` | Albatross/faster3a_2605/ | 核心: FP8 hwdot + RKV 融合内核 |
| `nvfp4_ops.py` | Albatross/faster3a_2605/ | 量化权重加载 + GEMM 分发 |
| `rwkv7_fast_v3a.py` | Albatross/faster3a_2605/ | 引擎: 量化路径集成 (80 行新增) |
| `profile_7b_v2.py` | eval scripts | 7.2B profiling |
| `fp8_hwdot_test.py` | eval scripts | FP8 hwdot 基准测试 |
| `opt_kernel_test.py` | eval scripts | Tile + Split-K 搜索 |
| `ffn_split_test.py` | eval scripts | FFN split 验证 |
| `bench_optimized.py` | eval scripts | 最终优化基准 |
| `ppl_optimized.py` | eval scripts | PPL + VRAM 验证 |
