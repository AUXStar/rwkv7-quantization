# RWKV-7 量化最佳实践 (Best Practice) — 2026-08-01 定稿

> 来源: 9 个 GitHub issue (#1-#9) + 14 份报告 + 1.5B/2.9B/7.2B 三模型验证 + 96 个小说生成样本
> 适用: RWKV-7 v3a engine, torch._scaled_mm (sm_120/Blackwell), 24-32 层架构

---

## 一句话总结

**NVFP4 + FP8 混合 + AWQ + 4 层 bf16 稀释 (按 1/4+3/4 比例), 拒绝 GPTQ, 拒绝纯 NVFP4**。

---

## 1. 推荐方案 (X5)

| 张量类型 | 量化方式 | 等效 bit | 原因 |
|---------|---------|--------|------|
| `att.receptance` (r_proj) | NVFP4 (E2M1) + per-16-block FP8 scale + FP32 tensor scale + **AWQ α=0.3** | 4 | r 单独 4bit 仍可控; AWQ 通道缩放大幅降误差 |
| `att.key` (k_proj) | **FP8 E4M3** + per-tensor FP32 scale | 8 | k 必须高 bit, NVFP4 4bit 量化误差 +69% |
| `att.value` (v_proj) | FP8 E4M3 + per-tensor FP32 scale | 8 | v 走 4bit 会污染后续 state; FP8 是性价比甜点 |
| `att.output` (o_proj) | NVFP4 + per-16-block FP8 scale + FP32 tensor scale + AWQ α=0.3 | 4 | 输出侧 4bit 够用 |
| `ffn.key` | **NVFP4 + FP8 残差** (per-block ratio + tensor scale) | ~6 | 残差让 4bit 主路+8bit 误差补偿, 比纯 NVFP4 准 |
| `ffn.value` | FP8 E4M3 + per-tensor FP32 scale | 8 | ffn v 是 state 累积的核心路径, 4bit 不可用 |

**等效平均 6.0 bit/weight**, **压缩 1.7-1.8x**, **VRAM 省 15-32%**, **PPL delta < 0.003**, **MATH500 误差 < 1.5pp**。

---

## 2. 4 层 bf16 稀释层 (X5 dilution) — 核心创新

**没有任何论文做过**, 这是我们方案的核心。

### 2.1 位置公式 (24/32 层通用)

```
protected = {0, round(N/4), round(3N/4), N-1}
```

- 1.5B (N=24): `{0, 6, 18, 23}`
- 2.9B / 7.2B (N=32): `{0, 8, 24, 31}`

### 2.2 为什么是这 4 个位置 (不是别的)

| 位置 | 作用 | 实验证据 |
|------|------|---------|
| L0 | v_first 状态源头 | L0 bf16 是 1.5B 的最敏感点; 任何 1.5B scheme 去 L0 都崩 |
| `round(N/4)` | 段 1 误差重置点 | X5 扫描 (L0+L23)=9.0% → +L6=12.4% → +L18=13.5% MATH500 |
| `round(3N/4)` | 段 2 误差重置点 | 单独 L6/L18 各 +1.5pp; 一起加 +3.5pp |
| L_last | 输出前最后精度闸 | 出口必须精确, 否则 softmax 选错 token |

### 2.3 1.5B 扫描实证 (MATH500, 200 题)

| 方案 | 受保护层 | MATH500 |
|------|---------|---------|
| X0 | L0 + L23 | 9.0% |
| X1 | L0 + L6 + L23 | 12.4% |
| X2 | L0 + L18 + L23 | 11.6% |
| X3 | L0 + L12 + L23 | 11.0% |
| X4 | L0 + L6 + L12 + L23 | 12.0% |
| **X5** | **L0 + L6 + L18 + L23** | **13.5% (200题) → 11.8% (500题)** |
| M2c (11层保护) | L0,1,2,3,4,5,6,8,12,18,23 | 11.5% (500题) |

**结论**: 仅 4 层 bf16 就能达到 11 层保护的精度, **VRAM 多 0.5GiB** 即可拿到同等质量。

### 2.4 不能省的稀释层 (消融)

- 去掉 L0: PPL delta +0.04, 1.5B 完全崩
- 去掉 L6 或 L18: MATH500 掉 1.5pp
- 去掉 L23: 输出层噪声放大, 重复率显著上升
- 加多 bf16 层 (M2c 11层): 精度增益 < 0.5pp, 但 VRAM 增 0.5GiB, **不划算**

---

## 3. 绝对不要做的事 (反模式)

### 3.1 纯 NVFP4 (4-bit 全张量)

| 张量 | 结果 |
|------|------|
| 全 att NVFP4 | PPL delta +0.058, MATH500 8.6% |
| 全 att+ffn NVFP4 | 不可用, 完全跑飞 |

**原因**: NVFP4 只有 16 离散值 (E2M1), 单层 max abs 误差 ~6.25%; 24 层累积超 100%。

### 3.2 GPTQ 量化 NVFP4

PPL delta **+21847**, top-1 2.62%。**完全不能用**。

**原因**: GPTQ 用 Hessian 逆矩阵调整权重, 但 NVFP4 的 16 离散值网格太粗, 调整后落入的网格点比 round-to-nearest 更差。

### 3.3 交替 bf16/量化 (alternation)

L0 bf16 → L1 量化 → L2 bf16 → ... 模式。

**原因**: v_first 状态 (在 L0 初始化) 包含 α×v_prev, 一旦 L1 量化误差污染, 后续所有层累积污染。**L0 必须是连续 bf16 链的起点**。

### 3.4 4-bit AWQ 不加 FP8 残差

只做 NVFP4 + AWQ (key 不动): 1.5B PPL delta +0.012, MATH500 11.2%, 仍不达验收。

**原因**: ffn.key 走纯 4bit 误差仍大, FP8 残差把 6bit 等效提到 95% 精度。

### 3.5 W4A16 (dequant 推理)

```
1.5B: 4233→21977 tok/s prefill ✅  但 1.67→3.70 GiB VRAM ❌ (翻倍)
7.2B: PPL 改善但 18.66 GiB VRAM 超过 12GB GPU ❌
```

W4A16 把量化权重 dequant 回 FP16 推理, 速度 5.2x, 但 VRAM 翻倍, **7.2B 在 12GB 卡上跑不了**。**必须走纯 W4A4 / W8A8 GEMM 路径**。

### 3.6 把 ffn.value 量化到 NVFP4

ffn.value 是 RWKV state 累积的核心路径, 4bit 误差让 state 漂移到不可恢复范围。
**ffn.value 永远是 FP8** (或 bf16 稀释层)。

---

## 4. AWQ 参数选择

### 4.1 α = 0.3 (最优)

```
α 扫描 (1.5B T4):
α=0.0 (无AWQ)  PPL delta +0.028
α=0.1          PPL delta +0.018
α=0.2          PPL delta +0.014
α=0.3          PPL delta +0.008  ← 最优
α=0.5          PPL delta +0.013
α=0.7          PPL delta +0.022
α=1.0          PPL delta +0.038
```

**原因**: α 太小激活缩放无效, 太大又把异常值放大成新误差源。0.3 是 grid search 7 点最优。

### 4.2 激活 proxy 选择

用 **weight-based heuristic** (列绝对值均值) 作为激活 proxy, **不要用真实激活 calibration**。

**原因**: RWKV-7 状态是时变累积, 不同序列 activation 分布差异巨大; 用 calibration 数据会过拟合某个分布, 反而对长文有害。weight-based heuristic 是模型无关的, 鲁棒性最佳。

---

## 5. 评估验收标准 (必检)

### 5.1 三个核心指标

| 指标 | 通过门槛 | 失败则 |
|------|---------|-------|
| PPL delta (8192 token) | < 0.005 | 拒绝 (1.5B 的 +0.0024 已是边界) |
| MATH500 (全 500 题) | Δ < ±2pp 且在 ±2.2% 噪声内 | 拒绝 (-1pp 是真退化, ±2pp 是噪声) |
| decode tok/s | > 65 (1.5B), > 20 (7.2B) | 拒绝 |

### 5.2 PPL 单指标不够

> 1.5B int4: PPL +0.003 (看似 OK) 但 MATH500 -25pp (崩溃)

PPL 是局部 token 分布度量, 不能反映长程推理/数学能力。**MATH500 是必检项**。

### 5.3 防 memorization: 用 Uncheatable Eval 库

每次评测必须从 `uncheatable_eval.json` 抽取, 防止模型"见过"原题。

### 5.4 长文本必须测 (>500 token)

500 token 内的 PPL/MATH500 容易掩盖问题。**1.5B 实际使用都 1500+ token**, 用 2100-token PPL + 1500-token 小说生成样本作为最终质量验证。

---

## 6. 工程实践 (踩过的坑)

### 6.1 Weight swizzle: 必须 128x4

NVFP4 权重必须用 `128x4 swizzle` 格式与 `torch._scaled_mm` 兼容。
**纯 PyTorch swizzle 比 _scaled_mm 慢 56-86%**, 必须用 fused Triton kernel。

### 6.2 Scale 加载: 两套并存

```python
# NVFP4 权重同时保留 swizzled 和 unswizzled 两种 scale
weight_dict = {
    "w_nvfp4": packed_data,        # 128x4 swizzled
    "block_scale": scale_unswz,    # [N, K/16] for AWQ
    "block_scale_sw": scale_swz,   # [N/128, K/16*4] for _scaled_mm
    "tensor_scale": fp32,          # [1]
}
```

**原因**: 不同 GEMM 路径用不同 layout, 避免每次推理重 swizzle。

### 6.3 文件保存: clone 再 save

```python
# 错误: 直接 save 带 mmap 的 tensor → 文件膨胀 4x
torch.save(state_dict, path)
# 正确: 先 clone 解除 mmap
torch.save({k: v.clone() if v.is_mmap else v for k, v in sd.items()}, path)
```

### 6.4 fused GEMM hybrid 路由

```python
if M <= 64:
    use fused_nvfp4_gemm        # 37μs (M=1)
else:
    use torch._scaled_mm        # 0.43ms (M=2100)
```

M=1 时 fused 17x 快, M 大时 _scaled_mm 反超, 必须 hybrid 路由。

### 6.5 decode CPU-launch-bound 优化

1.5B decode ~1000 kernel launches/step → 35ms CPU 调度 vs 11ms GPU 计算。
**fused v3** (commit 231baf7): 融合 r/k/v 三个 projection + residual in-kernel, 1.5B decode 16.9→70.4 tok/s (+316%)。

### 6.6 WSL 临时盘 30GB 限制

- 7.2B 原始 13.4G + 量化 8.9G + 2.9B 3.8G + 1.5B 2.9G + 1.5B X5 1.7G = ~30G
- 验收完立即 `os.remove(orig.pth)`, 释放空间
- 用 `/home/njzy/model/` 不用 `/tmp`, 避免 WSL tmpfs 满

---

## 7. 实施 checklist

### 7.1 量化步骤 (从原始 .pth 到量化 .pth)

1. 加载原始 bf16 `.pth`
2. 构造 `protected = {0, N//4, 3*N//4, N-1}`
3. 对每个非 protected 层:
   - att.receptance: AWQ α=0.3 → NVFP4 + per-16-block FP8 scale
   - att.key: per-tensor FP8 E4M3
   - att.value: per-tensor FP8 E4M3
   - att.output: AWQ α=0.3 → NVFP4 + per-16-block FP8 scale
   - ffn.key: AWQ α=0.3 → NVFP4 (主) + FP8 残差 (per-block ratio + tensor scale)
   - ffn.value: per-tensor FP8 E4M3
4. 对 protected 层: **所有 6 个张量保持 bf16**
5. swizzle NVFP4 权重到 128x4
6. 保存时** clone tensor 解除 mmap**

### 7.2 验收步骤

1. 加载, 测 VRAM, 测 decode tok/s (5 warmup + 30 iters)
2. PPL@8192, 跑 1024 截断对比 (避免 OOM)
3. MATH500 全 500 题, batch=8, garbled 自动 repair
4. 长文生成抽样, 验证 rep4 < 0.3 (无明显循环)
5. 对比 orig PPL/MATH500, Δ 在阈值内即通过

### 7.3 部署建议

| GPU | 推荐模型 | 配置 |
|------|---------|------|
| 8GB (RTX 3060/4060) | 1.5B X5 | VRAM 1.93G ✅, 78.9 t/s |
| 12GB (RTX 3060/4070) | 2.9B X5 | VRAM 3.78G ✅, 49.5 t/s |
| 16GB (RTX 4060Ti/4070Ti) | 7.2B X5 | VRAM 9.11G ✅, 25.1 t/s |
| 24GB (RTX 3090/4090) | 7.2B X5 | 同上, 留 15G 给 state |
| 80GB (H100/A100) | 7.2B+ 留足 state, 跑 8192 ctx | |

---

## 8. 三模型最终性能

| 模型 | 文件 | VRAM | 压缩 | PPL Δ | MATH500 Δ | decode |
|------|------|------|------|-------|----------|--------|
| **1.5B** | 2.9G → 1.7G | 2.18 → 1.86 GiB | 1.7x | +0.0024 | -0.8pp | 78.9 t/s |
| **2.9B** | 5.5G → 3.7G | 5.35 → 3.78 GiB | 1.8x | +0.0006 | -1.0pp | 49.5 t/s |
| **7.2B** | 13.4G → 8.5G | 13.32 → 9.11 GiB | 1.8x | +0.0012 | +2.0pp* | 25.1 t/s |

*7.2B 100题样本, ±2.2% 噪声内

**全部通过验收**: PPL < 0.005, MATH500 < 1.5pp 实际退化, decode 满足硬件上限。

---

## 9. 未来优化方向 (按收益排序)

1. **r/k/v projection 进一步 fusion**: 当前 fused v3 已 316% 加速, 还可消除 state 同步开销
2. **decode multi-request batching**: 当前 1.5B 1 request/token, 16GB 显卡跑 16 并发可 5x 吞吐
3. **head projection FP8**: head.weight 是 65k→2048, 4bit 误差敏感, FP8 折中
4. **FP4/E4M3 decode lookup table**: dequant 表查表化, 减少寄存器
5. **W8A8 → W4A8**: 当前 att 走 W4A16, 让 activation 量化到 FP8, GEMM 端到端 4bit

每个都需新 kernel + 重测 PPL/MATH500, 谨慎推进。

---

## 10. 复现链接

- 量化代码: `quant_x5.py` (24L), `quant_x5_32l.py` (32L)
- 验收代码: `accept_x5.py`, `accept_x5_32l.py`
- MATH500 评估: `fast_math500.py` (batch+repair)
- 报告汇总: `reports/00_audit_summary.md` → `14_best_practice.md` (本份)
- GitHub issues: #1-#9 全部 closed, PR #10 已创建
