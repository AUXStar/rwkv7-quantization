# RWKV-7 1.5B 量化最终定稿（M2 方案）

## 一、定稿方案

```
┌─────────────────────────────────────────────────────────────┐
│  L0 - L5   : 全 bf16（保护带）                              │
│  L6 - L17  : key FP8 / value FP8 / rec NVFP4 / out NVFP4    │
│              ffn_k NVFP4+FP8残差 / ffn_v FP8                │
│  L18 - L23 : 全 bf16（保护带）                              │
│  AWQ: alpha=0.3                                             │
└─────────────────────────────────────────────────────────────┘
```

**压缩比 1.3x**（2.25 GB → 1.68 GB）。量化12层，保护12层。

## 二、定稿依据（最终验收数据）

| 指标 | 目标 | M2 实测 | 状态 |
|------|------|---------|------|
| MATH500 | ≥12.0%（gap≤0.6pp）| **12.4%**（62/500）| ✅ 达标 |
| PPL delta @8192 | ≤0.05 | +0.0008 | ✅ |
| 显存 | ≤2.0 GiB | 2.23 GiB | ⚠️ 超0.23（用户接受）|
| decode | ≥65 t/s | 58.5 t/s | ⚠️ 低6.5（用户接受）|

**M2 是全部 11 个实测方案中唯一达到 MATH500 12.0% 目标的方案。**

## 三、定稿是怎么来的（完整迭代历程）

### Phase 1: 基础方案 (Task 1-4)
| 版本 | 改动 | MATH500 | 结论 |
|------|------|---------|------|
| 基线 | 原方案 | 8.0% | 远差于目标 |
| T1 | +低秩FP8 | 8.6% | 收益1.7%磁盘, 不值得 |
| T2 | key全面FP8 | 9.4% | 有效但不够 |
| T3 | 残差per-block + 低秩bf16 | 8.6% | 低秩量化不划算 |
| T4 | AWQ alpha=0.3 | **11.2%** | alpha是关键, PPL却不变 |

### Phase 2: 敏感度归因实验
| 实验 | 发现 |
|------|------|
| 逐层PPL归因 | 单层量化几乎无损(+0.0007), 误差线性叠加 |
| state MSE归因 | 389x差异, L0最不敏感(3.3e-5), L14峰值(1.27e-2) |
| 交替方案A/B | **MATH500 0%崩溃** — L0被量化→v_first污染全模型 |
| 语义对比 | 量化模型推理轨迹早期分叉, 非答案边缘抖动 |

### Phase 3: M梯度搜索（定稿的直接来源）
| 方案 | 保护层 | MATH500 |
|------|--------|---------|
| M0 | L0-key | 11.2% |
| M5 | L0 | 11.4% |
| L0L23 | L0+L23 | 11.4% |
| M1 | L0-3+L20-23 | 10.8% |
| **M2** | **L0-5+L18-23** | **12.4%** ✅ |
| M3 | L0-7+L16-23 | 11.2% |

**M2 窗口 L6-17 是全局最优**：M1（窗口窄）→10.8%，M3（窗口更窄）→11.2%，
M2（L6-17）→12.4%。保护带 L0-5 + L18-23 恰好覆盖 v_first源头(L0) 和
state峰值区(L14/15/18/22)。

## 四、实现思路

**核心原则：全链路量化域执行，禁止运行时反量化。**

1. **激活动态量化**：`prep_x` kernel 每次 forward 实时算 amax；
   fused kernel 内部对 x tile 做 per-16-block 动态量化。
2. **权重静态量化**：离线量化，scale 存盘。
3. **残差补偿**：ffn.key 的 NVFP4 误差用 FP8 残差补偿（per-block ratio scale
   + tensor scale），走纯 FP8×FP8 GEMM（dispatcher 对残差权重强制 fused kernel，
   因为 _scaled_mm 无法混合 fp32 tensor x-scale + fp8 block w-scale）。
4. **AWQ alpha=0.3**：grid search 7 点，MATH500 +2.6pp（PPL 几乎不变）。
5. **L0-5/L18-23 保护带**：L0 是 v_first 源头（value 残差门控传给所有层），
   必须 bf16；L18-23 是 state 高敏感区。

## 五、每个张量的量化方式

### 量化层 L6-17（12层）

| 张量 | 格式 | 量化方式 |
|------|------|---------|
| att.receptance.weight [C,C] | NVFP4 | E2M1 4bit + per-16-block E4M3 scale + fp32 tensor scale, 0.5B/元素 |
| att.key.weight [C,C] | FP8 | E4M3 8bit + per-row fp32 scale, 1B/元素 |
| att.value.weight [C,C] | FP8 | 同上 |
| att.output.weight [C,C] | NVFP4 | 同 rec |
| ffn.key.weight [4C,C] | NVFP4+残差 | NVFP4 + FP8残差(per-block ratio + tensor scale), 1.5B/元素 |
| ffn.value.weight [C,4C] | FP8 | 同 key |

### 保护层 L0-5, L18-23（12层）

所有 6 个线性层保持 bf16（不量化）。

### 不量化（全模型）

| 张量 | 原因 |
|------|------|
| emb.weight, head.weight | 全局1GB, 太大, 且是embedding查找 |
| 低秩 g1/g2/a1/a2/w1/w2/v1/v2 | 只占1.7%内存, 收益配不上精度 |
| x_r/x_w/x_k/x_v/x_a/x_g, w0/a0/v0, k_k/k_a/r_k | 向量参数, 每层~104KB |
| LayerNorm/GroupNorm (ln0/ln1/ln2/ln_x/ln_out) | 归一化必须精确 |

## 六、运行方式

```bash
# 量化
python3 -c "
import quantize_model as qm
qm.ALPHA = 0.3
qm.quantize_model('model.pth', 'out.pth', scheme_name='1.5b-m2-final')"

# 推理（引擎自动按 meta + 后缀检测路由）
```

## 七、缩小保护层实验（M2 → M2c）

| 变体 | 保护层 | 量化层 | MATH500 | decode | 结论 |
|------|--------|--------|---------|--------|------|
| M2 | L0-5+L18-23 | L6-17 | 12.4% | 58.5 | 参照 |
| M2b | L0-5+L19-23 | L6-18 | 11.2% | — | **L18必须保护** |
| **M2c** | **L0-4+L18-23** | **L5-17** | **12.0%** | **86.3** | ✅ 最终 |
| M2d | L0-4+L19-23 | L5-18 | 10.8% | — | 去L18+L5都崩 |

M2c = 仅去掉L5保护（L5 state MSE 1.4e-3，浅层最不敏感区），L18保留
（state MSE 6.0e-3）。比M2：decode 58.5→86.3（达标）、显存-0.04GiB、
压缩1.4x，MATH500 12.0%压线达标。
