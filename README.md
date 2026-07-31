# rwkv7-quantization

请阅读以下完整上下文，然后直接继续与我讨论。

# RWKV-7 NVFP4 量化推理项目 — 完整上下文

## 项目目标

对 RWKV-7 选择性量化，实现减存+加速+保效。

## 模型信息

| 模型 | 路径 | 大小 | dtype | C | H | N | layers |
|------|------|------|-------|---|---|---|--------|
| 2.9B | /home/njzy/model/rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth | 5.49 GB | bf16 | 2560 | 40 | 64 | 32 |
| 7.2B | /home/njzy/model/rwkv7-g1g-7.2b-20260523-ctx8192.pth | 13.41 GB | bf16 | 4096 | 64 | 64 | 32 |
| 13.3B | — | — | bf16 | — | — | — | — (mmap不够,暂不处理) |

2.9B 和 7.2B 结构完全一致, 仅维度不同。先用 2.9B 实验, 再在 7.2B 上执行。

## RWKV-7 完整前向传播公式 (从 CUDA 源码推导)

### 每层 Block 完整流程

```
Step 1: LayerNorm (仅 block 0)
  if layer_id == 0: x = LayerNorm(x, ln0.w, ln0.b)

Step 2: Token Shift (tmix_mix6_kernel)
  dx = x_prev - x_cur   # x_prev = shift_state (t=0) 或 x[t-1]
  xr = x + dx * x_r
  xw = x + dx * x_w
  xk = x + dx * x_k
  xv = x + dx * x_v
  xa = x + dx * x_a
  xg = x + dx * x_g

Step 3: 线性投影
  r = xr @ att.receptance.weight              [B,T,C]
  k = xk @ att.key.weight                     [B,T,C]
  v = xv @ att.value.weight                   [B,T,C]
  w = w0 + tanh(xw @ w1) @ w2                 [B,T,C], decay
  a = a0 + (xa @ a1) @ a2                     [B,T,C], learning rate
  g = sigmoid((xg @ g1) @ g2)                 [B,T,C], output gate

Step 4: Key 后处理 (tmix_kk_a_gate_kernel)
  kk_raw = k * k_k                             逐元素缩放
  kk = normalize(kk_raw, dim=-1)               L2 归一化 → 单位向量
  av = sigmoid(a)                              学习率门 [0,1]
  k_modified = k * (av * k_a + (1 - k_a))      衰减混合

Step 5: Value 残差门控 (vres, 仅 layer > 0)
  if layer_id > 0:
      gate = sigmoid(v0 + (xv @ v1) @ v2)
      v = v + gate * (v_first - v)             # v_first 来自 layer 0
  else:
      v_first = v                              # 保存给后续所有层

Step 6: WKV 状态更新 (wkv_fp16_v1_clone_kernel) ← 核心
  w_delta = 2^(-0.875 / (1 + exp(-1.443 * w))) - 1 + rotator(elapsed)

  S_new = diag(w_delta) · S                    ① 全局衰减
        + S @ (-kk) @ (kk · av)                ② 方向性擦除 (rank-1 投影)
        + k_modified ⊗ v                       ③ 新信息注入 (rank-1 外积)

  y_wkv = S_new @ r                            ④ receptance 读出

Step 7: 输出后处理 (tmix_lnx_rkvres_xg_kernel)
  ln_out = GroupNorm(y_wkv, ln_x.w, ln_x.b, H)
  rkr = per_head_sum(r * k * r_k)              # r-k 残差标量
  rkr_v = rkr * v
  y = (ln_out + rkr_v) * g

Step 8: 输出投影
  att_out = y @ att.output.weight
  x = x + att_out                              残差连接

Step 9: Channel Mixing (FFN)
  dx = x_prev - x_cur
  mixed = x + dx * ffn.x_k
  hid = relu(mixed @ ffn.key.weight) ** 2      ReLU² 激活
  cmix_out = hid @ ffn.value.weight
  x = x + cmix_out                             残差连接

Step 10: 状态推进
  shift_state = x_cur
  elapsed += T
```

### k_k 和 k_a 的精确角色

```
k_k: 缩放k → L2归一化 → kk(单位向量)
     kk 决定 state 擦除方向 ("方向盘")
     kk*av 决定擦除强度 (kka)
     → 最敏感的向量参数

k_a: 混合系数, 控制 k_modified 中衰减门 av 的比例
     k_modified = k * (1 + k_a * (av - 1))
     k_a=0 → k_modified=k (不衰减)
     k_a=1 → k_modified=k*av (完全衰减)
```

## 张量清单与内存占用 (7.2B 实测)

### A. 量化目标 — 6 个 nn.Linear / 层 (89.5%)

| # | key pattern | shape | 每层 | 32层 |
|---|-------------|-------|------|------|
| 0 | blocks.*.att.receptance.weight | [4096,4096] | 32 MB | 1024 MB |
| 1 | blocks.*.att.key.weight | [4096,4096] | 32 MB | 1024 MB |
| 2 | blocks.*.att.value.weight | [4096,4096] | 32 MB | 1024 MB |
| 3 | blocks.*.att.output.weight | [4096,4096] | 32 MB | 1024 MB |
| 4 | blocks.*.ffn.key.weight | [16384,4096] | 128 MB | 4096 MB |
| 5 | blocks.*.ffn.value.weight | [4096,16384] | 128 MB | 4096 MB |
|   | **合计** | | | **12288 MB** |

### B. 不量化 — 向量参数 (~104 KB/层, ~3.3 MB总计)

x_r, x_w, x_k, x_v, x_a, x_g, w0, a0, v0, k_k, k_a, r_k, ffn.x_k

### C. 不量化 — 低秩权重 (406 MB总计)

g1/g2 [4096,480], a1/a2 [4096,128], w1/w2 [4096,128], v1/v2 [4096,96]

### D. 不量化 — LayerNorm / GroupNorm

ln0(仅block0), ln1, ln2, att.ln_x(GroupNorm, eps=64e-5), ln_out

### E. 不量化 — 全局 (1024 MB)

emb.weight [65536,4096] = 512 MB, head.weight [65536,4096] = 512 MB

## NVFP4 技术概要

- 格式: E2M1 (1s+2e+1m), 每16元素共享 E4M3(FP8) scale + FP32 tensor scale
- 平均 4.5 bit/element
- 仅 Blackwell GPU 原生支持
- 精度 ≥ INT4 (分数倍缩放 vs 2次幂)
- 工具链: llmcompressor / nvidia-modelopt / 自定义

## 量化敏感度排序 (从公式推导)

```
★★★★★  att.key.weight     state擦除方向 + 信息注入, 双路径进state, 乘法性质
★★★★   att.value.weight   state信息注入, 低秩平滑, layer0跨层传播(v_first)
★★★    att.receptance.weight  只读不修改state, 误差不累积, 与state值成正比
★★     att.output.weight   残差流+GroupNorm缓冲, 不接触state
★      ffn.key.weight      无state, ReLU²抑制~50%通道, 体量最大
★      ffn.value.weight    无state, 残差+LN缓冲, 体量最大
```

att.key 的特别危险性:
- 产生 kk (归一化单位向量) → state 擦除方向
- 产生 k_modified → state 信息写入
- 产生 kka (kk*av) → state 衰减系数
- 三条路径全部进 state, 且 kk 方向误差是乘法性质不可恢复

## 社区实测数据

- W8A16 (权重8位, 激活BF16): 无损
- W8A8: 掉 2.3 点
- W4 (7.2B): 掉 2-3 点
- W4 (1.5B): 塌 20+ 点
- Molly方向: 最浅最深少量化, 中间多量化

## 最终量化方案 (7.2B)

```
component:   key    value  receptance  output  ffn_key  ffn_value
Layer 0:     bf16   bf16   nvfp4       nvfp4    nvfp4    nvfp4
Layer 1-3:   fp8    fp8    nvfp4       nvfp4    nvfp4    nvfp4
Layer 4-27:  nvfp4  nvfp4  nvfp4       nvfp4    nvfp4    nvfp4
Layer 28-31: fp8    fp8    nvfp4       nvfp4    nvfp4    nvfp4
```

层间选择依据:
- L0 bf16: v_first 跨层传播至全部32层, 误差放大32×
- L1-3 fp8: 特征提取层, W8A16社区验证无损
- L4-27 nvfp4: 中间层最鲁棒(Molly实测), 被LN缓冲
- L28-31 fp8: 接近输出, 误差直接影响生成质量

预期: 13.09 GB → 5.28 GB (压缩60%), 精度损失 ≤ 2-3 点

## Meta 格式 (最终版, 存入 .pth)

```python
meta = {
    "v": 1,
    # dtype: 0=bf16, 1=fp8_e4m3, 2=nvfp4_e2m1
    # comp:  0=rec, 1=key, 2=val, 3=out, 4=ffn_k, 5=ffn_v
    # rule: [layer_start, layer_end, comp, dtype]
    "r": [
        [ 0,  0, 1, 0], [ 0,  0, 2, 0],   # L0 key/value bf16
        [ 1,  3, 1, 1], [ 1,  3, 2, 1],   # L1-3 key/value fp8
        [ 4, 27, 1, 2], [ 4, 27, 2, 2],   # L4-27 key/value nvfp4
        [28, 31, 1, 1], [28, 31, 2, 1],   # L28-31 key/value fp8
        [ 0, 31, 0, 2], [ 0, 31, 3, 2],   # all layers rec/out nvfp4
        [ 0, 31, 4, 2], [ 0, 31, 5, 2],   # all layers ffn nvfp4
    ],
    "s": {"blk": 16, "sd": "fp8e4m3", "td": "fp32"},
    "n": [
        "emb.", "head.", "ln_out.", "ln0.", "ln1.", "ln2.", "ln_x.",
        "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
        "w0", "w1", "w2", "a0", "a1", "a2",
        "v0", "v1", "v2", "g1", "g2", "k_k", "k_a", "r_k"
    ]
}
```

推理侧消费逻辑:
```python
comp_map = {"receptance":0, "key":1, "value":2, "output":3}
# ffn: key->4, value->5

for key, tensor in checkpoint.items():
    if any(p in key for p in meta["n"]):
        load_as_is(tensor)
        continue
    layer = extract_layer(key)
    comp  = extract_component(key)
    dtype = get_dtype(meta["r"], layer, comp)
    dequant_or_load(tensor, dtype, meta["s"])
```

## Issues 计划与依赖

```
#1 [Meta]   分层量化方案设计 — 本文档即为 #1 内容
#2 [实验]   2.9B FFN-only NVFP4 基线 → 验证工具链 + FFN 无损
#3 [实验]   key/value FP8 验证 (W8A16) → 社区结论交叉验证
#4 [实验]   L4-27 key/value NVFP4 消融 → 核心争议点
#5 [实验]   L0 value BF16 必要性 → v_first 跨层传播
#6 [实验]   长序列 state MSE (2048/4096/8192)
#7 [工程]   量化工具链 (加载→量化→存储+meta)
#8 [工程]   推理引擎适配 (NVFP4/FP8 混合 kernel)
#9 [Benchmark] 7.2B 完整精度/速度/显存对比

依赖: #2→#3→#4→#9, #5→#9, #6→#9, #7→#8→#9
```

## #2 实验详情 (当前进行中)

仅将 2.9B 的 ffn.key.weight 和 ffn.value.weight 量化为 NVFP4, 其余全部 bf16。

```
量化: blocks.*.ffn.key.weight [10240,2560] + blocks.*.ffn.value.weight [2560,10240]
      32层 × 2 = 64 tensors, 3200 MB → ~914 MB, 节省 2286 MB
不量化: 其余 998 tensors, 2375 MB 不变
预期输出: 5490 MB → ~3204 MB ≈ 3.13 GB
预期 PPL delta: 0.00 ~ 0.03 (FFN 无 state, ReLU² 抑制)
验收: MSE ≤ 1e-4, top1 agree ≥ 99.5%, PPL delta ≤ 0.05
```

## 参考资源

- RWKV-7 源码: https://github.com/RWKV/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py
- RWKV-7 论文: https://arxiv.org/abs/2506.14761
- Molly移动模型: https://huggingface.co/mollysama/rwkv-mobile-models
- NVFP4博客: https://developer.nvidia.com/blog/achieving-rtx-pro-6000-peak-performance-with-nvfp4-on-vllm/
- 量化工具: pip install llmcompressor
- 内核: wkv_fp16_v1_clone_kernel, tmix_kk_a_gate_kernel

做为AI请从当前进度继续。
并且主动在issue区进行讨论


## 当前量化方案（1.5B 最终，Task 1-4 迭代后）

```
component:   key         value    rec      out      ffn_k           ffn_v   lowrank
Layer 0:     bf16        fp8      nvfp4    nvfp4    nvfp4+fp8残差    fp8     bf16
Layer 1-23:  fp8         fp8      nvfp4    nvfp4    nvfp4+fp8残差    fp8     bf16
```

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| MATH500 | ≥12.0% (gap≤0.6pp) | 11.2% (gap 1.4pp) | ✗ 差0.8pp |
| MATH500 验收线 | ≤2pp | 1.4pp | ✅ |
| PPL delta (1024/8192) | ≤0.05 | 0.028/0.002 | ✅ |
| 显存 | ≤2.0 GiB | 1.87 GiB | ✅ |
| decode | ≥65 t/s | 70 t/s | ✅ |
| 8192 state MSE L23 | 与FP16持平 | 6.81e-3 | ✅ |

设计原则：**全链路量化域执行，无运行时反量化**。
- 权重静态量化（离线算scale存盘）；激活动态量化（prep_x实时amax、
  fused kernel per-16-block动态scale）
- 残差per-block FP8走纯FP8×FP8 GEMM（dispatcher对残差权重强制fused kernel，
  因为 _scaled_mm 无法混合 fp32 tensor x-scale + fp8 block w-scale）
- AWQ alpha=0.3（grid search 7点，PPL几乎不变但MATH500 +2.6pp）
- 低秩(8×/层)保持bf16：只占1.7%内存，量化收益配不上精度

### 关键实验发现

1. **PPL 不预测 MATH500**：alpha 对 PPL 无影响(3.4219 vs 3.4224)但对解题 +2.6pp
2. **推理轨迹早期分叉**：量化模型第16个token即与原模型分歧(Q3)，之后
   质因数分解错误(196=2²×7¹×11⁰)一路错到底——不是答案边缘抖动
3. **state 存储 fp16 不变但承载量化误差**：k/v 由量化权重产生，误差渗透
   state 数值；8192长度 L23 MSE=6.81e-3 不累积
4. **低秩量化不划算**：FP8后磁盘省1.7%、运行时收益0、PPL+0.0052 → 回退bf16

## 研究方向（issue #12：逐层/逐头敏感度归因）

当前方案是"组件级经验分层"，层边界非逐层数据驱动（2.9B方案甚至直接缩放1.5B）。
下一步按数据驱动重新设计：

1. **逐层归因**：24次"单层量化其余bf16"PPL扫描 → 层敏感度曲线
2. **选择性量化**：基于曲线选量化层集合 → 精度-节省 Pareto 前沿
   （验证 bf16+nvfp4+bf16 交替是否成立）
3. **逐头归因**：若attention路径是主要误差源 → 中间层内32头归因
   （head×layer 才是理论最小粒度，per-head tensor scale 零成本）
4. **2.9B 重新归因**：不同参数量敏感度模式不同，不缩放

### 讨论共识

- 粒度：head×layer(768) > layer(24) > 组件(6)，但需先验证头间异质性
- 交替：bf16层=校准点（ln/残差重新归一化），但盲交替不如数据选择
- 动态量化：(a)激活动态 ✅ 已做；(b)逐层动态精度选择 ❌ 未做
- 模型规模：小模型冗余低→敏感度平；大模型中间层冗余高→选择性量化空间大
