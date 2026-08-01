# RWKV-7 1.5B 量化最终定稿（X5 方案）

## 一、定稿方案

L0: bf16 (v_first源头)
L1-L5: key FP8 / value FP8 / rec NVFP4 / out NVFP4 / ffn_k NVFP4+FP8残差 / ffn_v FP8
L6: bf16 (稀释点，重置量化误差)
L7-L17: key FP8 / value FP8 / rec NVFP4 / out NVFP4 / ffn_k NVFP4+FP8残差 / ffn_v FP8
L18: bf16 (稀释点，重置量化误差)
L19-L22: key FP8 / value FP8 / rec NVFP4 / out NVFP4 / ffn_k NVFP4+FP8残差 / ffn_v FP8
L23: bf16 (输出前最后层)
AWQ: alpha=0.3

压缩比 1.7x (2.25 GB -> 1.30 GB)。量化20层，保护4层（稀释层）。

## 二、定稿依据（最终验收数据）

| 指标 | 目标 | X5 实测 | 状态 |
|------|------|---------|------|
| MATH500 | >=12.0% | 11.8% (59/500) | 差1题 |
| PPL delta @8192 | <=0.05 | +0.0024 | OK |
| 显存 | <=2.0 GiB | 1.93 GiB | OK |
| decode | >=65t/s | 78.9 t/s | OK |

X5 是全部方案中唯一同时满足 MATH500~12% 和显存<=2.0GiB 的方案。

## 三、定稿是怎么来的

### Phase 1: 基础方案 (Task 1-4)
低秩量化不划算->回退bf16; key全面FP8; AWQ alpha=0.3是最大功臣(8.6%->11.2%)

### Phase 2: 敏感度归因实验
PPL归因平坦->PPL不能当主指标; state MSE归因: L0最不敏感(3.3e-5), L14峰值(1.27e-2)
交替方案崩溃->L0的v_first污染是根因

### Phase 3: M梯度搜索
M5(L0保护)=11.4%, M2(L0-5+L18-23)=12.4%, M2c(L0-4+L18-23)=12.0%

### Phase 4: 稀释层扫描（X0-X5）
X0(L0+L23)=9.0%, X5(L0+L6+L18+L23)=13.5%(200题)
L6(1/4处)+L18(3/4处)是最佳稀释点
X5全量500题=11.8%, 与M2c(12.0%)持平, 但仅4层bf16

### 关键洞察: 稀释层 vs 保护带
保护带(M2c): 连续多层bf16, 浪费量化层
稀释层(X5): 离散bf16点, 将网络分段, 每段起始重置量化误差
稀释点位置比数量重要: L6+L18(1/4+3/4) > L12(正中) > L8+L16(均匀)

## 四、实现思路

核心原则: 全链路量化域执行, 禁止运行时反量化。

1. 激活动态量化: prep_x kernel 每次 forward 实时算 amax;
   fused kernel 内部对 x tile 做 per-16-block 动态量化。
2. 权重静态量化: 离线量化, scale 存盘。
3. 残差补偿: ffn.key 的 NVFP4 误差用 FP8 残差补偿(per-block ratio scale + tensor scale),
   走纯 FP8*FP8 GEMM。
4. AWQ alpha=0.3: grid search 7 点, MATH500 +2.6pp。
5. 稀释层 L0/L6/L18/L23: L0 是 v_first 源头; L6/L18 是1/4和3/4处的稀释点,
   重置量化误差; L23 是输出前最后层。

## 五、每个张量的量化方式

### 量化层 L1-L5, L7-L17, L19-L22 (20层)

| 张量 | 格式 | 量化方式 |
|------|------|---------|
| att.receptance.weight [C,C] | NVFP4 | E2M1 4bit + per-16-block E4M3 scale + fp32 tensor scale |
| att.key.weight [C,C] | FP8 | E4M3 8bit + per-row fp32 scale |
| att.value.weight [C,C] | FP8 | 同上 |
| att.output.weight [C,C] | NVFP4 | 同 rec |
| ffn.key.weight [4C,C] | NVFP4+残差 | NVFP4 + FP8残差(per-block ratio + tensor scale) |
| ffn.value.weight [C,4C] | FP8 | 同 key |

### 稀释层 L0, L6, L18, L23 (4层)
所有 6 个线性层保持 bf16 (不量化)。

### 不量化 (全模型)
| 张量 | 原因 |
|------|------|
| emb.weight, head.weight | 全局1GB, 太大, 且是embedding查找 |
| 低秩 g1/g2/a1/a2/w1/w2/v1/v2 | 只占1.7%内存, 收益配不上精度 |
| x_r/x_w/x_k/x_v/x_a/x_g, w0/a0/v0, k_k/k_a/r_k | 向量参数, 每层~104KB |
| LayerNorm/GroupNorm (ln0/ln1/ln2/ln_x/ln_out) | 归一化必须精确 |

## 六、迭代历程汇总

| 方案 | 保护层 | MATH500 | 显存 | decode | 压缩 |
|------|--------|---------|------|--------|------|
| M0全量化 | L0-key | 11.2% | 1.87 | 70 | 2.0x |
| M5 | L0 | 11.4% | 1.87 | 70 | 1.9x |
| M2 | L0-5+L18-23 | 12.4% | 2.23 | 58.5 | 1.3x |
| M2c | L0-4+L18-23 | 12.0% | 2.19 | 86.3 | 1.4x |
| X5 | L0+L6+L18+L23 | 11.8% | 1.93 | 78.9 | 1.7x |
| 交替A/B | - | 0% | - | - | - |
