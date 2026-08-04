# #1 [Meta] RWKV-7 分层量化方案 v2（纯量化GEMM）

## 重大变更

v1 方案基于反量化推理（W4A16），已被用户否决。v2 改为**纯量化GEMM**，所有量化权重直接参与 `_scaled_mm` 计算，不反量化。

### v1 → v2 核心变化

| 项目 | v1（已废弃） | v2（当前） |
|------|-------------|-----------|
| 推理方式 | 反量化到FP16后GEMM | 纯量化GEMM（`_scaled_mm`） |
| FFN key | NVFP4 W4A16 | NVFP4+FP8残差 W4A4+W8A8 |
| FFN value | NVFP4 W4A16 | FP8 W8A8 |
| Attention key/value | 分层BF16/FP8/NVFP4 | 统一FP8 W8A8（L4-19可选NVFP4） |
| L0 value | BF16 | FP8（#5验证无损） |
| 激活量化 | 不量化（A16） | 在线量化到FP4/FP8（A4/A8） |

## 目标模型

| 模型 | 路径 | 原始大小 | dtype | 层数 | 用途 |
|------|------|----------|-------|------|------|
| 1.5B | `rwkv7-g1h-1.5b-20260710-ctx10240.pth` | 2.97 GB | bf16 | 24 | 实验验证 |
| 2.9B | `rwkv7-g1h_preview4673-2.9b-20260701-ctx8192.pth` | 5.49 GB | bf16 | 32 | 实验验证 |
| 7.2B | `rwkv7-g1g-7.2b-20260523-ctx8192.pth` | 13.41 GB | bf16 | 32 | 最终目标 |

## 量化格式

### NVFP4 (E2M1) — W4A4

- 权重：4-bit浮点（1s+2e+1m），每16元素共享FP8(E4M3) block scale + FP32 tensor scale
- 激活：在线量化到FP4，使用相同的block+tensor scale结构
- GEMM：`torch._scaled_mm(fp4, fp4, scale_a, scale_b) → bf16`
- AWQ：通道缩放（alpha=0.5），权重启发式（无校准数据时用权重abs mean代替激活统计）
- Clip ratio搜索：9个候选值[0.60~1.00]，逐block选MSE最小
- 权重存储：128x4 swizzle格式（`_scaled_mm`要求）

### FP8 (E4M3) — W8A8

- 权重：8-bit浮点（1s+4e+3m），per-tensor scale
- 激活：在线量化到FP8
- GEMM：`torch._scaled_mm(fp8, fp8, scale_a, scale_b) → bf16`

### NVFP4+FP8残差 — W4A4+W8A8

- 主权重：NVFP4 W4A4 GEMM
- 残差权重：(原权重 - NVFP4反量化值) 量化为FP8 W8A8
- 两路GEMM结果相加：`out = nvfp4_gemm(x, w) + fp8_gemm(x, residual)`
- 用途：补偿NVFP4 W4A4的量化误差（FFN key）

## 分层方案（v2，基于#2-#6实验结论）

### 1.5B（24层）

```
component:  key    value  rec    out    ffn_key      ffn_value
L0-3:       fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
L4-19:      nvfp4  fp8    nvfp4  nvfp4  nvfp4+res    fp8
L20-23:     fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
```

### 2.9B / 7.2B（32层）

```
component:  key    value  rec    out    ffn_key      ffn_value
L0-3:       fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
L4-27:      nvfp4  fp8    nvfp4  nvfp4  nvfp4+res    fp8
L28-31:     fp8    fp8    nvfp4  nvfp4  nvfp4+res    fp8
```

### 方案选择依据

| 组件 | 选择 | 依据 |
|------|------|------|
| att.key | 边缘FP8 / 中间NVFP4 | #4: NVFP4 key PPL delta 0.0041（可行但略逊FP8）；边缘层更敏感用FP8 |
| att.value | 全FP8 | #4: value比key更敏感；#3: FP8 W8A8 PPL delta 0.0018 近无损 |
| att.rec/out | 全NVFP4 | #4: rec/out低敏感度，NVFP4可行（方案C PPL delta 0.0101） |
| ffn.key | NVFP4+FP8残差 | #2: 纯NVFP4 W4A4误差大(0.0385)；残差补偿误差 |
| ffn.value | 全FP8 | #2: ReLU²后激活分布对FP4不友好；FP8 W8A8精度好 |
| L0 value | FP8（非BF16） | #5: FP8 vs BF16 PPL delta差0.0003，无需BF16 |

## 验证结论汇总（#2-#6, 1.5B, 2100 tokens）

| Issue | 方案 | PPL delta | Top-1 | 速度 | VRAM |
|-------|------|-----------|-------|------|------|
| #2 | FFN NVFP4 W4A4 | +0.0385 | 96.09% | 2426 t/s | 1.61G |
| #3 | Att FP8 W8A8 | +0.0018 | 99.43% | 6563 t/s | 2.50G |
| #4A | key NVFP4 L4-19 | +0.0041 | 98.19% | 2458 t/s | 1.54G |
| #4B | 全FP8 | +0.0033 | 98.38% | 6132 t/s | 1.60G |
| #5A | L0 value BF16 | +0.0030 | 98.33% | — | 1.57G |
| #5B | L0 value FP8 | +0.0033 | 98.38% | — | 1.60G |
| #6 | 混合方案 | +0.0242 | 97.14% | 3322 t/s | 1.67G |

### 关键发现

1. **FP8 W8A8远优于NVFP4 W4A4**：精度（0.0018 vs 0.0385）和速度（6563 vs 2426 t/s）都更好
2. **误差集中在warmup阶段**：前500 tokens Top-1 ~84-92%，后1600 tokens Top-1=100%
3. **无state累积**：PPL delta随序列增长递减趋近0（#6验证）
4. **L0 value不需要BF16**：FP8 W8A8足够（#5验证）
5. **FFN value不适合NVFP4**：ReLU²后激活分布对FP4不友好（#2发现）

## 不量化的参数

### 向量参数（每层~104KB）
- `blocks.*.att.x_{r,w,k,v,a,g}` [1,1,4096] ×6
- `blocks.*.att.{w0,a0,v0,k_k,k_a}` [1,1,4096] ×5
- `blocks.*.att.r_k` [64,64]
- `blocks.*.ffn.x_k` [1,1,4096]

### 低秩权重（每层~13MB，32层~406MB）
- `att.g1/g2` [4096,480]/[480,4096]
- `att.a1/a2` [4096,128]/[128,4096]
- `att.w1/w2` [4096,128]/[128,4096]
- `att.v1/v2` [4096,96]/[96,4096]

### LayerNorm / 全局
- `blocks.*.ln{0,1,2}.weight/bias`, `blocks.*.att.ln_x.weight/bias`
- `ln_out.weight/bias`
- `emb.weight` [65536,4096], `head.weight` [65536,4096]

## 工具链

| 文件 | 功能 |
|------|------|
| `quantize_model.py` | 统一量化工具（加载→分类→量化→存储+meta） |
| `nvfp4_ops.py` | NVFP4/FP8 GEMM kernel + 权重加载 + 激活量化 |
| `fused_nvfp4_quant.py` | Triton fused kernel（激活量化+pack+swizzle） |

## 验收标准

- PPL delta ≤ 0.05（1.5B）/ ≤ 0.02（7.2B）
- Top-1 ≥ 99.5%（warmup后）
- 压缩率 ≥ 2x
- 纯量化GEMM，不反量化
