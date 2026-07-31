# 最终量化方案（Task 1-4 完成）验收报告

## 方案（相对原方案的三处修改）

| 组件 | 原方案 | 最终方案 | 依据 |
|------|--------|---------|------|
| att.key | L0-3/L20-23 FP8, L4-19 NVFP4 | **L0 bf16, L1-23 FP8** | T2: 全面FP8 +0.8pp |
| 低秩(8×/层) | bf16 | **bf16（不量化）** | 用户决策（1.7%收益不值）|
| ffn.key残差 | per-tensor FP8 | **per-block FP8 ratio** | Task 3，量化域计算 |
| AWQ alpha | 0.5 | **0.3** | T4 grid search（MATH500 8.6%→11.2%）|

**原则落实**：全链路量化域执行，**无运行时反量化**。残差per-block走纯FP8×FP8 GEMM
（fused kernel 任意M处理；_scaled_mm 无法混合 fp32 tensor x-scale + fp8 block w-scale，
故dispatcher对残差权重强制fused）。

## Task 1-4 MATH500 数据（orig=12.6%缓存基准）

| 版本 | 配置 | MATH500 | delta |
|------|------|---------|-------|
| 基线 | 原方案 (alpha 0.5) | 8.0% | -4.6pp |
| T1 | +低秩FP8 | 8.6% | -4.0pp |
| T2 | +key全面FP8 | 9.4% | -3.2pp |
| T3 | 低秩bf16 + 残差per-block | 8.6% | -4.0pp |
| **T4** | **T3 + alpha=0.3** | **11.2%** | **-1.4pp** |

**关键发现**：AWQ alpha 对 PPL 几乎无影响（3.4219 vs 3.4224），但对 MATH500 有
**+2.6pp** 的影响——PPL 无法预测解题质量。

## 最终验收

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| MATH500 | ≥12.0% (gap≤0.6pp) | **11.2%** (gap 1.4pp) | ✗ 差0.8pp |
| MATH500 验收线 | ≤2pp | 1.4pp | ✅ PASS |
| PPL delta (1024/2048/4096/8192) | ≤0.05 | 0.028/0.008/0.005/0.002 | ✅ |
| 显存 | ≤2.0 GiB | 1.87 GiB | ✅ |
| decode 速度 | ≥65 t/s | 70.0 t/s | ✅ |
| 8192 state MSE L23 | 与FP16持平 | 6.81e-3 | ✅ |

## 测试过程记录

1. **T1 低秩FP8**：发现key无.weight后缀bug（修复）；per-column scale避免小K误差；
   MATH500 8.6%，收益1.7%磁盘、0运行时收益 → 用户决策回退bf16
2. **T2 key全面FP8**：9.4%，有效但不够
3. **T3 残差per-block**：初版因 **Albatross/rwkv7-quantization 的 nvfp4_ops.py 副本漂移**
   导致加载器不识别per-block格式 → 生成全崩；同步后修复。PPL 1.5338（最佳）但 MATH500 8.6%
4. **T4 AWQ alpha search**：alpha∈[0.1,0.2,0.3,0.4,0.5,0.7,0.9] 上 MATH500-dist PPL，
   alpha=0.3 最优 → MATH500 **11.2%**（重大跃升）

## 下一步可选（追 12.0%）

1. alpha=0.2/0.4 直接跑 MATH500（PPL 不预测 MATH500，可能更好）
2. 恢复低秩 FP8 + alpha=0.3（T2 数据 +0.8pp 是在 alpha=0.5 下测得，alpha=0.3 下可能叠加）
3. 接受 11.2%（已过 2pp 验收线）

## 输出物

- `eval_tmp/math500_T4.json` — T4 每题明细
- `eval_tmp/final_acceptance.json` — 最终验收指标
- `run_math500_quant.py` / `run_final_accept.py` / `run_t4_search.py` — 复现脚本
