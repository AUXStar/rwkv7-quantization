# RWKV-7 量化工具包

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/rwkv-quant-nv.svg)](https://pypi.org/project/rwkv-quant-nv/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)

> RWKV-7 FP8 / INT8 / INT4 量化工具，直接在量化权重上推理，不反量化。

[中文](README_zh.md) | **English**

## 安装

```bash
pip install rwkv-quant-nv
```

## 快速开始

```bash
# 列出可用量化方案
rwkv-quant l

# 量化模型
rwkv-quant q -m model.pth -o quantized.pth -s fp8

# 查看模型信息
rwkv-quant i -m model.pth

# 对比量化前后
rwkv-quant c -m model.pth -q quantized.pth

# 评估（EAR、Top-1、速度、显存）
rwkv-quant e -b model.pth -q quantized.pth

# 敏感度分析
rwkv-quant s -m model.pth
```

CLI 子命令支持缩写：`l` (list)、`i` (info)、`q` (quantize)、`c` (compare)、`e` (eval)、`s` (sens)。

## 量化方案

| 方案 | 格式 | 压缩比 | 硬件要求 | 评分 |
|------|------|--------|----------|------|
| `fp8` | float8_e4m3fn (W8A8) | 2.0x | SM 8.9+（FP8 张量核） | 4.5/5 |
| `fp8_perchannel` | float8_e4m3fn 逐通道 | 2.0x | SM 8.9+ | 4.0/5 |
| `int8_symmetric` | int8 (W8A8) | 2.0x | 任意 CUDA GPU | 3.5/5 |
| `int8_affine` | uint8 + 双仿射 | ~1.9x | 任意 CUDA GPU | 4.0/5 |
| `int4_symmetric` | int4 打包 (W4A16) | 4.0x | 任意 CUDA GPU | 3.0/5 |
| `int4_groupwise_128` | int4 + 逐组 scale | ~3.5x | 任意 CUDA GPU | 3.5/5 |
| `int4_groupwise_256` | int4 + 逐组 scale | ~3.7x | 任意 CUDA GPU | 3.5/5 |

## 核心设计

- **不反量化**：所有量化权重在推理时保持在量化域内。FP8 权重通过 FP8 张量核（`_scaled_mm` / `tl.dot(fp8, fp8)`）计算，INT8 通过 DP4A 指令（`tl.dot(int8, int8)`）计算。
- **head.weight 不量化**：始终保持 FP16，保证输出投影层质量。
- **融合 Triton 内核**：R/K/V 投影在单个 kernel launch 中完成。针对 Blackwell FP8 的 shape-aware tile 配置。
- **懒加载**：重量级模块（torch、CUDA 扩展）按需加载，CLI 响应快。
- **CUDA JIT 编译**：CUDA 内核在运行时通过 `torch.utils.cpp_extension.load` 编译，自动适配本地 GPU 架构。

## 项目结构

```
rwkv-quant          # CLI 入口
rwkv_quant/         # 主包
  cli.py            # 参数解析
  commands.py       # 子命令实现（list/info/quantize/compare/eval/sens）
  engine.py         # 推理引擎加载
  evaluate.py       # EAR / Top-1 指标
  schemes.py        # 方案注册表、权重分类、模型状态加载
  utils.py          # 终端样式、文件大小、vocab 定位
schemes.py          # 量化函数（fp8/int8/int4）
fp8_ops.py          # FP8/INT8 权重加载 + GEMM 分发
fused_fp8_gemm.py   # Triton 融合内核（FP8 + INT8 GEMM，RKV 融合）
rwkv7_fast_v3a.py   # RWKV-7 推理引擎（CUDA 扩展）
cuda/               # CUDA 源文件（.cu / .cpp）
int4/               # INT4 独立量化工具 + Triton 内核
int8/               # INT8 独立量化工具 + Triton 内核
```

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- NVIDIA GPU（FP8 需 SM 8.9+；INT8/INT4 任意 CUDA GPU）
- Triton（PyTorch >= 2.0 自带）
- NVIDIA CUDA Toolkit（用于 JIT 编译 CUDA 扩展）

## 迭代历史

本项目经过 4 个阶段的系统探索，完整报告在 `iterations/` 目录：

| 阶段 | 主题 | 关键结论 |
|------|------|----------|
| Phase 1 | NVFP4 探索、敏感度分析 | NVFP4 误差 8.8%，所有组件敏感度均匀 |
| Phase 2 | 引擎适配、融合内核开发 | 融合内核 1.84x 加速，CUDA Graph 无益 |
| Phase 3 | X5 残差方案、多模型验证 | X5 精度略高但复杂度不值得 |
| Phase 4 | 最终 FP8 方案、算子优化 | 全 FP8 是 SM 8.9+ 上的最优方案 |

详见 [QUANTIZATION_CONCLUSION.md](QUANTIZATION_CONCLUSION.md)。

## 致谢

- [RWKV-7](https://github.com/RWKV/RWKV-LM) 模型架构
- [Blink_DL](https://modelscope.cn/models/Blink_DL/temp-latest-training-models) 模型权重
- [Albatross](https://github.com/BlinkDL/Albatross) 推理引擎

## License

MIT — 见 [LICENSE](LICENSE)。
