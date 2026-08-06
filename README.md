# RWKV-7 Quantization Toolkit

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/rwkv-quant-nv.svg)](https://pypi.org/project/rwkv-quant-nv/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)

> FP8 / INT8 / INT4 quantization for RWKV-7, with direct-on-quantized-weight inference (no dequantization).

[中文](README_zh.md) | **English**

## Install

```bash
pip install rwkv-quant-nv
```

## Quick Start

```bash
# List available quantization schemes
rwkv-quant l

# Quantize a model
rwkv-quant q -m model.pth -o quantized.pth -s fp8

# Show model info
rwkv-quant i -m model.pth

# Compare quantized vs baseline
rwkv-quant c -m model.pth -q quantized.pth

# Evaluate (EAR, Top-1, speed, VRAM)
rwkv-quant e -b model.pth -q quantized.pth

# Sensitivity analysis
rwkv-quant s -m model.pth
```

CLI subcommands support abbreviations: `l` (list), `i` (info), `q` (quantize), `c` (compare), `e` (eval), `s` (sens).

## Quantization Schemes

| Scheme | Format | Compression | Hardware | Score |
|--------|--------|-------------|----------|-------|
| `fp8` | float8_e4m3fn (W8A8) | 2.0x | SM 8.9+ (FP8 tensor cores) | 4.5/5 |
| `fp8_perchannel` | float8_e4m3fn per-channel | 2.0x | SM 8.9+ | 4.0/5 |
| `int8_symmetric` | int8 (W8A8) | 2.0x | Any CUDA GPU | 3.5/5 |
| `int8_affine` | uint8 + dual affine | ~1.9x | Any CUDA GPU | 4.0/5 |
| `int4_symmetric` | int4 packed (W4A16) | 4.0x | Any CUDA GPU | 3.0/5 |
| `int4_groupwise_128` | int4 + per-group scale | ~3.5x | Any CUDA GPU | 3.5/5 |
| `int4_groupwise_256` | int4 + per-group scale | ~3.7x | Any CUDA GPU | 3.5/5 |

## Key Design

- **No dequantization**: All quantized weights stay in their quantized domain during inference. FP8 weights computed via FP8 tensor cores (`_scaled_mm` / `tl.dot(fp8, fp8)`), INT8 via DP4A (`tl.dot(int8, int8)`).
- **head.weight unquantized**: Always kept in FP16 for output projection quality.
- **Fused Triton kernels**: R/K/V projections computed in a single kernel launch. Shape-aware tile configs for Blackwell FP8.
- **Lazy loading**: Heavy modules (torch, CUDA extensions) loaded on-demand for fast CLI response.
- **CUDA JIT compilation**: CUDA kernels compiled at runtime via `torch.utils.cpp_extension.load`, auto-adapting to the local GPU architecture.

## Architecture

```
rwkv-quant          # CLI entry point
rwkv_quant/         # Main package
  cli.py            # Argument parsing
  commands.py       # Subcommand implementations (list/info/quantize/compare/eval/sens)
  engine.py         # Inference engine loader
  evaluate.py       # EAR / Top-1 metrics
  schemes.py        # Scheme registry, weight classification, model state loading
  utils.py          # Terminal styling, file size, vocab locator
schemes.py          # Quantization functions (fp8/int8/int4)
fp8_ops.py          # FP8/INT8 weight loading + GEMM dispatch
fused_fp8_gemm.py   # Triton fused kernels (FP8 + INT8 GEMM, RKV fusion)
rwkv7_fast_v3a.py   # RWKV-7 inference engine (CUDA extensions)
cuda/               # CUDA source files (.cu / .cpp)
int4/               # INT4 standalone quantization tool + Triton kernels
int8/               # INT8 standalone quantization tool + Triton kernels
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NVIDIA GPU (SM 8.9+ for FP8; any CUDA GPU for INT8/INT4)
- Triton (bundled with PyTorch >= 2.0)
- NVIDIA CUDA Toolkit (for JIT compilation of CUDA extensions)

## Iteration History

This project went through 4 phases of systematic exploration. Full reports are in the `iterations/` directory:

| Phase | Topic | Key Conclusion |
|-------|-------|----------------|
| Phase 1 | NVFP4 exploration, sensitivity analysis | NVFP4 error 8.8%, all components equally sensitive |
| Phase 2 | Engine integration, fused kernel development | Fused kernel 1.84x speedup, CUDA Graph not beneficial |
| Phase 3 | X5 residual scheme, multi-model validation | X5 slightly more accurate but not worth the complexity |
| Phase 4 | Final FP8 scheme, operator optimization | Full FP8 is the optimal scheme for SM 8.9+ |

See [QUANTIZATION_CONCLUSION.md](QUANTIZATION_CONCLUSION.md) for the complete experimental comparison.

## Acknowledgments

- [RWKV-7](https://github.com/RWKV/RWKV-LM) model architecture
- [Blink_DL](https://modelscope.cn/models/Blink_DL/temp-latest-training-models) model weights
- [Albatross](https://github.com/BlinkDL/Albatross) inference engine

## License

MIT — see [LICENSE](LICENSE).
