# #6 W4A4: Long-sequence State Accumulation Error

## Results (1.5B mixed scheme, 2100 tokens)

| Window | Top-1 | PPL delta |
|------|-------|-----------|
| 0-100 | 84.00% | +1.6491 |
| 100-300 | 85.50% | +0.3374 |
| 300-500 | 92.50% | +0.1651 |
| 500-700 | 100.00% | +0.0029 |
| 700-1000 | 100.00% | +0.0004 |
| 1000-1500 | 100.00% | +0.0003 |
| 1500-2100 | 100.00% | +0.0002 |

Overall: Top-1=97.14%, PPL delta=+0.0242, VRAM=1.67G, Speed=3322 t/s

## Conclusion

- Errors are concentrated in the warmup phase (0-500 tokens), PPL delta decreases from 1.65 to 0.17
- After 500 tokens, Top-1=100%, PPL delta<0.003
- PPL delta **decreases and approaches 0** as sequence grows, proving that quantization error does not accumulate after state convergence
- No state divergence issue
