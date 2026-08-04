# #5 W4A4: L0 value BF16 vs FP8

## Results (1.5B, 2100 tokens)

| Scheme | PPL delta | Top-1 | VRAM |
|------|-----------|-------|------|
| A: L0 value BF16 | +0.0030 | 98.33% | 1.57G |
| B: L0 value FP8 | +0.0033 | 98.38% | 1.60G |

## Conclusion

L0 value FP8 W8A8 is nearly identical to BF16 (PPL delta difference 0.0003, Top-1 difference 0.05%). The v_first cross-layer propagation error is negligible under W8A8. L0 value does not need BF16; FP8 is sufficient.
