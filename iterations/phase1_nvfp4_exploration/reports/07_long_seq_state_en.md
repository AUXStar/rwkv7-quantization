# #6 Experiment Report: Long-sequence State MSE

## Overview

Testing whether quantization error accumulates over long sequences causing state divergence. RWKV's state update includes decay + erase + inject; theoretically, small errors may compound and amplify.

## Experimental Method

1. Generated 8192 tokens greedily using the original model (starting from the first 100 tokens of test_2100)
2. Ran both the original model and the quantized model on the 8192 tokens
3. Computed Top-1 and PPL delta by position window

## Experimental Results

| Window | 6a NVFP4 Top1 | 6a PPL_d | 6b FP8 Top1 | 6b PPL_d |
|------|---------------|----------|--------------|----------|
| 0-512 | 97.66% | +0.0147 | 99.22% | +0.0013 |
| 512-1024 | 100.00% | +0.0004 | 100.00% | 0.0000 |
| 1024-2048 | 100.00% | +0.0002 | 100.00% | 0.0000 |
| 2048-4096 | 100.00% | +0.0002 | 100.00% | 0.0000 |
| 4096-8192 | 100.00% | +0.0002 | 100.00% | 0.0000 |

## Key Findings

### Quantization Error Does Not Accumulate
- Top-1 reaches 100% in subsequent windows, with no downward trend
- PPL delta stabilizes at ~0.0002 (NVFP4) and 0.0000 (FP8)
- **Conclusion: state does not diverge due to quantization error**

### 0-512 Window Degrades More
- Includes the original prompt (first 100 tokens), where prediction is more difficult
- The quantized model performs slightly worse at "difficult" positions but perfectly at "easy" positions (model self-generated text)

### RWKV State Decay Mechanism Naturally Suppresses Error Accumulation
- Global decay `S = diag(w_delta) * S` shrinks the state at each step
- Old errors are decayed, new information is continuously injected
- Unlike RNN, RWKV's state has self-repair properties

## Conclusion

- Quantization has no accumulated error on 8192-token long sequences
- No special quantization handling needed for long-sequence scenarios
- The state decay mechanism naturally suppresses error propagation
