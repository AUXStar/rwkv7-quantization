#!/bin/bash
source /home/njzy/test/.venv/bin/activate
cd /home/njzy/test/rwkv7-quantization

models=(
  "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v8.pth:v8_baseline"
  "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9.pth:v9_gptq_d0.1"
  "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9-d5.pth:v9_gptq_d5"
  "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9-d10.pth:v9_gptq_d10"
  "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v10.pth:v10_hessian_awq"
  "/home/njzy/model/rwkv7-g1h-2.9b-hybrid-v9b.pth:v9b_gptq_awq_hess"
)

for entry in "${models[@]}"; do
  path="${entry%%:*}"
  label="${entry##*:}"
  echo "============================================"
  echo "Benchmarking: $label"
  echo "============================================"
  python quick_bench.py "$path" "$label" 2>&1
  echo ""
done
