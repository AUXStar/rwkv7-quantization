#!/bin/bash
# A/B PPL comparison script
# Run original kernels PPL, then optimized kernels PPL, compare

set -e
export PATH="/home/njzy/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
FUSED_FILE="/home/njzy/test/Albatross/faster3a_2605/fused_nvfp4_gemm.py"
FUSED_BAK="${FUSED_FILE}.bak"
PPL_SCRIPT="/mnt/c/Users/njzy/.trae-cn/work/6a6b7147083abdc8623c651a/ppl_optimized.py"
PATCH_SCRIPT="/mnt/c/Users/njzy/.trae-cn/work/6a6b7147083abdc8623c651a/patch_fused_gemm.py"
EVAL_DIR="/home/njzy/test/eval_tmp"

echo "============================================================"
echo "A) PPL with ORIGINAL kernels (FP16 dot)"
echo "============================================================"
cp "$FUSED_BAK" "$FUSED_FILE"
cd /home/njzy/test/rwkv7-quantization
/usr/bin/python3 "$PPL_SCRIPT" 2>&1
cp "$EVAL_DIR/ppl_optimized_7b.json" "$EVAL_DIR/ppl_original_7b.json"
echo ""
echo "Original PPL saved to ppl_original_7b.json"

echo ""
echo "============================================================"
echo "B) PPL with OPTIMIZED kernels (FP8 hwdot)"
echo "============================================================"
/usr/bin/python3 "$PATCH_SCRIPT"
/usr/bin/python3 "$PPL_SCRIPT" 2>&1
echo ""
echo "Optimized PPL saved to ppl_optimized_7b.json"

echo ""
echo "============================================================"
echo "A/B Comparison"
echo "============================================================"
/usr/bin/python3 -c "
import json
with open('$EVAL_DIR/ppl_original_7b.json') as f: orig = json.load(f)
with open('$EVAL_DIR/ppl_optimized_7b.json') as f: opt = json.load(f)
print(f'{\"Length\":<10} {\"Original\":<14} {\"Optimized\":<14} {\"Delta\":<14}')
for k in sorted(orig.keys()):
    if k.startswith('ppl_'):
        o = orig[k]; p = opt.get(k, 0)
        print(f'{k[4:]:<10} {o:<14.6f} {p:<14.6f} {p-o:<+14.6f}')
print()
print(f'VRAM orig: {orig.get(\"vram_allocated_gib\",0):.2f} GiB')
print(f'VRAM opt:  {opt.get(\"vram_allocated_gib\",0):.2f} GiB')
"
