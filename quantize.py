#!/usr/bin/env python3
# coding=utf-8
"""Unified quantization CLI for RWKV-7 models.

    python quantize.py --model model.pth --output q.pth --scheme fp8
    python quantize.py --model model.pth --output q.pth --scheme int8_affine
    python quantize.py --list-schemes
    python quantize.py --compare model.pth --output-dir ./compare_out
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import torch

QUANTIZED = {"att_r","att_k","att_v","att_o","ffn_k","ffn_v"}
COMP_MAP = {"receptance":"att_r","key":"att_k","value":"att_v","output":"att_o"}
FFN_MAP = {"key":"ffn_k","value":"ffn_v"}

def classify(key, nl):
    p = key.split(".")
    if len(p)<4 or p[0]!="blocks": return None
    try: layer=int(p[1])
    except: return None
    if layer<0 or layer>=nl: return None
    comp = COMP_MAP.get(p[3]) if p[2]=="att" else (FFN_MAP.get(p[3]) if p[2]=="ffn" else None)
    return (layer, comp) if comp in QUANTIZED else None

def quantize_model(model_path, output_path, scheme_name, *, verbose=True):
    sys.path.insert(0, str(Path(__file__).parent))
    from schemes import get_scheme
    scheme = get_scheme(scheme_name)
    qfn = scheme["quantize"]
    qkw = dict(scheme.get("quantize_kwargs", {}))

    if verbose: print(f"Scheme: {scheme['name']} | Loading {model_path}...")
    t0=time.time()
    z=torch.load(model_path, map_location="cpu", weights_only=True)
    nl=max((int(k.split(".")[1]) for k in z if k.startswith("blocks.") and len(k.split("."))>=2), default=0)+1
    if verbose: print(f"  {nl} layers, {len(z)} tensors")

    qc=0; meta={"scheme":scheme_name,"format":scheme["format"],"bits":scheme["bits_per_weight"],"act_bits":scheme["activation_bits"],"layers":nl,"ts":time.strftime("%Y-%m-%d %H:%M:%S")}
    for key in list(z.keys()):
        info=classify(key,nl)
        if info is None: continue
        w=z[key].float()
        z[key]=qfn(w,**qkw)
        qc+=1

    z["quant_meta"]=meta
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(z, output_path)
    sz=os.path.getsize(output_path)/1e9; orig=os.path.getsize(model_path)/1e9
    ratio=orig/max(sz,1e-9); dt=time.time()-t0
    if verbose:
        print(f"  Quantized {qc} weights in {dt:.1f}s")
        print(f"  {orig:.2f}GB -> {sz:.2f}GB ({ratio:.2f}x)")
    return {"scheme":scheme_name,"weights":qc,"orig_gb":round(orig,3),"q_gb":round(sz,3),"ratio":round(ratio,3),"time_s":round(dt,1)}

def compare_schemes(model_path, output_dir, selected=None):
    from schemes import list_schemes, SCHEMES
    schemes = selected or list_schemes()
    results=[]
    for sn in schemes:
        out=os.path.join(output_dir, f"model_{sn}.pth")
        print(f"\n--- {sn} ---")
        r=quantize_model(model_path, out, sn, verbose=True)
        results.append(r)
    summary=os.path.join(output_dir,"comparison.json")
    with open(summary,"w") as f: json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"{'Scheme':20s} {'Orig':>6s} {'Quant':>6s} {'Ratio':>6s} {'Time':>5s}")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['scheme']:20s} {r['orig_gb']:5.2f}G {r['q_gb']:5.2f}G {r['ratio']:5.2f}x {r['time_s']:4.1f}s")
    print(f"\nComparison saved to {summary}")

def main():
    p=argparse.ArgumentParser(description="RWKV-7 Unified Quantization Tool", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python quantize.py -m model.pth -o q.pth -s fp8\n  python quantize.py -m model.pth -o q.pth -s int8_affine\n  python quantize.py --list-schemes\n  python quantize.py -m model.pth -C ./compare\n")
    p.add_argument("-m","--model",help="Input model .pth path")
    p.add_argument("-o","--output",help="Output quantized .pth path")
    p.add_argument("-s","--scheme",default="fp8",help="Quantization scheme (default: fp8)")
    p.add_argument("-C","--compare",metavar="DIR",help="Compare all schemes, save to DIR")
    p.add_argument("--schemes",nargs="+",help="Limit comparison to these schemes")
    p.add_argument("--list-schemes",action="store_true",help="List all available schemes")
    args=p.parse_args()
    if args.list_schemes:
        from schemes import print_scheme_table; print_scheme_table(); return
    if args.compare:
        if not args.model: p.error("--compare requires --model")
        compare_schemes(args.model, args.compare, args.schemes); return
    if not args.model or not args.output: p.error("--model and --output are required (or use --list-schemes / --compare)")
    quantize_model(args.model, args.output, args.scheme)

if __name__=="__main__": main()
