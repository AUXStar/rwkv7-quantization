#!/usr/bin/env python3
"""Per-tensor sensitivity analysis for RWKV-7 quantization - Part 1: Stats + Error"""
import json, math, os, torch, torch.nn.functional as F
from collections import defaultdict

MODEL = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"
OUT = "/home/njzy/test/rwkv7-quantization/experiments/tensor_sensitivity.json"

def q_fp8(w):
    s = w.float().abs().max().clamp(min=1e-12) / 448.0
    return ((w.float()/s).clamp(-448,448).round(), s)

def q_i8s(w):
    s = w.float().abs().max().clamp(min=1e-12) / 127.0
    return ((w.float()/s).round().clamp(-128,127).float(), s)

def q_i8a(w):
    wf = w.float(); sh = wf.shape
    if wf.dim()!=2: wf=wf.reshape(1,-1)
    my=wf.amin(1,keepdim=True); w2=wf-my
    mx=w2.amin(0,keepdim=True); w2=w2-mx
    rx=w2.amax(0,keepdim=True).clamp(min=1e-12); w2=w2/rx
    ry=w2.amax(1,keepdim=True).clamp(min=1e-12); w2=w2/ry
    wq=(w2*256).floor().clamp(0,255)
    return ((wq+0.5)*(ry*16)*(rx*16)+my+mx).reshape(sh)

def q_i4s(w):
    s = w.float().abs().max().clamp(min=1e-12) / 7.0
    return ((w.float()/s).round().clamp(-8,7).float()) * s

def q_i4gw(w, gs=128):
    wf=w.float()
    sh=wf.shape; flat=wf.dim()==2
    if not flat: wf=wf.reshape(1,-1)
    N,M=wf.shape; pad=(gs-M%gs)%gs
    if pad: wf=F.pad(wf,(0,pad)); M+=pad
    wg=wf.reshape(N,M//gs,gs)
    mn,mx=wg.amin(2,keepdim=True),wg.amax(2,keepdim=True)
    s=((mx-mn)/15).clamp(min=1e-12); z=mn
    wq=((wg-z)/s).round().clamp(0,15)
    r=(wq*s+z).reshape(N,M)[:,:sh[-1] if flat else sh[-1]]
    return r

def metrics(o,a):
    o,a=o.float(),a.float(); e=o-a
    mse=float((e**2).mean())
    snr_db=10*math.log10(max(float((o**2).mean())/max(mse,1e-20),1e-20))
    cos=float(F.cosine_similarity(o.reshape(1,-1),a.reshape(1,-1)))
    return {"mse":round(mse,10),"snr_db":round(snr_db,2),"cos":round(cos,8)}

def main():
    z = torch.load(MODEL, map_location="cpu", weights_only=True)
    nl = max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))+1
    print(f"Model: {nl} layers, {len(z)} tensors")

    def comp_type(key):
        if "ffn.key.weight" in key: return "ffn_key"
        if "ffn.value.weight" in key: return "ffn_value"
        if "att.receptance.weight" in key: return "att_rec"
        if "att.key.weight" in key: return "att_key"
        if "att.value.weight" in key: return "att_value"
        if "att.output.weight" in key: return "att_output"
        if "head.weight" in key or "lm_head" in key: return "lm_head"
        if "ln" in key: return "layernorm"
        if any(x in key for x in ["w1","w2","a1","a2","g1","g2","v1","v2"]): return "lowrank"
        if "r_k" in key: return "r_k"
        if any(x in key for x in ["x_r","x_w","x_k","x_v","x_a","x_g","k_k","k_a","a0","v0","w0"]): return "vector"
        return "other"

    results = []
    for key in sorted(z.keys()):
        val = z[key]
        if not isinstance(val, torch.Tensor) or val.dim()<1 or val.numel()<100:
            continue
        wf = val.float(); std_v = float(wf.std())
        st = {"mean":round(float(wf.mean()),6),"std":round(std_v,6),
              "min":round(float(wf.min()),6),"max":round(float(wf.max()),6),
              "l2":round(float(wf.norm()),4),
              "skew":round(float(((wf-wf.mean())**3).mean()/max(std_v**3,1e-20)),4),
              "kurt":round(float(((wf-wf.mean())**4).mean()/max(std_v**4,1e-20)-3),4),
              "sp1%":round(float((wf.abs()<wf.abs().max()*0.01).float().mean()),4),
              "sp10%":round(float((wf.abs()<wf.abs().max()*0.1).float().mean()),4)}
        ct = comp_type(key)
        qe = {}
        if val.dim()==2 and val.numel()>=10000:
            wq,_=q_fp8(wf); qe["fp8"]=metrics(wf,wq)
            wq,_=q_i8s(wf); qe["i8sym"]=metrics(wf,wq)
            wq=q_i8a(wf);   qe["i8aff"]=metrics(wf,wq)
            wq=q_i4s(wf);   qe["i4sym"]=metrics(wf,wq)
            wq=q_i4gw(wf,128); qe["i4gw128"]=metrics(wf,wq)
            wq=q_i4gw(wf,256); qe["i4gw256"]=metrics(wf,wq)
        results.append({"key":key,"shape":list(val.shape),"numel":val.numel(),
                        "type":ct,"stats":st,"qe":qe})

    # Aggregate by type
    groups = defaultdict(list)
    for r in results: groups[r["type"]].append(r)

    summary = {}
    for ct, entries in sorted(groups.items()):
        fp8s = [e["qe"]["fp8"] for e in entries if "fp8" in e["qe"]]
        i8as = [e["qe"]["i8aff"] for e in entries if "i8aff" in e["qe"]]
        i4gs = [e["qe"]["i4gw128"] for e in entries if "i4gw128" in e["qe"]]
        s = {"n":len(entries),"params":sum(e["numel"] for e in entries),
             "skew":round(sum(e["stats"]["skew"] for e in entries)/len(entries),4),
             "kurt":round(sum(e["stats"]["kurt"] for e in entries)/len(entries),4),
             "sp1%":round(sum(e["stats"]["sp1%"] for e in entries)/len(entries),4)}
        if fp8s: s["fp8_snr"]=round(sum(x["snr_db"] for x in fp8s)/len(fp8s),2); s["fp8_cos"]=round(sum(x["cos"] for x in fp8s)/len(fp8s),6)
        if i8as: s["i8a_snr"]=round(sum(x["snr_db"] for x in i8as)/len(i8as),2); s["i8a_cos"]=round(sum(x["cos"] for x in i8as)/len(i8as),6)
        if i4gs: s["i4g_snr"]=round(sum(x["snr_db"] for x in i4gs)/len(i4gs),2); s["i4g_cos"]=round(sum(x["cos"] for x in i4gs)/len(i4gs),6)
        summary[ct] = s
        print(f"  {ct:12s}: {s['n']:3d} tensors, {s['params']:>10,} params, skew={s['skew']:+.3f}, kurt={s['kurt']:+.3f}, sp1%={s['sp1%']:.1%}")

    # Cross-scheme comparison for main linear weights
    print("\n--- Cross-Scheme SNR (dB) for main linear weights ---")
    for ct in ["ffn_key","ffn_value","att_rec","att_key","att_value","att_output","lm_head"]:
        entries = groups.get(ct, [])
        if not entries: continue
        for e in entries:
            qe = e["qe"]
            if not qe: continue
            row = f"  {e['key'][-40:]:40s} ({list(e['shape'])}) "
            for scheme in ["fp8","i8sym","i8aff","i4sym","i4gw128","i4gw256"]:
                if scheme in qe:
                    row += f" {scheme}={qe[scheme]['snr_db']:6.1f}dB"
            print(row)

    # Save
    out = {"pass1_results": results, "pass1_summary": summary}
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved pass1 to {OUT}")

if __name__ == "__main__":
    main()
