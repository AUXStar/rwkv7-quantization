#!/usr/bin/env python3
"""Analyze weight distribution asymmetry for RWKV-7 1.5B model.

For each quantized weight component, compute:
- mean, std
- skewness (mean/std)
- positive/negative max ratio
- fraction of positive values
- potential benefit of asymmetric quantization (zero-point shift)

If |mean| / std > 0.1, asymmetric quantization could help.
"""
import sys, os, json, torch
import numpy as np

MODEL_PATH = "/home/njzy/model/rwkv7-g1h-1.5b-20260710-ctx10240.pth"

COMPONENTS = {
    "att.receptance": "blocks.{l}.att.receptance.weight",
    "att.key": "blocks.{l}.att.key.weight",
    "att.value": "blocks.{l}.att.value.weight",
    "att.output": "blocks.{l}.att.output.weight",
    "ffn.key": "blocks.{l}.ffn.key.weight",
    "ffn.value": "blocks.{l}.ffn.value.weight",
}

def analyze_weight(w):
    """Analyze a single weight tensor [N, K]."""
    w_np = w.float().numpy().flatten()

    mean = w_np.mean()
    std = w_np.std()
    amax = np.abs(w_np).max()
    pos_max = w_np.max()
    neg_max = w_np.min()
    pos_frac = (w_np > 0).mean()

    # Skewness = mean / std (simple measure)
    skew = mean / std if std > 0 else 0

    # Asymmetric quantization benefit:
    # Symmetric: scale = max(|w|) / 448, uses range [-448, 448]
    # Asymmetric: shift = mean, then scale = max(|w - mean|) / 448
    # Benefit = (max(|w|) - max(|w - mean|)) / max(|w|)  (how much scale we save)
    shifted_amax = np.abs(w_np - mean).max()
    benefit = (amax - shifted_amax) / amax if amax > 0 else 0

    # Quantization error comparison
    # Symmetric FP8
    sym_scale = amax / 448.0
    sym_q = np.clip(w_np / sym_scale, -448, 448).round()
    sym_deq = sym_q * sym_scale
    sym_mse = np.mean((w_np - sym_deq) ** 2)

    # Asymmetric FP8 (shift by mean)
    asym_scale = shifted_amax / 448.0
    asym_q = np.clip((w_np - mean) / asym_scale, -448, 448).round()
    asym_deq = asym_q * asym_scale + mean
    asym_mse = np.mean((w_np - asym_deq) ** 2)

    mse_improvement = (sym_mse - asym_mse) / sym_mse if sym_mse > 0 else 0

    return {
        "shape": list(w.shape),
        "mean": float(mean),
        "std": float(std),
        "amax": float(amax),
        "pos_max": float(pos_max),
        "neg_max": float(neg_max),
        "pos_frac": float(pos_frac),
        "skew": float(skew),
        "sym_mse": float(sym_mse),
        "asym_mse": float(asym_mse),
        "mse_improvement": float(mse_improvement),
        "scale_benefit": float(benefit),
    }


def main():
    print(f"Loading model: {MODEL_PATH}", flush=True)
    z = torch.load(MODEL_PATH, map_location="cpu", mmap=True)

    # Determine number of layers
    layer_keys = [k for k in z.keys() if k.startswith("blocks.")]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    print(f"Model: {num_layers} layers\n", flush=True)

    results = {}

    # Analyze each component (average across layers)
    for comp_name, key_pattern in COMPONENTS.items():
        print(f"--- {comp_name} ---", flush=True)
        comp_stats = []

        for layer in range(num_layers):
            key = key_pattern.format(l=layer)
            if key not in z:
                continue
            w = z[key]
            if w.dim() != 2:
                continue
            stats = analyze_weight(w)
            comp_stats.append(stats)

        # Aggregate
        if comp_stats:
            n = len(comp_stats)
            avg = {
                "shape": comp_stats[0]["shape"],
                "num_layers": n,
                "avg_mean": np.mean([s["mean"] for s in comp_stats]),
                "avg_std": np.mean([s["std"] for s in comp_stats]),
                "avg_skew": np.mean([s["skew"] for s in comp_stats]),
                "avg_pos_frac": np.mean([s["pos_frac"] for s in comp_stats]),
                "avg_sym_mse": np.mean([s["sym_mse"] for s in comp_stats]),
                "avg_asym_mse": np.mean([s["asym_mse"] for s in comp_stats]),
                "avg_mse_improvement": np.mean([s["mse_improvement"] for s in comp_stats]),
                "avg_scale_benefit": np.mean([s["scale_benefit"] for s in comp_stats]),
                "max_skew": max(abs(s["skew"]) for s in comp_stats),
                "max_mse_improvement": max(s["mse_improvement"] for s in comp_stats),
            }
            results[comp_name] = avg

            print(f"  Shape: {avg['shape']}, Layers: {n}", flush=True)
            print(f"  Mean: {avg['avg_mean']:.6f}, Std: {avg['avg_std']:.6f}", flush=True)
            print(f"  Skew (mean/std): {avg['avg_skew']:.6f} (max: {avg['max_skew']:.6f})", flush=True)
            print(f"  Positive fraction: {avg['avg_pos_frac']:.4f}", flush=True)
            print(f"  Symmetric MSE: {avg['avg_sym_mse']:.8f}", flush=True)
            print(f"  Asymmetric MSE: {avg['avg_asym_mse']:.8f}", flush=True)
            print(f"  MSE improvement: {avg['avg_mse_improvement']*100:.2f}% (max: {avg['max_mse_improvement']*100:.2f}%)", flush=True)
            print(f"  Scale benefit: {avg['avg_scale_benefit']*100:.2f}%", flush=True)

            # Verdict
            if avg['avg_mse_improvement'] > 0.05:
                print(f"  >>> ASYMMETRIC QUANTIZATION HELPFUL (>5% MSE reduction)", flush=True)
            elif avg['avg_mse_improvement'] > 0.01:
                print(f"  >>> Marginal benefit from asymmetric quantization (1-5%)", flush=True)
            else:
                print(f"  >>> No benefit from asymmetric quantization (<1%)", flush=True)
            print(flush=True)

    # Summary
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Component':<20} {'Skew':<10} {'Pos%':<10} {'MSE Improve':<15} {'Verdict'}", flush=True)
    for comp, stats in results.items():
        verdict = "HELPFUL" if stats['avg_mse_improvement'] > 0.05 else \
                  "marginal" if stats['avg_mse_improvement'] > 0.01 else "no benefit"
        print(f"{comp:<20} {stats['avg_skew']:<10.4f} {stats['avg_pos_frac']:<10.4f} {stats['avg_mse_improvement']*100:<15.2f} {verdict}", flush=True)

    # Save results
    output_path = "/home/njzy/test/rwkv7-quantization/experiments/weight_asymmetry_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
