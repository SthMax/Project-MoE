#!/usr/bin/env python3
"""
E1.3: Analyze load balance and routing distribution results.

This script analyzes the routing statistics collected during experiments
to evaluate load imbalance metrics and PriceMoE effectiveness.

Usage:
    python -m src.analyze_load_balance outputs/runs/20260115_120000_e1_3_price_moe/
    python -m src.analyze_load_balance outputs/runs/ --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_json, save_json, compute_load_imbalance, gini_coefficient


def load_npz_stats(run_dir: Path) -> Optional[dict]:
    """
    Load routing statistics from npz files in moe_stats/ directory.
    
    Returns:
        Dictionary with aggregated routing statistics, or None if no npz files found.
    """
    moe_stats_dir = run_dir / "moe_stats"
    if not moe_stats_dir.exists():
        return None
    
    npz_files = list(moe_stats_dir.glob("*.npz"))
    if not npz_files:
        return None
    
    # Aggregate stats from all npz files (in case of multiple processes)
    all_cnt_topk = []
    all_cnt_top1 = []
    total_tokens = 0
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            if "cnt_topk" in data:
                all_cnt_topk.append(data["cnt_topk"])
            if "cnt_top1" in data:
                all_cnt_top1.append(data["cnt_top1"])
            if "tokens_processed" in data:
                total_tokens += data["tokens_processed"].sum()
        except Exception as e:
            print(f"[Warning] Failed to load {npz_file}: {e}")
            continue
    
    if not all_cnt_topk:
        return None
    
    # Sum across all processes
    cnt_topk = np.sum(all_cnt_topk, axis=0)  # [num_layers, num_experts]
    cnt_top1 = np.sum(all_cnt_top1, axis=0) if all_cnt_top1 else None
    
    num_layers, num_experts = cnt_topk.shape
    
    # Compute per-layer load imbalance (max/mean)
    layer_imbalance = []
    for layer_idx in range(num_layers):
        layer_counts = cnt_topk[layer_idx]
        if layer_counts.sum() > 0:
            mean_load = layer_counts.mean()
            max_load = layer_counts.max()
            imbalance = max_load / mean_load if mean_load > 0 else 0
            layer_imbalance.append(imbalance)
    
    # Compute global Gini coefficient
    total_counts = cnt_topk.sum(axis=0)  # Sum across layers
    gini = gini_coefficient(total_counts) if total_counts.sum() > 0 else 0
    
    # Find hottest experts per layer
    hottest_experts = cnt_topk.argmax(axis=1).tolist()
    
    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_tokens": int(total_tokens),
        "load_imbalance": {
            "avg": float(np.mean(layer_imbalance)) if layer_imbalance else 0,
            "min": float(np.min(layer_imbalance)) if layer_imbalance else 0,
            "max": float(np.max(layer_imbalance)) if layer_imbalance else 0,
            "per_layer": layer_imbalance,
        },
        "gini": float(gini),
        "hottest_experts": hottest_experts,
        "cnt_topk": cnt_topk,  # Keep for detailed analysis
    }


def analyze_single_run(run_dir: Path) -> dict:
    """
    Analyze a single experiment run.
    
    Returns:
        Dictionary with analysis results.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    
    summary = load_json(summary_path)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}")
    print(f"{'='*60}")
    print(f"Routing mode: {summary.get('routing_mode', 'unknown')}")
    print(f"Routing params: {summary.get('routing_params', {})}")
    print(f"Average PPL: {summary.get('average_ppl', 'N/A'):.4f}")
    print(f"Throughput: {summary.get('overall_throughput', 'N/A'):.2f} examples/s")
    print(f"Total examples: {summary.get('total_examples', 0)}")
    
    # Load npz stats from moe_stats/ directory
    npz_stats = load_npz_stats(run_dir)
    if npz_stats:
        print(f"\n--- Load Balance Metrics (from npz) ---")
        print(f"  Layers: {npz_stats['num_layers']}, Experts: {npz_stats['num_experts']}")
        print(f"  Total tokens: {npz_stats['total_tokens']:,}")
        imb = npz_stats['load_imbalance']
        print(f"  Load imbalance (max/mean):")
        print(f"    Avg: {imb['avg']:.3f}")
        print(f"    Min: {imb['min']:.3f}")
        print(f"    Max: {imb['max']:.3f}")
        print(f"  Gini coefficient: {npz_stats['gini']:.4f}")
    else:
        print(f"\n[Info] No npz stats found in {run_dir / 'moe_stats'}")
    
    # Analyze per-domain results
    domains = summary.get("domains", [])
    
    aggregate_load = []
    aggregate_ppl = []
    
    for domain_result in domains:
        domain = domain_result.get("domain", "unknown")
        ppl = domain_result.get("ppl", {}).get("ppl", float("inf"))
        
        print(f"\n--- Domain: {domain} ---")
        print(f"  PPL: {ppl:.4f}")
        print(f"  Examples: {domain_result.get('n_examples', 0)}")
        
        # Check routing stats
        routing_stats = domain_result.get("routing_stats", {})
        if "load_imbalance" in routing_stats:
            load_metrics = routing_stats["load_imbalance"]
            print(f"  Load imbalance (max/mean): {load_metrics.get('max_to_mean', 'N/A'):.3f}")
            print(f"  Gini coefficient: {load_metrics.get('gini', 'N/A'):.3f}")
            print(f"  Hottest expert: {load_metrics.get('max_expert', 'N/A')}")
        
        # Check PriceMoE stats
        price_moe_stats = domain_result.get("price_moe_stats", {})
        if price_moe_stats:
            print(f"  PriceMoE layers active: {len(price_moe_stats)}")
            # Show stats from first layer as example
            for layer_key, stats in list(price_moe_stats.items())[:1]:
                print(f"    Layer {layer_key}:")
                print(f"      Avg load imbalance: {stats.get('avg_load_imbalance', 'N/A'):.3f}")
                print(f"      Total updates: {stats.get('total_updates', 0)}")
        
        aggregate_ppl.append(ppl)
    
    result = {
        "run_name": run_dir.name,
        "routing_mode": summary.get("routing_mode"),
        "routing_params": summary.get("routing_params"),
        "average_ppl": summary.get("average_ppl"),
        "throughput": summary.get("overall_throughput"),
        "total_examples": summary.get("total_examples"),
        "num_domains": len(domains),
    }
    
    # Add npz stats to result
    if npz_stats:
        result["load_imbalance_avg"] = npz_stats["load_imbalance"]["avg"]
        result["load_imbalance_max"] = npz_stats["load_imbalance"]["max"]
        result["gini"] = npz_stats["gini"]
        result["total_tokens"] = npz_stats["total_tokens"]
    
    return result


def compare_runs(run_dirs: list[Path]) -> dict:
    """
    Compare multiple experiment runs.
    """
    print("\n" + "="*80)
    print("COMPARISON ACROSS RUNS")
    print("="*80)
    
    results = []
    for run_dir in run_dirs:
        try:
            result = analyze_single_run(run_dir)
            results.append(result)
        except Exception as e:
            print(f"[Warning] Failed to analyze {run_dir}: {e}")
    
    if not results:
        print("No valid runs to compare")
        return {}
    
    # Sort by load imbalance (lower is better)
    results.sort(key=lambda x: x.get("load_imbalance_avg", float("inf")))
    
    print("\n" + "-"*100)
    print("RANKING BY LOAD BALANCE (lower is better)")
    print("-"*100)
    header = f"{'Rank':<5} {'Run Name':<45} {'Mode':<12} {'Imbalance':<12} {'Gini':<10} {'PPL':<10} {'Tput':<10}"
    print(header)
    print("-"*100)
    
    baseline_imb = None
    baseline_gini = None
    for i, r in enumerate(results):
        mode = r.get("routing_mode", "?")
        imb = r.get("load_imbalance_avg", float("inf"))
        gini = r.get("gini", float("inf"))
        ppl = r.get("average_ppl", float("inf"))
        throughput = r.get("throughput", 0)
        
        if mode == "baseline":
            baseline_imb = imb
            baseline_gini = gini
        
        # Show delta compared to baseline
        imb_str = f"{imb:.3f}" if imb != float("inf") else "N/A"
        gini_str = f"{gini:.4f}" if gini != float("inf") else "N/A"
        ppl_str = f"{ppl:.4f}" if ppl != float("inf") else "inf"
        tput_str = f"{throughput:.2f}/s" if throughput else "N/A"
        
        print(f"{i+1:<5} {r['run_name'][:45]:<45} {mode:<12} {imb_str:<12} {gini_str:<10} {ppl_str:<10} {tput_str:<10}")
    
    # Compute improvements
    print("\n" + "-"*80)
    print("LOAD BALANCE IMPROVEMENTS vs BASELINE")
    print("-"*80)
    
    if baseline_imb is not None and baseline_imb != float("inf"):
        for r in results:
            mode = r.get("routing_mode", "?")
            if mode == "baseline":
                continue
            imb = r.get("load_imbalance_avg", float("inf"))
            gini = r.get("gini", float("inf"))
            
            if imb != float("inf") and baseline_imb > 0:
                imb_reduction = (1 - imb / baseline_imb) * 100
                print(f"  {r['run_name'][:50]}:")
                print(f"    Imbalance: {baseline_imb:.3f} → {imb:.3f} (↓ {imb_reduction:.1f}%)")
                if baseline_gini and gini != float("inf"):
                    gini_reduction = (1 - gini / baseline_gini) * 100
                    print(f"    Gini:      {baseline_gini:.4f} → {gini:.4f} (↓ {gini_reduction:.1f}%)")
    else:
        print("  No baseline run found for comparison")
    
    # Summary by mode
    print("\n" + "-"*80)
    print("SUMMARY BY ROUTING MODE")
    print("-"*80)
    
    modes = set(r.get("routing_mode") for r in results)
    for mode in sorted(modes):
        mode_results = [r for r in results if r.get("routing_mode") == mode]
        imbs = [r["load_imbalance_avg"] for r in mode_results if r.get("load_imbalance_avg") not in (None, float("inf"))]
        ginis = [r["gini"] for r in mode_results if r.get("gini") not in (None, float("inf"))]
        ppls = [r["average_ppl"] for r in mode_results if r.get("average_ppl") not in (None, float("inf"))]
        
        print(f"\n  {mode}:")
        if imbs:
            print(f"    Load imbalance: avg={np.mean(imbs):.3f}, range=[{min(imbs):.3f}, {max(imbs):.3f}], n={len(imbs)}")
        if ginis:
            print(f"    Gini: avg={np.mean(ginis):.4f}, range=[{min(ginis):.4f}, {max(ginis):.4f}], n={len(ginis)}")
        if ppls:
            print(f"    PPL: avg={np.mean(ppls):.4f}, range=[{min(ppls):.4f}, {max(ppls):.4f}], n={len(ppls)}")
    
    return {"runs": results}


def main():
    parser = argparse.ArgumentParser(
        description="E1.3: Analyze load balance and routing distribution"
    )
    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Path(s) to run directories or parent directory containing runs",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple runs (auto-enabled if directory contains multiple runs)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for analysis results JSON",
    )
    
    args = parser.parse_args()
    
    # Collect all run directories
    run_dirs = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: path not found: {path}")
            continue
        
        # Check if it's a run directory (has summary.json)
        if (path / "summary.json").exists():
            run_dirs.append(path)
        else:
            # Assume it's a parent directory containing runs
            for child in sorted(path.iterdir()):
                if child.is_dir() and (child / "summary.json").exists():
                    run_dirs.append(child)
    
    if not run_dirs:
        print("No valid run directories found")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} run(s) to analyze")
    
    if len(run_dirs) == 1 and not args.compare:
        result = analyze_single_run(run_dirs[0])
    else:
        result = compare_runs(run_dirs)
    
    if args.output:
        save_json(result, args.output)
        print(f"\nAnalysis saved to {args.output}")


if __name__ == "__main__":
    main()
