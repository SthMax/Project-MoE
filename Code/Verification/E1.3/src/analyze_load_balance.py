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
    
    # Sort by PPL
    results.sort(key=lambda x: x.get("average_ppl", float("inf")))
    
    print("\n" + "-"*80)
    print("RANKING BY PPL (lower is better)")
    print("-"*80)
    print(f"{'Rank':<5} {'Run Name':<40} {'Mode':<12} {'PPL':<10} {'Throughput':<12}")
    print("-"*80)
    
    baseline_ppl = None
    for i, r in enumerate(results):
        mode = r.get("routing_mode", "?")
        ppl = r.get("average_ppl", float("inf"))
        throughput = r.get("throughput", 0)
        
        if mode == "baseline" and baseline_ppl is None:
            baseline_ppl = ppl
        
        delta_str = ""
        if baseline_ppl is not None and ppl != float("inf"):
            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            delta_str = f" ({delta:+.2f}%)"
        
        print(f"{i+1:<5} {r['run_name'][:40]:<40} {mode:<12} {ppl:.4f}{delta_str:<10} {throughput:.2f}/s")
    
    # Summary
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    
    modes = set(r.get("routing_mode") for r in results)
    for mode in sorted(modes):
        mode_results = [r for r in results if r.get("routing_mode") == mode]
        ppls = [r["average_ppl"] for r in mode_results if r["average_ppl"] != float("inf")]
        if ppls:
            print(f"{mode}: PPL range = [{min(ppls):.4f}, {max(ppls):.4f}], n={len(ppls)}")
    
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
