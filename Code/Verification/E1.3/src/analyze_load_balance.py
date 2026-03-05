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


def _infer_ep_size(config: dict) -> int:
    enable_ep = bool(config.get("enable_expert_parallel", False))
    tp_size = int(config.get("tensor_parallel_size", 1) or 1)
    return tp_size if enable_ep else 1


def load_npz_stats(run_dir: Path, *, ep_size: int = 1) -> Optional[dict]:
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
    
    ep_size = max(int(ep_size or 1), 1)
    ep_size = min(ep_size, num_experts)

    # EP-rank (per-GPU) load metrics: group experts into contiguous EP buckets.
    # vLLM uses ep_rank_start = ep_rank * num_local_experts with num_local_experts = num_experts // ep_size.
    ep_rank_imbalance = []
    ep_rank_ginis = []
    if ep_size > 1:
        experts_per_rank = max(num_experts // ep_size, 1)
        for layer_idx in range(num_layers):
            layer_counts = cnt_topk[layer_idx]
            if layer_counts.sum() <= 0:
                continue
            ep_loads = []
            for ep_rank in range(ep_size):
                start = ep_rank * experts_per_rank
                end = (ep_rank + 1) * experts_per_rank
                if ep_rank == ep_size - 1:
                    end = num_experts
                start = min(start, num_experts)
                end = min(end, num_experts)
                ep_loads.append(layer_counts[start:end].sum())
            ep_loads = np.asarray(ep_loads, dtype=np.float64)
            mean_load = float(ep_loads.mean())
            max_load = float(ep_loads.max())
            ep_rank_imbalance.append(max_load / mean_load if mean_load > 0 else 0.0)
            ep_rank_ginis.append(float(gini_coefficient(ep_loads)) if ep_loads.sum() > 0 else 0.0)

    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "ep_size": int(ep_size),
        "total_tokens": int(total_tokens),
        "load_imbalance": {
            "avg": float(np.mean(layer_imbalance)) if layer_imbalance else 0,
            "min": float(np.min(layer_imbalance)) if layer_imbalance else 0,
            "max": float(np.max(layer_imbalance)) if layer_imbalance else 0,
            "per_layer": layer_imbalance,
        },
        "ep_rank_load_imbalance": {
            "avg": float(np.mean(ep_rank_imbalance)) if ep_rank_imbalance else 0.0,
            "min": float(np.min(ep_rank_imbalance)) if ep_rank_imbalance else 0.0,
            "max": float(np.max(ep_rank_imbalance)) if ep_rank_imbalance else 0.0,
            "per_layer": ep_rank_imbalance,
        },
        "ep_rank_gini": {
            "avg": float(np.mean(ep_rank_ginis)) if ep_rank_ginis else 0.0,
            "min": float(np.min(ep_rank_ginis)) if ep_rank_ginis else 0.0,
            "max": float(np.max(ep_rank_ginis)) if ep_rank_ginis else 0.0,
            "per_layer": ep_rank_ginis,
        },
        "gini": float(gini),
        "hottest_experts": hottest_experts,
        "cnt_topk": cnt_topk,  # Keep for detailed analysis
    }


def analyze_single_run(run_dir: Path) -> dict:
    """
    Analyze a single experiment run.
    
    Supports two formats:
    1. PPL mode: summary.json with domain results
    2. Throughput mode: benchmark_results.json + config.json
    
    Returns:
        Dictionary with analysis results.
    """
    summary_path = run_dir / "summary.json"
    benchmark_path = run_dir / "benchmark_results.json"
    config_path = run_dir / "config.json"

    config_for_ep = load_json(config_path) if config_path.exists() else {}
    ep_size = _infer_ep_size(config_for_ep)
    
    # Determine which format we have
    if summary_path.exists():
        # PPL mode - original format
        summary = load_json(summary_path)
        routing_mode = summary.get('routing_mode', 'unknown')
        routing_params = summary.get('routing_params', {})
        avg_ppl = summary.get('average_ppl', float('inf'))
        throughput = summary.get('overall_throughput', 0)
        total_examples = summary.get('total_examples', 0)
        domains = summary.get("domains", [])
    elif benchmark_path.exists():
        # Throughput benchmark mode
        benchmark = load_json(benchmark_path)
        config = load_json(config_path) if config_path.exists() else {}
        
        routing_mode = config.get('mode', config.get('routing', {}).get('mode', 'unknown'))
        routing_params = config.get('routing', config)
        avg_ppl = float('inf')  # No PPL in throughput mode
        throughput = benchmark.get('requests_per_second', 0)
        total_examples = benchmark.get('num_prompts', 0)
        domains = []
        
        # Create a pseudo-summary for throughput mode
        summary = {
            "routing_mode": routing_mode,
            "routing_params": routing_params,
            "benchmark": benchmark,
        }
    else:
        raise FileNotFoundError(f"No summary.json or benchmark_results.json found in: {run_dir}")
    
    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}")
    print(f"{'='*60}")
    print(f"Routing mode: {routing_mode}")
    print(f"Routing params: {routing_params}")
    
    # Update interval: read from config if present
    # Shell-script configs store it at top level; run_experiment.py configs nest it under 'routing'
    update_interval = config_for_ep.get('update_interval', None)
    if update_interval is None:
        update_interval = config_for_ep.get('routing', {}).get('update_interval', None)
    if update_interval is not None:
        print(f"Update interval: {update_interval}")
    
    if avg_ppl != float('inf'):
        print(f"Average PPL: {avg_ppl:.4f}")
    
    if benchmark_path.exists():
        benchmark = load_json(benchmark_path)
        print(f"Throughput: {benchmark.get('requests_per_second', 0):.2f} req/s")
        print(f"Token/s: {benchmark.get('tokens_per_second', 0):.2f}")
        print(f"Elapsed time: {benchmark.get('elapsed_time', 0):.2f} s")
    else:
        print(f"Throughput: {throughput:.2f} examples/s")
    
    print(f"Total examples: {total_examples}")

    if ep_size > 1:
        print(f"EP size: {ep_size}")
    
    # Load npz stats from moe_stats/ directory
    npz_stats = load_npz_stats(run_dir, ep_size=ep_size)
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

        if ep_size > 1:
            ep_imb = npz_stats.get("ep_rank_load_imbalance", {})
            ep_gini = npz_stats.get("ep_rank_gini", {})
            print(f"  EP-rank load imbalance (max/mean):")
            print(f"    Avg: {ep_imb.get('avg', 0.0):.3f}")
            print(f"    Min: {ep_imb.get('min', 0.0):.3f}")
            print(f"    Max: {ep_imb.get('max', 0.0):.3f}")
            print(f"  EP-rank Gini: {ep_gini.get('avg', 0.0):.4f}")
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
    
    # Build per-domain PPL dict for comparison
    domain_ppls = {}
    for domain_result in domains:
        domain = domain_result.get("domain", "unknown")
        ppl_val = domain_result.get("ppl", {}).get("ppl", float("inf"))
        if ppl_val != float("inf"):
            domain_ppls[domain] = ppl_val

    result = {
        "run_name": run_dir.name,
        "routing_mode": routing_mode,
        "routing_params": routing_params,
        "average_ppl": avg_ppl,
        "domain_ppls": domain_ppls,
        "throughput": throughput,
        "total_examples": total_examples,
        "num_domains": len(domains),
        "ep_size": int(ep_size),
    }
    if update_interval is not None:
        result["update_interval"] = update_interval
    
    # Add benchmark-specific fields if available
    if benchmark_path.exists():
        benchmark = load_json(benchmark_path)
        result["requests_per_second"] = benchmark.get("requests_per_second", 0)
        result["tokens_per_second"] = benchmark.get("tokens_per_second", 0)
        result["elapsed_time"] = benchmark.get("elapsed_time", 0)
    
    # Add npz stats to result
    if npz_stats:
        result["load_imbalance_avg"] = npz_stats["load_imbalance"]["avg"]
        result["load_imbalance_max"] = npz_stats["load_imbalance"]["max"]
        result["gini"] = npz_stats["gini"]
        result["total_tokens"] = npz_stats["total_tokens"]

        if ep_size > 1:
            result["ep_rank_imbalance_avg"] = npz_stats.get("ep_rank_load_imbalance", {}).get("avg", 0.0)
            result["ep_rank_imbalance_max"] = npz_stats.get("ep_rank_load_imbalance", {}).get("max", 0.0)
            result["ep_rank_gini_avg"] = npz_stats.get("ep_rank_gini", {}).get("avg", 0.0)
    
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
    
    # Determine if this is a throughput benchmark (has requests_per_second)
    is_throughput_benchmark = any(r.get("requests_per_second") for r in results)
    # Check if any run has PPL data
    has_ppl_data = any(r.get("average_ppl", float("inf")) != float("inf") for r in results)
    
    print("\n" + "-"*140)
    print("RANKING BY LOAD BALANCE (lower is better)")
    print("-"*140)
    
    show_ep_cols = any(r.get("ep_rank_imbalance_avg") is not None for r in results)

    # Check if any non-baseline run has update_interval
    has_interval = any(r.get("update_interval") is not None and r.get("routing_mode") != "baseline" for r in results)

    if is_throughput_benchmark:
        if show_ep_cols:
            header = (
                f"{'Rank':<5} {'Run Name':<55} {'Mode':<12} "
                + (f"{'Intv':<6} " if has_interval else "")
                + f"{'Imb':<8} {'Gini':<8} {'EPimb':<8} {'EPgini':<8} {'Req/s':<10} {'Tok/s':<12}"
            )
        else:
            header = f"{'Rank':<5} {'Run Name':<55} {'Mode':<12} " + (f"{'Intv':<6} " if has_interval else "") + f"{'Imbalance':<10} {'Gini':<8} {'Req/s':<10} {'Tok/s':<12}"
    else:
        if has_ppl_data:
            header = f"{'Rank':<5} {'Run Name':<55} {'Mode':<12} " + (f"{'Intv':<6} " if has_interval else "") + f"{'Imbalance':<12} {'Gini':<10} {'PPL':<10} {'Tput':<10}"
        else:
            header = f"{'Rank':<5} {'Run Name':<55} {'Mode':<12} " + (f"{'Intv':<6} " if has_interval else "") + f"{'Imbalance':<12} {'Gini':<10} {'Tput':<10}"
    print(header)
    print("-"*140)
    
    baseline_imb = None
    baseline_gini = None
    baseline_ep_imb = None
    baseline_ep_gini = None
    baseline_tput = None
    for i, r in enumerate(results):
        mode = r.get("routing_mode", "?")
        imb = r.get("load_imbalance_avg", float("inf"))
        gini = r.get("gini", float("inf"))
        ppl = r.get("average_ppl", float("inf"))
        throughput = r.get("throughput", 0)
        req_per_sec = r.get("requests_per_second", 0)
        tok_per_sec = r.get("tokens_per_second", 0)
        ep_imb = r.get("ep_rank_imbalance_avg", None)
        ep_gini = r.get("ep_rank_gini_avg", None)
        intv = r.get("update_interval", None)
        intv_str = f"{intv}" if intv is not None else "-"
        
        if mode == "baseline":
            baseline_imb = imb
            baseline_gini = gini
            baseline_ep_imb = ep_imb
            baseline_ep_gini = ep_gini
            baseline_tput = req_per_sec if req_per_sec else throughput
        
        # Show delta compared to baseline
        imb_str = f"{imb:.3f}" if imb != float("inf") else "N/A"
        gini_str = f"{gini:.4f}" if gini != float("inf") else "N/A"
        
        if is_throughput_benchmark:
            req_str = f"{req_per_sec:.2f}" if req_per_sec else "N/A"
            tok_str = f"{tok_per_sec:.1f}" if tok_per_sec else "N/A"
            if show_ep_cols:
                ep_imb_str = f"{ep_imb:.3f}" if ep_imb is not None else "N/A"
                ep_gini_str = f"{ep_gini:.4f}" if ep_gini is not None else "N/A"
                print(
                    f"{i+1:<5} {r['run_name'][:55]:<55} {mode:<12} "
                    + (f"{intv_str:<6} " if has_interval else "")
                    + f"{imb_str:<8} {gini_str:<8} {ep_imb_str:<8} {ep_gini_str:<8} {req_str:<10} {tok_str:<12}"
                )
            else:
                print(f"{i+1:<5} {r['run_name'][:55]:<55} {mode:<12} " + (f"{intv_str:<6} " if has_interval else "") + f"{imb_str:<10} {gini_str:<8} {req_str:<10} {tok_str:<12}")
        else:
            tput_str = f"{throughput:.2f}/s" if throughput else "N/A"
            if has_ppl_data:
                ppl_str = f"{ppl:.4f}" if ppl != float("inf") else "N/A"
                print(f"{i+1:<5} {r['run_name'][:55]:<55} {mode:<12} " + (f"{intv_str:<6} " if has_interval else "") + f"{imb_str:<12} {gini_str:<10} {ppl_str:<10} {tput_str:<10}")
            else:
                print(f"{i+1:<5} {r['run_name'][:55]:<55} {mode:<12} " + (f"{intv_str:<6} " if has_interval else "") + f"{imb_str:<12} {gini_str:<10} {tput_str:<10}")
    
    # Compute improvements
    print("\n" + "-"*80)
    print("IMPROVEMENTS vs BASELINE")
    print("-"*80)
    
    # Find baseline PPL and domain PPLs for comparison
    baseline_ppl = None
    baseline_domain_ppls = {}
    for r in results:
        if r.get("routing_mode") == "baseline":
            ppl_val = r.get("average_ppl", float("inf"))
            if ppl_val != float("inf"):
                baseline_ppl = ppl_val
                baseline_domain_ppls = r.get("domain_ppls", {})
            break

    if baseline_imb is not None and baseline_imb != float("inf"):
        for r in results:
            mode = r.get("routing_mode", "?")
            if mode == "baseline":
                continue
            imb = r.get("load_imbalance_avg", float("inf"))
            gini = r.get("gini", float("inf"))
            req_per_sec = r.get("requests_per_second", 0)
            tok_per_sec = r.get("tokens_per_second", 0)
            ep_imb = r.get("ep_rank_imbalance_avg", None)
            ep_gini = r.get("ep_rank_gini_avg", None)
            
            if imb != float("inf") and baseline_imb > 0:
                imb_reduction = (1 - imb / baseline_imb) * 100
                print(f"  {r['run_name'][:60]}:")
                print(f"    Imbalance: {baseline_imb:.3f} → {imb:.3f} (↓ {imb_reduction:.1f}%)")
                if baseline_gini and gini != float("inf"):
                    gini_reduction = (1 - gini / baseline_gini) * 100
                    print(f"    Gini:      {baseline_gini:.4f} → {gini:.4f} (↓ {gini_reduction:.1f}%)")

                if baseline_ep_imb and ep_imb is not None and baseline_ep_imb > 0:
                    ep_imb_reduction = (1 - ep_imb / baseline_ep_imb) * 100
                    direction = "↓" if ep_imb_reduction >= 0 else "↑"
                    print(f"    EP-imb:    {baseline_ep_imb:.3f} → {ep_imb:.3f} ({direction} {abs(ep_imb_reduction):.1f}%)")
                if baseline_ep_gini and ep_gini is not None and baseline_ep_gini > 0:
                    ep_gini_reduction = (1 - ep_gini / baseline_ep_gini) * 100
                    direction = "↓" if ep_gini_reduction >= 0 else "↑"
                    print(f"    EP-gini:   {baseline_ep_gini:.4f} → {ep_gini:.4f} ({direction} {abs(ep_gini_reduction):.1f}%)")
                
                # Throughput improvement (for EP benchmark)
                if baseline_tput and baseline_tput > 0:
                    current_tput = req_per_sec if req_per_sec else r.get("throughput", 0)
                    if current_tput:
                        tput_change = (current_tput / baseline_tput - 1) * 100
                        arrow = "↑" if tput_change > 0 else "↓"
                        print(f"    Throughput: {baseline_tput:.2f} → {current_tput:.2f} req/s ({arrow} {abs(tput_change):.1f}%)")
                        if tok_per_sec:
                            print(f"    Token Rate: {tok_per_sec:.1f} tok/s")

                # PPL comparison
                run_ppl = r.get("average_ppl", float("inf"))
                if baseline_ppl is not None and run_ppl != float("inf"):
                    ppl_change = (run_ppl / baseline_ppl - 1) * 100
                    arrow = "↑" if ppl_change > 0 else "↓"
                    print(f"    PPL:       {baseline_ppl:.4f} → {run_ppl:.4f} ({arrow} {abs(ppl_change):.1f}%)")
                    # Per-domain PPL breakdown
                    run_domain_ppls = r.get("domain_ppls", {})
                    if baseline_domain_ppls and run_domain_ppls:
                        for domain in sorted(baseline_domain_ppls.keys()):
                            if domain in run_domain_ppls:
                                bl_d = baseline_domain_ppls[domain]
                                rn_d = run_domain_ppls[domain]
                                d_change = (rn_d / bl_d - 1) * 100
                                d_arrow = "↑" if d_change > 0 else "↓"
                                print(f"      {domain:>12}: {bl_d:.4f} → {rn_d:.4f} ({d_arrow} {abs(d_change):.1f}%)")
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


def is_valid_run_dir(path: Path) -> bool:
    """Check if a directory is a valid run directory."""
    return (path / "summary.json").exists() or (path / "benchmark_results.json").exists()


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
        
        # Check if it's a run directory (has summary.json or benchmark_results.json)
        if is_valid_run_dir(path):
            run_dirs.append(path)
        else:
            # Assume it's a parent directory containing runs
            # Support nested structure: runs/{model_short}/{timestamp}_{run_name}
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    if is_valid_run_dir(child):
                        # Direct run directory
                        run_dirs.append(child)
                    else:
                        # Could be a model subdirectory, check children
                        for grandchild in sorted(child.iterdir()):
                            if grandchild.is_dir() and is_valid_run_dir(grandchild):
                                run_dirs.append(grandchild)
    
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
