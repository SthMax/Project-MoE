#!/usr/bin/env python3
"""
E1.3: PriceMoE MVP Experiment Runner

This script runs experiments with different routing modes:
  - baseline: Original top-k routing
  - variable_k: Variable top-k (E1.2 compatibility)
  - price_moe: PriceMoE with congestion penalty

Usage:
    python -m src.run_experiment --config configs/e1_3_smoke.json
    python -m src.run_experiment --config configs/e1_3_price_moe.json --routing.alpha 0.2
    python -m src.run_experiment --config configs/e1_2_variable_k.json --routing.top_k 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    load_json,
    save_json,
    ensure_dir,
    deep_update,
    parse_cli_overrides,
    setup_routing_env,
    cleanup_routing_env,
    load_domain_texts,
    compute_ppl_from_outputs,
    compute_load_imbalance,
)


def _slugify(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ["-", "_"]) else "-" for ch in s.lower()).strip("-")


def _format_float_for_name(x: Any) -> str:
    if x is None:
        return "na"
    try:
        xf = float(x)
    except Exception:
        return _slugify(str(x))
    if xf == 0.0:
        return "0.00"
    return f"{xf:.6f}".rstrip("0").rstrip(".")


def _infer_model_short(model_id: str) -> str:
    model_short = (model_id or "unknown").split("/")[-1].lower()
    for suffix in ["-instruct", "-fp8", "-a3b", "-2507"]:
        model_short = model_short.replace(suffix, "")
    model_short = model_short.replace("--", "-").strip("-")
    return _slugify(model_short)


def _infer_dataset_label(config: dict) -> str:
    run_meta = config.get("run", {}) or {}
    if isinstance(run_meta, dict):
        for key in ["dataset", "dataset_name", "workload"]:
            v = run_meta.get(key)
            if v:
                return _slugify(str(v))

    # Throughput benchmark defaults to synthetic/random prompts unless specified by scripts.
    if "throughput" in config:
        return "synthetic"

    domains = config.get("domains", {}) or {}
    if isinstance(domains, dict):
        keys = list(domains.keys())
        if len(keys) == 1:
            return _slugify(keys[0])
        if len(keys) > 1:
            return "multi"
    return "unknown"


def _infer_load_profile(config: dict) -> str:
    run_meta = config.get("run", {}) or {}
    if isinstance(run_meta, dict):
        for key in ["load", "load_profile", "profile"]:
            v = run_meta.get(key)
            if v:
                return _slugify(str(v))

    # Heuristic for throughput configs
    thr = config.get("throughput", {}) or {}
    model = config.get("model", {}) or {}
    num_prompts = int(thr.get("num_prompts", 0) or 0)
    max_num_seqs = int(model.get("max_num_seqs", 0) or 0)
    max_model_len = int(model.get("max_model_len", 0) or 0)
    if num_prompts >= 2000 or max_num_seqs >= 128 or max_model_len >= 8192:
        return "heavy"
    return "normal"


def _infer_ep_mode(config: dict) -> str:
    model = config.get("model", {}) or {}
    return "ep" if bool(model.get("enable_expert_parallel", False)) else "noep"


def get_routing_stats_collector():
    """
    Get the routing stats collector from vLLM.
    
    This assumes moe_stats.py exists in vllm (from E1.1).
    """
    try:
        from vllm import moe_stats
        return moe_stats
    except ImportError:
        print("[E1.3] Warning: moe_stats not available, routing stats disabled")
        return None


def get_price_moe_router():
    """
    Get the PriceMoE router module.
    """
    try:
        from vllm.model_executor.layers.fused_moe import price_moe_router
        return price_moe_router
    except ImportError:
        print("[E1.3] Warning: price_moe_router not available")
        return None


def run_single_domain(
    llm,
    config: dict,
    domain: str,
    output_dir: Path,
) -> dict:
    """
    Run experiment on a single domain.
    
    Returns:
        Dictionary with results including PPL and routing stats.
    """
    from vllm import SamplingParams
    
    print(f"\n[E1.3] Processing domain: {domain}")
    
    # Load texts
    texts = load_domain_texts(config, domain)
    print(f"[E1.3] Loaded {len(texts)} examples")
    
    # Setup sampling params
    model_cfg = config.get("model", {})
    max_tokens = model_cfg.get("max_tokens", 1)  # For PPL, we just do forward pass
    
    # Note: prompt_logprobs=1 is extremely expensive (stores vocab_size logits per token)
    # For smoke tests, we skip PPL and just test routing functionality
    eval_cfg = config.get("evaluation", {})
    compute_ppl = eval_cfg.get("compute_ppl", False)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy for reproducibility
        prompt_logprobs=1 if compute_ppl else None,  # Only enable if PPL needed
    )
    
    # Clear routing stats before run
    moe_stats = get_routing_stats_collector()
    if moe_stats is not None:
        moe_stats.clear()
    
    # Clear PriceMoE router state
    price_moe = get_price_moe_router()
    if price_moe is not None:
        price_moe.reset_all_routers()
    
    # Run generation
    start_time = time.time()
    outputs = llm.generate(texts, sampling_params)
    end_time = time.time()
    
    elapsed = end_time - start_time
    throughput = len(texts) / elapsed if elapsed > 0 else 0
    
    print(f"[E1.3] Generation completed in {elapsed:.2f}s ({throughput:.1f} examples/s)")
    
    # Compute PPL
    ppl_result = compute_ppl_from_outputs(outputs)
    print(f"[E1.3] PPL: {ppl_result['ppl']:.4f} (over {ppl_result['num_tokens']} tokens)")
    
    # Collect routing stats
    routing_stats = {}
    if moe_stats is not None:
        try:
            stats = moe_stats.get_stats()
            routing_stats = stats
            
            # Compute load imbalance
            if "expert_counts" in stats:
                load_metrics = compute_load_imbalance(stats["expert_counts"])
                routing_stats["load_imbalance"] = load_metrics
                print(f"[E1.3] Load imbalance (max/mean): {load_metrics['max_to_mean']:.3f}")
        except Exception as e:
            print(f"[E1.3] Warning: failed to get routing stats: {e}")
    
    # Collect PriceMoE specific stats
    price_moe_stats = {}
    if price_moe is not None:
        try:
            # get_all_router_stats() returns a list of stats dicts
            all_stats = price_moe.get_all_router_stats()
            for stats in all_stats:
                layer_name = stats.get("layer_name", "unknown")
                price_moe_stats[layer_name] = stats
        except Exception as e:
            print(f"[E1.3] Warning: failed to get PriceMoE stats: {e}")
    
    result = {
        "domain": domain,
        "n_examples": len(texts),
        "elapsed_seconds": elapsed,
        "throughput": throughput,
        "ppl": ppl_result,
        "routing_stats": routing_stats,
        "price_moe_stats": price_moe_stats,
    }
    
    # Save domain results
    domain_output = output_dir / f"domain_{domain}.json"
    save_json(result, domain_output)
    print(f"[E1.3] Saved domain results to {domain_output}")
    
    return result


def run_experiment(config: dict, output_dir: Path) -> dict:
    """
    Run full experiment with given configuration.
    """
    from vllm import LLM
    
    print("[E1.3] === Starting Experiment ===")
    routing_cfg = config.get("routing", {})
    routing_mode = routing_cfg.get("mode", "baseline")
    print(f"[E1.3] Routing mode: {routing_mode}")
    if routing_mode == "price_moe":
        print(
            f"[E1.3] PriceMoE params: alpha={routing_cfg.get('alpha')}, "
            f"ema_decay={routing_cfg.get('ema_decay')}, "
            f"use_original_weights={routing_cfg.get('use_original_weights')}"
        )
    elif routing_mode == "variable_k":
        print(f"[E1.3] Variable-k params: top_k={routing_cfg.get('top_k')}")
    print(f"[E1.3] Output directory: {output_dir}")
    
    # Setup environment variables for routing
    set_vars = setup_routing_env(routing_cfg)
    
    # Enable moe_stats collection by setting VLLM_MOE_STATS_DIR
    # This allows record_topk() to collect stats in-memory
    os.environ["VLLM_MOE_STATS_DIR"] = str(output_dir / "moe_stats")
    set_vars["VLLM_MOE_STATS_DIR"] = str(output_dir / "moe_stats")
    
    print(f"[E1.3] Environment variables set: {set_vars}")
    
    try:
        # Initialize LLM
        model_cfg = config["model"]
        print(f"[E1.3] Loading model: {model_cfg['model_id']}")
        
        llm = LLM(
            model=model_cfg["model_id"],
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
            max_model_len=model_cfg.get("max_model_len"),
            max_num_seqs=model_cfg.get("max_num_seqs", 256),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
            enforce_eager=model_cfg.get("enforce_eager", False),
        )
        
        # Run on each domain
        domains = list(config.get("domains", {}).keys())
        if not domains:
            raise ValueError("No domains specified in config")
        
        print(f"[E1.3] Domains to process: {domains}")
        
        all_results = []
        for domain in domains:
            result = run_single_domain(llm, config, domain, output_dir)
            all_results.append(result)
        
        # Aggregate results
        total_examples = sum(r["n_examples"] for r in all_results)
        total_elapsed = sum(r["elapsed_seconds"] for r in all_results)
        
        # Weighted average PPL
        weighted_ppl = 0.0
        total_tokens = 0
        for r in all_results:
            n_tokens = r["ppl"]["num_tokens"]
            weighted_ppl += r["ppl"]["ppl"] * n_tokens
            total_tokens += n_tokens
        avg_ppl = weighted_ppl / total_tokens if total_tokens > 0 else float("inf")
        
        summary = {
            "config": config,
            "routing_mode": routing_cfg.get("mode", "baseline"),
            "routing_params": routing_cfg,
            "total_examples": total_examples,
            "total_elapsed_seconds": total_elapsed,
            "overall_throughput": total_examples / total_elapsed if total_elapsed > 0 else 0,
            "average_ppl": avg_ppl,
            "total_tokens": total_tokens,
            "domains": all_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save summary
        summary_path = output_dir / "summary.json"
        save_json(summary, summary_path)
        print(f"\n[E1.3] === Experiment Complete ===")
        print(f"[E1.3] Average PPL: {avg_ppl:.4f}")
        print(f"[E1.3] Throughput: {summary['overall_throughput']:.2f} examples/s")
        print(f"[E1.3] Summary saved to {summary_path}")
        
        return summary
    
    finally:
        # Cleanup environment variables
        cleanup_routing_env(set_vars)


def run_throughput_benchmark(config: dict, output_dir: Path) -> dict:
    """
    Run throughput benchmark using vLLM's built-in benchmark.
    
    This is more suitable for measuring actual throughput gains from PriceMoE
    in EP mode or high-concurrency scenarios.
    """
    import subprocess
    import shutil
    
    print("[E1.3] === Starting Throughput Benchmark ===")
    
    # Setup environment variables for routing
    routing_cfg = config.get("routing", {})
    routing_mode = routing_cfg.get("mode", "baseline")
    set_vars = setup_routing_env(routing_cfg)
    
    # Enable moe_stats collection
    os.environ["VLLM_MOE_STATS_DIR"] = str(output_dir / "moe_stats")
    set_vars["VLLM_MOE_STATS_DIR"] = str(output_dir / "moe_stats")
    (output_dir / "moe_stats").mkdir(parents=True, exist_ok=True)
    
    print(f"[E1.3] Routing mode: {routing_mode}")
    if routing_mode == "price_moe":
        print(
            f"[E1.3] PriceMoE params: alpha={routing_cfg.get('alpha')}, "
            f"ema_decay={routing_cfg.get('ema_decay')}, "
            f"use_original_weights={routing_cfg.get('use_original_weights')}"
        )
    print(f"[E1.3] Environment variables set: {set_vars}")
    
    try:
        # Build benchmark command
        model_cfg = config["model"]
        throughput_cfg = config.get("throughput", {})
        
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", model_cfg["model_id"],
            "--tensor-parallel-size", str(model_cfg.get("tensor_parallel_size", 1)),
            "--num-prompts", str(throughput_cfg.get("num_prompts", 500)),
            "--input-len", str(throughput_cfg.get("input_len", 256)),
            "--output-len", str(throughput_cfg.get("output_len", 128)),
            "--output-json", str(output_dir / "benchmark_results.json"),
        ]
        
        # Add optional flags
        if model_cfg.get("trust_remote_code", False):
            cmd.append("--trust-remote-code")
        
        if model_cfg.get("enforce_eager", False):
            cmd.append("--enforce-eager")
        
        if model_cfg.get("enable_expert_parallel", False):
            cmd.append("--enable-expert-parallel")
        
        max_model_len = model_cfg.get("max_model_len")
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
        
        print(f"[E1.3] Running: {' '.join(cmd)}")
        
        # Run benchmark
        log_file = output_dir / "benchmark.log"
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            f.write(result.stdout)
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"[E1.3] Warning: Benchmark exited with code {result.returncode}")
        
        # Load and return results
        results_file = output_dir / "benchmark_results.json"
        if results_file.exists():
            benchmark_results = load_json(results_file)
            
            # Create summary
            summary = {
                "config": config,
                "routing_mode": routing_cfg.get("mode", "baseline"),
                "routing_params": routing_cfg,
                "benchmark_results": benchmark_results,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Save summary
            save_json(summary, output_dir / "summary.json")
            
            print(f"\n[E1.3] === Benchmark Complete ===")
            print(f"[E1.3] Throughput: {benchmark_results.get('requests_per_second', 0):.2f} req/s")
            print(f"[E1.3] Tokens/s: {benchmark_results.get('tokens_per_second', 0):.2f}")
            print(f"[E1.3] Results saved to {output_dir}")
            
            return summary
        else:
            print(f"[E1.3] Error: No benchmark results found")
            return {}
    
    finally:
        cleanup_routing_env(set_vars)


def main():
    parser = argparse.ArgumentParser(
        description="E1.3: PriceMoE MVP Experiment Runner"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["ppl", "throughput", "both"],
        default="ppl",
        help="Experiment mode: 'ppl' for PPL evaluation (default), "
             "'throughput' for throughput benchmark, 'both' for both",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: auto-generated in outputs/runs/)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for output directory",
    )
    
    # Parse known args, rest are config overrides
    args, extra_args = parser.parse_known_args()
    
    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    
    config = load_json(config_path)
    
    # Apply CLI overrides
    if extra_args:
        overrides = parse_cli_overrides(extra_args)
        config = deep_update(config, overrides)
        print(f"[E1.3] Applied CLI overrides: {overrides}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = Path(__file__).resolve().parent.parent / "outputs" / "runs"

        now = datetime.now()
        date_tag = now.strftime("%y%m%d")
        time_tag = now.strftime("%H%M%S")

        routing_cfg = config.get("routing", {}) or {}
        routing_mode = routing_cfg.get("mode", "baseline")

        model_id = config.get("model", {}).get("model_id", "unknown")
        model_short = _infer_model_short(model_id)
        ep_mode = _infer_ep_mode(config)
        load_profile = _infer_load_profile(config)
        dataset = _infer_dataset_label(config)

        outer_dir_name = f"{date_tag}-{model_short}-{ep_mode}-{load_profile}-{dataset}"

        if args.run_name:
            run_dir_name = f"{date_tag}_{_slugify(args.run_name)}"
        else:
            run_dir_name = f"{date_tag}_{dataset}_{routing_mode}"
            if routing_mode == "price_moe":
                alpha = _format_float_for_name(routing_cfg.get("alpha"))
                ema = _format_float_for_name(routing_cfg.get("ema_decay"))
                run_dir_name = f"{run_dir_name}_alpha{alpha}_ema{ema}"
                update_interval = routing_cfg.get("update_interval", 1)
                if update_interval is not None and int(update_interval) != 1:
                    run_dir_name = f"{run_dir_name}_interval{update_interval}"
            elif routing_mode == "variable_k":
                k = routing_cfg.get("top_k", "default")
                run_dir_name = f"{run_dir_name}_k{_slugify(str(k))}"

        # Avoid collisions if user reruns multiple times in the same day
        candidate = base_dir / outer_dir_name / run_dir_name
        if candidate.exists():
            run_dir_name = f"{run_dir_name}_{time_tag}"

        output_dir = base_dir / outer_dir_name / run_dir_name
    
    output_dir = ensure_dir(output_dir)
    
    # Save effective config
    save_json(config, output_dir / "config.json")
    
    # Run experiment based on mode
    if args.mode == "ppl":
        run_experiment(config, output_dir)
    elif args.mode == "throughput":
        run_throughput_benchmark(config, output_dir)
    elif args.mode == "both":
        # Run both PPL and throughput
        ppl_dir = output_dir / "ppl"
        throughput_dir = output_dir / "throughput"
        ppl_dir.mkdir(parents=True, exist_ok=True)
        throughput_dir.mkdir(parents=True, exist_ok=True)
        
        print("[E1.3] Running PPL evaluation...")
        run_experiment(config, ppl_dir)
        
        print("\n[E1.3] Running throughput benchmark...")
        run_throughput_benchmark(config, throughput_dir)


if __name__ == "__main__":
    main()
