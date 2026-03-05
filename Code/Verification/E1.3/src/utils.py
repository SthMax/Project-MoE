"""
Shared utilities for E1.3 experiments.
"""

from __future__ import annotations

import json
import math
import os
import importlib
from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_json(path: str | Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_import(name: str):
    """Import module with helpful error message."""
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{name}'. Install it in conda env 'moe'.\n"
            f"Example: `pip install {name}`\n"
            f"Original error: {e}"
        ) from e


def deep_update(base: dict, updates: dict) -> dict:
    """Deep merge updates into base dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def parse_cli_overrides(args: list[str]) -> dict:
    """
    Parse CLI arguments like --routing.alpha=0.1 into nested dict.
    
    Example:
        parse_cli_overrides(["--routing.alpha", "0.1", "--model.max_num_seqs", "32"])
        -> {"routing": {"alpha": 0.1}, "model": {"max_num_seqs": 32}}
    """
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]
            # Handle --key=value format
            if "=" in key:
                key, value = key.split("=", 1)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 1
            else:
                value = "true"
            
            # Parse value type
            value = _parse_value(value)
            
            # Convert dotted key to nested dict
            parts = key.split(".")
            current = overrides
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        i += 1
    
    return overrides


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ============================================================================
# Load Imbalance Metrics
# ============================================================================

def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient for load distribution.
    
    0 = perfect equality, 1 = maximum inequality
    """
    values = np.sort(values.flatten())
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def compute_load_imbalance(expert_load: np.ndarray) -> dict:
    """
    Compute load imbalance metrics for expert distribution.
    
    Args:
        expert_load: Array of expert loads, shape (E,) or (L, E)
    
    Returns:
        Dictionary with metrics:
            - max_to_mean: max load / mean load
            - cv: coefficient of variation (std/mean)
            - gini: Gini coefficient
            - entropy: normalized entropy
    """
    expert_load = np.asarray(expert_load).flatten()
    
    if len(expert_load) == 0 or expert_load.sum() == 0:
        return {
            "max_to_mean": 0.0,
            "cv": 0.0,
            "gini": 0.0,
            "entropy": 0.0,
        }
    
    mean_load = expert_load.mean()
    max_load = expert_load.max()
    std_load = expert_load.std()
    
    # Normalize for entropy calculation
    p = expert_load / expert_load.sum()
    p = p[p > 0]  # Avoid log(0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(expert_load))  # Uniform distribution
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        "max_to_mean": float(max_load / mean_load) if mean_load > 0 else 0.0,
        "cv": float(std_load / mean_load) if mean_load > 0 else 0.0,
        "gini": float(gini_coefficient(expert_load)),
        "entropy": float(normalized_entropy),
        "max_expert": int(expert_load.argmax()),
        "min_expert": int(expert_load.argmin()),
    }


# ============================================================================
# PPL Calculation
# ============================================================================

def compute_ppl_from_logprobs(log_probs: list[float]) -> float:
    """
    Compute perplexity from log probabilities.
    
    PPL = exp(-mean(log_probs))
    """
    if not log_probs:
        return float("inf")
    return math.exp(-sum(log_probs) / len(log_probs))


def compute_ppl_from_outputs(outputs: list, skip_first: bool = True) -> dict:
    """
    Compute perplexity from vLLM outputs with prompt_logprobs.
    
    Args:
        outputs: List of vLLM RequestOutput objects
        skip_first: Whether to skip the first token (often has no logprob)
    
    Returns:
        Dictionary with PPL statistics
    """
    all_log_probs = []
    
    for output in outputs:
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None:
            continue
        
        for i, token_logprobs in enumerate(prompt_logprobs):
            if token_logprobs is None:
                continue
            if skip_first and i == 0:
                continue
            
            # Get the log prob of the actual token
            for token_id, logprob_obj in token_logprobs.items():
                if hasattr(logprob_obj, "logprob"):
                    all_log_probs.append(logprob_obj.logprob)
                else:
                    all_log_probs.append(logprob_obj)
                break  # Only take the first (actual) token
    
    ppl = compute_ppl_from_logprobs(all_log_probs)
    
    return {
        "ppl": ppl,
        "num_tokens": len(all_log_probs),
        "mean_log_prob": sum(all_log_probs) / len(all_log_probs) if all_log_probs else 0.0,
    }


# ============================================================================
# Environment Variable Helpers
# ============================================================================

def setup_routing_env(routing_config: dict) -> dict:
    """
    Set environment variables based on routing configuration.
    
    Returns dict of variables that were set (for cleanup).
    """
    set_vars = {}
    mode = routing_config.get("mode", "baseline")
    
    if mode == "price_moe":
        os.environ["VLLM_PRICE_MOE_ENABLED"] = "1"
        set_vars["VLLM_PRICE_MOE_ENABLED"] = "1"
        
        alpha = routing_config.get("alpha")
        if alpha is not None:
            os.environ["VLLM_PRICE_MOE_ALPHA"] = str(alpha)
            set_vars["VLLM_PRICE_MOE_ALPHA"] = str(alpha)
        
        ema_decay = routing_config.get("ema_decay")
        if ema_decay is not None:
            os.environ["VLLM_PRICE_MOE_EMA_DECAY"] = str(ema_decay)
            set_vars["VLLM_PRICE_MOE_EMA_DECAY"] = str(ema_decay)
        
        use_original_weights = routing_config.get("use_original_weights")
        if use_original_weights is not None:
            os.environ["VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS"] = (
                "1" if use_original_weights else "0"
            )
            set_vars["VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS"] = (
                "1" if use_original_weights else "0"
            )
        
        update_interval = routing_config.get("update_interval")
        if update_interval is not None:
            os.environ["VLLM_PRICE_MOE_UPDATE_INTERVAL"] = str(update_interval)
            set_vars["VLLM_PRICE_MOE_UPDATE_INTERVAL"] = str(update_interval)
    
    elif mode == "variable_k":
        top_k = routing_config.get("top_k")
        if top_k is not None:
            os.environ["VLLM_MOE_VARIABLE_K"] = str(top_k)
            set_vars["VLLM_MOE_VARIABLE_K"] = str(top_k)
    
    return set_vars


def cleanup_routing_env(set_vars: dict) -> None:
    """Clean up environment variables set by setup_routing_env."""
    for key in set_vars:
        if key in os.environ:
            del os.environ[key]


# ============================================================================
# Data Loading (reused from E1.1)
# ============================================================================

def load_domain_texts(config: dict, domain: str) -> list[str]:
    """
    Load text prompts from HuggingFace dataset for a given domain.
    
    This is adapted from E1.1's collect_routing_stats.py.
    """
    domain_cfg = config["domains"].get(domain)
    if domain_cfg is None:
        raise ValueError(
            f"Unknown domain: {domain}. Available: {list(config['domains'].keys())}"
        )
    
    datasets = maybe_import("datasets")
    transformers = maybe_import("transformers")
    
    model_id = config["model"]["model_id"]
    trust_remote_code = bool(config["model"].get("trust_remote_code", False))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    
    data_cfg = config["data"]
    seed = int(data_cfg["seed"])
    n_examples_default = int(data_cfg["n_examples"])
    n_examples = int(domain_cfg.get("n_examples", n_examples_default))
    max_prompt_tokens = int(data_cfg["max_prompt_tokens"])
    
    allow_fewer_examples = bool(
        domain_cfg.get(
            "allow_fewer_examples",
            data_cfg.get("allow_fewer_examples", True)
        )
    )
    
    hf = domain_cfg.get("hf_dataset")
    if hf is None:
        raise ValueError(f"Domain {domain} missing hf_dataset config")
    
    streaming = bool(hf.get("streaming", False))
    shuffle = bool(hf.get("shuffle", True))
    shuffle_buffer = int(hf.get("shuffle_buffer", 10_000))
    
    ds = datasets.load_dataset(
        hf["path"],
        hf.get("name"),
        split=hf.get("split", "train"),
        streaming=streaming,
    )
    
    text_field = hf["text_field"]
    
    # Shuffle deterministically
    if shuffle:
        try:
            if streaming:
                ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            else:
                ds = ds.shuffle(seed=seed)
        except Exception as e:
            print(f"[E1.3] Warning: failed to shuffle domain={domain}: {e}")
    
    texts: list[str] = []
    for ex in ds:
        text = ex.get(text_field)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        
        # Token-level truncation
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        if len(ids) == 0:
            continue
        if len(ids) > max_prompt_tokens:
            ids = ids[:max_prompt_tokens]
            text = tokenizer.decode(ids, skip_special_tokens=True)
        
        texts.append(text)
        if len(texts) >= n_examples:
            break
    
    if len(texts) < n_examples:
        if allow_fewer_examples and len(texts) > 0:
            print(
                f"[E1.3] Warning: only collected {len(texts)} examples for domain "
                f"{domain} (requested {n_examples}). Proceeding with fewer examples."
            )
        else:
            raise RuntimeError(
                f"Only collected {len(texts)} examples for domain {domain} "
                f"(requested {n_examples})."
            )
    
    return texts
