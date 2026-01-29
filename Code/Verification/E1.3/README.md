# Experiment E1.3 — PriceMoE MVP & Variable Top-k (E1.2 Compatible)

This folder contains the **executable pipeline** for:
- **Experiment 1.3**: PriceMoE MVP (Queue-aware routing with congestion penalty)
- **Experiment 1.2**: Variable top-k quality-sparsity trade-off (as a subset)

See the design document:
- [Documentation/Experiment Documentations/Verification Experiments/实验 1.3：PriceMoE MVP 实现与验证（Queue-aware Routing Policy）.md](../../../Documentation/Experiment%20Documentations/Verification%20Experiments/实验%201.3：PriceMoE%20MVP%20实现与验证（Queue-aware%20Routing%20Policy）.md)

## Environment

- vLLM is expected to be importable from the **conda env** `moe`.
- vLLM source is expected at `~/sthmax/git/vllm` (symlinked at `external/vllm`).
- This experiment requires the **PriceMoE router patch** to be applied to vLLM.

## Experiment Modes

This experiment supports three routing modes, configured via `routing.mode`:

| Mode | Description | E1.x |
|------|-------------|------|
| `baseline` | Original vLLM top-k routing | - |
| `variable_k` | Change top-k value (quality-sparsity curve) | E1.2 |
| `price_moe` | PriceMoE with congestion penalty α·q_e | E1.3 |

## Config Structure

```json
{
  "routing": {
    "mode": "price_moe",           // "baseline" | "variable_k" | "price_moe"
    "top_k": 8,                    // Override top-k (for variable_k mode)
    "alpha": 0.1,                  // PriceMoE congestion penalty
    "ema_decay": 0.9,              // EMA smoothing factor
    "use_original_weights": true   // Use original logits for weight normalization
  },
  "evaluation": {
    "compute_ppl": true,           // Calculate perplexity
    "collect_routing_stats": true, // Collect expert load distribution
    "benchmark_throughput": false  // Run throughput benchmark
  }
}
```

## What this experiment produces

For each run, we write:

- `outputs/runs/<run_id>/<domain>/routing_stats_merged.npz` - Expert load distribution
- `outputs/runs/<run_id>/<domain>/ppl_results.json` - Perplexity results
- `outputs/runs/<run_id>/<domain>/meta.json` - Run metadata
- `outputs/runs/<run_id>/throughput_results.json` - Throughput benchmark (if enabled)

### Routing Stats Schema

- `cnt_topk[L,E]`: token counts per expert (top-k membership)
- `cnt_top1[L,E]`: token counts per expert (top-1)
- `expert_load_ema[L,E]`: PriceMoE's EMA expert load (if mode=price_moe)
- `load_imbalance[L]`: max/mean load ratio per layer

## Quickstart

### E1.2: Variable Top-k Experiment

```bash
conda activate moe
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Sweep k values
for K in 1 2 4 8; do
    python src/run_experiment.py \
        --config configs/e1_2_variable_k.json \
        --domain wikitext \
        --routing.top_k $K \
        --run_id "e1_2_k${K}"
done
```

### E1.3: PriceMoE Experiment

```bash
conda activate moe
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Sweep alpha values
for ALPHA in 0.01 0.05 0.1 0.2 0.5 1.0; do
    python src/run_experiment.py \
        --config configs/e1_3_price_moe.json \
        --domain wikitext \
        --routing.alpha $ALPHA \
        --run_id "e1_3_alpha${ALPHA}"
done
```

### Baseline (for comparison)

```bash
python src/run_experiment.py \
    --config configs/e1_3_baseline.json \
    --domain wikitext \
    --run_id "e1_3_baseline"
```

## Analysis

After running experiments, analyze results:

```bash
# Compare load imbalance across configs
python src/analyze_load_balance.py \
    --run_dirs outputs/runs/e1_3_* \
    --output outputs/analysis/load_balance_comparison.png

# Plot Pareto curve (throughput vs PPL)
python src/plot_pareto.py \
    --run_dirs outputs/runs/e1_3_* \
    --output outputs/analysis/pareto_curve.png
```

## Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Load Imbalance** | max(q_e) / mean(q_e) | ↓ with PriceMoE |
| **Throughput** | tokens/s | ↑ 5-20% (high concurrency) |
| **PPL** | exp(mean(-log p)) | ↑ < 5% acceptable |
| **Gini Coefficient** | Expert load inequality | ↓ with PriceMoE |

## Directory Structure

```
E1.3/
├── README.md
├── configs/
│   ├── e1_2_variable_k.json      # E1.2: Variable top-k
│   ├── e1_3_baseline.json        # Baseline comparison
│   ├── e1_3_price_moe.json       # PriceMoE MVP
│   └── e1_3_smoke.json           # Quick sanity check
├── outputs/
│   └── runs/
│       └── <run_id>/
│           └── <domain>/
├── scripts/
│   ├── run_sweep.sh              # Parameter sweep
│   └── run_all_experiments.sh    # Full experiment suite
└── src/
    ├── __init__.py
    ├── run_experiment.py         # Main entry point
    ├── compute_ppl.py            # PPL calculation
    ├── analyze_load_balance.py   # Load imbalance analysis
    ├── plot_pareto.py            # Pareto curve plotting
    └── utils.py                  # Shared utilities
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PRICE_MOE_ENABLED` | `0` | Enable PriceMoE router |
| `VLLM_PRICE_MOE_ALPHA` | `0.1` | Congestion penalty coefficient |
| `VLLM_PRICE_MOE_EMA_DECAY` | `0.9` | EMA decay rate |
| `VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS` | `1` | Use original logits for weights |
| `VLLM_MOE_STATS_DIR` | - | Directory for routing stats |
| `VLLM_MOE_VARIABLE_K` | - | Override top-k value (for E1.2) |

## Dependencies

- vLLM with PriceMoE patch (see `external/vllm`)
- `datasets` (HuggingFace)
- `transformers` (HuggingFace)
- `numpy`, `matplotlib`, `scipy`

## Notes

1. **E1.2 is a subset of E1.3**: Use `routing.mode = "variable_k"` to run E1.2 experiments.
2. **PPL calculation**: Uses vLLM's `prompt_logprobs` feature.
3. **Throughput benchmark**: Requires separate server mode; see `scripts/run_throughput.sh`.
4. **Stability**: For TP=4, use `enforce_eager: true` to avoid CUDA graph issues.
