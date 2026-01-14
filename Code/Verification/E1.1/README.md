# Experiment E1.1 — Expert activation distribution (routing arrival process)

This folder contains the **executable pipeline** for Experiment 1.1 described in:
- [Documentation/Experiment Documentations/Verification Experiments/实验 1.1：多数据集观察 expert activation 分布（“arrival process 长什么样”）.md](../../../Documentation/Experiment%20Documentations/Verification%20Experiments/%E5%AE%9E%E9%AA%8C%201.1%EF%BC%9A%E5%A4%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E8%A7%82%E5%AF%9F%20expert%20activation%20%E5%88%86%E5%B8%83%EF%BC%88%E2%80%9Carrival%20process%20%E9%95%BF%E4%BB%80%E4%B9%88%E6%A0%B7%E2%80%9D%EF%BC%89.md)

## Environment

- vLLM is expected to be importable from the **conda env** `moe`.
- vLLM source is expected at `~/sthmax/git/vllm`.
- This repo uses a workspace-local symlink at `external/vllm` pointing to that path (so you can patch vLLM for instrumentation, while still keeping vLLM as its own repo).

### Stability notes (TP=4 on A100)

If you see errors like:
- `custom_all_reduce.cuh:... 'invalid argument'`
- `cudaErrorStreamCaptureInvalidated` during engine warmup

Then you want to disable CUDA graph capture and vLLM's custom all-reduce. The smoke config does this via:
- `model.enforce_eager: true`
- `model.disable_custom_all_reduce: true`

## What this experiment produces

For each domain (dataset), we write a merged routing stats file:

- `outputs/runs/<run_id>/<domain>/routing_stats_merged.npz`

Schema highlights:
- `cnt_topk[L,E]`: token counts for **non-shared** experts (top-k membership)
- `cnt_top1[L,E]`: token counts for **non-shared** experts (top-1)
- `cnt_shared[L]`: token counts for the **shared expert** (counted separately; may be absent for models without shared experts)
- `tokens_processed[L]`: number of tokens routed at that MoE layer
- `layer_names[L]`: the vLLM layer identifiers (strings)

Phase-split (new; written into the same `.npz` when available):
- `cnt_topk_prefill[L,E]`, `cnt_top1_prefill[L,E]`, `tokens_prefill[L]`
- `cnt_topk_decode[L,E]`, `cnt_top1_decode[L,E]`, `tokens_decode[L]`
- `cnt_shared_prefill[L]`, `cnt_shared_decode[L]` (optional)

## Quickstart (2-domain smoke)

From repo root:

```bash
conda activate moe
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

python src/collect_routing_stats.py \
  --config configs/e1_1_smoke.json \
  --domain wikitext

python src/collect_routing_stats.py \
  --config configs/e1_1_smoke.json \
  --domain humaneval
```

Then merge partial files (written by vLLM workers) into one merged `.npz` per domain:

```bash
python src/merge_partials.py \
  --run_dir outputs/runs/<run_id> \
  --domain wikitext

python src/merge_partials.py \
  --run_dir outputs/runs/<run_id> \
  --domain humaneval
```

## Sanity checks

For each layer `l`:
- `sum_e cnt_topk[l,e] == tokens_processed[l] * k_experts`
- `sum_e cnt_top1[l,e] == tokens_processed[l]`
- `cnt_shared[l] == tokens_processed[l]` (if shared expert is always-on in that layer)

## Official runs (recommended)

Use the general runner script (supports clean run dir + auto run_id):

```bash
conda activate moe
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Clean, timestamped run (no collisions with old debug runs)
RUN_ID=$(date +%Y-%m-%d_%H%M%S)_official \
  CLEAN_RUN_DIR=1 \
  bash Code/Verification/E1.1/scripts/run_from_config.sh \
  Code/Verification/E1.1/configs/e1_1_official.json
```

Notes:
- The official config is [Code/Verification/E1.1/configs/e1_1_official.json](configs/e1_1_official.json). Adjust `data.n_examples`, `data.max_prompt_tokens`, and add more domains as needed.
- The runner will: collect -> merge -> summarize -> plot (per domain) -> analyze (cross-domain).

### Large datasets (streaming)

For very large HF datasets (e.g., `mc4`), set in the domain's `hf_dataset`:
- `streaming: true` (avoid full download)
- `shuffle: true/false`
- `shuffle_buffer: 10000` (larger buffer = better shuffle, more RAM)

## Summaries, analysis, plots

Per domain (under `outputs/runs/<run_id>/<domain>/`):
- `routing_stats_merged.npz`: merged counters
- `routing_summary.json`: token-weighted metrics + per-layer metrics (total)
- `routing_summary_prefill.json`: same metrics for prefill (if phase counters exist; otherwise falls back to total)
- `routing_summary_decode.json`: same metrics for decode (may be empty when `max_tokens=1`)
- `routing_summary_layers.csv`: per-layer table (total)
- `routing_summary_layers_prefill.csv`: per-layer table (prefill)
- `routing_summary_layers_decode.csv`: per-layer table (decode)
- `routing_analysis.json`: detailed per-layer top experts + outlier flags (total/prefill/decode)
- `plots/`:
  - `entropy_by_layer.png` (+ `_prefill` / `_decode` variants)
  - `max_share_by_layer.png` (+ `_prefill` / `_decode` variants)
  - `shared_frac_by_layer.png` (+ `_prefill` / `_decode` variants)
  - `heatmap_top50_experts_top1.png` (+ `_prefill` / `_decode` variants)
  - `heatmap_top50_experts_topk.png` (+ `_prefill` / `_decode` variants)

Per run (under `outputs/runs/<run_id>/`):
- `run_analysis.json`: pairwise cross-domain JS divergence across layers (total, and prefill/decode when available)

Dependencies:
- Plotting needs `matplotlib` in env `moe`: `pip install matplotlib`

# Commands:
RUN_ID=$(date +%Y-%m-%d_%H%M%S)_official CLEAN_RUN_DIR=1 Code/Verification/E1.1/scripts/run_from_config.sh Code/Verification/E1.1/configs/e1_1_official.json
