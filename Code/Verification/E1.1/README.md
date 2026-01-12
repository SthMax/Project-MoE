# Experiment E1.1 — Expert activation distribution (routing arrival process)

This folder contains the **executable pipeline** for Experiment 1.1 described in:
- [Documentation/Experiment Documentations/Verification Experiments/实验 1.1：多数据集观察 expert activation 分布（“arrival process 长什么样”）.md](../../../Documentation/Experiment%20Documentations/Verification%20Experiments/%E5%AE%9E%E9%AA%8C%201.1%EF%BC%9A%E5%A4%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E8%A7%82%E5%AF%9F%20expert%20activation%20%E5%88%86%E5%B8%83%EF%BC%88%E2%80%9Carrival%20process%20%E9%95%BF%E4%BB%80%E4%B9%88%E6%A0%B7%E2%80%9D%EF%BC%89.md)

## Environment

- vLLM is expected to be importable from the **conda env** `moe`.
- You indicated vLLM source is at `~/sthmax/git/vllm`.
- This repo creates a workspace-local symlink at `external/vllm` pointing to that path, so we can patch vLLM for instrumentation.

## What this experiment produces

For each domain (dataset), we write a merged routing stats file:

- `outputs/runs/<run_id>/<domain>/routing_stats_merged.npz`

Schema highlights:
- `cnt_topk[L,E]`: token counts for **non-shared** experts (top-k membership)
- `cnt_top1[L,E]`: token counts for **non-shared** experts (top-1)
- `cnt_shared[L]`: token counts for the **shared expert** (counted separately)
- `tokens_processed[L]`: number of tokens routed at that MoE layer
- `layer_names[L]`: the vLLM layer identifiers (strings)

## Quickstart (2-domain smoke)

From repo root:

```bash
conda activate moe
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

python Code/Verification/E1.1/src/collect_routing_stats.py \
  --config Code/Verification/E1.1/configs/e1_1_smoke.json \
  --domain wikitext

python Code/Verification/E1.1/src/collect_routing_stats.py \
  --config Code/Verification/E1.1/configs/e1_1_smoke.json \
  --domain humaneval
```

Then merge partial files (written by vLLM workers) into one merged `.npz` per domain:

```bash
python Code/Verification/E1.1/src/merge_partials.py \
  --run_dir Code/Verification/E1.1/outputs/runs/<run_id> \
  --domain wikitext

python Code/Verification/E1.1/src/merge_partials.py \
  --run_dir Code/Verification/E1.1/outputs/runs/<run_id> \
  --domain humaneval
```

## Sanity checks

For each layer `l`:
- `sum_e cnt_topk[l,e] == tokens_processed[l] * k_experts`
- `sum_e cnt_top1[l,e] == tokens_processed[l]`
- `cnt_shared[l] == tokens_processed[l]` (if shared expert is always-on in that layer)
