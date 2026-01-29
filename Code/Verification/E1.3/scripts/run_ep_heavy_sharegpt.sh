#!/bin/bash
# E1.3: Heavy Load EP Throughput Benchmark with ShareGPT Dataset
#
# Uses real ShareGPT conversations for realistic workload distribution
# This is more representative of actual LLM usage patterns
#
# Usage:
#   ./scripts/run_ep_heavy_sharegpt.sh baseline
#   ./scripts/run_ep_heavy_sharegpt.sh price_moe 0.005
#   ./scripts/run_ep_heavy_sharegpt.sh price_moe 0.005 0.9   # with EMA decay
#   ./scripts/run_ep_heavy_sharegpt.sh price_moe 0 0.9       # alpha=0 for overhead test
#
# EMA decay (ρ) controls memory length:
#   0.9 (default): smooth, 90% history retention
#   0.99: very long memory, slow adaptation
#   0.5: balanced, faster response to load changes
#   0.1: aggressive, mostly current batch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E13_DIR="$(dirname "$SCRIPT_DIR")"

# Parameters
MODE="${1:-baseline}"
ALPHA="${2:-0.005}"
EMA_DECAY="${3:-0.9}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Heavy load config with ShareGPT
NUM_PROMPTS=2000
MAX_NUM_SEQS=128
MAX_MODEL_LEN=8192

# HuggingFace ShareGPT dataset - vLLM supported dataset
# See vllm/benchmarks/datasets.py ConversationDataset.SUPPORTED_DATASET_PATHS
HF_DATASET="Aeala/ShareGPT_Vicuna_unfiltered"
HF_SPLIT="train"

# Model config
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SHORT="qwen3-30b-ep-heavy-sharegpt"
TP=4

# Output directory
OUTPUT_DIR="$E13_DIR/outputs/runs/$MODEL_SHORT"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "E1.3 Heavy Load EP Benchmark (ShareGPT)"
echo "=============================================="
echo "Mode: $MODE"
echo "Alpha: $ALPHA (if price_moe)"
echo "Dataset: $HF_DATASET (HuggingFace)"
echo "Num Prompts: $NUM_PROMPTS"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Expert Parallelism: TP=$TP → EP=$TP"
echo "=============================================="

# Set routing environment variables
if [ "$MODE" == "price_moe" ]; then
    export VLLM_PRICE_MOE_ENABLED="1"
    export VLLM_PRICE_MOE_ALPHA="$ALPHA"
    export VLLM_PRICE_MOE_EMA_DECAY="$EMA_DECAY"
    export VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS="1"
    # Include ema in run name only if non-default
    if [ "$EMA_DECAY" != "0.9" ]; then
        RUN_NAME="${TIMESTAMP}_sharegpt_price_moe_alpha${ALPHA}_ema${EMA_DECAY}"
    else
        RUN_NAME="${TIMESTAMP}_sharegpt_price_moe_alpha${ALPHA}_ema0.9"
    fi
else
    unset VLLM_PRICE_MOE_ENABLED
    unset VLLM_PRICE_MOE_ALPHA
    unset VLLM_PRICE_MOE_EMA_DECAY
    unset VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS
    RUN_NAME="${TIMESTAMP}_sharegpt_baseline"
fi

# Create run directory
mkdir -p "$OUTPUT_DIR/$RUN_NAME"

# Set up MOE stats collection
export VLLM_MOE_STATS_ENABLED="1"
export VLLM_MOE_STATS_DIR="$OUTPUT_DIR/$RUN_NAME/moe_stats"
mkdir -p "$VLLM_MOE_STATS_DIR"

echo ""
echo "Starting benchmark: $RUN_NAME"
echo "Output: $OUTPUT_DIR/$RUN_NAME"
echo ""

RESULT_FILE="$OUTPUT_DIR/$RUN_NAME/benchmark_results.json"

# Run vLLM throughput benchmark with HuggingFace ShareGPT dataset
# ConversationDataset requires --backend vllm-chat
python -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --enable-expert-parallel \
    --backend vllm-chat \
    --dataset-name hf \
    --dataset-path "$HF_DATASET" \
    --hf-split "$HF_SPLIT" \
    --num-prompts $NUM_PROMPTS \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --enforce-eager \
    --output-json "$RESULT_FILE"

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Results saved to: $RESULT_FILE"
echo "MOE stats saved to: $VLLM_MOE_STATS_DIR"
echo ""

# Print summary
if [ -f "$RESULT_FILE" ]; then
    echo "Quick Summary:"
    python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
print(f'  Elapsed time: {r.get(\"elapsed_time\", 0):.2f} s')
print(f'  Throughput: {r.get(\"requests_per_second\", 0):.2f} req/s')
print(f'  Tokens/s: {r.get(\"tokens_per_second\", 0):.2f}')
print(f'  Total tokens: {r.get(\"total_num_tokens\", 0)}')
"
fi

# Save config
cat > "$OUTPUT_DIR/$RUN_NAME/config.json" << EOF
{
    "mode": "$MODE",
    "alpha": $ALPHA,
    "ema_decay": $EMA_DECAY,
    "model": "$MODEL",
    "tensor_parallel_size": $TP,
    "enable_expert_parallel": true,
    "dataset_name": "hf",
    "dataset_path": "$HF_DATASET",
    "hf_split": "$HF_SPLIT",
    "num_prompts": $NUM_PROMPTS,
    "max_num_seqs": $MAX_NUM_SEQS,
    "max_model_len": $MAX_MODEL_LEN,
    "timestamp": "$TIMESTAMP",
    "load_profile": "heavy-sharegpt"
}
EOF

echo ""
echo "Config saved. Run analysis with:"
echo "  python -m src.analyze_load_balance $OUTPUT_DIR --compare"
