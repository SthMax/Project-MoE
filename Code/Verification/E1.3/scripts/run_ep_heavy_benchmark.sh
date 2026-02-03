#!/bin/bash
# E1.3: Heavy Load EP Throughput Benchmark
#
# High-load test to stress-test Expert Parallelism and load balancing
# Config: 2000 prompts, 512+256 tokens, max_num_seqs=128
#
# Usage:
#   ./scripts/run_ep_heavy_benchmark.sh baseline
#   ./scripts/run_ep_heavy_benchmark.sh price_moe 0.005
#   ./scripts/run_ep_heavy_benchmark.sh price_moe 0.005 0.9   # with EMA decay

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E13_DIR="$(dirname "$SCRIPT_DIR")"

# Parameters
MODE="${1:-baseline}"
ALPHA="${2:-0.005}"
EMA_DECAY="${3:-0.9}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Heavy load config
NUM_PROMPTS=2000
INPUT_LEN=512
OUTPUT_LEN=256
MAX_NUM_SEQS=128
MAX_MODEL_LEN=8192

# Model config
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SHORT="qwen3-30b-ep-heavy"
TP=4

# Output directory
OUTPUT_DIR="$E13_DIR/outputs/runs/$MODEL_SHORT"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "E1.3 Heavy Load EP Throughput Benchmark"
echo "=============================================="
echo "Mode: $MODE"
echo "Alpha: $ALPHA (if price_moe)"
echo "Num Prompts: $NUM_PROMPTS"
echo "Input/Output Len: $INPUT_LEN / $OUTPUT_LEN"
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
    RUN_NAME="${TIMESTAMP}_heavy_price_moe_alpha${ALPHA}_ema${EMA_DECAY}"
else
    unset VLLM_PRICE_MOE_ENABLED
    unset VLLM_PRICE_MOE_ALPHA
    unset VLLM_PRICE_MOE_EMA_DECAY
    unset VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS
    RUN_NAME="${TIMESTAMP}_heavy_baseline"
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

# Run vLLM throughput benchmark with EP enabled
python -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --enable-expert-parallel \
    --num-prompts $NUM_PROMPTS \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
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
    "num_prompts": $NUM_PROMPTS,
    "input_len": $INPUT_LEN,
    "output_len": $OUTPUT_LEN,
    "max_num_seqs": $MAX_NUM_SEQS,
    "max_model_len": $MAX_MODEL_LEN,
    "timestamp": "$TIMESTAMP",
    "load_profile": "heavy"
}
EOF

echo ""
echo "Config saved. Run analysis with:"
echo "  python -m src.analyze_load_balance $OUTPUT_DIR --compare"
