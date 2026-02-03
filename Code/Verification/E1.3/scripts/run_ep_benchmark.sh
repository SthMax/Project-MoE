#!/bin/bash
# E1.3: Expert Parallelism (EP) Throughput Benchmark
#
# This is the CORRECT way to demonstrate PriceMoE's throughput benefit!
# In EP mode, different experts are on different GPUs, so load balancing
# directly impacts throughput by reducing GPU idle time.
#
# Usage:
#   ./scripts/run_ep_benchmark.sh baseline
#   ./scripts/run_ep_benchmark.sh price_moe 0.005
#   ./scripts/run_ep_benchmark.sh price_moe 0.005 500 256 128 0.9  # with EMA decay

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E13_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_DIR="$E13_DIR/../../../external/vllm"

# Parameters
MODE="${1:-baseline}"
ALPHA="${2:-0.005}"
NUM_PROMPTS="${3:-500}"
INPUT_LEN="${4:-256}"
OUTPUT_LEN="${5:-128}"
EMA_DECAY="${6:-0.9}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Model config - using 30B for EP (less memory per expert)
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SHORT="qwen3-30b-ep"
TP=4  # This becomes EP size when enable_expert_parallel=True

# Output directory
OUTPUT_DIR="$E13_DIR/outputs/runs/$MODEL_SHORT"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "E1.3 Expert Parallelism Throughput Benchmark"
echo "=============================================="
echo "Mode: $MODE"
echo "Alpha: $ALPHA (if price_moe)"
echo "Num Prompts: $NUM_PROMPTS"
echo "Input/Output Len: $INPUT_LEN / $OUTPUT_LEN"
echo "Expert Parallelism: TP=$TP → EP=$TP"
echo "=============================================="

# Set routing environment variables
if [ "$MODE" == "price_moe" ]; then
    # Use VLLM_PRICE_MOE_* naming (consistent with utils.py)
    export VLLM_PRICE_MOE_ENABLED="1"
    export VLLM_PRICE_MOE_ALPHA="$ALPHA"
    export VLLM_PRICE_MOE_EMA_DECAY="$EMA_DECAY"
    export VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS="1"
    RUN_NAME="${TIMESTAMP}_ep_price_moe_alpha${ALPHA}_ema${EMA_DECAY}"

else
    unset VLLM_PRICE_MOE_ENABLED
    unset VLLM_PRICE_MOE_ALPHA
    unset VLLM_PRICE_MOE_EMA_DECAY
    unset VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS
    RUN_NAME="${TIMESTAMP}_ep_baseline"
fi

# Enable stats collection
export VLLM_MOE_STATS_DIR="$OUTPUT_DIR/$RUN_NAME/moe_stats"
mkdir -p "$VLLM_MOE_STATS_DIR"

RESULT_FILE="$OUTPUT_DIR/$RUN_NAME/benchmark_results.json"
LOG_FILE="$OUTPUT_DIR/$RUN_NAME/benchmark.log"
mkdir -p "$(dirname "$RESULT_FILE")"

echo ""
echo "[Step 1] Running vLLM throughput benchmark with Expert Parallelism..."
echo "=========================================="

# Use vLLM's CLI for throughput benchmark with EP enabled
python -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --enable-expert-parallel \
    --trust-remote-code \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS \
    --enforce-eager \
    --output-json "$RESULT_FILE" \
    2>&1 | tee "$LOG_FILE"

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
    "timestamp": "$TIMESTAMP"
}
EOF

echo ""
echo "Config saved to: $OUTPUT_DIR/$RUN_NAME/config.json"
