#!/bin/bash
# E1.3: Online Serving Benchmark
# Compare baseline vs PriceMoE under high concurrency
#
# This is the RIGHT way to demonstrate PriceMoE's value:
# - High concurrency requests compete for expert resources
# - Baseline: hot experts become bottlenecks
# - PriceMoE: load balancing reduces tail latency
#
# Usage:
#   ./scripts/run_online_benchmark.sh baseline
#   ./scripts/run_online_benchmark.sh price_moe 0.1
#   ./scripts/run_online_benchmark.sh price_moe 0.2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E13_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_DIR="$E13_DIR/../../../external/vllm"

# Parameters
MODE="${1:-baseline}"
ALPHA="${2:-0.1}"
REQUEST_RATE="${3:-8}"  # requests per second
NUM_PROMPTS="${4:-200}"
DATE=$(date +%y%m%d)
TIME=$(date +%H%M%S)
EMA_DECAY="${5:-0.9}"

# Model config
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_NAME="qwen3-next-80b"
EP_MODE="noep"
LOAD_PROFILE="normal"
DATASET="online"
TP=4
PORT=8000

# Output directory
OUTER_DIR_NAME="${DATE}-${MODEL_NAME}-${EP_MODE}-${LOAD_PROFILE}-${DATASET}"
OUTPUT_DIR="$E13_DIR/outputs/runs/$OUTER_DIR_NAME"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "E1.3 Online Benchmark"
echo "=============================================="
echo "Mode: $MODE"
echo "Alpha: $ALPHA (if price_moe)"
echo "EMA decay: $EMA_DECAY (if price_moe)"
echo "Request Rate: $REQUEST_RATE req/s"
echo "Num Prompts: $NUM_PROMPTS"
echo "=============================================="

# Set routing environment variables
if [ "$MODE" == "price_moe" ]; then
    # Use VLLM_PRICE_MOE_* naming (consistent with utils.py)
    export VLLM_PRICE_MOE_ENABLED="1"
    export VLLM_PRICE_MOE_ALPHA="$ALPHA"
    export VLLM_PRICE_MOE_EMA_DECAY="$EMA_DECAY"
    export VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS="1"
    RUN_NAME="${DATE}_${DATASET}_price_moe_alpha${ALPHA}_ema${EMA_DECAY}"
else
    unset VLLM_PRICE_MOE_ENABLED
    unset VLLM_PRICE_MOE_ALPHA
    unset VLLM_PRICE_MOE_EMA_DECAY
    unset VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS
    RUN_NAME="${DATE}_${DATASET}_baseline"
fi

# Avoid overwriting an existing run directory
if [ -d "$OUTPUT_DIR/$RUN_NAME" ]; then
    RUN_NAME="${RUN_NAME}_${TIME}"
fi

# Enable stats collection
export VLLM_MOE_STATS_DIR="$OUTPUT_DIR/$RUN_NAME/moe_stats"
mkdir -p "$VLLM_MOE_STATS_DIR"

RESULT_FILE="$OUTPUT_DIR/$RUN_NAME/benchmark_results.json"
mkdir -p "$(dirname "$RESULT_FILE")"

echo ""
echo "[Step 1] Starting vLLM server..."
echo "=========================================="

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --port $PORT \
    --disable-log-requests \
    2>&1 | tee "$OUTPUT_DIR/$RUN_NAME/server.log" &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start (PID: $SERVER_PID)..."
MAX_WAIT=300
WAITED=0
while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Server failed to start within $MAX_WAIT seconds"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Still waiting... ($WAITED s)"
done

echo "Server is ready!"

# Warmup
echo ""
echo "[Step 2] Warmup (20 requests)..."
echo "=========================================="

python "$VLLM_DIR/benchmarks/benchmark_serving.py" \
    --backend openai-chat \
    --base-url "http://localhost:$PORT" \
    --endpoint "/v1/chat/completions" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 20 \
    --request-rate 2 \
    --save-result "$OUTPUT_DIR/$RUN_NAME/warmup.json" \
    2>&1 | tee -a "$OUTPUT_DIR/$RUN_NAME/warmup.log"

echo ""
echo "[Step 3] Running benchmark..."
echo "=========================================="
echo "Request rate: $REQUEST_RATE req/s"
echo "Num prompts: $NUM_PROMPTS"

python "$VLLM_DIR/benchmarks/benchmark_serving.py" \
    --backend openai-chat \
    --base-url "http://localhost:$PORT" \
    --endpoint "/v1/chat/completions" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --save-result "$RESULT_FILE" \
    2>&1 | tee "$OUTPUT_DIR/$RUN_NAME/benchmark.log"

echo ""
echo "[Step 4] Shutting down server..."
echo "=========================================="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

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
print(f'  Throughput: {r.get(\"request_throughput\", 0):.2f} req/s')
print(f'  Output Throughput: {r.get(\"output_throughput\", 0):.2f} tok/s')
print(f'  TTFT P50: {r.get(\"ttft_p50\", 0)*1000:.1f} ms')
print(f'  TTFT P99: {r.get(\"ttft_p99\", 0)*1000:.1f} ms')
print(f'  TPOT P50: {r.get(\"tpot_p50\", 0)*1000:.1f} ms')
print(f'  TPOT P99: {r.get(\"tpot_p99\", 0)*1000:.1f} ms')
print(f'  Request Latency P50: {r.get(\"request_latency_p50\", 0):.2f} s')
print(f'  Request Latency P99: {r.get(\"request_latency_p99\", 0):.2f} s')
"
fi
