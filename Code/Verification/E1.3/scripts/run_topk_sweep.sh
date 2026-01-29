#!/bin/bash
# E1.3/E1.2: Parameter sweep for variable top-k values
#
# This implements E1.2 functionality: observe PPL vs sparsity tradeoff
# when changing the number of activated experts.
#
# Usage:
#   ./scripts/run_topk_sweep.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_BASE="${PROJECT_DIR}/configs/e1_2_variable_k.json"

# Top-k values to sweep (model default is typically 8)
TOPK_VALUES=(1 2 4 8 10 12 16)

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN MODE]"
fi

echo "======================================"
echo "E1.2/E1.3: Variable Top-K Sweep"
echo "======================================"
echo "Base config: $CONFIG_BASE"
echo "Top-K values: ${TOPK_VALUES[*]}"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate moe || echo "Warning: failed to activate 'moe' environment"
fi

# Run each top-k configuration
for i in "${!TOPK_VALUES[@]}"; do
    k="${TOPK_VALUES[$i]}"
    run_num=$((i + 1))
    total="${#TOPK_VALUES[@]}"
    
    echo ""
    echo "[$run_num/$total] Running with top_k=$k..."
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would run: python -m src.run_experiment --config $CONFIG_BASE --routing.top_k $k"
    else
        cd "$PROJECT_DIR"
        python -m src.run_experiment --config "$CONFIG_BASE" --routing.top_k "$k"
    fi
done

echo ""
echo "======================================"
echo "Sweep complete!"
echo "======================================"

# Analyze results
echo ""
echo "Running analysis..."
if [[ "$DRY_RUN" == true ]]; then
    echo "  Would run: python -m src.analyze_load_balance outputs/runs/ --compare"
else
    cd "$PROJECT_DIR"
    python -m src.analyze_load_balance outputs/runs/ --compare
fi
