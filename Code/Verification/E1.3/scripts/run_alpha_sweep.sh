#!/bin/bash
# E1.3: Parameter sweep script for PriceMoE alpha values
#
# Usage:
#   ./scripts/run_alpha_sweep.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_BASE="${PROJECT_DIR}/configs/e1_3_price_moe.json"

# Alpha values to sweep
ALPHA_VALUES=(0.01 0.05 0.1 0.2 0.5 1.0)

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN MODE]"
fi

echo "======================================"
echo "E1.3: PriceMoE Alpha Sweep"
echo "======================================"
echo "Base config: $CONFIG_BASE"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate moe || echo "Warning: failed to activate 'moe' environment"
fi

# Run baseline first
echo "[1/${#ALPHA_VALUES[@]}] Running baseline..."
if [[ "$DRY_RUN" == true ]]; then
    echo "  Would run: python -m src.run_experiment --config configs/e1_3_baseline.json"
else
    cd "$PROJECT_DIR"
    python -m src.run_experiment --config configs/e1_3_baseline.json
fi

# Run PriceMoE with each alpha
for i in "${!ALPHA_VALUES[@]}"; do
    alpha="${ALPHA_VALUES[$i]}"
    run_num=$((i + 2))
    total=$((${#ALPHA_VALUES[@]} + 1))
    
    echo ""
    echo "[$run_num/$total] Running PriceMoE with alpha=$alpha..."
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would run: python -m src.run_experiment --config $CONFIG_BASE --routing.alpha $alpha"
    else
        cd "$PROJECT_DIR"
        python -m src.run_experiment --config "$CONFIG_BASE" --routing.alpha "$alpha"
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
