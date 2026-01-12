#!/usr/bin/env bash
set -euo pipefail

cd /home/zhhuang/sthmax/git/Project-MoE

# NOTE: run this after: conda activate moe
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false

CFG=Code/Verification/E1.1/configs/e1_1_smoke.json
RUN_ID=$(python -c "import json; print(json.load(open('$CFG'))['run_id'])")
RUN_DIR=Code/Verification/E1.1/outputs/runs/$RUN_ID

python Code/Verification/E1.1/src/collect_routing_stats.py --config $CFG --domain wikitext
python Code/Verification/E1.1/src/collect_routing_stats.py --config $CFG --domain humaneval

python Code/Verification/E1.1/src/merge_partials.py --run_dir $RUN_DIR --domain wikitext
python Code/Verification/E1.1/src/merge_partials.py --run_dir $RUN_DIR --domain humaneval

echo "[E1.1] Done. Outputs under: $RUN_DIR"