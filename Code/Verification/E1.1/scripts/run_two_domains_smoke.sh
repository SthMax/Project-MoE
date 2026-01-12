#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate moe
#   ./Code/Verification/E1.1/scripts/run_two_domains_smoke.sh
#
# Clean start (recommended after debugging runs):
#   CLEAN_START=1 ./Code/Verification/E1.1/scripts/run_two_domains_smoke.sh
#
# Clean run dir (most thorough):
#   CLEAN_RUN_DIR=1 ./Code/Verification/E1.1/scripts/run_two_domains_smoke.sh

cd /home/zhhuang/sthmax/git/Project-MoE

# NOTE: run this after: conda activate moe
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false

CFG=Code/Verification/E1.1/configs/e1_1_smoke.json
RUN_ID=$(python -c "import json; print(json.load(open('$CFG'))['run_id'])")
RUN_DIR=Code/Verification/E1.1/outputs/runs/$RUN_ID

if [[ "${CLEAN_RUN_DIR:-0}" == "1" ]]; then
	echo "[E1.1] CLEAN_RUN_DIR=1: deleting entire run dir: $RUN_DIR"
	python Code/Verification/E1.1/src/clean_run.py --run_dir "$RUN_DIR" --delete_run_dir --yes
elif [[ "${CLEAN_START:-0}" == "1" ]]; then
	echo "[E1.1] CLEAN_START=1: deleting old partials under: $RUN_DIR"
	python Code/Verification/E1.1/src/clean_run.py --run_dir "$RUN_DIR" --yes
fi

python Code/Verification/E1.1/src/collect_routing_stats.py --config $CFG --domain wikitext
python Code/Verification/E1.1/src/collect_routing_stats.py --config $CFG --domain humaneval

python Code/Verification/E1.1/src/merge_partials.py --run_dir $RUN_DIR --domain wikitext
python Code/Verification/E1.1/src/merge_partials.py --run_dir $RUN_DIR --domain humaneval

echo "[E1.1] Done. Outputs under: $RUN_DIR"