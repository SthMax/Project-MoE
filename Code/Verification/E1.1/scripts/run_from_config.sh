#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate moe
#   bash Code/Verification/E1.1/scripts/run_from_config.sh Code/Verification/E1.1/configs/e1_1_official.json
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 CLEAN_RUN_DIR=1 bash Code/Verification/E1.1/scripts/run_from_config.sh <cfg>
#   RUN_ID=my_run_001 bash Code/Verification/E1.1/scripts/run_from_config.sh <cfg>

cd /home/zhhuang/sthmax/git/Project-MoE

CFG=${1:-Code/Verification/E1.1/configs/e1_1_official.json}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false

# Compute run_id:
# - prefer $RUN_ID
# - else prefer config['run_id'] if present and not null
# - else use timestamp
RUN_ID=${RUN_ID:-$(python -c "import json,sys,time; cfg=json.load(open(sys.argv[1])); print(cfg.get('run_id') or time.strftime('%Y-%m-%d_%H%M%S'))" "$CFG")}

OUTPUT_ROOT=$(python -c "import json,sys; cfg=json.load(open(sys.argv[1])); print(cfg.get('output_root', 'Code/Verification/E1.1/outputs/runs'))" "$CFG")

RUN_DIR=$OUTPUT_ROOT/$RUN_ID

if [[ "${CLEAN_RUN_DIR:-0}" == "1" ]]; then
  echo "[E1.1] CLEAN_RUN_DIR=1: deleting entire run dir: $RUN_DIR"
  python Code/Verification/E1.1/src/clean_run.py --run_dir "$RUN_DIR" --delete_run_dir --yes
elif [[ "${CLEAN_START:-0}" == "1" ]]; then
  echo "[E1.1] CLEAN_START=1: deleting old partials under: $RUN_DIR"
  python Code/Verification/E1.1/src/clean_run.py --run_dir "$RUN_DIR" --yes
fi

domains=$(python -c "import json,sys; cfg=json.load(open(sys.argv[1])); print(' '.join(cfg['domains'].keys()))" "$CFG")

echo "[E1.1] Running config: $CFG"
echo "[E1.1] RUN_ID: $RUN_ID"
echo "[E1.1] RUN_DIR: $RUN_DIR"
echo "[E1.1] Domains: $domains"

for domain in $domains; do
  echo "[E1.1] Collect domain=$domain"
  python Code/Verification/E1.1/src/collect_routing_stats.py --config "$CFG" --domain "$domain" --run_id "$RUN_ID"

done

for domain in $domains; do
  echo "[E1.1] Merge domain=$domain"
  python Code/Verification/E1.1/src/merge_partials.py --run_dir "$RUN_DIR" --domain "$domain"

  echo "[E1.1] Summarize domain=$domain"
  python Code/Verification/E1.1/src/summarize_npz.py --run_dir "$RUN_DIR" --domain "$domain"

  echo "[E1.1] Plot domain=$domain"
  python Code/Verification/E1.1/src/plot_routing.py --run_dir "$RUN_DIR" --domain "$domain"

done

echo "[E1.1] Analyze run"
python Code/Verification/E1.1/src/analyze_run.py --run_dir "$RUN_DIR"

echo "[E1.1] Done. Outputs under: $RUN_DIR"
