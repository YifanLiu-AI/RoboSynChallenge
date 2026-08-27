#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wcm_dir="$(cd -- "$script_dir/.." && pwd)"
run_dir="$wcm_dir/outputs/wcm/items_handover_pi05_all383_2to1"
dataset_dir="$wcm_dir/../../lerobot_dataset/wcm_v30/RoboSynChallenge/items_handover_pi05_all383_2to1_v30_with_return"
train_pid="$(cat "$run_dir/train.pid")"

while kill -0 "$train_pid" 2>/dev/null; do
    sleep 30
done

if [[ ! -f "$run_dir/deploy.pt" || ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "Training process $train_pid exited without deploy.pt and best.pt." >&2
    exit 1
fi

cd "$wcm_dir"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

.venv/bin/python -u -m world_critic.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --output-dir "$run_dir/eval_best_val" \
    --split val \
    --batch-size 16 \
    --num-workers 4 \
    --expected-world-size 1 \
    --episode-curves \
    --log-every-batches 100

.venv/bin/python -u scripts/analyze_success_failure.py \
    --input-csv "$run_dir/eval_best_val/episode_curves/episode_curves.csv" \
    --selection-manifest "$dataset_dir/meta/rollout_selection.json" \
    --output-dir "$run_dir/eval_best_val/success_failure"
