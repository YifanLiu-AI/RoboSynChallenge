#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wcm_dir="$(cd -- "$script_dir/.." && pwd)"
base_config="$wcm_dir/configs/items_handover_pi05_2to1.yaml"
current_run_dir="$wcm_dir/outputs/wcm/items_handover_pi05_all383_2to1"
dataset_dir="$wcm_dir/../../lerobot_dataset/wcm_v30/RoboSynChallenge/items_handover_pi05_all383_2to1_v30_with_return"
output_root="$wcm_dir/outputs/wcm/items_handover_pi05_all383_2to1_big_batch"
eval_batch=32
wait_current=1
resume_existing=0
declare -a batches=()

usage() {
    cat <<'EOF'
Usage: scripts/run_items_handover_big_batch.sh [options]

Train one or more batch sizes serially, then evaluate and plot each run.

Options:
  --batch N          Batch size to train (repeatable; default: 32)
  --eval-batch N     Evaluation batch size (default: 32)
  --output-root DIR  Parent directory for batch-specific runs
  --no-wait-current  Do not wait for the existing batch=2 train/posttrain PIDs
  --resume-existing  Resume each existing batch directory from checkpoints/last.pt
  -h, --help         Show this help

Examples:
  scripts/run_items_handover_big_batch.sh --batch 32
  scripts/run_items_handover_big_batch.sh --batch 16 --batch 32
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "$1 requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)
            require_value "$@"
            batches+=("$2")
            shift 2
            ;;
        --eval-batch)
            require_value "$@"
            eval_batch="$2"
            shift 2
            ;;
        --output-root)
            require_value "$@"
            output_root="$2"
            shift 2
            ;;
        --no-wait-current)
            wait_current=0
            shift
            ;;
        --resume-existing)
            resume_existing=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ${#batches[@]} -eq 0 ]]; then
    batches=(32)
fi
for batch in "${batches[@]}" "$eval_batch"; do
    if [[ ! "$batch" =~ ^[1-9][0-9]*$ ]]; then
        echo "Batch sizes must be positive integers, got: $batch" >&2
        exit 2
    fi
done
if [[ ! -x "$wcm_dir/.venv/bin/python" ]]; then
    echo "WCM Python is missing: $wcm_dir/.venv/bin/python" >&2
    exit 1
fi
if [[ ! -f "$dataset_dir/meta/rollout_selection.json" ]]; then
    echo "Dataset or selection manifest is missing: $dataset_dir" >&2
    exit 1
fi

mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd)"
exec 9>"$output_root/.serial.lock"
if ! flock -n 9; then
    echo "Another big-batch queue already holds $output_root/.serial.lock" >&2
    exit 1
fi

wait_for_pid_file() {
    local label="$1"
    local pid_file="$2"
    local pid
    [[ -f "$pid_file" ]] || return 0
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        echo "[$(date --iso-8601=seconds)] Waiting for $label PID $pid..."
        while kill -0 "$pid" 2>/dev/null; do
            sleep 30
        done
        echo "[$(date --iso-8601=seconds)] $label PID $pid finished."
    fi
}

cd "$wcm_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Prevent an inherited experiment override from silently changing the checked-in
# five-epoch configuration. Batch/output overrides are set per queue item below.
unset WCM_DATASET_REPO_ID WCM_DATASET_ROOT WCM_DATASET_REVISION
unset WCM_VISION_MODEL_NAME WCM_LANGUAGE_MODEL_NAME WCM_EPOCHS WCM_RESUME WCM_PRECISION
export WCM_NUM_WORKERS=4
export WCM_EXPECTED_WORLD_SIZE=1

if (( wait_current )); then
    wait_for_pid_file "current batch=2 training" "$current_run_dir/train.pid"
    wait_for_pid_file "current batch=2 post-training pipeline" "$current_run_dir/posttrain.pid"
fi

for batch in "${batches[@]}"; do
    run_dir="$output_root/batch_${batch}"
    if [[ -e "$run_dir" ]]; then
        if (( ! resume_existing )); then
            echo "Refusing to overwrite existing run directory: $run_dir" >&2
            exit 1
        fi
        checkpoint="$run_dir/checkpoints/last.pt"
        if [[ ! -f "$checkpoint" ]]; then
            echo "Cannot resume without a last checkpoint: $checkpoint" >&2
            exit 1
        fi
        export WCM_RESUME="$checkpoint"
        echo "[$(date --iso-8601=seconds)] Resuming batch=$batch checkpoint=$checkpoint"
    else
        unset WCM_RESUME
        mkdir -p "$run_dir"
    fi

    export WCM_OUTPUT_DIR="$run_dir"
    export WCM_PER_DEVICE_BATCH_SIZE="$batch"
    export WCM_EVAL_BATCH_SIZE="$eval_batch"

    echo "[$(date --iso-8601=seconds)] Starting train batch=$batch output=$run_dir"
    "$wcm_dir/.venv/bin/python" -u -m world_critic.train --config "$base_config" \
        2>&1 | tee -a "$run_dir/train.log"

    checkpoint="$run_dir/checkpoints/best.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Training completed without best checkpoint: $checkpoint" >&2
        exit 1
    fi

    eval_dir="$run_dir/eval_best_val"
    echo "[$(date --iso-8601=seconds)] Evaluating batch=$batch checkpoint=$checkpoint"
    "$wcm_dir/.venv/bin/python" -u -m world_critic.evaluate \
        --checkpoint "$checkpoint" \
        --output-dir "$eval_dir" \
        --split val \
        --batch-size "$eval_batch" \
        --num-workers 4 \
        --expected-world-size 1 \
        --episode-curves \
        --log-every-batches 100 \
        2>&1 | tee "$run_dir/evaluate.log"

    echo "[$(date --iso-8601=seconds)] Plotting success/failure analysis for batch=$batch"
    "$wcm_dir/.venv/bin/python" -u "$script_dir/analyze_success_failure.py" \
        --input-csv "$eval_dir/episode_curves/episode_curves.csv" \
        --selection-manifest "$dataset_dir/meta/rollout_selection.json" \
        --output-dir "$eval_dir/success_failure" \
        2>&1 | tee "$run_dir/analyze_success_failure.log"
    echo "[$(date --iso-8601=seconds)] Completed batch=$batch"
    echo "[$(date --iso-8601=seconds)] Completed batch=$batch" >>"$run_dir/restart_swanlab.log"
done
