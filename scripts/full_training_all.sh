#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
MODEL_NAME="${MODEL_NAME:-mobilenet_v2}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"

SCHEMES=(
  "single_raw_image:mobilenet_v2_single_raw_image"
  "single_clean_image:mobilenet_v2_single_clean_image"
  "single_12_lead:mobilenet_v2_single_12_lead"
  "single_long_lead_ii:mobilenet_v2_single_long_lead_ii"
  "multibranch_12lead_longlead:mobilenet_v2_multibranch_12lead_longlead"
  "multibranch_6lead_6lead_longlead:mobilenet_v2_multibranch_6lead_6lead_longlead"
  "multibranch_13lead_individual:mobilenet_v2_multibranch_13lead_individual"
  "stacked_12lead_longlead:mobilenet_v2_stacked_12lead_longlead"
  "stacked_6lead_6lead_longlead:mobilenet_v2_stacked_6lead_6lead_longlead"
  "stacked_13lead_individual:mobilenet_v2_stacked_13lead_individual"
)

if [[ "${FULL_TRAINING_ALL_INTERNAL:-0}" != "1" ]]; then
  mkdir -p runs
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  nohup env FULL_TRAINING_ALL_INTERNAL=1 bash "$0" "$@" \
    > "runs/${timestamp}_full_training_all.log" 2>&1 &
  exit 0
fi

for item in "${SCHEMES[@]}"; do
  input_scheme="${item%%:*}"
  run_name="${item##*:}"
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  run_dir="runs/${timestamp}_${run_name}"
  mkdir -p "${run_dir}/logs"

  uv run python -m src.runner \
    --config "${CONFIG}" \
    --mode train_eval \
    --model_name "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    "$@" \
    --input_scheme "${input_scheme}" \
    --run_name "${run_name}" \
    --output_dir "${run_dir}" \
    > "${run_dir}/logs/runner.nohup.log" 2>&1
done
