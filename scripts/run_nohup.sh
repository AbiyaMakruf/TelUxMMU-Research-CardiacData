#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-train_eval}"
MODEL_NAME="${MODEL_NAME:-mobilenet_v2}"
INPUT_SCHEME="${INPUT_SCHEME:-single_clean_image}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_${INPUT_SCHEME}}"
CONFIG="${CONFIG:-configs/default.yaml}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --mode)
      MODE="${ARGS[$((i + 1))]:-${MODE}}"
      ;;
    --model_name)
      MODEL_NAME="${ARGS[$((i + 1))]:-${MODEL_NAME}}"
      ;;
    --input_scheme)
      INPUT_SCHEME="${ARGS[$((i + 1))]:-${INPUT_SCHEME}}"
      ;;
    --run_name)
      RUN_NAME="${ARGS[$((i + 1))]:-${RUN_NAME}}"
      ;;
    --config)
      CONFIG="${ARGS[$((i + 1))]:-${CONFIG}}"
      ;;
  esac
done

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="runs/${TIMESTAMP}_${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs"

{
  printf '%s | INFO | launcher_start=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(date --iso-8601=seconds)"
  printf '%s | INFO | run_dir=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${RUN_DIR}"
  printf '%s | INFO | mode=%s model_name=%s input_scheme=%s run_name=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${MODE}" "${MODEL_NAME}" "${INPUT_SCHEME}" "${RUN_NAME}"
  printf '%s | INFO | process_status=starting_background_process\n' "$(date '+%Y-%m-%d %H:%M:%S')"
} > training.log
cp training.log "${RUN_DIR}/logs/training.log"

nohup uv run python -m src.runner \
  --config "${CONFIG}" \
  --mode "${MODE}" \
  --model_name "${MODEL_NAME}" \
  --input_scheme "${INPUT_SCHEME}" \
  --run_name "${RUN_NAME}" \
  --output_dir "${RUN_DIR}" \
  "${ARGS[@]}" \
  > "${RUN_DIR}/logs/runner.nohup.log" 2>&1 &
