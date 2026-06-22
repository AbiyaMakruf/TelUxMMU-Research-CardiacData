#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
MODEL_NAME="${MODEL_NAME:-mobilenet_v2}"
INPUT_SCHEME="${INPUT_SCHEME:-single_clean_image}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_${INPUT_SCHEME}_inference_only}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --config)
      CONFIG="${ARGS[$((i + 1))]:-${CONFIG}}"
      ;;
    --model-name|--model_name)
      MODEL_NAME="${ARGS[$((i + 1))]:-${MODEL_NAME}}"
      ;;
    --input-scheme|--input_scheme)
      INPUT_SCHEME="${ARGS[$((i + 1))]:-${INPUT_SCHEME}}"
      ;;
    --run-name|--run_name)
      RUN_NAME="${ARGS[$((i + 1))]:-${RUN_NAME}}"
      ;;
  esac
done

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="runs/${TIMESTAMP}_${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs"

{
  printf '%s | INFO | launcher_start=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(date --iso-8601=seconds)"
  printf '%s | INFO | run_dir=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${RUN_DIR}"
  printf '%s | INFO | mode=inference_only model_name=%s input_scheme=%s run_name=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${MODEL_NAME}" "${INPUT_SCHEME}" "${RUN_NAME}"
  printf '%s | INFO | process_status=starting_background_process\n' "$(date '+%Y-%m-%d %H:%M:%S')"
} > training.log
cp training.log "${RUN_DIR}/logs/training.log"

nohup uv run python -m src.runner \
  --config "${CONFIG}" \
  --mode inference_only \
  --model-name "${MODEL_NAME}" \
  --input-scheme "${INPUT_SCHEME}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${RUN_DIR}" \
  "${ARGS[@]}" \
  > "${RUN_DIR}/logs/runner.nohup.log" 2>&1 &
