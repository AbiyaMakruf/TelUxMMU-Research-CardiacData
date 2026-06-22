#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="${NOTEBOOK:-notebooks/preprocessing.ipynb}"
OUTPUT_DIR="${OUTPUT_DIR:-notebooks}"
OUTPUT_NAME="${OUTPUT_NAME:-preprocessing_executed.ipynb}"
TIMEOUT="${TIMEOUT:--1}"

uv run jupyter nbconvert \
  --to notebook \
  --execute "${NOTEBOOK}" \
  --output "${OUTPUT_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --ExecutePreprocessor.timeout="${TIMEOUT}"
