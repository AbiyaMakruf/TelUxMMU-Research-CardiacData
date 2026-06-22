#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-densenet201}" BATCH_SIZE="${BATCH_SIZE:-4}" RUN_ROOT="${RUN_ROOT:-runs/densenet}" bash scripts/_full_training_model.sh "$@"
