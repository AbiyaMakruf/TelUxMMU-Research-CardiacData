#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-mobilenet_v2}" RUN_ROOT="${RUN_ROOT:-runs/mobilenet}" bash scripts/_full_training_model.sh "$@"
