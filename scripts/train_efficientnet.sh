#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-efficientnet_v2_s}" RUN_ROOT="${RUN_ROOT:-runs/efficientnet}" bash scripts/_full_training_model.sh "$@"
