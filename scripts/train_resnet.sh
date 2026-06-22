#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-resnet50}" RUN_ROOT="${RUN_ROOT:-runs/resnet}" bash scripts/_full_training_model.sh "$@"
