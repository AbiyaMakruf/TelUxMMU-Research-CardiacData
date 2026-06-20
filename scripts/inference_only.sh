#!/usr/bin/env bash
set -euo pipefail

MODE=inference_only CONFIG="${CONFIG:-configs/default.yaml}" bash scripts/run_nohup.sh "$@" >/dev/null 2>&1
