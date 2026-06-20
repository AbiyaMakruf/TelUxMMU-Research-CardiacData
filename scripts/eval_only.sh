#!/usr/bin/env bash
set -euo pipefail

MODE=eval_only CONFIG="${CONFIG:-configs/default.yaml}" bash scripts/run_nohup.sh "$@" >/dev/null 2>&1
