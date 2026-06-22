#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-runs}"
OUTPUT_DIR="${OUTPUT_DIR:-summary/run_rankings}"
TOP_N="${TOP_N:-20}"

uv run python -m src.reports.rank_runs \
  --runs_root "${RUNS_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --top_n "${TOP_N}" \
  "$@"
