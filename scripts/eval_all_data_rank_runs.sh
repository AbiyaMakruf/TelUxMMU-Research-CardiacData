#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/all_data_eval}"
RANK_OUTPUT_DIR="${RANK_OUTPUT_DIR:-summary/all_data_run_rankings}"
TOP_N="${TOP_N:-20}"
FORCE="${FORCE:-0}"

mkdir -p "${OUTPUT_ROOT}" "${RANK_OUTPUT_DIR}"

mapfile -t RUN_SPECS < <(
  uv run python - "${RUNS_ROOT}" "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import sys

runs_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])

for checkpoint in sorted(runs_root.rglob("checkpoints/best.pt")):
    run_dir = checkpoint.parents[1]
    if output_root in run_dir.parents or run_dir == output_root:
        continue
    config = run_dir / "config_resolved.yaml"
    if not config.exists():
        config = run_dir / "config.yaml"
    if not config.exists():
        continue
    relative = run_dir.relative_to(runs_root)
    eval_dir = output_root / relative
    print(f"{run_dir}\t{checkpoint}\t{config}\t{eval_dir}")
PY
)

for spec in "${RUN_SPECS[@]}"; do
  IFS=$'\t' read -r SOURCE_RUN CHECKPOINT CONFIG EVAL_DIR <<< "${spec}"
  if [[ "${FORCE}" != "1" && -f "${EVAL_DIR}/metrics/all_data_metrics.json" ]]; then
    printf '%s | INFO | skip_existing eval_dir=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${EVAL_DIR}"
    continue
  fi

  mkdir -p "${EVAL_DIR}/logs"
  printf '%s | INFO | eval_all_data source_run=%s eval_dir=%s checkpoint=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${SOURCE_RUN}" "${EVAL_DIR}" "${CHECKPOINT}" | tee "${EVAL_DIR}/logs/launcher.log"

  uv run python -m src.runner \
    --config "${CONFIG}" \
    --mode eval_all_data \
    --checkpoint-path "${CHECKPOINT}" \
    --output-dir "${EVAL_DIR}" \
    --run-name "$(basename "${SOURCE_RUN}")_all_data" \
    > "${EVAL_DIR}/logs/runner.nohup.log" 2>&1
done

uv run python -m src.reports.rank_runs \
  --runs_root "${OUTPUT_ROOT}" \
  --output_dir "${RANK_OUTPUT_DIR}" \
  --top_n "${TOP_N}" \
  --metrics_filename all_data_metrics.json
