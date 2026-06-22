#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
MODEL_NAME="${MODEL_NAME:-mobilenet_v2}"
RUN_ROOT="${RUN_ROOT:-runs/${MODEL_NAME}}"
STATE_FILE="${STATE_FILE:-${RUN_ROOT}/full_training_all_state.csv}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESET_FULL_TRAINING="${RESET_FULL_TRAINING:-0}"
MULTIBRANCH_BACKBONE_SHARING="${MULTIBRANCH_BACKBONE_SHARING:-shared}"
FOREGROUND="${FOREGROUND:-0}"

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --model-name|--model_name)
      MODEL_NAME="$2"; shift 2 ;;
    --run-root|--run_root)
      RUN_ROOT="$2"; STATE_FILE="${RUN_ROOT}/full_training_all_state.csv"; shift 2 ;;
    --state-file|--state_file)
      STATE_FILE="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch-size|--batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate|--learning_rate)
      LEARNING_RATE="$2"; shift 2 ;;
    --num-workers|--num_workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --reset)
      RESET_FULL_TRAINING=1; shift ;;
    --foreground)
      FOREGROUND=1; shift ;;
    --multibranch-backbone-sharing|--multibranch_backbone_sharing)
      MULTIBRANCH_BACKBONE_SHARING="$2"; shift 2 ;;
    *)
      PASSTHROUGH+=("$1"); shift ;;
  esac
done

SCHEMES=(
  "single_raw_image:${MODEL_NAME}_single_raw_image:${BATCH_SIZE}"
  "single_clean_image:${MODEL_NAME}_single_clean_image:${BATCH_SIZE}"
  "single_12_lead:${MODEL_NAME}_single_12_lead:${BATCH_SIZE}"
  "single_long_lead_ii:${MODEL_NAME}_single_long_lead_ii:${BATCH_SIZE}"
  "multibranch_12lead_longlead:${MODEL_NAME}_multibranch_12lead_longlead:${BATCH_SIZE}"
  "multibranch_6lead_6lead_longlead:${MODEL_NAME}_multibranch_6lead_6lead_longlead:${BATCH_SIZE}"
  "multibranch_13lead_individual:${MODEL_NAME}_multibranch_13lead_individual:${BATCH_SIZE}"
  "stacked_12lead_longlead:${MODEL_NAME}_stacked_12lead_longlead:${BATCH_SIZE}"
  "stacked_6lead_6lead_longlead:${MODEL_NAME}_stacked_6lead_6lead_longlead:${BATCH_SIZE}"
  "stacked_13lead_individual:${MODEL_NAME}_stacked_13lead_individual:${BATCH_SIZE}"
)

if [[ "${FULL_TRAINING_MODEL_INTERNAL:-0}" != "1" && "${FOREGROUND}" != "1" ]]; then
  mkdir -p "${RUN_ROOT}"
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  nohup env FULL_TRAINING_MODEL_INTERNAL=1 CONFIG="${CONFIG}" MODEL_NAME="${MODEL_NAME}" RUN_ROOT="${RUN_ROOT}" \
    STATE_FILE="${STATE_FILE}" EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" LEARNING_RATE="${LEARNING_RATE}" \
    NUM_WORKERS="${NUM_WORKERS}" RESET_FULL_TRAINING="${RESET_FULL_TRAINING}" \
    MULTIBRANCH_BACKBONE_SHARING="${MULTIBRANCH_BACKBONE_SHARING}" \
    bash "$0" "${PASSTHROUGH[@]}" > "${RUN_ROOT}/${timestamp}_full_training_all.log" 2>&1 &
  exit 0
fi

init_state() {
  uv run python - <<'PY' "${STATE_FILE}" "${RESET_FULL_TRAINING}" "${SCHEMES[@]}"
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

state_path = Path(sys.argv[1])
reset = sys.argv[2] == "1"
items = sys.argv[3:]
fieldnames = ["input_scheme", "run_name", "batch_size", "run_dir", "status", "updated_at"]

def progress_complete(run_dir: Path) -> bool:
    progress_path = run_dir / "artifacts" / "training_progress.json"
    if not progress_path.exists():
        return False
    try:
        progress = json.loads(progress_path.read_text())
    except Exception:
        return False
    return progress.get("status") == "complete"

rows = []
if state_path.exists():
    with state_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

if rows and not reset and all(row.get("run_dir") and progress_complete(Path(row["run_dir"])) for row in rows):
    rows = []

if reset or not rows:
    rows = []
    for item in items:
        input_scheme, run_name, batch_size = item.split(":", 2)
        rows.append(
            {
                "input_scheme": input_scheme,
                "run_name": run_name,
                "batch_size": batch_size,
                "run_dir": "",
                "status": "pending",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
else:
    by_scheme = {row["input_scheme"]: row for row in rows}
    for item in items:
        input_scheme, run_name, batch_size = item.split(":", 2)
        if input_scheme not in by_scheme:
            rows.append(
                {
                    "input_scheme": input_scheme,
                    "run_name": run_name,
                    "batch_size": batch_size,
                    "run_dir": "",
                    "status": "pending",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        else:
            by_scheme[input_scheme]["run_name"] = run_name
            by_scheme[input_scheme]["batch_size"] = batch_size

state_path.parent.mkdir(parents=True, exist_ok=True)
with state_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

get_state_field() {
  local input_scheme="$1"
  local field="$2"
  uv run python - <<'PY' "${STATE_FILE}" "${input_scheme}" "${field}"
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["input_scheme"] == sys.argv[2]:
            print(row[sys.argv[3]])
            break
PY
}

update_state_field() {
  local input_scheme="$1"
  local field="$2"
  local value="$3"
  uv run python - <<'PY' "${STATE_FILE}" "${input_scheme}" "${field}" "${value}"
import csv
import sys
from datetime import datetime
from pathlib import Path

state_path = Path(sys.argv[1])
input_scheme = sys.argv[2]
field = sys.argv[3]
value = sys.argv[4]
fieldnames = ["input_scheme", "run_name", "batch_size", "run_dir", "status", "updated_at"]
with state_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if row["input_scheme"] == input_scheme:
        row[field] = value
        row["updated_at"] = datetime.now().isoformat(timespec="seconds")
with state_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

progress_is_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/artifacts/training_progress.json" ]] || return 1
  uv run python - <<'PY' "${run_dir}/artifacts/training_progress.json"
import json
import sys
from pathlib import Path

progress = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if progress.get("status") == "complete" else 1)
PY
}

init_state

for item in "${SCHEMES[@]}"; do
  input_scheme="${item%%:*}"
  run_name="$(get_state_field "${input_scheme}" run_name)"
  batch_size="$(get_state_field "${input_scheme}" batch_size)"
  run_dir="$(get_state_field "${input_scheme}" run_dir)"
  status="$(get_state_field "${input_scheme}" status)"

  if [[ -z "${run_dir}" ]]; then
    timestamp="$(date +"%Y%m%d_%H%M%S")"
    run_dir="${RUN_ROOT}/${timestamp}_${run_name}"
    update_state_field "${input_scheme}" run_dir "${run_dir}"
  elif [[ "${status}" == "pending" && ! -e "${run_dir}/artifacts/training_progress.json" && ! -e "${run_dir}/checkpoints/latest.pt" ]]; then
    timestamp="$(date +"%Y%m%d_%H%M%S")"
    run_dir="${RUN_ROOT}/${timestamp}_${run_name}"
    update_state_field "${input_scheme}" run_dir "${run_dir}"
  fi

  if progress_is_complete "${run_dir}"; then
    echo "$(date --iso-8601=seconds) | skip_complete input_scheme=${input_scheme} run_dir=${run_dir}"
    update_state_field "${input_scheme}" status complete
    continue
  fi

  mkdir -p "${run_dir}/logs"
  update_state_field "${input_scheme}" status running
  echo "$(date --iso-8601=seconds) | start model=${MODEL_NAME} input_scheme=${input_scheme} batch_size=${batch_size} run_dir=${run_dir}"

  if uv run python -m src.runner \
    --config "${CONFIG}" \
    --mode train_eval \
    --model-name "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --learning-rate "${LEARNING_RATE}" \
    --num-workers "${NUM_WORKERS}" \
    --multibranch-backbone-sharing "${MULTIBRANCH_BACKBONE_SHARING}" \
    "${PASSTHROUGH[@]}" \
    --input-scheme "${input_scheme}" \
    --run-name "${run_name}" \
    --output-dir "${run_dir}" \
    > "${run_dir}/logs/runner.nohup.log" 2>&1; then
    update_state_field "${input_scheme}" status complete
    echo "$(date --iso-8601=seconds) | complete input_scheme=${input_scheme} run_dir=${run_dir}"
  else
    update_state_field "${input_scheme}" status failed
    echo "$(date --iso-8601=seconds) | failed input_scheme=${input_scheme} run_dir=${run_dir}"
    echo "$(date --iso-8601=seconds) | continuing_to_next_scheme"
    continue
  fi
done

echo "$(date --iso-8601=seconds) | full_training_model_finished model=${MODEL_NAME}"
