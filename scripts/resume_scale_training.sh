#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/c/Users/HP/Documents/ChatGPT/q4d
source /home/mrogwm/.venvs/q4d/bin/activate

dataset_root="artifacts/datasets/pushcube_scale_v1"
pipeline_log="$dataset_root/pipeline.log"
status_path="$dataset_root/pipeline_status.json"
export PYTHONUNBUFFERED=1
exec >>"$pipeline_log" 2>&1

write_status() {
  local status="$1"
  local phase="$2"
  local exit_code="${3:-0}"
  printf '{"status":"%s","phase":"%s","exit_code":%s,"updated_unix":%s}\n' \
    "$status" "$phase" "$exit_code" "$(date +%s)" >"$status_path.tmp"
  mv "$status_path.tmp" "$status_path"
}

phase="training_and_ablations"
trap 'code=$?; write_status failed "$phase" "$code"; exit "$code"' ERR
write_status running "$phase"
python scripts/run_scale_experiment.py --config configs/scale_experiment.toml

phase="complete"
write_status complete "$phase"
