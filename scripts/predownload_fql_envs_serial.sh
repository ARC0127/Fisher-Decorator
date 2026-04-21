#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
STATUS_TSV="${STATUS_TSV:-$LOG_DIR/predownload_fql_envs_status.tsv}"

export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export OGBENCH_DATA_DIR="${OGBENCH_DATA_DIR:-/root/autodl-tmp/rl_datasets/ogbench}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/root/autodl-tmp/rl_datasets/d4rl}"

mkdir -p "$LOG_DIR" "$OGBENCH_DATA_DIR" "$D4RL_DATASET_DIR"

# 先把旧 status 文件规范化，兼容之前带 idx 列的版本
"$PYTHON_BIN" - "$STATUS_TSV" <<'PY'
import csv
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
header = ["env_name", "status", "frame_stack", "train_size", "val_size", "note"]

if not path.exists():
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
    raise SystemExit(0)

rows = []
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append({k: r.get(k, "") for k in header})

with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
    w.writeheader()
    w.writerows(rows)
PY

declare -a ENVS=()

for fam in \
  antmaze-large-navigate \
  antmaze-giant-navigate \
  humanoidmaze-medium-navigate \
  humanoidmaze-large-navigate \
  antsoccer-arena-navigate \
  cube-single-play \
  cube-double-play \
  scene-play \
  puzzle-3x3-play \
  puzzle-4x4-play
do
  for t in 1 2 3 4 5; do
    ENVS+=("${fam}-singletask-task${t}-v0")
  done
done

ENVS+=(
  visual-cube-single-play-singletask-task1-v0
  visual-cube-double-play-singletask-task1-v0
  visual-scene-play-singletask-task1-v0
  visual-puzzle-3x3-play-singletask-task1-v0
  visual-puzzle-4x4-play-singletask-task1-v0
)

ENVS+=(
  antmaze-umaze-v2
  antmaze-umaze-diverse-v2
  antmaze-medium-play-v2
  antmaze-medium-diverse-v2
  antmaze-large-play-v2
  antmaze-large-diverse-v2
)

ENVS+=(
  pen-human-v1
  pen-cloned-v1
  pen-expert-v1
  door-human-v1
  door-cloned-v1
  door-expert-v1
  hammer-human-v1
  hammer-cloned-v1
  hammer-expert-v1
  relocate-human-v1
  relocate-cloned-v1
  relocate-expert-v1
)

already_ok() {
  local env_name="$1"
  awk -F'\t' -v e="$env_name" 'NR>1 && $1==e && $2=="ok" {found=1} END{exit(found?0:1)}' "$STATUS_TSV"
}

total="${#ENVS[@]}"
idx=0

for env_name in "${ENVS[@]}"; do
  idx=$((idx + 1))

  if already_ok "$env_name"; then
    echo "[SKIP-OK $idx/$total] $env_name"
    continue
  fi

  frame_stack_arg=()
  if [[ "$env_name" == visual-* ]]; then
    frame_stack_arg=(--frame_stack 3)
  fi

  echo "[$idx/$total] $env_name ${frame_stack_arg[*]:-}"

  if "$PYTHON_BIN" scripts/_predownload_one_env.py \
      --env_name "$env_name" \
      --status_file "$STATUS_TSV" \
      "${frame_stack_arg[@]}"; then
    :
  else
    rc=$?
    echo "[WARN] child process failed for $env_name with rc=$rc"
    "$PYTHON_BIN" - "$STATUS_TSV" "$env_name" "$rc" <<'PY'
import csv
import sys
from pathlib import Path

tsv = Path(sys.argv[1])
env_name = sys.argv[2]
rc = sys.argv[3]
header = ["env_name", "status", "frame_stack", "train_size", "val_size", "note"]

rows = []
if tsv.exists():
    with open(tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append({k: r.get(k, "") for k in header if k in header})

rows = [r for r in rows if r.get("env_name") != env_name]
rows.append({
    "env_name": env_name,
    "status": "fail",
    "frame_stack": "3" if env_name.startswith("visual-") else "",
    "train_size": "",
    "val_size": "",
    "note": "child_exit_" + str(rc),
})

with open(tsv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
    w.writeheader()
    w.writerows(rows)
PY
  fi

  sleep 1
done

echo "[DONE] serial predownload finished"
echo "[DONE] inspect: $STATUS_TSV"
