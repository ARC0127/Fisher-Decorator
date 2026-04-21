#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/ogbench_env_flags.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE="${WANDB_MODE:-offline}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
PARALLEL_ENVS="${PARALLEL_ENVS:-2}"
SEED="${SEED:-0}"
ONLINE_STEPS="${ONLINE_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"

# offline 批次根目录，例如：
# /root/TrustRegion/DeFlow-main/results/ogbench_state5_s1_b1_v1_20260309_123456
OFFLINE_BATCH_ROOT="${OFFLINE_BATCH_ROOT:?must set OFFLINE_BATCH_ROOT}"

RUN_TAG="${RUN_TAG:-ogbench_default_task5_online_from_offline_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/$RUN_TAG}"
SAVE_DIR="$OUT_ROOT/exp"
mkdir -p "$OUT_ROOT" "$SAVE_DIR"

BASE_AGENT="${BASE_AGENT:-agents/deflow.py}"
FIDEC_AGENT="${FIDEC_AGENT:-agents/fidec.py}"

# 只放“方法专属”项；不要在这里重复写 env-specific 的 target_divergence / discount / q_agg
BASE_EXTRA="${BASE_EXTRA:-}"
FIDEC_EXTRA="${FIDEC_EXTRA:---agent.fisher_t=0.90 --agent.fisher_t_width=0.05 --agent.num_fisher_samples=8 --agent.fisher_damping=1e-3}"

ENVS=(
  humanoidmaze-medium-navigate-singletask-task1-v0
  antsoccer-arena-navigate-singletask-task4-v0
  cube-double-play-singletask-task2-v0
  scene-play-singletask-task2-v0
  puzzle-4x4-play-singletask-task4-v0
)

tokenize_extra() {
  local extra="$1"
  EXTRA_ARR=()
  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($extra)
  fi
}

run_one() {
  local method="$1"
  local env_name="$2"
  local agent_cfg="$3"
  local extra_str="$4"

  get_state_ogbench_flags "$env_name"

  local env_slug="${env_name//\//_}"
  env_slug="${env_slug//:/_}"

  local offline_group
  offline_group="$(find "$OFFLINE_BATCH_ROOT/exp/fql" -maxdepth 1 -mindepth 1 -type d \
    | grep "__${method}__${env_slug}$" | head -n 1 || true)"

  if [[ -z "$offline_group" ]]; then
    echo "[ERROR] cannot find offline group for method=$method env=$env_name" >&2
    exit 1
  fi

  local restore_glob="$offline_group/sd$(printf '%03d' "$SEED")_*"
  local restore_dir
  restore_dir="$(ls -td $restore_glob 2>/dev/null | head -n 1 || true)"

  if [[ -z "$restore_dir" ]]; then
    echo "[ERROR] cannot find restore dir for method=$method env=$env_name seed=$SEED" >&2
    exit 1
  fi

  if [[ ! -f "$restore_dir/params_1000000.pkl" ]]; then
    echo "[ERROR] missing checkpoint: $restore_dir/params_1000000.pkl" >&2
    exit 1
  fi

  local run_group="${RUN_TAG}__${method}__${env_slug}"

  tokenize_extra "$extra_str"
  local extra_arr=("${EXTRA_ARR[@]}")

  echo "[RUN-ONLINE] method=$method env=$env_name agent=$agent_cfg restore=$restore_dir"

  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.42}" \
  WANDB_MODE="$WANDB_MODE" \
  "$PYTHON_BIN" -u main.py \
    --save_dir "$SAVE_DIR" \
    --run_group "$run_group" \
    --seed "$SEED" \
    --env_name "$env_name" \
    --offline_steps 0 \
    --online_steps "$ONLINE_STEPS" \
    --restore_path "$restore_dir" \
    --restore_epoch 1000000 \
    --log_interval "$LOG_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_episodes "$EVAL_EPISODES" \
    --save_interval "$SAVE_INTERVAL" \
    --agent "$agent_cfg" \
    "${ENV_FLAGS[@]}" \
    "${extra_arr[@]}"
}

run_env_pair() {
  local env_name="$1"
  run_one deflow "$env_name" "$BASE_AGENT" "$BASE_EXTRA"
  run_one fidec "$env_name" "$FIDEC_AGENT" "$FIDEC_EXTRA"
}

pids=()
active=0

for env_name in "${ENVS[@]}"; do
  run_env_pair "$env_name" &
  pids+=("$!")
  active=$((active + 1))
  if [[ "$active" -ge "$PARALLEL_ENVS" ]]; then
    wait -n
    active=$((active - 1))
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[OK] all offline->online jobs finished"

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  echo "[OK] AUTO_SHUTDOWN=1, attempting poweroff"
  (shutdown -h now || poweroff || sudo shutdown -h now || sudo poweroff)
fi