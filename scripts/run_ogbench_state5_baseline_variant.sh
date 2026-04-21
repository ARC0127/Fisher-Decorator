#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/ogbench_env_flags.sh

SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-ogbench_state5_s1_b1_v1_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/$RUN_TAG}"
SAVE_DIR="$OUT_ROOT/exp"
MANIFEST="$OUT_ROOT/run_manifest.tsv"
TMP_MANIFEST_DIR="$OUT_ROOT/_manifest_parts"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
BASE_AGENT="${BASE_AGENT:-agents/deflow.py}"
VARIANT_AGENT="${VARIANT_AGENT:-agents/fidec.py}"
BASE_METHOD_NAME="${BASE_METHOD_NAME:-deflow}"
VARIANT_METHOD_NAME="${VARIANT_METHOD_NAME:-fidec}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
COMMON_EXTRA="${COMMON_EXTRA:-}"
BASE_EXTRA="${BASE_EXTRA:-}"
VARIANT_EXTRA="${VARIANT_EXTRA:---agent.fisher_t=0.90 --agent.fisher_t_width=0.05 --agent.num_fisher_samples=8 --agent.fisher_damping=1e-3}"
PRESET="${PRESET:-default_task5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL_ENVS="${PARALLEL_ENVS:-2}"

mkdir -p "$OUT_ROOT" "$SAVE_DIR" "$TMP_MANIFEST_DIR"

case "$PRESET" in
  default_task5|state5_default_task)
    ENVS=(
      humanoidmaze-medium-navigate-singletask-task1-v0
      antsoccer-arena-navigate-singletask-task4-v0
      cube-double-play-singletask-task2-v0
      scene-play-singletask-task2-v0
      puzzle-4x4-play-singletask-task4-v0
    )
    ;;
  state5_balanced)
    ENVS=(
      antmaze-large-navigate-singletask-task1-v0
      humanoidmaze-medium-navigate-singletask-task1-v0
      antsoccer-arena-navigate-singletask-task4-v0
      cube-single-play-singletask-v0
      scene-play-singletask-v0
    )
    ;;
  state5_hard)
    ENVS=(
      antmaze-large-navigate-singletask-task1-v0
      antmaze-giant-navigate-singletask-task1-v0
      humanoidmaze-large-navigate-singletask-task1-v0
      antsoccer-arena-navigate-singletask-task4-v0
      puzzle-4x4-play-singletask-v0
    )
    ;;
  *)
    echo "[ERROR] Unknown PRESET=$PRESET" >&2
    exit 1
    ;;
esac

printf 'method\tenv_name\tagent_config\trun_group\tsave_dir\teval_csv\tpaper_eval_steps\n' > "$MANIFEST"

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
  local method_extra_str="$4"
  local manifest_part="$5"
  local env_slug group latest_run eval_csv paper_steps

  get_state_ogbench_flags "$env_name"
  paper_steps="$(get_state_ogbench_eval_steps_csv)"
  tokenize_extra "$COMMON_EXTRA"
  local common_extra_arr=("${EXTRA_ARR[@]}")
  tokenize_extra "$method_extra_str"
  local method_extra_arr=("${EXTRA_ARR[@]}")

  env_slug="${env_name//\//_}"
  env_slug="${env_slug//:/_}"
  group="${RUN_TAG}__${method}__${env_slug}"

  echo "[RUN] method=$method env=$env_name agent=$agent_cfg"
  WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" -u main.py \
    --save_dir "$SAVE_DIR" \
    --run_group "$group" \
    --seed "$SEED" \
    --env_name "$env_name" \
    --offline_steps=1000000 \
    --online_steps=0 \
    --log_interval "$LOG_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_episodes "$EVAL_EPISODES" \
    --save_interval "$SAVE_INTERVAL" \
    --agent="$agent_cfg" \
    "${ENV_FLAGS[@]}" \
    "${common_extra_arr[@]}" \
    "${method_extra_arr[@]}"

  latest_run="$(ls -td "$SAVE_DIR"/fql/"$group"/sd$(printf '%03d' "$SEED")_* 2>/dev/null | head -n 1)"
  if [[ -z "$latest_run" ]]; then
    echo "[ERROR] Could not locate run directory for group=$group" >&2
    exit 1
  fi
  eval_csv="$latest_run/eval.csv"
  if [[ ! -f "$eval_csv" ]]; then
    echo "[ERROR] Missing eval.csv: $eval_csv" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$method" "$env_name" "$agent_cfg" "$group" "$latest_run" "$eval_csv" "$paper_steps" >> "$manifest_part"
}

run_env_pair() {
  local env_name="$1"
  local env_slug manifest_part
  env_slug="${env_name//\//_}"
  env_slug="${env_slug//:/_}"
  manifest_part="$TMP_MANIFEST_DIR/${env_slug}.tsv"
  : > "$manifest_part"

  run_one "$BASE_METHOD_NAME" "$env_name" "$BASE_AGENT" "$BASE_EXTRA" "$manifest_part"
  run_one "$VARIANT_METHOD_NAME" "$env_name" "$VARIANT_AGENT" "$VARIANT_EXTRA" "$manifest_part"
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

cat "$TMP_MANIFEST_DIR"/*.tsv >> "$MANIFEST"

"$PYTHON_BIN" scripts/summarize_ogbench_paper_table.py --manifest "$MANIFEST" --out_dir "$OUT_ROOT/summary"

echo "[OK] all runs finished"
echo "[OK] manifest: $MANIFEST"
echo "[OK] summary dir: $OUT_ROOT/summary"

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  echo "[OK] AUTO_SHUTDOWN=1, attempting to power off the machine because all runs and summary finished successfully."
  (shutdown -h now || poweroff || sudo shutdown -h now || sudo poweroff)
fi
