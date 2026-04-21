#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export OGBENCH_DATA_DIR="${OGBENCH_DATA_DIR:-/root/autodl-tmp/rl_datasets/ogbench}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/root/autodl-tmp/rl_datasets/d4rl}"
mkdir -p "$OGBENCH_DATA_DIR" "$D4RL_DATASET_DIR"

source scripts/fql_scope_registry.sh

SCOPE="${SCOPE:?must set SCOPE}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-FiDec}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

SEEDS_STR="${SEEDS:-0}"
read -r -a SEEDS <<< "$SEEDS_STR"

FIDEC_T_VALUES_STR="${FIDEC_T_VALUES:-0.90}"
read -r -a FIDEC_T_VALUES <<< "$FIDEC_T_VALUES_STR"

NORMALIZE_Q="${NORMALIZE_Q:-false}"

GPU_IDS_STR="${GPU_IDS:-0 1}"
read -r -a GPU_IDS <<< "$GPU_IDS_STR"
GPU_SLOTS="${GPU_SLOTS:-4}"
XLA_PREALLOCATE="${XLA_PREALLOCATE:-false}"
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.22}"

RUN_TAG="${RUN_TAG:-${SCOPE}_offline_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/$RUN_TAG}"
SAVE_DIR="$OUT_ROOT/exp"
MANIFEST="$OUT_ROOT/run_manifest.tsv"
TMP_MANIFEST_DIR="$OUT_ROOT/_manifest_parts"
LOG_DIR="$OUT_ROOT/logs"

BASE_AGENT="${BASE_AGENT:-agents/deflow.py}"
FIDEC_AGENT="${FIDEC_AGENT:-agents/fidec.py}"
FIDEC_WIDTH="${FIDEC_WIDTH:-0.05}"
FIDEC_NUM_SAMPLES="${FIDEC_NUM_SAMPLES:-8}"
FIDEC_DAMPING="${FIDEC_DAMPING:-1e-3}"

LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

mkdir -p "$SAVE_DIR" "$TMP_MANIFEST_DIR" "$LOG_DIR"
mapfile -t ENVS < <(get_scope_envs "$SCOPE")

declare -a PIDS=()
declare -a PID_GPU=()

compact_jobs() {
  local -a new_pids=()
  local -a new_gpus=()
  local i
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      new_pids+=("${PIDS[$i]}")
      new_gpus+=("${PID_GPU[$i]}")
    fi
  done
  PIDS=("${new_pids[@]}")
  PID_GPU=("${new_gpus[@]}")
}

active_on_gpu() {
  local gpu="$1"
  local cnt=0
  local i
  for i in "${!PIDS[@]}"; do
    if [[ "${PID_GPU[$i]}" == "$gpu" ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
      cnt=$((cnt + 1))
    fi
  done
  echo "$cnt"
}

wait_for_gpu_slot() {
  while true; do
    compact_jobs
    local best_gpu=""
    local best_cnt=999999
    local gpu cnt
    for gpu in "${GPU_IDS[@]}"; do
      cnt="$(active_on_gpu "$gpu")"
      if (( cnt < GPU_SLOTS )) && (( cnt < best_cnt )); then
        best_gpu="$gpu"
        best_cnt="$cnt"
      fi
    done
    if [[ -n "$best_gpu" ]]; then
      echo "$best_gpu"
      return 0
    fi
    sleep 10
  done
}

safe_slug() {
  echo "$1" | tr '/:.' '_' | tr -cs '[:alnum:]_-' '_'
}

launch_task() {
  local seed="$1"
  local method="$2"
  local env_name="$3"
  local agent_cfg="$4"
  local fidec_t="$5"

  local gpu env_slug group logfile part
  get_env_flags_and_steps "$env_name"

  gpu="$(wait_for_gpu_slot)"
  env_slug="$(safe_slug "$env_name")"
  group="${RUN_TAG}__${method}__s${seed}__${env_slug}"
  logfile="$LOG_DIR/${method}__s${seed}__${env_slug}.log"
  part="$TMP_MANIFEST_DIR/${method}__s${seed}__${env_slug}.tsv"

  local extra_arr=()
  extra_arr+=(--agent.normalize_q_loss="${NORMALIZE_Q}")
  if [[ -n "$fidec_t" ]]; then
    extra_arr+=(--agent.fisher_t="$fidec_t" --agent.fisher_t_width="$FIDEC_WIDTH" --agent.num_fisher_samples="$FIDEC_NUM_SAMPLES" --agent.fisher_damping="$FIDEC_DAMPING")
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PREALLOCATE"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
    export WANDB_MODE="$WANDB_MODE"
    export WANDB_PROJECT="$WANDB_PROJECT"

    "$PYTHON_BIN" -u main.py \
      --save_dir "$SAVE_DIR" \
      --run_group "$group" \
      --seed "$seed" \
      --env_name "$env_name" \
      --offline_steps "$OFFLINE_STEPS" \
      --online_steps 0 \
      --log_interval "$LOG_INTERVAL" \
      --eval_interval "$EVAL_INTERVAL" \
      --eval_episodes "$EVAL_EPISODES" \
      --save_interval "$SAVE_INTERVAL" \
      --agent "$agent_cfg" \
      "${ENV_FLAGS[@]}" \
      "${extra_arr[@]}"

    latest_run="$(ls -td "$SAVE_DIR"/FiDec/"$group"/sd$(printf '%03d' "$seed")_* 2>/dev/null | head -n 1)"
    if [[ -z "$latest_run" ]]; then
      latest_run="$(ls -td "$SAVE_DIR"/fql/"$group"/sd$(printf '%03d' "$seed")_* 2>/dev/null | head -n 1)"
    fi
    if [[ -z "$latest_run" ]]; then
      echo "[ERROR] run directory not found for $group" >&2
      exit 1
    fi
    eval_csv="$latest_run/eval.csv"
    printf 'scope\tmetric_group\tcategory_group\tmethod\tenv_name\tseed\tfidec_t\tagent_config\trun_group\tsave_dir\teval_csv\tpaper_eval_steps\n' > "$part"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$SCOPE" "$METRIC_GROUP" "$CATEGORY_GROUP" "$method" "$env_name" "$seed" "${fidec_t:-na}" "$agent_cfg" "$group" "$latest_run" "$eval_csv" "$PAPER_EVAL_STEPS" >> "$part"
  ) > "$logfile" 2>&1 &

  pid=$!
  PIDS+=("$pid")
  PID_GPU+=("$gpu")
  echo "[LAUNCH] pid=$pid gpu=$gpu seed=$seed method=$method env=$env_name"
}

for seed in "${SEEDS[@]}"; do
  for env_name in "${ENVS[@]}"; do
    launch_task "$seed" "deflow" "$env_name" "$BASE_AGENT" ""
    for fidec_t in "${FIDEC_T_VALUES[@]}"; do
      ttag="${fidec_t/./p}"
      launch_task "$seed" "fidec_t${ttag}" "$env_name" "$FIDEC_AGENT" "$fidec_t"
    done
  done
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "[ERROR] at least one offline job failed" >&2
  exit 1
fi

printf 'scope\tmetric_group\tcategory_group\tmethod\tenv_name\tseed\tfidec_t\tagent_config\trun_group\tsave_dir\teval_csv\tpaper_eval_steps\n' > "$MANIFEST"
awk 'FNR>1' "$TMP_MANIFEST_DIR"/*.tsv >> "$MANIFEST"

"$PYTHON_BIN" scripts/summarize_fql_scope.py --manifest "$MANIFEST" --out_dir "$OUT_ROOT/summary" --mode offline

echo "[OK] offline finished"
echo "[OK] manifest: $MANIFEST"

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  (shutdown -h now || poweroff || sudo shutdown -h now || sudo poweroff)
fi
