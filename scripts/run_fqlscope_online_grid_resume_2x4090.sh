#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export OGBENCH_DATA_DIR="${OGBENCH_DATA_DIR:-/root/autodl-tmp/rl_datasets/ogbench}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/root/autodl-tmp/rl_datasets/d4rl}"
mkdir -p "$OGBENCH_DATA_DIR" "$D4RL_DATASET_DIR"

source scripts/fql_scope_registry.sh

SCOPE="${SCOPE:?must set SCOPE to o2o_ogbench5 / o2o_d4rl_antmaze6 / o2o_d4rl_adroit4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-FiDec}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

if [[ -z "${OFFLINE_MANIFEST:-}" ]]; then
  echo "[ERROR] must set OFFLINE_MANIFEST" >&2
  exit 1
fi

SEEDS_STR="${SEEDS:-0}"
read -r -a SEEDS <<< "$SEEDS_STR"

FIDEC_T_VALUES_STR="${FIDEC_T_VALUES:-0.90}"
read -r -a FIDEC_T_VALUES <<< "$FIDEC_T_VALUES_STR"

NORMALIZE_Q="${NORMALIZE_Q:-false}"

GPU_IDS_STR="${GPU_IDS:-0 1}"
read -r -a GPU_IDS <<< "$GPU_IDS_STR"
GPU_SLOTS="${GPU_SLOTS:-4}"
XLA_PREALLOCATE="${XLA_PREALLOCATE:-false}"
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.10}"

ONLINE_STEPS="${ONLINE_STEPS:-1000000}"
RESTORE_EPOCH="${RESTORE_EPOCH:-1000000}"

RUN_TAG="${RUN_TAG:-${SCOPE}_online_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/$RUN_TAG}"
SAVE_DIR="${SAVE_DIR:-$OUT_ROOT/exp}"
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
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"

# Resume policy
RESUME_MODE="${RESUME_MODE:-restart_from_offline}"   # restart_from_offline | skip_done_only
ONLINE_RESUME_TAG="${ONLINE_RESUME_TAG:-}"           # existing online batch run tag to inspect/continue
ONLINE_DONE_EPOCH="${ONLINE_DONE_EPOCH:-1000000}"    # regard params_<DONE_EPOCH>.pkl as finished online job

mkdir -p "$SAVE_DIR" "$TMP_MANIFEST_DIR" "$LOG_DIR"
mapfile -t ENVS < <(get_scope_envs "$SCOPE")

declare -a PIDS=()
declare -a PID_GPU=()

resolve_manifest() {
  "$PYTHON_BIN" - "$OFFLINE_MANIFEST" <<'PY'
import os, sys
p = os.path.abspath(os.path.expanduser(sys.argv[1]))
if not os.path.isfile(p):
    raise SystemExit(f"[ERROR] OFFLINE_MANIFEST not found: {p}")
print(p)
PY
}

lookup_restore_dir() {
  local manifest="$1"
  local method="$2"
  local env_name="$3"
  local seed="$4"
  "$PYTHON_BIN" - "$manifest" "$method" "$env_name" "$seed" <<'PY'
import csv, sys
manifest, method, env_name, seed = sys.argv[1:]
seed = int(seed)
with open(manifest, newline='') as f:
    rows = [r for r in csv.DictReader(f, delimiter='\t')
            if r['method'] == method and r['env_name'] == env_name and int(r['seed']) == seed]
if not rows:
    raise SystemExit(1)
print(rows[-1]['save_dir'])
PY
}

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

existing_online_run_dir() {
  local run_tag="$1"
  local method="$2"
  local env_name="$3"
  local seed="$4"
  local env_slug group d1 d2 base_dir latest

  env_slug="$(safe_slug "$env_name")"
  group="${run_tag}__${method}__s${seed}__${env_slug}"

  d1="$ROOT_DIR/results/$run_tag/exp/FiDec/$group"
  d2="$ROOT_DIR/results/$run_tag/exp/fql/$group"

  base_dir=""
  if [[ -d "$d1" ]]; then
    base_dir="$d1"
  elif [[ -d "$d2" ]]; then
    base_dir="$d2"
  else
    return 1
  fi

  latest="$(ls -td "$base_dir"/sd$(printf '%03d' "$seed")_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  echo "$latest"
}

inspect_existing_online_job() {
  local run_tag="$1"
  local method="$2"
  local env_name="$3"
  local seed="$4"
  local run_dir ckpt_done max_ckpt eval_csv

  run_dir="$(existing_online_run_dir "$run_tag" "$method" "$env_name" "$seed" 2>/dev/null || true)"
  if [[ -z "$run_dir" ]]; then
    echo none
    return 0
  fi

  ckpt_done="$run_dir/params_${ONLINE_DONE_EPOCH}.pkl"
  if [[ -f "$ckpt_done" ]]; then
    echo "done::$run_dir"
    return 0
  fi

  max_ckpt="$(find "$run_dir" -maxdepth 1 -type f -name 'params_*.pkl' | sed -E 's/.*params_([0-9]+)\.pkl/\1/' | sort -n | tail -n 1 || true)"
  if [[ -n "$max_ckpt" && "$max_ckpt" != "${ONLINE_DONE_EPOCH}" ]]; then
    echo "partial::${run_dir}::${max_ckpt}"
    return 0
  fi

  eval_csv="$run_dir/eval.csv"
  if [[ -f "$eval_csv" ]]; then
    echo "started::$run_dir"
    return 0
  fi

  echo none
}

write_part_from_existing_done_run() {
  local run_dir="$1"
  local scope="$2"
  local method="$3"
  local env_name="$4"
  local seed="$5"
  local fidec_t="$6"
  local agent_cfg="$7"
  local part="$8"
  local start_eval_csv="$9"

  local eval_csv group
  eval_csv="$run_dir/eval.csv"
  group="$(basename "$(dirname "$run_dir")")"

  printf 'scope\tmetric_group\tcategory_group\tmethod\tenv_name\tseed\tfidec_t\tagent_config\trun_group\tsave_dir\teval_csv\tstart_eval_csv\tstart_steps\tend_steps\n' > "$part"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scope" "$(get_online_metric_group "$scope")" "$scope" "$method" "$env_name" "$seed" "${fidec_t:-na}" "$agent_cfg" "$group" "$run_dir" "$eval_csv" "$start_eval_csv" "1000000" "1000000" >> "$part"
}

OFFLINE_MANIFEST="$(resolve_manifest)"
echo "[INFO] resolved OFFLINE_MANIFEST=$OFFLINE_MANIFEST"
echo "[INFO] RESUME_MODE=$RESUME_MODE"
if [[ -n "$ONLINE_RESUME_TAG" ]]; then
  echo "[INFO] ONLINE_RESUME_TAG=$ONLINE_RESUME_TAG"
fi

launch_task() {
  local seed="$1"
  local method="$2"
  local env_name="$3"
  local agent_cfg="$4"
  local fidec_t="$5"

  local restore_dir ckpt gpu env_slug group logfile part
  local existing_status run_dir partial_epoch

  restore_dir="$(lookup_restore_dir "$OFFLINE_MANIFEST" "$method" "$env_name" "$seed")"
  ckpt="$restore_dir/params_${RESTORE_EPOCH}.pkl"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] missing offline checkpoint: $ckpt" >&2
    exit 1
  fi

  env_slug="$(safe_slug "$env_name")"
  group="${RUN_TAG}__${method}__s${seed}__${env_slug}"
  logfile="$LOG_DIR/${method}__s${seed}__${env_slug}.log"
  part="$TMP_MANIFEST_DIR/${method}__s${seed}__${env_slug}.tsv"

  if [[ -n "$ONLINE_RESUME_TAG" ]]; then
    existing_status="$(inspect_existing_online_job "$ONLINE_RESUME_TAG" "$method" "$env_name" "$seed")"
    case "$existing_status" in
      done::*)
        run_dir="${existing_status#done::}"
        echo "[SKIP-DONE] method=$method seed=$seed env=$env_name run_dir=$run_dir"
        write_part_from_existing_done_run "$run_dir" "$SCOPE" "$method" "$env_name" "$seed" "$fidec_t" "$agent_cfg" "$part" "$restore_dir/eval.csv"
        return 0
        ;;
      partial::*)
        partial_epoch="$(echo "$existing_status" | awk -F'::' '{print $3}')"
        echo "[FOUND-PARTIAL] method=$method seed=$seed env=$env_name partial_epoch=$partial_epoch"
        echo "[RESTART-FROM-OFFLINE] exact online resume is unsupported by current code; restarting from offline checkpoint."
        ;;
      started::*)
        echo "[FOUND-STARTED] method=$method seed=$seed env=$env_name existing online dir found but no final checkpoint; restarting from offline checkpoint."
        ;;
      none)
        ;;
    esac
  fi

  get_env_flags_and_steps "$env_name"
  gpu="$(wait_for_gpu_slot)"

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
      --offline_steps 0 \
      --online_steps "$ONLINE_STEPS" \
      --restore_path "$restore_dir" \
      --restore_epoch "$RESTORE_EPOCH" \
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

    printf 'scope\tmetric_group\tcategory_group\tmethod\tenv_name\tseed\tfidec_t\tagent_config\trun_group\tsave_dir\teval_csv\tstart_eval_csv\tstart_steps\tend_steps\n' > "$part"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$SCOPE" "$(get_online_metric_group "$SCOPE")" "$SCOPE" "$method" "$env_name" "$seed" "${fidec_t:-na}" "$agent_cfg" "$group" "$latest_run" "$eval_csv" "$restore_dir/eval.csv" "1000000" "1000000" >> "$part"
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
  echo "[ERROR] at least one online job failed" >&2
  exit 1
fi

printf 'scope\tmetric_group\tcategory_group\tmethod\tenv_name\tseed\tfidec_t\tagent_config\trun_group\tsave_dir\teval_csv\tstart_eval_csv\tstart_steps\tend_steps\n' > "$MANIFEST"
shopt -s nullglob
parts=("$TMP_MANIFEST_DIR"/*.tsv)
if (( ${#parts[@]} == 0 )); then
  echo "[ERROR] no part manifests found" >&2
  exit 1
fi
awk 'FNR>1' "${parts[@]}" >> "$MANIFEST"

"$PYTHON_BIN" scripts/summarize_fql_scope.py --manifest "$MANIFEST" --out_dir "$OUT_ROOT/summary" --mode online

echo "[OK] online finished"
echo "[OK] manifest: $MANIFEST"

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  (shutdown -h now || poweroff || sudo shutdown -h now || sudo poweroff)
fi
