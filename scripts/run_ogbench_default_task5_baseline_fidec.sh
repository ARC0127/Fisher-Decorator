#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WANDB_MODE="${WANDB_MODE:-offline}" \
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}" \
PRESET="${PRESET:-default_task5}" \
BASE_AGENT="${BASE_AGENT:-agents/deflow.py}" \
VARIANT_AGENT="${VARIANT_AGENT:-agents/fidec.py}" \
BASE_METHOD_NAME="${BASE_METHOD_NAME:-deflow}" \
VARIANT_METHOD_NAME="${VARIANT_METHOD_NAME:-fidec}" \
VARIANT_EXTRA="${VARIANT_EXTRA:---agent.fisher_t=0.90 --agent.fisher_t_width=0.05 --agent.num_fisher_samples=8 --agent.fisher_damping=1e-3}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
 bash scripts/run_ogbench_state5_baseline_variant.sh
