#!/usr/bin/env bash

get_state_ogbench_flags() {
  local env_name="$1"
  ENV_FLAGS=(--agent.normalize_q_loss=True)
  case "$env_name" in
    antmaze-large-navigate-singletask-task1-v0|antmaze-large-navigate-singletask-v0)
      ENV_FLAGS+=(--agent.q_agg=min --agent.target_divergence=0.01)
      ;;
    antmaze-giant-navigate-singletask-task1-v0|antmaze-giant-navigate-singletask-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.q_agg=min --agent.target_divergence=0.01)
      ;;
    humanoidmaze-medium-navigate-singletask-task1-v0|humanoidmaze-medium-navigate-singletask-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.target_divergence=0.001)
      ;;
    humanoidmaze-large-navigate-singletask-task1-v0|humanoidmaze-large-navigate-singletask-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.target_divergence=0.001)
      ;;
    antsoccer-arena-navigate-singletask-task4-v0|antsoccer-arena-navigate-singletask-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.target_divergence=0.01)
      ;;
    cube-single-play-singletask-task2-v0|cube-single-play-singletask-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.001)
      ;;
    cube-double-play-singletask-task2-v0|cube-double-play-singletask-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.001)
      ;;
    scene-play-singletask-task2-v0|scene-play-singletask-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.001)
      ;;
    puzzle-3x3-play-singletask-task4-v0|puzzle-3x3-play-singletask-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.005)
      ;;
    puzzle-4x4-play-singletask-task4-v0|puzzle-4x4-play-singletask-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.005)
      ;;
    *)
      echo "[ERROR] Unrecognized or unsupported state-based OGBench env for exact paper flags: $env_name" >&2
      return 1
      ;;
  esac
}

get_state_ogbench_eval_steps_csv() {
  echo "800000,900000,1000000"
}
