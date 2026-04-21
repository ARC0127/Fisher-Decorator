#!/usr/bin/env bash

get_scope_envs() {
  local scope="$1"
  case "$scope" in
    ogbench_state_all50)
      local fams=(
        antmaze-large-navigate
        antmaze-giant-navigate
        humanoidmaze-medium-navigate
        humanoidmaze-large-navigate
        antsoccer-arena-navigate
        cube-single-play
        cube-double-play
        scene-play
        puzzle-3x3-play
        puzzle-4x4-play
      )
      local fam
      for fam in "${fams[@]}"; do
        for t in 1 2 3 4 5; do
          echo "${fam}-singletask-task${t}-v0"
        done
      done
      ;;
    ogbench_pixel_default5)
      cat <<'EOF'
visual-cube-single-play-singletask-task1-v0
visual-cube-double-play-singletask-task1-v0
visual-scene-play-singletask-task1-v0
visual-puzzle-3x3-play-singletask-task1-v0
visual-puzzle-4x4-play-singletask-task1-v0
EOF
      ;;
    d4rl_antmaze6|o2o_d4rl_antmaze6)
      cat <<'EOF'
antmaze-umaze-v2
antmaze-umaze-diverse-v2
antmaze-medium-play-v2
antmaze-medium-diverse-v2
antmaze-large-play-v2
antmaze-large-diverse-v2
EOF
      ;;
    d4rl_adroit12)
      cat <<'EOF'
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
EOF
      ;;
    o2o_ogbench5)
      cat <<'EOF'
humanoidmaze-medium-navigate-singletask-task1-v0
antsoccer-arena-navigate-singletask-task4-v0
cube-double-play-singletask-task2-v0
scene-play-singletask-task2-v0
puzzle-4x4-play-singletask-task4-v0
EOF
      ;;
    o2o_d4rl_adroit4)
      cat <<'EOF'
pen-cloned-v1
door-cloned-v1
hammer-cloned-v1
relocate-cloned-v1
EOF
      ;;
    *)
      echo "[ERROR] unknown scope=$scope" >&2
      return 1
      ;;
  esac
}

get_env_flags_and_steps() {
  local env_name="$1"

  ENV_FLAGS=()
  OFFLINE_STEPS=1000000
  PAPER_EVAL_STEPS="800000,900000,1000000"
  METRIC_GROUP="success"
  CATEGORY_GROUP="ogbench_state"

  case "$env_name" in
    antmaze-large-navigate-singletask-task*-v0)
      ENV_FLAGS+=(--agent.q_agg=min --agent.target_divergence=0.01)
      ;;
    antmaze-giant-navigate-singletask-task*-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.q_agg=min --agent.target_divergence=0.01)
      ;;
    humanoidmaze-medium-navigate-singletask-task*-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.target_divergence=0.001)
      ;;
    humanoidmaze-large-navigate-singletask-task*-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.target_divergence=0.001)
      ;;
    antsoccer-arena-navigate-singletask-task*-v0)
      ENV_FLAGS+=(--agent.discount=0.995 --agent.target_divergence=0.01)
      ;;
    cube-single-play-singletask-task*-v0|cube-double-play-singletask-task*-v0|scene-play-singletask-task*-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.001)
      ;;
    puzzle-3x3-play-singletask-task*-v0|puzzle-4x4-play-singletask-task*-v0)
      ENV_FLAGS+=(--agent.target_divergence=0.005)
      ;;

    visual-cube-single-play-singletask-task1-v0|visual-cube-double-play-singletask-task1-v0|visual-scene-play-singletask-task1-v0)
      OFFLINE_STEPS=500000
      PAPER_EVAL_STEPS="300000,400000,500000"
      METRIC_GROUP="success"
      CATEGORY_GROUP="ogbench_pixel"
      ENV_FLAGS+=(--agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --agent.target_divergence=0.001)
      ;;
    visual-puzzle-3x3-play-singletask-task1-v0|visual-puzzle-4x4-play-singletask-task1-v0)
      OFFLINE_STEPS=500000
      PAPER_EVAL_STEPS="300000,400000,500000"
      METRIC_GROUP="success"
      CATEGORY_GROUP="ogbench_pixel"
      ENV_FLAGS+=(--agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --agent.target_divergence=0.005)
      ;;

    antmaze-umaze-v2|antmaze-umaze-diverse-v2|antmaze-medium-play-v2|antmaze-medium-diverse-v2|antmaze-large-play-v2|antmaze-large-diverse-v2)
      OFFLINE_STEPS=500000
      PAPER_EVAL_STEPS="500000"
      METRIC_GROUP="d4rl_antmaze"
      CATEGORY_GROUP="d4rl_antmaze"
      ENV_FLAGS+=(--agent.target_divergence=0.001)
      ;;

    pen-human-v1|pen-cloned-v1|pen-expert-v1|door-human-v1|door-cloned-v1|door-expert-v1|hammer-human-v1|hammer-cloned-v1|hammer-expert-v1|relocate-human-v1|relocate-cloned-v1|relocate-expert-v1)
      OFFLINE_STEPS=500000
      PAPER_EVAL_STEPS="500000"
      METRIC_GROUP="d4rl_adroit"
      CATEGORY_GROUP="d4rl_adroit"
      ENV_FLAGS+=(--agent.q_agg=min --agent.target_divergence=0.001)
      ;;
    *)
      echo "[ERROR] unhandled env=$env_name" >&2
      return 1
      ;;
  esac
}

get_online_metric_group() {
  local scope="$1"
  case "$scope" in
    o2o_ogbench5|o2o_d4rl_antmaze6) echo "success" ;;
    o2o_d4rl_adroit4) echo "d4rl_adroit" ;;
    *) echo "[ERROR] unknown online scope=$scope" >&2; return 1 ;;
  esac
}
