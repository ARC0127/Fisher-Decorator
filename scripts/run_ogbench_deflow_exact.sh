#!/usr/bin/env bash
set -e

# Exact OGBench state-based commands mirrored from README.
# Use one command at a time.

# antmaze-large
# python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.q_agg=min --agent.target_divergence=0.01 --agent.normalize_q_loss=True

# antmaze-giant
# python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_agg=min --agent.target_divergence=0.01 --agent.normalize_q_loss=True

# humanoidmaze-medium
# python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.target_divergence=0.001 --agent.normalize_q_loss=True

# humanoidmaze-large
# python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.target_divergence=0.001 --agent.normalize_q_loss=True

# antsoccer
# python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.target_divergence=0.01 --agent.normalize_q_loss=True

# cube-single
# python main.py --env_name=cube-single-play-singletask-v0 --agent.target_divergence=0.001 --agent.normalize_q_loss=True

# cube-double
# python main.py --env_name=cube-double-play-singletask-v0 --agent.target_divergence=0.001 --agent.normalize_q_loss=True

# scene
# python main.py --env_name=scene-play-singletask-v0 --agent.target_divergence=0.001 --agent.normalize_q_loss=True

# puzzle-3x3
# python main.py --env_name=puzzle-3x3-play-singletask-v0 --agent.target_divergence=0.005 --agent.normalize_q_loss=True

# puzzle-4x4
# python main.py --env_name=puzzle-4x4-play-singletask-v0 --agent.target_divergence=0.005 --agent.normalize_q_loss=True
