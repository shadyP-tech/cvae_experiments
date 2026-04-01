#!/usr/bin/env bash
set -euo pipefail

# Run the protocol-locked BreakHis learned utility routing sweep.
# Scope: 3 seeds x 1 backbone (resnet50) x Variant B protocol config.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="/home/stud/spark/.venvs/cvae-breakhis/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] Expected venv Python not found at ${VENV_PYTHON}; falling back to python3"
  VENV_PYTHON="python3"
fi

mkdir -p "$HOME/.cache/huggingface/hub" "$HOME/.cache/torch/hub/checkpoints"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"
export TORCH_HOME="$HOME/.cache/torch"

SEEDS=(42 43 44)
CONFIG="configs/experiments/breakhis/learned_utility_routing_v1.yaml"

echo "Starting learned utility routing seed sweep (3 runs)."
for seed in "${SEEDS[@]}"; do
  echo "[RUN] config=${CONFIG} seed=${seed}"
  "$VENV_PYTHON" -m src.run_experiment --config "$CONFIG" --seed "$seed"
done

echo "Completed learned utility routing seed sweep."
