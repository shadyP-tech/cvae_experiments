#!/usr/bin/env bash
set -euo pipefail

# Run the controlled backbone ablation matrix:
# 2 datasets x 3 backbones x 3 seeds = 18 total runs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$HOME/.cache/huggingface/hub" "$HOME/.cache/torch/hub/checkpoints"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"
export TORCH_HOME="$HOME/.cache/torch"

SEEDS=(42 43 44)
CONFIGS=(
  "configs/experiments/breakhis/hybrid_ablation_extractor_resnet18_v1.yaml"
  "configs/experiments/breakhis/hybrid_ablation_extractor_resnet50_v1.yaml"
  "configs/experiments/breakhis/hybrid_ablation_extractor_dinov2_vitb14_v1.yaml"
  "configs/experiments/camelyon17/hybrid_ablation_extractor_resnet18_v1.yaml"
  "configs/experiments/camelyon17/hybrid_ablation_extractor_resnet50_v1.yaml"
  "configs/experiments/camelyon17/hybrid_ablation_extractor_dinov2_vitb14_v1.yaml"
)

echo "Starting controlled backbone ablation sweep (18 runs)."

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[RUN] config=${cfg} seed=${seed}"
    python3 -m src.run_experiment --config "$cfg" --seed "$seed"
  done
done

echo "Completed 18-run backbone ablation sweep."
