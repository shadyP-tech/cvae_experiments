#!/usr/bin/env bash
set -euo pipefail

# Run compact aggregation ablation sweep:
# 4 aggregation modes x 3 seeds = 12 runs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$HOME/.cache/huggingface/hub" "$HOME/.cache/torch/hub/checkpoints"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"
export TORCH_HOME="$HOME/.cache/torch"

SEEDS=(42 43 44)
CONFIGS=(
  "configs/experiments/breakhis/hybrid_ablation_extractor_resnet50_v1_top1_hard.yaml"
  "configs/experiments/breakhis/hybrid_ablation_extractor_resnet50_v1_topk2_uniform.yaml"
  "configs/experiments/breakhis/hybrid_ablation_extractor_resnet50_v1_topk3_uniform.yaml"
  "configs/experiments/breakhis/hybrid_ablation_extractor_resnet50_v1_soft_all_softmax.yaml"
)

MANIFEST="results/comparison_tables/aggregation_ablation_run_manifest.txt"
mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"

echo "Starting aggregation ablation sweep (12 runs)."

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[RUN] config=${cfg} seed=${seed}"
    python3 -m src.run_experiment --config "$cfg" --seed "$seed"

    report_csv="$(python3 - "$cfg" <<'PY'
import pathlib
import yaml
import sys

cfg_path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
root = pathlib.Path(cfg.get("output", {}).get("root", "outputs"))
dataset = str(cfg["experiment"]["dataset_name"])
exp_name = str(cfg["experiment"]["name"])
latest = (root / dataset / exp_name / "latest.txt").read_text(encoding="utf-8").strip()
print(root / dataset / exp_name / latest / "reports" / "hybrid_variant_comparison.csv")
PY
)"

    echo "$report_csv" >> "$MANIFEST"
    echo "[MANIFEST] $report_csv"
  done
done

echo "Completed aggregation ablation sweep."
echo "Run manifest: $MANIFEST"
