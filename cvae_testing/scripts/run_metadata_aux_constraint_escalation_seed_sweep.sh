#!/usr/bin/env bash
set -euo pipefail

# Run one-step metadata auxiliary-constraint escalation experiments with thesis-canonical seeds.
# Scope: 2 datasets x 1 escalated constrained config each x 3 seeds = 6 runs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_ACTIVATE="$HOME/.venvs/cvae-breakhis/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
else
  echo "[warn] Expected activate script not found at $VENV_ACTIVATE; using system python3."
fi

mkdir -p "$HOME/.cache/huggingface/hub" "$HOME/.cache/torch/hub/checkpoints"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"
export TORCH_HOME="$HOME/.cache/torch"

SEEDS=(11 42 73)
CONFIGS=(
  "configs/experiments/breakhis/routed_cvae_metadata_aux_constraint_v1_escalated.yaml"
  "configs/experiments/camelyon17/routed_cvae_metadata_aux_constraint_v1_escalated.yaml"
)

BREAKHIS_MANIFEST="results/comparison_tables/legacy_conditioning_breakhis_aux_constraint_escalated_manifest.txt"
CAMELYON_MANIFEST="results/comparison_tables/legacy_conditioning_camelyon17_aux_constraint_escalated_manifest.txt"
mkdir -p "$(dirname "$BREAKHIS_MANIFEST")"
: > "$BREAKHIS_MANIFEST"
: > "$CAMELYON_MANIFEST"

resolve_routing_results_json() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import pathlib
import sys
import yaml

cfg_path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
root = pathlib.Path(cfg.get('output', {}).get('root', 'outputs'))
dataset = str(cfg['experiment']['dataset_name'])
exp_name = str(cfg['experiment']['name'])
latest = (root / dataset / exp_name / 'latest.txt').read_text(encoding='utf-8').strip()
print(root / dataset / exp_name / latest / 'reports' / 'routing_results.json')
PY
}

resolve_dataset_name() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import pathlib
import sys
import yaml

cfg_path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
print(str(cfg['experiment']['dataset_name']))
PY
}

echo "Starting metadata auxiliary-constraint escalation seed sweep (6 runs)."
for cfg in "${CONFIGS[@]}"; do
  dataset_name="$(resolve_dataset_name "$cfg")"
  for seed in "${SEEDS[@]}"; do
    echo "[RUN] config=${cfg} dataset=${dataset_name} seed=${seed}"
    python -m src.run_experiment --config "$cfg" --seed "$seed"

    report_json="$(resolve_routing_results_json "$cfg")"
    if [[ "$dataset_name" == "breakhis" ]]; then
      echo "$report_json" >> "$BREAKHIS_MANIFEST"
    elif [[ "$dataset_name" == "camelyon17" ]]; then
      echo "$report_json" >> "$CAMELYON_MANIFEST"
    else
      echo "[warn] Unknown dataset_name=${dataset_name}; skipping manifest append for report=${report_json}"
    fi
    echo "[MANIFEST ${dataset_name}] $report_json"
  done
done

echo "Completed metadata auxiliary-constraint escalation sweep."
echo "BreakHis manifest: $BREAKHIS_MANIFEST"
echo "Camelyon17 manifest: $CAMELYON_MANIFEST"
