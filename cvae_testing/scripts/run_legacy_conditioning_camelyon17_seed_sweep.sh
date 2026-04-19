#!/usr/bin/env bash
set -euo pipefail

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
BASELINE_CONFIG="configs/experiments/camelyon17/routed_cvae_v1.yaml"
CONDITIONED_CONFIG="configs/experiments/camelyon17/routed_cvae_metadata_cond_v1.yaml"

BASELINE_MANIFEST="results/comparison_tables/legacy_conditioning_camelyon17_baseline_manifest.txt"
CONDITIONED_MANIFEST="results/comparison_tables/legacy_conditioning_camelyon17_conditioned_manifest.txt"

mkdir -p "$(dirname "$BASELINE_MANIFEST")"
: > "$BASELINE_MANIFEST"
: > "$CONDITIONED_MANIFEST"

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

echo "Starting Camelyon17 legacy conditioning sweep (baseline + conditioned, 3 seeds)."
for seed in "${SEEDS[@]}"; do
	echo "[RUN] baseline config=${BASELINE_CONFIG} seed=${seed}"
	python -m src.run_experiment --config "$BASELINE_CONFIG" --seed "$seed"
	baseline_report="$(resolve_routing_results_json "$BASELINE_CONFIG")"
	echo "$baseline_report" >> "$BASELINE_MANIFEST"
	echo "[MANIFEST baseline] $baseline_report"

	echo "[RUN] conditioned config=${CONDITIONED_CONFIG} seed=${seed}"
	python -m src.run_experiment --config "$CONDITIONED_CONFIG" --seed "$seed"
	conditioned_report="$(resolve_routing_results_json "$CONDITIONED_CONFIG")"
	echo "$conditioned_report" >> "$CONDITIONED_MANIFEST"
	echo "[MANIFEST conditioned] $conditioned_report"
done

python scripts/build_legacy_conditioning_decision_table.py \
	--baseline-manifest "$BASELINE_MANIFEST" \
	--conditioned-manifest "$CONDITIONED_MANIFEST" \
	--output-csv results/comparison_tables/legacy_conditioning_camelyon17_decision_table.csv \
	--output-json results/comparison_tables/legacy_conditioning_camelyon17_decision_summary.json \
	--output-md results/summaries/legacy_conditioning_camelyon17_decision_table.md

echo "Completed Camelyon17 legacy conditioning sweep."
echo "Baseline manifest: $BASELINE_MANIFEST"
echo "Conditioned manifest: $CONDITIONED_MANIFEST"
