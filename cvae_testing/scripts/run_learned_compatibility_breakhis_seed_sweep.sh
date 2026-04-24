#!/usr/bin/env bash
set -euo pipefail

# Run LOQDO compatibility analysis on BreakHis with locked seed/backbone matrix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="/home/stud/spark/.venvs/cvae-breakhis/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] Expected venv Python not found at ${VENV_PYTHON}; falling back to python3"
  VENV_PYTHON="python3"
fi

SEEDS=(42 43 44)
BACKBONES=(resnet18 resnet50 dinov2_vitb14)
VARIANT="B"
DATASET="breakhis"
ORACLE_PROBES="${ORACLE_PROBES:-0}"
SEMI_ORACLE_RISK_LAMBDA="${SEMI_ORACLE_RISK_LAMBDA:-}"

RAW_OUT="results/comparison_tables/learned_compatibility_loqdo_${DATASET}_raw.csv"
STATS_OUT="results/comparison_tables/learned_compatibility_loqdo_${DATASET}_stats.csv"
SUMMARY_OUT="results/comparison_tables/learned_compatibility_loqdo_${DATASET}_summary.json"
MANIFEST="results/comparison_tables/compatibility_run_manifest_${DATASET}_loqdo.txt"

RUN_DIRS=()
shopt -s nullglob
for backbone in "${BACKBONES[@]}"; do
  exp_dir="outputs/${DATASET}/hybrid_ablation_extractor_${backbone}_v1"
  if [[ ! -d "$exp_dir" ]]; then
    echo "[error] Missing experiment directory: ${exp_dir}" >&2
    exit 1
  fi

  for seed in "${SEEDS[@]}"; do
    matches=("${exp_dir}"/*_seed"${seed}")
    if [[ ${#matches[@]} -eq 0 ]]; then
      echo "[error] No run directory found for ${exp_dir} seed=${seed}" >&2
      exit 1
    fi
    IFS=$'\n' sorted=( $(printf '%s\n' "${matches[@]}" | sort) )
    unset IFS
    RUN_DIRS+=("${sorted[-1]}")
  done
done
shopt -u nullglob

mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"

echo "[run] dataset=${DATASET} runs=${#RUN_DIRS[@]}"
EXTRA_ARGS=()
if [[ "$ORACLE_PROBES" != "1" ]]; then
  EXTRA_ARGS+=(--skip-oracle-probes)
fi
if [[ -n "$SEMI_ORACLE_RISK_LAMBDA" ]]; then
  EXTRA_ARGS+=(--semi-oracle-risk-lambda "$SEMI_ORACLE_RISK_LAMBDA")
fi

"$VENV_PYTHON" scripts/run_learned_compatibility_loqdo.py \
  --experiment-dirs "${RUN_DIRS[@]}" \
  --variant "$VARIANT" \
  --raw-out "$RAW_OUT" \
  --stats-out "$STATS_OUT" \
  --summary-json-out "$SUMMARY_OUT" \
  "${EXTRA_ARGS[@]}"

echo "$ROOT_DIR/$SUMMARY_OUT" >> "$MANIFEST"

echo "Completed BreakHis compatibility sweep."
echo "Summary: $SUMMARY_OUT"
echo "Manifest: $MANIFEST"
