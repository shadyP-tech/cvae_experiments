#!/usr/bin/env bash
set -euo pipefail

# Run BreakHis LOQDO support-set utility calibration over the locked seed/backbone matrix.

SUPPORT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${SUPPORT_ROOT}/.." && pwd)"
CVAE_TESTING_ROOT="${REPO_ROOT}/cvae_testing"
cd "$CVAE_TESTING_ROOT"

VENV_PYTHON="/home/stud/spark/.venvs/cvae-breakhis/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] Expected venv Python not found at ${VENV_PYTHON}; falling back to python3"
  VENV_PYTHON="python3"
fi

SEEDS=(42 43 44)
BACKBONES=(resnet18 resnet50 dinov2_vitb14)
VARIANT="B"
DATASET="breakhis"

RAW_OUT="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_loqdo_${DATASET}_raw.csv"
STATS_OUT="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_loqdo_${DATASET}_stats.csv"
SUMMARY_OUT="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_loqdo_${DATASET}_summary.json"
DECISION_OUT="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_loqdo_${DATASET}_decision.csv"
PAIRED_OUT="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_loqdo_${DATASET}_paired_deltas.csv"
DECISION_SUMMARY_OUT="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_loqdo_${DATASET}_decision_summary.json"
MANIFEST="${SUPPORT_ROOT}/artifacts/comparison_tables/support_set_calibration_run_manifest_${DATASET}_loqdo.txt"

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
"$VENV_PYTHON" "${SUPPORT_ROOT}/scripts/run/run_support_set_calibration_loqdo.py" \
  --experiment-dirs "${RUN_DIRS[@]}" \
  --variant "$VARIANT" \
  --raw-out "$RAW_OUT" \
  --stats-out "$STATS_OUT" \
  --summary-json-out "$SUMMARY_OUT"

"$VENV_PYTHON" "${SUPPORT_ROOT}/scripts/reports/build_support_set_calibration_decision_table.py" \
  --raw "$RAW_OUT" \
  --out "$DECISION_OUT" \
  --paired-out "$PAIRED_OUT" \
  --summary-json-out "$DECISION_SUMMARY_OUT"

{
  echo "$SUMMARY_OUT"
  echo "$DECISION_SUMMARY_OUT"
} >> "$MANIFEST"

echo "Completed BreakHis support-set calibration sweep."
echo "Summary: $SUMMARY_OUT"
echo "Decision: $DECISION_OUT"
echo "Manifest: $MANIFEST"
