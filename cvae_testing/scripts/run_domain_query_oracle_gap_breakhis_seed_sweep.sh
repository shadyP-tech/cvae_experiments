#!/usr/bin/env bash
set -euo pipefail

# Run fixed-domain vs per-query oracle-gap diagnostics on the locked BreakHis
# LOQDO seed/backbone matrix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${VENV_PYTHON:-/home/stud/spark/.venvs/cvae-breakhis/bin/python}"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] Expected venv Python not found at ${VENV_PYTHON}; falling back to python3"
  VENV_PYTHON="python3"
fi

SEEDS=(42 43 44)
BACKBONES=(resnet18 resnet50 dinov2_vitb14)
VARIANT="${VARIANT:-B}"
DATASET="breakhis"
BATCH_SIZE="${BATCH_SIZE:-2048}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-1337}"

RAW_OUT="results/comparison_tables/domain_query_oracle_gap_loqdo_${DATASET}_raw.csv"
PER_SAMPLE_OUT="results/comparison_tables/domain_query_oracle_gap_loqdo_${DATASET}_per_sample.csv"
STATS_OUT="results/comparison_tables/domain_query_oracle_gap_loqdo_${DATASET}_stats.csv"
SUMMARY_OUT="results/comparison_tables/domain_query_oracle_gap_loqdo_${DATASET}_summary.json"
SUMMARY_MD_OUT="results/summaries/domain_query_oracle_gap_loqdo_${DATASET}_summary.md"
MANIFEST="results/comparison_tables/domain_query_oracle_gap_run_manifest_${DATASET}_loqdo.txt"

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

echo "[run] dataset=${DATASET} runs=${#RUN_DIRS[@]} variant=${VARIANT}"
"$VENV_PYTHON" scripts/run_domain_query_oracle_gap_loqdo.py \
  --experiment-dirs "${RUN_DIRS[@]}" \
  --variant "$VARIANT" \
  --batch-size "$BATCH_SIZE" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --raw-out "$RAW_OUT" \
  --per-sample-out "$PER_SAMPLE_OUT" \
  --stats-out "$STATS_OUT" \
  --summary-json-out "$SUMMARY_OUT" \
  --summary-md-out "$SUMMARY_MD_OUT"

{
  echo "$ROOT_DIR/$RAW_OUT"
  echo "$ROOT_DIR/$PER_SAMPLE_OUT"
  echo "$ROOT_DIR/$STATS_OUT"
  echo "$ROOT_DIR/$SUMMARY_OUT"
  echo "$ROOT_DIR/$SUMMARY_MD_OUT"
} >> "$MANIFEST"

echo "Completed BreakHis fixed-domain vs per-query oracle-gap sweep."
echo "Raw folds: $RAW_OUT"
echo "Per-sample rows: $PER_SAMPLE_OUT"
echo "Stats: $STATS_OUT"
echo "Summary JSON: $SUMMARY_OUT"
echo "Summary Markdown: $SUMMARY_MD_OUT"
echo "Manifest: $MANIFEST"
