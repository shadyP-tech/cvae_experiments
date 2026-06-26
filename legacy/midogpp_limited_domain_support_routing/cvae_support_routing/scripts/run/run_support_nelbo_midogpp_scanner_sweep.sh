#!/usr/bin/env bash
set -euo pipefail

# Run the MIDOG++ scanner-model diagnostic-to-thesis support-NELBO stress test.

SUPPORT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${SUPPORT_ROOT}/.." && pwd)"
CVAE_TESTING_ROOT="${REPO_ROOT}/cvae_testing"
cd "$CVAE_TESTING_ROOT"

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

CONFIG="${SUPPORT_ROOT}/configs/experiments/midogpp/midogpp_scanner_support_estimated_utility_routing_v1.yaml"
EXPERIMENT_ROOT="outputs/midogpp/midogpp_scanner_support_estimated_utility_routing_v1"
RESULTS_DIR="${SUPPORT_ROOT}/artifacts/comparison_tables"
PREFIX="midogpp_scanner_support_estimated_utility_routing_v1"
MANIFEST="${RESULTS_DIR}/${PREFIX}_run_manifest.txt"
RUN_ID_TEMPLATE="support_utility_v1_seed{seed}"
SEEDS=(42 43 44)

mkdir -p "$RESULTS_DIR"
: > "$MANIFEST"

echo "[preflight] config=${CONFIG}"
"$VENV_PYTHON" "${SUPPORT_ROOT}/scripts/preflight/preflight_midogpp_scanner.py" \
  --config "$CONFIG" \
  --output-dir "$RESULTS_DIR" \
  --output-prefix "$PREFIX" \
  --require-full-feasible

echo "[run] starting MIDOG++ scanner support-NELBO stress test"
for seed in "${SEEDS[@]}"; do
  run_id="support_utility_v1_seed${seed}"
  echo "[run] seed=${seed} run_id=${run_id}"
  "$VENV_PYTHON" -m src.run_experiment --config "$CONFIG" --seed "$seed" --run-id "$run_id"
  echo "${CVAE_TESTING_ROOT}/${EXPERIMENT_ROOT}/${run_id}/reports/learned_utility_results.json" >> "$MANIFEST"
done

echo "[report] consolidating MIDOG++ scanner support-NELBO artifacts"
heldout_domains=$("$VENV_PYTHON" - "$RESULTS_DIR/${PREFIX}_preflight.json" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
domains = sorted(int(k) for k in payload["domain_id_to_raw_scanner_label"].keys())
print(",".join(str(d) for d in domains))
PY
)

"$VENV_PYTHON" "${SUPPORT_ROOT}/scripts/reports/build_support_nelbo_consolidation_report.py" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --output-dir "$RESULTS_DIR" \
  --decision-table "${RESULTS_DIR}/${PREFIX}_decision_table.csv" \
  --run-seeds "42,43,44" \
  --heldout-domains "$heldout_domains" \
  --support-seeds "17,23,31" \
  --support-sizes "4,8,16,32" \
  --run-id-template "$RUN_ID_TEMPLATE" \
  --output-prefix "$PREFIX" \
  --dataset-context "midogpp_scanner"

echo "[postrun] validating required MIDOG++ support-NELBO artifacts"
"$VENV_PYTHON" - "$RESULTS_DIR" "$PREFIX" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])
prefix = str(sys.argv[2])

required = [
    f"{prefix}_preflight.json",
    f"{prefix}_scanner_confounding_table.csv",
    f"{prefix}_scanner_fold_feasibility.csv",
    f"{prefix}_decision_table.csv",
    f"{prefix}_seed_stability.csv",
    f"{prefix}_per_magnification_decisions.csv",
    f"{prefix}_rank_consistency.csv",
    f"{prefix}_decision_summary.json",
    f"{prefix}.md",
    f"{prefix}_protocol_audit.csv",
    f"{prefix}_support_response_selections.csv",
    f"{prefix}_raw_support_nelbo_rows.csv",
    f"{prefix}_metadata_baseline_diagnostics.csv",
    f"{prefix}_support_size_monotonicity.csv",
    f"{prefix}_margin_diagnostics.csv",
    f"{prefix}_selection_entropy.csv",
]
missing = [name for name in required if not (results_dir / name).exists()]
if missing:
    raise SystemExit(f"Missing required MIDOG++ artifacts: {missing}")

nonempty_required = [
    f"{prefix}_scanner_confounding_table.csv",
    f"{prefix}_scanner_fold_feasibility.csv",
    f"{prefix}_support_response_selections.csv",
    f"{prefix}_raw_support_nelbo_rows.csv",
    f"{prefix}_metadata_baseline_diagnostics.csv",
    f"{prefix}_support_size_monotonicity.csv",
    f"{prefix}_margin_diagnostics.csv",
    f"{prefix}_selection_entropy.csv",
]
empty = [name for name in nonempty_required if (results_dir / name).stat().st_size <= 0]
if empty:
    raise SystemExit(f"Empty required MIDOG++ diagnostic artifacts: {empty}")

summary = json.loads((results_dir / f"{prefix}_decision_summary.json").read_text(encoding="utf-8"))
protocol = summary.get("decision_layers", {}).get("protocol_validity", {})
if protocol.get("overall_protocol_validity") not in {"pass", None}:
    raise SystemExit(f"Unexpected protocol validity: {protocol}")
PY
