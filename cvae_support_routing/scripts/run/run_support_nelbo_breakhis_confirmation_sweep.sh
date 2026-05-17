#!/usr/bin/env bash
set -euo pipefail

# Run the BreakHis magnification-domain stress test for direct support-NELBO.

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

CONFIG="${SUPPORT_ROOT}/configs/experiments/breakhis/breakhis_support_estimated_utility_routing_v1.yaml"
EXPERIMENT_ROOT="outputs/breakhis/breakhis_support_estimated_utility_routing_v1"
RESULTS_DIR="${SUPPORT_ROOT}/artifacts/comparison_tables"
PREFIX="breakhis_support_estimated_utility_routing_v1"
MANIFEST="${RESULTS_DIR}/${PREFIX}_run_manifest.txt"
PREFLIGHT_OUT="${RESULTS_DIR}/${PREFIX}_preflight.json"

SEEDS=(42 43 44)
RUN_ID_TEMPLATE="support_utility_v1_seed{seed}"

mkdir -p "$RESULTS_DIR"
: > "$MANIFEST"

echo "[preflight] config=${CONFIG}"
"$VENV_PYTHON" - "$CONFIG" "$PREFLIGHT_OUT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from src.config.load_config import load_config
from src.config.schema import validate_config
from src.data.registry import prepare_dataset_records


project_root = Path.cwd()
config_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

cfg = load_config(config_path)
validate_config(cfg)

configured = [int(v) for v in cfg["data"]["magnifications"]]
if configured != [40, 100, 200, 400]:
    raise SystemExit(f"Configured BreakHis domains must be exactly [40, 100, 200, 400], got {configured}")

records, leakage = prepare_dataset_records(project_root, cfg)
observed = sorted({int(getattr(rec, "magnification")) for rec in records})
if observed != [40, 100, 200, 400]:
    raise SystemExit(f"Observed normalized BreakHis domains must be exactly [40, 100, 200, 400], got {observed}")

missing_patient = [getattr(rec, "image_path") for rec in records if not str(getattr(rec, "patient_id", "") or "").strip()]
if missing_patient:
    raise SystemExit(f"Missing or unparseable patient IDs for {len(missing_patient)} images; refusing thesis-facing run.")

paths = [str(getattr(rec, "image_path")) for rec in records]
if len(paths) != len(set(paths)):
    raise SystemExit("Duplicate image paths found after BreakHis preflight preparation.")

patient_by_split = {"train": set(), "val": set(), "test": set()}
for rec in records:
    patient_by_split[str(getattr(rec, "split"))].add(str(getattr(rec, "patient_id")))

split_overlaps = {
    "train_val": sorted(patient_by_split["train"].intersection(patient_by_split["val"])),
    "train_test": sorted(patient_by_split["train"].intersection(patient_by_split["test"])),
    "val_test": sorted(patient_by_split["val"].intersection(patient_by_split["test"])),
}
if any(split_overlaps.values()):
    raise SystemExit(f"Patient overlap across train/val/test detected: {split_overlaps}")

max_support = max(int(v) for v in cfg["learned_utility"]["support_response_routing"]["support_sizes"])
target_splits = {str(v) for v in cfg.get("learned_utility", {}).get("splits", ["test"])}
target_patient_counts = {}
target_image_counts = {}
for domain in configured:
    target_records = [
        rec
        for rec in records
        if int(getattr(rec, "magnification")) == domain
        and str(getattr(rec, "split")) in target_splits
    ]
    patient_count = len({str(getattr(rec, "patient_id")) for rec in target_records})
    target_patient_counts[str(domain)] = patient_count
    target_image_counts[str(domain)] = len(target_records)
    if patient_count < max_support + 1:
        raise SystemExit(
            f"Held-out magnification {domain} has {patient_count} target patients; "
            f"need at least k+1={max_support + 1} for patient-disjoint support/eval."
        )

normalization_rows = sorted(
    {
        (str(getattr(rec, "domain_name")), int(getattr(rec, "magnification")))
        for rec in records
    },
    key=lambda item: item[1],
)
payload = {
    "status": "pass",
    "configured_domains": configured,
    "observed_normalized_domains": observed,
    "domain_normalization": [
        {"raw_domain_name": raw, "normalized_magnification": normalized}
        for raw, normalized in normalization_rows
    ],
    "n_records": len(records),
    "target_splits": sorted(target_splits),
    "target_patient_counts_by_magnification": target_patient_counts,
    "target_image_counts_by_magnification": target_image_counts,
    "max_support_size": int(max_support),
    "patient_id_coverage_pct": 100.0,
    "split_patient_overlap": split_overlaps,
    "dataset_leakage_report": leakage,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[run] starting BreakHis support-NELBO stress test"
for seed in "${SEEDS[@]}"; do
  run_id="support_utility_v1_seed${seed}"
  echo "[run] seed=${seed} run_id=${run_id}"
  "$VENV_PYTHON" -m src.run_experiment --config "$CONFIG" --seed "$seed" --run-id "$run_id"
  echo "${CVAE_TESTING_ROOT}/${EXPERIMENT_ROOT}/${run_id}/reports/learned_utility_results.json" >> "$MANIFEST"
done

echo "[report] consolidating BreakHis support-NELBO artifacts"
"$VENV_PYTHON" "${SUPPORT_ROOT}/scripts/reports/build_support_nelbo_consolidation_report.py" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --output-dir "$RESULTS_DIR" \
  --decision-table "${RESULTS_DIR}/${PREFIX}_decision_table.csv" \
  --run-seeds "42,43,44" \
  --heldout-domains "40,100,200,400" \
  --support-seeds "17,23,31" \
  --support-sizes "4,8,16,32" \
  --run-id-template "$RUN_ID_TEMPLATE" \
  --output-prefix "$PREFIX" \
  --dataset-context "breakhis"

echo "[postrun] validating generated decision artifacts"
"$VENV_PYTHON" - "$RESULTS_DIR" "$PREFIX" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])
prefix = str(sys.argv[2])

required = [
    f"{prefix}_decision_table.csv",
    f"{prefix}_seed_stability.csv",
    f"{prefix}_per_magnification_decisions.csv",
    f"{prefix}_rank_consistency.csv",
    f"{prefix}_decision_summary.json",
    f"{prefix}.md",
    f"{prefix}_protocol_audit.csv",
]
missing = [name for name in required if not (results_dir / name).exists()]
if missing:
    raise SystemExit(f"Missing required BreakHis artifacts: {missing}")

with (results_dir / f"{prefix}_decision_summary.json").open("r", encoding="utf-8") as f:
    summary = json.load(f)
counts = summary["row_count_assertions"]
expected_counts = {
    "selected_direct_conservative_rows_expected": 288,
    "selected_direct_conservative_rows_observed": 288,
    "alpha_selection_rows_expected": 48,
    "alpha_selection_rows_observed": 48,
    "raw_support_nelbo_rows_expected": 6480,
    "raw_support_nelbo_rows_observed": 6480,
}
for key, expected in expected_counts.items():
    if int(counts.get(key, -1)) != int(expected):
        raise SystemExit(f"Unexpected {key}: got {counts.get(key)}, expected {expected}")

with (results_dir / f"{prefix}_protocol_audit.csv").open("r", encoding="utf-8", newline="") as f:
    audit_rows = list(csv.DictReader(f))
if not audit_rows:
    raise SystemExit("Protocol audit is empty.")
for row in audit_rows:
    for check in [
        "support_eval_disjoint_ok",
        "support_eval_patient_disjoint_ok",
        "support_labels_unused_for_routing_ok",
        "target_expert_excluded_ok",
        "candidate_pool_excludes_target_expert_ok",
        "routing_uses_eval_nelbo_ok",
        "routing_uses_eval_domain_statistics_ok",
    ]:
        if str(row.get(check)) != "1":
            raise SystemExit(f"Protocol audit failed check {check}: {row}")

with (results_dir / f"{prefix}_decision_table.csv").open("r", encoding="utf-8", newline="") as f:
    methods = {row["source_method"] for row in csv.DictReader(f)}
required_methods = {
    "support_metadata_routing",
    "support_static_embedding_routing",
    "support_random_expert_floor",
    "support_set_nelbo_top1",
    "support_set_nelbo_conservative",
}
if not required_methods.issubset(methods):
    raise SystemExit(f"Decision table missing methods: {sorted(required_methods - methods)}")

print("postrun validation passed")
PY

echo "Completed BreakHis support-NELBO stress test."
echo "Manifest: ${MANIFEST}"
echo "Report: ${RESULTS_DIR}/${PREFIX}.md"
