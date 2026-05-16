#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.feature_regimes import FEATURE_REGISTRY


SCHEMA_VERSION = "response_routing_v1"


CSV_SCHEMAS: Dict[str, List[str]] = {
    "baseline_static_dinov2_reproduction.csv": [
        "dataset_name",
        "scope",
        "seed",
        "feature_regime",
        "method",
        "top1_agreement_with_best_expert",
        "spearman_similarity_vs_neg_nelbo",
        "metadata_to_oracle_gap",
        "normalized_metadata_to_oracle_gap",
        "calibration_error_bin10",
        "no_data_reason",
    ],
    "loqdo_response_feature_table.csv": [
        "dataset_name",
        "scope",
        "seed",
        "fold_id",
        "feature_regime",
        "feature_name",
        "feature_schema_hash",
        "included",
        "dropped_zero_variance",
        "blocked",
        "no_data_reason",
    ],
    "response_indirect_decision_table.csv": [
        "dataset_name",
        "scope",
        "method_key",
        "feature_regime",
        "adoption_gate_pass_proxy",
        "veto_reason",
        "no_data_reason",
    ],
    "response_indirect_shuffled_control_table.csv": [
        "dataset_name",
        "scope",
        "method_key",
        "feature_regime",
        "control_only",
        "adoption_gate_pass_proxy",
        "veto_reason",
        "no_data_reason",
    ],
    "response_target_adjacent_diagnostic_table.csv": [
        "dataset_name",
        "scope",
        "method_key",
        "feature_regime",
        "diagnostic_only",
        "adoption_gate_pass_proxy",
        "veto_reason",
        "no_data_reason",
    ],
    "response_oracle_diagnostic_table.csv": [
        "dataset_name",
        "scope",
        "method_key",
        "feature_regime",
        "diagnostic_only",
        "adoption_gate_pass_proxy",
        "veto_reason",
        "no_data_reason",
    ],
    "response_budget_reliability_table.csv": [
        "dataset_name",
        "scope",
        "seed",
        "response_budget",
        "feature_regime",
        "metric",
        "value",
        "no_data_reason",
    ],
    "cross_dataset_response_assessment.csv": [
        "scope",
        "source_dataset",
        "target_dataset",
        "best_adoption_eligible_method",
        "classification",
        "leakage_veto_reason",
        "no_data_reason",
    ],
    "target_adjacency_audit.csv": [
        "dataset_name",
        "scope",
        "feature_regime",
        "feature_name",
        "utility_correlation",
        "risk_level",
        "no_data_reason",
    ],
    "feature_regime_audit.csv": [
        "regime",
        "adoption_eligible",
        "diagnostic_only",
        "control_only",
        "num_features",
        "feature_schema_hash",
        "num_blocked_features",
        "num_dropped_zero_variance",
        "no_data_reason",
    ],
}


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _empty_reason(input_path: Path | None) -> str:
    if input_path is None:
        return "no benchmark input path provided"
    return f"expected input file {input_path} was not found"


def build_artifact_suite(output_dir: Path, *, scope: str, input_csv: Path | None = None) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    no_data_reason = _empty_reason(input_csv) if input_csv is None or not input_csv.exists() else ""
    created_files: List[str] = []

    for filename, headers in CSV_SCHEMAS.items():
        rows: List[dict] = []
        if filename == "feature_regime_audit.csv":
            rows = [
                {
                    "regime": regime.name,
                    "adoption_eligible": int(regime.adoption_eligible),
                    "diagnostic_only": int(regime.diagnostic_only),
                    "control_only": int(regime.control_only),
                    "num_features": 0,
                    "feature_schema_hash": "",
                    "num_blocked_features": 0,
                    "num_dropped_zero_variance": 0,
                    "no_data_reason": no_data_reason,
                }
                for regime in FEATURE_REGISTRY.values()
            ]
        _write_csv(output_dir / filename, headers, rows)
        created_files.append(filename)

    decision_summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": str(scope),
        "status": "no_data" if no_data_reason else "schema_emitted",
        "no_data_reason": no_data_reason,
        "adoption_gate_pass_proxy": 0,
    }
    (output_dir / "decision_summary.json").write_text(
        json.dumps(decision_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    created_files.append("decision_summary.json")

    failure_md = (
        "# Failure Mode Summary\n\n"
        f"Status: {'no benchmark rows available' if no_data_reason else 'schema emitted'}.\n\n"
        f"Reason: {no_data_reason or 'benchmark input was present but aggregation is not performed by this schema builder'}.\n\n"
        "Effect: schema emitted, metrics unavailable.\n"
    )
    (output_dir / "failure_mode_summary.md").write_text(failure_md, encoding="utf-8")
    created_files.append("failure_mode_summary.md")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": str(scope),
        "created_files": created_files,
        "missing_inputs": [] if not no_data_reason else [str(input_csv) if input_csv is not None else ""],
        "no_data_reasons": {name: no_data_reason for name in created_files if no_data_reason},
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    created_files.append("artifact_manifest.json")
    manifest["created_files"] = created_files
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build empty-safe response-routing artifact suite.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", type=str, default="development")
    parser.add_argument("--input-csv", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_artifact_suite(args.output_dir, scope=str(args.scope), input_csv=args.input_csv)
    print(f"Wrote {len(manifest['created_files'])} artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
