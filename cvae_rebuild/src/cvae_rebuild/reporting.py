from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from .config import RebuildConfig, write_resolved_config
from .protocol import LeakageReport, build_leakage_report


REQUIRED_OUTPUTS = (
    "tables/support_nelbo_routing_scores.csv",
    "tables/preservation_gap_summary.csv",
    "tables/baseline_comparison.csv",
    "tables/all_expert_downstream_matrix.csv",
    "tables/routing_to_downstream_alignment.csv",
    "tables/generation_classifier_stability.csv",
    "manifests/protocol_manifest.json",
    "manifests/expert_manifest.csv",
    "reports/leakage_report.json",
    "reports/decision_summary.md",
    "run_config_resolved.yaml",
)


def prepare_artifact_dirs(root: str | Path) -> Path:
    artifact_root = Path(root)
    for rel in ("tables", "manifests", "reports"):
        (artifact_root / rel).mkdir(parents=True, exist_ok=True)
    return artifact_root


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        columns = tuple(keys)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_protocol_manifest(root: str | Path, cfg: RebuildConfig) -> None:
    write_json(
        Path(root) / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "support_size": cfg.support_size,
            "candidate_count_per_cell": cfg.candidate_count_per_cell,
            "support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "oracle_role": "diagnostic_only",
        },
    )


def write_leakage_report(root: str | Path, report: LeakageReport | None = None) -> LeakageReport:
    if report is None:
        report = build_leakage_report(
            target_support_labels_for_selection=False,
            target_eval_labels_for_scoring_only=True,
            target_expert_excluded=True,
            oracle_rows_diagnostic_only=True,
        )
    write_json(Path(root) / "reports" / "leakage_report.json", report.to_json_dict())
    return report


def write_decision_summary(
    root: str | Path,
    *,
    mean_bacc: float | None = None,
    leakage_status: str = "PASS",
) -> None:
    mean_text = "nan" if mean_bacc is None else f"{float(mean_bacc):.4f}"
    text = "\n".join(
        [
            "# CVAE Rebuild: Target-Support32 Virchow2-CVAE Top2 Routing",
            "",
            "## Primary Method",
            "",
            "- `support_nelbo_top2_geom`",
            "",
            "## Summary",
            "",
            f"- Primary mean BACC: {mean_text}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Claim Boundary",
            "",
            "This experiment is unlabeled target-support deployable, not source-only.",
            "Target evaluation labels are final utility scoring only.",
            "",
        ]
    )
    path = Path(root) / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_empty_contract_artifacts(root: str | Path, cfg: RebuildConfig) -> None:
    """Write the full artifact contract with headers for smoke/protocol checks."""

    root = prepare_artifact_dirs(root)
    write_resolved_config(root / "run_config_resolved.yaml", cfg)
    write_protocol_manifest(root, cfg)
    write_json(root / "manifests" / "expert_manifest.json", {"experts": []})
    write_csv_rows(
        root / "manifests" / "expert_manifest.csv",
        [],
        columns=("experiment_seed", "expert_id", "heldout_center", "checkpoint_path", "n_train", "n_val"),
    )
    write_csv_rows(
        root / "tables" / "support_nelbo_routing_scores.csv",
        [],
        columns=(
            "experiment_seed",
            "heldout_center",
            "support_seed",
            "support_size",
            "expert_id",
            "eligible_expert_count",
            "candidate_rank",
            "raw_support_nelbo",
            "calibrated_support_nelbo",
            "selected_top1",
            "selected_top2",
            "selected_top3",
            "selected_expert_count",
            "selected_fraction",
            "oracle_rank_diagnostic",
            "downstream_bacc",
        ),
    )
    write_csv_rows(
        root / "tables" / "preservation_gap_summary.csv",
        [],
        columns=(
            "experiment_seed",
            "heldout_center",
            "real_feature_source_top1_bacc",
            "cvae_source_top1_synthetic_bacc",
            "cvae_support_nelbo_top1_synthetic_bacc",
            "cvae_support_nelbo_top2_synthetic_bacc",
            "cvae_support_nelbo_top3_synthetic_bacc",
            "cvae_all4_synthetic_bacc",
            "cvae_oracle_synthetic_bacc_diagnostic_only",
        ),
    )
    for table in (
        "baseline_comparison.csv",
        "all_expert_downstream_matrix.csv",
        "routing_to_downstream_alignment.csv",
        "generation_classifier_stability.csv",
    ):
        write_csv_rows(root / "tables" / table, [], columns=("method", "status"))
    report = write_leakage_report(root)
    write_decision_summary(root, leakage_status=report.status)
