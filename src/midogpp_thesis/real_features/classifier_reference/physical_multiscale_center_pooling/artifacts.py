"""Artifact schemas, hashing, and completed-bundle writers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts import prepare_artifact_dirs, stable_hash, write_csv_rows, write_json
from .config import PhysicalMultiscalePilotConfig
from .decision_lock import DecisionLock
from .outer_evaluation import OuterEvaluationTables
from .reporting import decision_summary, render_decision_report


REQUIRED_STATIC_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/content_index.json",
    "manifests/decision_lock_index.json",
    "reports/leakage_provenance_report.json",
    "reports/decision_summary.json",
    "reports/decision_report.md",
    "reports/runtime_summary.json",
    "reports/conditional_bootstrap.json",
    "reports/validation_report.json",
    "tables/cache_alignment_audit.csv",
    "tables/source_inner_selector_cells.csv",
    "tables/source_inner_candidate_summary.csv",
    "tables/representation_decisions.csv",
    "tables/outer_locked_results.csv",
    "tables/outer_locked_predictions.csv",
    "tables/outer_fit_audit.csv",
    "tables/canonical_a_replay.csv",
    "tables/posthoc_candidate_isolation.csv",
)


def write_frozen_protocol(
    root: Path,
    config: PhysicalMultiscalePilotConfig,
    *,
    input_hashes: Mapping[str, str],
) -> tuple[str, str]:
    prepare_artifact_dirs(root)
    payload = {
        "schema_version": "midogpp_physical_multiscale_frozen_protocol_v1",
        "experiment_name": config.name,
        "experiment_seed": config.experiment_seed,
        "classifier_seed": config.classifier_seed,
        "heldout_centers": list(config.heldout_centers),
        "profile_id": config.profile.profile_id,
        "representations": dict(config.representation_dims),
        "classifier_grid": [spec.to_payload() for spec in config.classifier_specs],
        "gate": config.gate.__dict__,
        "bootstrap": config.bootstrap.__dict__,
        "input_hashes": dict(sorted(input_hashes.items())),
        "posthoc_rows_may_feed_selection": False,
        "target_labels_used_for_selection": False,
        "inner_delta_role": "optimistic_selection_statistic",
        "not_performance_estimate": True,
        "gate_is_statistical_test": False,
        "claim_scope": "real_feature_transfer_only",
        "non_adoptive": True,
    }
    config_hash = stable_hash(payload)
    payload["config_hash"] = config_hash
    protocol_hash = stable_hash(payload)
    payload["protocol_hash"] = protocol_hash
    path = root / "manifests" / "frozen_protocol_snapshot.json"
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite changed frozen protocol: {path}")
    path.write_text(rendered, encoding="utf-8")
    return config_hash, protocol_hash


def write_decision_lock_index(root: Path, locks: Sequence[DecisionLock]) -> str:
    rows = [
        {
            "outer_target_center": str(lock.payload["outer_target_center"]),
            "path": str(lock.path.relative_to(root)),
            "decision_hash": lock.decision_hash,
        }
        for lock in sorted(locks, key=lambda item: int(item.payload["outer_target_center"]))
    ]
    bundle_hash = stable_hash(rows)
    write_json(
        root / "manifests" / "decision_lock_index.json",
        {
            "schema_version": "midogpp_physical_multiscale_decision_lock_index_v1",
            "status": "LOCKED_BEFORE_OUTER_EVALUATION",
            "lock_count": len(rows),
            "locks": rows,
            "bundle_lock_hash": bundle_hash,
            "posthoc_rows_in_hash": False,
        },
    )
    return bundle_hash


def write_completed_bundle(
    root: Path,
    *,
    config: PhysicalMultiscalePilotConfig,
    protocol_hash: str,
    bundle_lock_hash: str,
    input_hashes: Mapping[str, str],
    selector_cells: Sequence[Mapping[str, object]],
    candidate_summaries: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
    cache_alignment_rows: Sequence[Mapping[str, object]],
    outer: OuterEvaluationTables,
    bootstrap: Mapping[str, object],
    runtime_seconds: float,
) -> None:
    write_csv_rows(root / "tables" / "cache_alignment_audit.csv", cache_alignment_rows)
    write_csv_rows(root / "tables" / "source_inner_selector_cells.csv", selector_cells)
    write_csv_rows(root / "tables" / "source_inner_candidate_summary.csv", candidate_summaries)
    write_csv_rows(root / "tables" / "representation_decisions.csv", decision_rows)
    write_csv_rows(root / "tables" / "outer_locked_results.csv", outer.results)
    write_csv_rows(root / "tables" / "outer_locked_predictions.csv", outer.predictions)
    write_csv_rows(root / "tables" / "outer_fit_audit.csv", outer.fit_audit)
    write_csv_rows(root / "tables" / "canonical_a_replay.csv", outer.canonical_a_replay)
    write_csv_rows(root / "tables" / "posthoc_candidate_isolation.csv", outer.posthoc)
    summary = decision_summary(
        decision_rows,
        outer.results,
        bootstrap,
        profile_id=config.profile.profile_id,
    )
    write_json(root / "reports" / "conditional_bootstrap.json", bootstrap)
    write_json(root / "reports" / "decision_summary.json", summary)
    (root / "reports" / "decision_report.md").write_text(
        render_decision_report(summary), encoding="utf-8"
    )
    write_json(
        root / "reports" / "leakage_provenance_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_leakage_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "target_labels_used_for_selection": False,
            "fit_used_target_center": False,
            "decision_locks_written_before_outer_evaluation": True,
            "posthoc_rows_used_for_selection": False,
            "support_labels_used": False,
            "oracle_rows_used": False,
        },
    )
    write_json(
        root / "reports" / "runtime_summary.json",
        {
            "schema_version": "midogpp_physical_multiscale_runtime_v1",
            "status": "COMPLETE",
            "elapsed_seconds": float(runtime_seconds),
            "selector_cell_count": len(selector_cells),
            "candidate_summary_count": len(candidate_summaries),
            "decision_count": len(decision_rows),
            "outer_result_count": len(outer.results),
        },
    )
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "midogpp_physical_multiscale_protocol_manifest_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "protocol_hash": protocol_hash,
            "bundle_lock_hash": bundle_lock_hash,
            "input_hashes": dict(sorted(input_hashes.items())),
            "claim_scope": "real_feature_transfer_only",
            "claim_role": (
                "complete_deterministic_representation_plus_classifier_"
                "pipeline_diagnostic"
            ),
            "non_adoptive": True,
            "may_feed_recipe_selection": False,
            "may_feed_deployable_selection": False,
            "uses_cvae": False,
            "uses_nelbo": False,
            "uses_router": False,
            "performs_expert_aggregation": False,
            "representation_c_combination": "feature_concatenation_not_mixture",
            "uses_likelihood": False,
            "uses_latent_prior": False,
            "uses_posterior": False,
            "uses_mixture_model": False,
            "uses_experts": False,
            "uses_generative_sampling": False,
            "global_representation_adoption_allowed": False,
            "inner_delta_role": "optimistic_selection_statistic",
            "not_performance_estimate": True,
            "gate_is_statistical_test": False,
            "probabilities_calibrated": False,
            "covers_new_center_uncertainty": False,
            "bootstrap_conditions_on_fixed_fits_and_locked_selection": True,
        },
    )
    _write_content_index(root)


def _write_content_index(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    write_json(
        root / "manifests" / "content_index.json",
        {
            "schema_version": "midogpp_physical_multiscale_content_index_v1",
            "files": files,
            "content_hash": stable_hash(files),
        },
    )


def finalize_validated_bundle(
    root: Path,
    *,
    validation: Mapping[str, object],
) -> None:
    """Promote a pending bundle only after independent reconstruction passes."""

    protocol_path = root / "manifests" / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["status"] = "PASS"
    protocol["independent_validation_status"] = "PASS"
    write_json(protocol_path, protocol)
    leakage_path = root / "reports" / "leakage_provenance_report.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage["status"] = "PASS"
    leakage["independent_validation_status"] = "PASS"
    write_json(leakage_path, leakage)
    write_json(
        root / "reports" / "validation_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_validation_report_v1",
            "status": "PASS",
            "validator": "validate_physical_multiscale_pilot_bundle",
            "checks": dict(validation),
            "authoritative_bundle_verdict": True,
        },
    )
    _write_content_index(root)


def mark_bundle_validation_failed(
    root: Path,
    *,
    error: str,
) -> None:
    """Ensure a failed final check cannot leave a PASS-labeled bundle."""

    protocol_path = root / "manifests" / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["status"] = "INVALID_INDEPENDENT_VALIDATION"
    protocol["independent_validation_status"] = "FAIL"
    write_json(protocol_path, protocol)
    leakage_path = root / "reports" / "leakage_provenance_report.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage["status"] = "INVALID_INDEPENDENT_VALIDATION"
    leakage["independent_validation_status"] = "FAIL"
    write_json(leakage_path, leakage)
    write_json(
        root / "reports" / "validation_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_validation_report_v1",
            "status": "FAIL",
            "validator": "validate_physical_multiscale_pilot_bundle",
            "error": str(error),
            "authoritative_bundle_verdict": True,
        },
    )
    _write_content_index(root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
