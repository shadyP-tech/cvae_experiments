"""Deterministic, resume-safe persistence for the fixed-bank audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .artifact_io import atomic_json, persist_or_validate_csv, persist_or_validate_json
from .metric_contracts import FixedBankDecisionAuditResult
from .partitions import SUPPORT_PARTITION_COLUMNS
from .reports import (
    prelabel_phase_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .response_production import FixedBankResponseProduction


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partitions: object,
) -> None:
    input_hashes = {
        artifact_id: stable_hash(provenance[artifact_id])
        for artifact_id in getattr(config, "input_artifact_ids")
    }
    persist_or_validate_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            input_artifact_hashes=input_hashes,
            test_cache_binding_hash=str(frame.cache_binding_hash),
            firewall=firewall,
        ),
    )
    persist_or_validate_csv(
        root / "tables/support_partitions.csv",
        partitions.table_rows,
        SUPPORT_PARTITION_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/support_partition_lock.json", partitions.lock_payload
    )


def persist_prelabel_surfaces(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    development_prediction_seal_hash: str,
    feature_surface: object,
    feature_lock: Mapping[str, object],
) -> None:
    _persist_rows(
        root / "tables/fixed_bank_feature_rows.csv",
        tuple(_payload_row(value) for value in feature_surface.rows),
    )
    persist_or_validate_json(root / "manifests/fixed_bank_feature_lock.json", feature_lock)
    persist_or_validate_json(
        root / "reports/phase_01_prelabel_surfaces_complete.json",
        prelabel_phase_payload(
            config_contract_hash=config_contract_hash,
            source_cache_lock_hash=source_cache_lock_hash,
            development_prediction_seal_hash=development_prediction_seal_hash,
            feature_lock_hash=str(feature_lock["fixed_bank_feature_lock_hash"]),
        ),
    )


def persist_postseal_audit(
    root: Path,
    *,
    development_labels: object,
    responses: FixedBankResponseProduction,
    response_lock: Mapping[str, object],
    exact_crossfit_lock: Mapping[str, object],
    smooth_crossfit_lock: Mapping[str, object],
    audit: FixedBankDecisionAuditResult,
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    audit_payload = audit.to_payload()
    persist_or_validate_json(
        root / "reports/development_label_access_report.json",
        development_label_access_report_payload(development_labels),
    )
    _persist_rows(
        root / "tables/fixed_bank_response_rows.csv",
        tuple(_payload_row(value) for value in responses.rows),
    )
    _persist_rows(
        root / "tables/source_inner_seed_diagnostics.csv",
        tuple(_mapping_row(value) for value in responses.descriptive_seed_rows),
    )
    persist_or_validate_json(root / "manifests/fixed_bank_response_lock.json", response_lock)
    persist_or_validate_json(root / "manifests/exact_crossfit_lock.json", exact_crossfit_lock)
    persist_or_validate_json(
        root / "manifests/smooth_descriptive_crossfit_lock.json", smooth_crossfit_lock
    )
    persist_or_validate_json(root / "manifests/audit_result.json", audit_payload)
    _persist_rows(
        root / "tables/exact_crossfit_predictions.csv",
        tuple(_mapping_row(value) for value in audit.exact_crossfit.table_rows),
    )
    _persist_rows(
        root / "tables/exact_crossfit_fold_audits.csv",
        tuple(_mapping_row(value) for value in audit.exact_crossfit.fold_table_rows),
    )
    _persist_rows(
        root / "tables/exact_query_metrics.csv",
        tuple(_mapping_row(value) for value in audit.query_metric_table_rows),
    )
    _persist_rows(
        root / "tables/exact_outer_metrics.csv",
        tuple(_mapping_row(value) for value in audit.outer_metric_table_rows),
    )
    _persist_rows(
        root / "tables/exact_family_summary.csv",
        tuple(_mapping_row(value) for value in audit.family_summary_table_rows),
    )
    _persist_rows(
        root / "tables/abstention_decisions.csv",
        tuple(_mapping_row(value) for value in audit.abstention_decision_table_rows),
    )
    _persist_rows(
        root / "tables/abstention_summary.csv",
        tuple(_mapping_row(value) for value in audit.abstention_summary_table_rows),
    )
    smooth = audit.smooth_descriptive_crossfit
    if smooth is None:
        raise ValueError("Canonical fixed-bank bundle requires smooth descriptive rows.")
    _persist_rows(
        root / "tables/smooth_descriptive_crossfit_predictions.csv",
        tuple(_mapping_row(value) for value in smooth.table_rows),
    )
    _persist_rows(
        root / "tables/smooth_descriptive_crossfit_fold_audits.csv",
        tuple(_mapping_row(value) for value in smooth.fold_table_rows),
    )
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(
        root / "reports/publication_decision.json",
        publication_decision_payload(audit_payload),
    )
    persist_or_validate_json(root / "reports/runtime_summary.json", runtime_summary)


def development_label_access_report_payload(labels: object) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_fixed_bank_test_label_access_report_v1",
        "status": "PASS",
        "prediction_seal_hash": str(labels.prediction_seal_hash),
        "manifest_sha256": str(labels.manifest_sha256),
        "capability_hash": str(labels.capability_hash),
        "evaluation_row_hash_by_center": dict(labels.evaluation_row_hash_by_center),
        "label_hash_by_center": dict(labels.label_hash_by_center),
        "evaluation_split": "test",
        "test_split_previously_consumed": True,
        "features_sealed_before_label_access": True,
        "test_labels_construct_postseal_response_rows": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "support_labels_opened": False,
        "labels_persisted": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
    }


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _payload_row(value: object) -> dict[str, object]:
    raw = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(raw, Mapping):
        raise TypeError("Fixed-bank table rows must be mappings or expose to_payload().")
    return _mapping_row(raw)


def _mapping_row(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _table_value(item) for key, item in value.items()}


def _table_value(value: object) -> object:
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _persist_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = tuple(rows)
    if not values:
        raise ValueError(f"Cannot persist empty fixed-bank table: {path}.")
    columns = tuple(values[0])
    if any(tuple(row) != columns for row in values):
        raise ValueError(f"Fixed-bank table columns drifted: {path}.")
    persist_or_validate_csv(path, values, columns)


__all__ = (
    "development_label_access_report_payload",
    "persist_initial_surfaces",
    "persist_postseal_audit",
    "persist_prelabel_surfaces",
    "persist_validation_report",
    "write_run_state",
)
