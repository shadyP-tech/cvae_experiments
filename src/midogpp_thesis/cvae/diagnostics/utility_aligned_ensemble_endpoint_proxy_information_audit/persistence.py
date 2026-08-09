"""Deterministic persistence for the independent proxy-information audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .artifact_io import atomic_json, persist_or_validate_csv, persist_or_validate_json
from .partitions import SUPPORT_PARTITION_COLUMNS
from .reports import (
    prelabel_phase_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)


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
            validation_cache_binding_hash=str(frame.cache_binding_hash),
            firewall=firewall,
        ),
    )
    persist_or_validate_csv(
        root / "tables/support_partitions.csv",
        partitions.table_rows,
        SUPPORT_PARTITION_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/support_partition_lock.json",
        partitions.lock_payload,
    )


def persist_prelabel_surfaces(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    development_prediction_seal_hash: str,
    proxy_rows: Sequence[object],
    proxy_feature_lock: Mapping[str, object],
) -> None:
    rows = tuple(_payload_row(value) for value in proxy_rows)
    _persist_rows(root / "tables/proxy_feature_rows.csv", rows)
    persist_or_validate_json(
        root / "manifests/proxy_feature_lock.json", proxy_feature_lock
    )
    persist_or_validate_json(
        root / "reports/phase_01_prelabel_surfaces_complete.json",
        prelabel_phase_payload(
            config_contract_hash=config_contract_hash,
            source_cache_lock_hash=source_cache_lock_hash,
            development_prediction_seal_hash=development_prediction_seal_hash,
            proxy_feature_lock_hash=str(
                proxy_feature_lock["proxy_feature_lock_hash"]
            ),
        ),
    )


def persist_postseal_audit(
    root: Path,
    *,
    development_labels: object,
    utility_surface: object,
    descriptive_seed_rows: Sequence[Mapping[str, object]],
    fold_lock: Mapping[str, object],
    audit_result: Mapping[str, object],
    crossfit_rows: Sequence[Mapping[str, object]],
    query_rows: Sequence[Mapping[str, object]],
    outer_rows: Sequence[Mapping[str, object]],
    family_rows: Sequence[Mapping[str, object]],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    persist_or_validate_json(
        root / "reports/development_label_access_report.json",
        {
            "schema_version": (
                "midogpp_stage90_proxy_information_development_label_access_v1"
            ),
            "status": "PASS",
            "prediction_seal_hash": development_labels.prediction_seal_hash,
            "manifest_sha256": development_labels.manifest_sha256,
            "capability_hash": development_labels.capability_hash,
            "evaluation_row_hash_by_center": dict(
                development_labels.evaluation_row_hash_by_center
            ),
            "label_hash_by_center": dict(development_labels.label_hash_by_center),
            "support_labels_opened": False,
            "labels_persisted": False,
            "target_labels_opened": False,
        },
    )
    _persist_rows(
        root / "tables/source_inner_ensemble_endpoints.csv",
        tuple(_payload_row(value) for value in utility_surface.rows),
    )
    _persist_rows(
        root / "tables/source_inner_seed_diagnostics.csv",
        tuple(_mapping_row(value) for value in descriptive_seed_rows),
    )
    persist_or_validate_json(root / "manifests/crossfit_fold_lock.json", fold_lock)
    persist_or_validate_json(root / "manifests/audit_result.json", audit_result)
    _persist_rows(
        root / "tables/crossfit_predictions.csv",
        tuple(_mapping_row(value) for value in crossfit_rows),
    )
    _persist_rows(
        root / "tables/query_metrics.csv",
        tuple(_mapping_row(value) for value in query_rows),
    )
    _persist_rows(
        root / "tables/outer_metrics.csv",
        tuple(_mapping_row(value) for value in outer_rows),
    )
    _persist_rows(
        root / "tables/family_summary.csv",
        tuple(_mapping_row(value) for value in family_rows),
    )
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(
        root / "reports/publication_decision.json",
        publication_decision_payload(audit_result),
    )
    persist_or_validate_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(
        root / "reports/run_state.json",
        run_state_payload(status, phase, error=error),
    )


def _payload_row(value: object) -> dict[str, object]:
    raw = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(raw, Mapping):
        raise TypeError("Scientific table rows must be mappings or expose to_payload().")
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
        raise ValueError(f"Cannot persist empty proxy-audit table: {path}.")
    columns = tuple(values[0])
    if any(tuple(row) != columns for row in values):
        raise ValueError(f"Proxy-audit table columns drifted: {path}.")
    persist_or_validate_csv(path, values, columns)


__all__ = (
    "persist_initial_surfaces",
    "persist_postseal_audit",
    "persist_prelabel_surfaces",
    "persist_validation_report",
    "write_run_state",
)
