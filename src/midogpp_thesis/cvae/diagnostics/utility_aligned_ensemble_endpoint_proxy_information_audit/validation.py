"""Reconstructive, non-repairing validation for the proxy-information audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import read_json, render_csv
from .audit_adapter import run_persistable_proxy_information_audit
from .bundle import assert_closed_world, validate_content_index
from .config import (
    ProxyInformationAuditConfig,
    load_utility_aligned_ensemble_endpoint_proxy_information_audit_config,
)
from .execution_adapter import (
    DevelopmentPredictionCapability,
    open_globally_sealed_development_labels,
    produce_label_free_seed_features,
    score_development_ensemble_endpoints,
    validate_global_development_seal,
)
from .feature_production import (
    build_proxy_feature_lock,
    produce_label_free_proxy_feature_payloads,
)
from .inputs import (
    load_label_free_validation_frame,
    load_metadata_similarity,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .partitions import SUPPORT_PARTITION_COLUMNS, build_fixed_partition_surface
from .reports import (
    leakage_report_payload,
    prelabel_phase_payload,
    protocol_manifest_payload,
    publication_decision_payload,
)
from .source_cache import load_source_cache, validate_source_cache_lock

from ..utility_aligned_ensemble_endpoint_router.combined_prediction_io import (
    read_combined_store,
)
from ..utility_aligned_ensemble_endpoint_router.development_prediction_execution import (
    DEVELOPMENT_ARRAY_MEMBER,
    DEVELOPMENT_INDEX_MEMBER,
    validate_development_prediction_store,
)
from ..utility_aligned_ensemble_endpoint_router.development_seal import (
    GLOBAL_DEVELOPMENT_SEAL_MEMBER,
    GlobalDevelopmentPredictionSeal,
)


def validate_proxy_information_audit_bundle(
    root: str | Path,
    *,
    config: ProxyInformationAuditConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct every scientific table without creating or repairing files."""

    path = Path(root).resolve()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending,
    )
    resolved = load_utility_aligned_ensemble_endpoint_proxy_information_audit_config(
        path / "config.resolved.yaml"
    )
    if config is not None and (
        resolved.contract_hash != config.contract_hash
        or resolved.artifact_root.resolve() != config.artifact_root.resolve()
        or resolved.input_artifact_ids != config.input_artifact_ids
    ):
        raise ProtocolError("Proxy-audit supplied/resolved config drifted.")

    # Fail byte tamper before opening labels or invoking reconstruction.
    validate_content_index(path, config_contract_hash=resolved.contract_hash)
    workspace = validate_active_diagnostic_workspace_binding(resolved)
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_validation_frame(resolved)
    firewall = {
        **validate_pre_gpu_firewall(resolved, frame),
        "workspace_binding": workspace,
    }
    partitions = build_fixed_partition_surface(
        frame, config_contract_hash=resolved.contract_hash
    )
    input_hashes = {
        artifact_id: stable_hash(provenance[artifact_id])
        for artifact_id in resolved.input_artifact_ids
    }
    _assert_json(
        path / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            resolved,
            input_artifact_hashes=input_hashes,
            validation_cache_binding_hash=str(frame.cache_binding_hash),
            firewall=firewall,
        ),
    )
    _assert_csv(
        path / "tables/support_partitions.csv",
        tuple(partitions.table_rows),
        SUPPORT_PARTITION_COLUMNS,
    )
    _assert_json(
        path / "manifests/support_partition_lock.json", partitions.lock_payload
    )

    source_cache = load_source_cache(path)
    source_lock = validate_source_cache_lock(
        path,
        config=resolved,
        generation_lock=locks.generation,
        frame=frame,
        partitions=partitions,
        source_cache=source_cache,
    )
    source_lock_hash = str(source_lock["source_cache_lock_hash"])
    seed_features = produce_label_free_seed_features(
        source_cache, frame, partitions, load_metadata_similarity(resolved)
    )

    store = read_combined_store(
        path / DEVELOPMENT_ARRAY_MEMBER, path / DEVELOPMENT_INDEX_MEMBER
    )
    validate_development_prediction_store(
        store,
        source_cache_lock_hash=source_lock_hash,
        partition_lock_hash=partitions.lock_hash,
    )
    capability = DevelopmentPredictionCapability(
        store=store,
        seal=GlobalDevelopmentPredictionSeal(
            read_json(path / GLOBAL_DEVELOPMENT_SEAL_MEMBER)
        ),
        seal_path=path / GLOBAL_DEVELOPMENT_SEAL_MEMBER,
        prediction_index_path=path / DEVELOPMENT_INDEX_MEMBER,
        prediction_arrays_path=path / DEVELOPMENT_ARRAY_MEMBER,
    )
    validate_global_development_seal(capability)

    proxy_payloads = produce_label_free_proxy_feature_payloads(
        seed_features, capability, partitions
    )
    proxy_lock = build_proxy_feature_lock(
        proxy_payloads,
        partition_lock_hash=partitions.lock_hash,
        development_prediction_seal_hash=capability.seal.prediction_seal_hash,
    )
    _assert_payload_csv(path / "tables/proxy_feature_rows.csv", proxy_payloads)
    _assert_json(path / "manifests/proxy_feature_lock.json", proxy_lock)
    _assert_json(
        path / "reports/phase_01_prelabel_surfaces_complete.json",
        prelabel_phase_payload(
            config_contract_hash=resolved.contract_hash,
            source_cache_lock_hash=source_lock_hash,
            development_prediction_seal_hash=capability.seal.prediction_seal_hash,
            proxy_feature_lock_hash=str(proxy_lock["proxy_feature_lock_hash"]),
        ),
    )

    labels = open_globally_sealed_development_labels(
        resolved.validation_manifest_path, partitions, capability=capability
    )
    utility, seed_rows = score_development_ensemble_endpoints(
        capability, labels, partitions
    )
    _assert_payload_csv(
        path / "tables/source_inner_ensemble_endpoints.csv", utility.rows
    )
    _assert_payload_csv(
        path / "tables/source_inner_seed_diagnostics.csv", seed_rows
    )
    expected_label_report = {
        "schema_version": (
            "midogpp_stage90_proxy_information_development_label_access_v1"
        ),
        "status": "PASS",
        "prediction_seal_hash": labels.prediction_seal_hash,
        "manifest_sha256": labels.manifest_sha256,
        "capability_hash": labels.capability_hash,
        "evaluation_row_hash_by_center": dict(labels.evaluation_row_hash_by_center),
        "label_hash_by_center": dict(labels.label_hash_by_center),
        "support_labels_opened": False,
        "labels_persisted": False,
        "target_labels_opened": False,
    }
    _assert_json(
        path / "reports/development_label_access_report.json", expected_label_report
    )

    audit = run_persistable_proxy_information_audit(
        proxy_payloads,
        utility.rows,
        ridge_alpha=float(resolved.model["ridge_alpha"]),
    )
    _assert_json(path / "manifests/crossfit_fold_lock.json", audit.fold_lock)
    _assert_json(path / "manifests/audit_result.json", audit.result_payload)
    _assert_payload_csv(path / "tables/crossfit_predictions.csv", audit.crossfit_rows)
    _assert_payload_csv(path / "tables/query_metrics.csv", audit.query_metric_rows)
    _assert_payload_csv(path / "tables/outer_metrics.csv", audit.outer_metric_rows)
    _assert_payload_csv(path / "tables/family_summary.csv", audit.family_summary_rows)

    leakage = leakage_report_payload(
        support_partition_lock_hash=partitions.lock_hash,
        development_prediction_seal_hash=capability.seal.prediction_seal_hash,
        proxy_feature_lock_hash=str(proxy_lock["proxy_feature_lock_hash"]),
        crossfit_fold_lock_hash=str(
            audit.fold_lock["crossfit_fold_lock_hash"]
        ),
    )
    _assert_json(path / "reports/leakage_report.json", leakage)
    _assert_json(
        path / "reports/publication_decision.json",
        publication_decision_payload(audit.result_payload),
    )
    _validate_runtime_summary(read_json(path / "reports/runtime_summary.json"))

    checks = {
        "schema_version": "midogpp_stage90_proxy_information_validation_report_v1",
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_cache_lock_hash": source_lock_hash,
        "development_prediction_seal_hash": capability.seal.prediction_seal_hash,
        "proxy_feature_lock_hash": str(proxy_lock["proxy_feature_lock_hash"]),
        "crossfit_fold_lock_hash": str(
            audit.fold_lock["crossfit_fold_lock_hash"]
        ),
        "audit_result_hash": str(audit.result_payload["audit_result_hash"]),
        "proxy_feature_row_count": len(proxy_payloads),
        "ensemble_endpoint_response_count": len(utility.rows),
        "descriptive_seed_row_count": len(seed_rows),
        "family_count": len(audit.family_summary_rows),
        "query_metric_row_count": len(audit.query_metric_rows),
        "outer_metric_row_count": len(audit.outer_metric_rows),
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "target_actions_built": False,
        "target_labels_opened": False,
        "prior_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "diagnostic_only": True,
        "policy_update_authorized": False,
    }
    report_path = path / "reports/validation_report.json"
    if not allow_pending and read_json(report_path) != checks:
        raise ProtocolError("Persisted proxy-audit validation report drifted.")
    if not allow_pending:
        state = read_json(path / "reports/run_state.json")
        if state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
            raise ProtocolError("Completed proxy-audit bundle lacks COMPLETE state.")
    return checks


def _assert_json(path: Path, expected: Mapping[str, object]) -> None:
    if read_json(path) != _json_ready(expected):
        raise ProtocolError(f"Proxy-audit JSON reconstruction drifted: {path}.")


def _assert_payload_csv(path: Path, rows: Sequence[object]) -> None:
    payloads = tuple(_payload_row(row) for row in rows)
    if not payloads:
        raise ProtocolError(f"Proxy-audit reconstructed table is empty: {path}.")
    _assert_csv(path, payloads, tuple(payloads[0]))


def _assert_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    expected = render_csv(rows, columns).encode("utf-8")
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"Cannot read proxy-audit CSV: {path}.") from exc
    if observed != expected:
        raise ProtocolError(f"Proxy-audit CSV reconstruction drifted: {path}.")


def _payload_row(value: object) -> dict[str, object]:
    raw = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(raw, Mapping):
        raise ProtocolError("Proxy-audit table row is not a mapping.")
    return {str(key): _table_value(item) for key, item in raw.items()}


def _table_value(value: object) -> object:
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _validate_runtime_summary(payload: Mapping[str, object]) -> None:
    preflight = payload.get("workstation_preflight")
    staging = payload.get("source_cache_staging")
    if (
        payload.get("schema_version")
        != "midogpp_stage90_proxy_information_runtime_summary_v1"
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or not isinstance(staging, Mapping)
        or int(payload.get("source_stream_count", -1)) != 81
        or int(payload.get("development_prediction_cell_count", -1)) != 5184
        or int(payload.get("proxy_feature_row_count", -1)) != 504
        or int(payload.get("endpoint_response_count", -1)) != 504
        or int(payload.get("descriptive_seed_row_count", -1)) != 4536
        or int(payload.get("target_task_count", -1)) != 0
        or payload.get("target_labels_opened") is not False
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or int(payload.get("classifier_workers", -1)) != 4
        or int(payload.get("classifier_threads_per_worker", -1)) != 3
        or payload.get("hash_validated_resume") is not True
    ):
        raise ProtocolError("Proxy-audit runtime summary semantics drifted.")


__all__ = ("validate_proxy_information_audit_bundle",)
