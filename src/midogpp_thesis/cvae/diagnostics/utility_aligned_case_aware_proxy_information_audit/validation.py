"""Reconstructive, non-repairing validation for the case-aware audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
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
from .artifact_io import read_json, render_csv
from .audit import run_case_aware_proxy_information_audit
from .bundle import assert_closed_world, validate_content_index
from .config import (
    CaseAwareProxyInformationAuditConfig,
    load_utility_aligned_case_aware_proxy_information_audit_config,
)
from .execution_adapter import (
    DevelopmentPredictionCapability,
    open_globally_sealed_development_labels,
    validate_global_development_seal,
)
from .feature_production import (
    build_case_aware_feature_lock,
    produce_label_free_case_aware_features,
)
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_metadata_similarity,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .partitions import SUPPORT_PARTITION_COLUMNS, build_fixed_test_partition_surface
from .persistence import development_label_access_report_payload
from .reports import (
    leakage_report_payload,
    prelabel_phase_payload,
    protocol_manifest_payload,
    publication_decision_payload,
)
from .response_production import (
    build_crossfit_fold_lock,
    produce_case_aware_responses,
)
from .source_cache import load_source_cache, validate_source_cache_lock


def validate_case_aware_proxy_information_audit_bundle(
    root: str | Path,
    *,
    config: CaseAwareProxyInformationAuditConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct every scientific surface without creating or repairing it."""

    path = Path(root).resolve()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending,
    )
    resolved = load_utility_aligned_case_aware_proxy_information_audit_config(
        path / "config.resolved.yaml"
    )
    if resolved.artifact_root.resolve() != path:
        raise ProtocolError("Case-aware resolved artifact root drifted.")
    if config is not None and (
        resolved.contract_hash != config.contract_hash
        or resolved.artifact_root.resolve() != config.artifact_root.resolve()
        or resolved.input_artifact_ids != config.input_artifact_ids
    ):
        raise ProtocolError("Case-aware audit supplied/resolved config drifted.")

    # Byte tamper fails before label access or any scientific reconstruction.
    validate_content_index(path, config_contract_hash=resolved.contract_hash)
    assert_input_fence(resolved)
    workspace = validate_active_diagnostic_workspace_binding(resolved)
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_test_frame(resolved)
    firewall = {
        **validate_pre_gpu_firewall(resolved, frame),
        "workspace_binding": workspace,
    }
    partitions = build_fixed_test_partition_surface(
        frame,
        config_contract_hash=resolved.contract_hash,
        support_case_count=resolved.fixed_support_case_count_per_center,
        split_seed=int(resolved.protocol["support_split_seed"]),
        namespace=str(resolved.protocol["support_partition_namespace"]),
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
            test_cache_binding_hash=str(frame.cache_binding_hash),
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

    features = produce_label_free_case_aware_features(
        source_cache,
        frame,
        partitions,
        load_metadata_similarity(resolved),
        capability,
    )
    feature_lock = build_case_aware_feature_lock(
        features,
        partition_lock_hash=partitions.lock_hash,
        development_prediction_seal_hash=capability.seal.prediction_seal_hash,
    )
    _assert_payload_csv(path / "tables/proxy_feature_rows.csv", features.rows)
    _assert_json(path / "manifests/proxy_feature_lock.json", feature_lock)
    persisted_feature_lock = read_json(path / "manifests/proxy_feature_lock.json")
    _assert_json(
        path / "reports/phase_01_prelabel_surfaces_complete.json",
        prelabel_phase_payload(
            config_contract_hash=resolved.contract_hash,
            source_cache_lock_hash=source_lock_hash,
            development_prediction_seal_hash=(
                capability.seal.prediction_seal_hash
            ),
            feature_lock_hash=str(
                feature_lock["case_aware_feature_lock_hash"]
            ),
        ),
    )

    labels = open_globally_sealed_development_labels(
        resolved.test_manifest_path, partitions, capability=capability
    )
    _assert_json(
        path / "reports/development_label_access_report.json",
        development_label_access_report_payload(labels),
    )
    responses = produce_case_aware_responses(
        features, persisted_feature_lock, capability, labels, partitions
    )
    _assert_payload_csv(
        path / "tables/source_inner_ensemble_endpoints.csv",
        responses.surface.rows,
    )
    _assert_payload_csv(
        path / "tables/source_inner_seed_diagnostics.csv",
        responses.descriptive_seed_rows,
    )

    audit = run_case_aware_proxy_information_audit(features, responses.surface)
    fold_lock = build_crossfit_fold_lock(audit.crossfit)
    audit_payload = audit.to_payload()
    _assert_json(path / "manifests/crossfit_fold_lock.json", fold_lock)
    _assert_json(path / "manifests/audit_result.json", audit_payload)
    _assert_payload_csv(
        path / "tables/crossfit_predictions.csv", audit.crossfit_table_rows
    )
    _assert_payload_csv(
        path / "tables/crossfit_fold_audits.csv", audit.fold_audit_table_rows
    )
    _assert_payload_csv(
        path / "tables/query_metrics.csv", audit.query_metric_table_rows
    )
    _assert_payload_csv(
        path / "tables/outer_metrics.csv", audit.outer_metric_table_rows
    )
    _assert_payload_csv(
        path / "tables/family_summary.csv", audit.family_summary_table_rows
    )

    leakage = leakage_report_payload(
        support_partition_lock_hash=partitions.lock_hash,
        development_prediction_seal_hash=capability.seal.prediction_seal_hash,
        feature_lock_hash=str(feature_lock["case_aware_feature_lock_hash"]),
        crossfit_fold_lock_hash=str(fold_lock["crossfit_fold_lock_hash"]),
    )
    _assert_json(path / "reports/leakage_report.json", leakage)
    _assert_json(
        path / "reports/publication_decision.json",
        publication_decision_payload(audit_payload),
    )
    _validate_runtime_summary(read_json(path / "reports/runtime_summary.json"))

    checks = {
        "schema_version": "midogpp_stage90_case_aware_proxy_validation_report_v1",
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_cache_lock_hash": source_lock_hash,
        "development_prediction_seal_hash": (
            capability.seal.prediction_seal_hash
        ),
        "case_aware_feature_lock_hash": str(
            feature_lock["case_aware_feature_lock_hash"]
        ),
        "feature_surface_hash": features.surface_hash,
        "response_surface_hash": responses.surface.surface_hash,
        "crossfit_fold_lock_hash": str(
            fold_lock["crossfit_fold_lock_hash"]
        ),
        "audit_result_hash": str(audit_payload["audit_result_hash"]),
        "primary_proxy_information_gate_passed": (
            audit.primary_proxy_information_gate_passed
        ),
        "informative_family_ids": list(audit.informative_family_ids),
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "proxy_feature_row_count": len(features.rows),
        "ensemble_endpoint_response_count": len(responses.surface.rows),
        "descriptive_seed_row_count": len(responses.descriptive_seed_rows),
        "crossfit_prediction_row_count": len(audit.crossfit.predictions),
        "crossfit_fold_audit_row_count": len(audit.crossfit.fold_audits),
        "logical_crossfit_fold_count": len(audit.crossfit.fold_audits),
        "unique_crossfit_ridge_fit_count": 1_176,
        "query_metric_row_count": len(audit.query_metrics),
        "outer_metric_row_count": len(audit.outer_metrics),
        "family_summary_row_count": len(audit.family_summaries),
        "support_case_count_per_center": 8,
        "evaluation_case_count_total": 146,
        "primary_response": "exact_bacc_delta",
        "diagnostic_response": "smooth_bacc_delta",
        "test_labels_construct_postseal_response_rows": True,
        "label_derived_responses_feed_strict_crossfit_diagnostic_models": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "features_sealed_before_test_label_access": True,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "routing_quality_claimed": False,
        "target_actions_built": False,
        "deployable_target_labels_opened": False,
        "prior_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "terminal_diagnostic_only": True,
        "policy_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
    }
    report_path = path / "reports/validation_report.json"
    if not allow_pending and read_json(report_path) != checks:
        raise ProtocolError("Persisted case-aware validation report drifted.")
    if not allow_pending:
        state = read_json(path / "reports/run_state.json")
        if state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
            raise ProtocolError("Completed case-aware bundle lacks COMPLETE state.")
    return checks


def _assert_json(path: Path, expected: Mapping[str, object]) -> None:
    if read_json(path) != _json_ready(expected):
        raise ProtocolError(f"Case-aware JSON reconstruction drifted: {path}.")


def _assert_payload_csv(path: Path, rows: Sequence[object]) -> None:
    payloads = tuple(_payload_row(row) for row in rows)
    if not payloads:
        raise ProtocolError(f"Case-aware reconstructed table is empty: {path}.")
    _assert_csv(path, payloads, tuple(payloads[0]))


def _assert_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    expected = render_csv(rows, columns).encode("utf-8")
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"Cannot read case-aware CSV: {path}.") from exc
    if observed != expected:
        raise ProtocolError(f"Case-aware CSV reconstruction drifted: {path}.")


def _payload_row(value: object) -> dict[str, object]:
    raw = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(raw, Mapping):
        raise ProtocolError("Case-aware table row is not a mapping.")
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
        != "midogpp_stage90_case_aware_proxy_runtime_summary_v1"
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or not isinstance(staging, Mapping)
        or int(payload.get("source_stream_count", -1)) != 81
        or int(payload.get("development_prediction_cell_count", -1)) != 5_184
        or int(payload.get("proxy_feature_row_count", -1)) != 504
        or int(payload.get("endpoint_response_count", -1)) != 504
        or int(payload.get("descriptive_seed_row_count", -1)) != 4_536
        or int(payload.get("crossfit_prediction_row_count", -1)) != 7_056
        or int(payload.get("crossfit_fold_audit_row_count", -1)) != 7_056
        or int(payload.get("logical_crossfit_fold_count", -1)) != 7_056
        or int(payload.get("unique_crossfit_ridge_fit_count", -1)) != 1_176
        or int(payload.get("query_metric_row_count", -1)) != 1_008
        or int(payload.get("outer_metric_row_count", -1)) != 126
        or int(payload.get("family_summary_row_count", -1)) != 14
        or int(payload.get("target_task_count", -1)) != 0
        or int(payload.get("target_action_count", -1)) != 0
        or payload.get("test_labels_opened_after_global_prediction_seal") is not True
        or payload.get("deployable_target_labels_opened") is not False
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or int(payload.get("classifier_workers", -1)) != 4
        or int(payload.get("classifier_threads_per_worker", -1)) != 3
        or payload.get("scratch_preference") != ["/data/local", "artifact_parent"]
        or payload.get("hash_validated_resume") is not True
        or payload.get("features_persisted_before_test_label_access") is not True
        or payload.get("terminal_diagnostic_only") is not True
    ):
        raise ProtocolError("Case-aware runtime summary semantics drifted.")


# Short alias for callers mirroring other diagnostics.
validate_proxy_information_audit_bundle = (
    validate_case_aware_proxy_information_audit_bundle
)


__all__ = (
    "validate_case_aware_proxy_information_audit_bundle",
    "validate_proxy_information_audit_bundle",
)
