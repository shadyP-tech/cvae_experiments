"""Reconstructive, non-repairing validation for the fixed-bank audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import read_json, render_csv
from .bundle import assert_closed_world, validate_content_index
from .config import (
    FixedBankDecisionAuditConfig,
    load_fixed_bank_decision_audit_config,
)
from .constants import (
    EXACT_FAMILY_IDS,
    EXPECTED_EXACT_FOLD_COUNT,
    EXPECTED_EXACT_PREDICTION_COUNT,
    EXPECTED_SMOOTH_FOLD_COUNT,
    EXPECTED_SMOOTH_PREDICTION_COUNT,
)
from .execution_adapter import (
    EXPECTED_DEVELOPMENT_CELL_COUNT,
    load_development_prediction_capability,
    open_globally_sealed_development_labels,
    validate_development_prediction_store,
    validate_global_development_seal,
)
from .feature_production import (
    build_fixed_bank_feature_lock,
    produce_label_free_fixed_bank_features,
)
from .experiment_contracts import (
    CENTERS,
    EXPECTED_DESCRIPTIVE_SEED_ROW_COUNT,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_FEATURE_ROW_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_RESPONSE_ROW_COUNT,
    EXPECTED_STRICT_TRAINING_ROW_COUNT,
    EXPECTED_SUPPORT_CASE_COUNT,
    SEED_PAIR_COUNT,
)
from .features import build_fixed_bank_dataset
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
    build_exact_crossfit_lock,
    build_fixed_bank_response_lock,
    build_smooth_descriptive_crossfit_lock,
    produce_fixed_bank_responses,
)
from .source_cache import load_source_cache, validate_source_cache_lock


def validate_fixed_bank_decision_audit_bundle(
    root: str | Path,
    *,
    config: FixedBankDecisionAuditConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct all scientific surfaces without creating or repairing one."""

    path = Path(root).resolve()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending,
    )
    resolved = load_fixed_bank_decision_audit_config(path / "config.resolved.yaml")
    if resolved.artifact_root.resolve() != path:
        raise ProtocolError("Fixed-bank resolved artifact root drifted.")
    if config is not None and (
        resolved.contract_hash != config.contract_hash
        or resolved.artifact_root.resolve() != config.artifact_root.resolve()
        or resolved.input_artifact_ids != config.input_artifact_ids
    ):
        raise ProtocolError("Fixed-bank supplied/resolved config drifted.")

    # Byte tamper fails before any label access or scientific reconstruction.
    validate_content_index(path, config_contract_hash=resolved.contract_hash)
    assert_input_fence(resolved)
    workspace = validate_active_diagnostic_workspace_binding(resolved)
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_test_frame(resolved)
    firewall = {
        **validate_pre_gpu_firewall(resolved, frame, locks),
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
            test_cache_binding_hash=frame.cache_binding_hash,
            firewall=firewall,
        ),
    )
    _assert_csv(
        path / "tables/support_partitions.csv",
        partitions.table_rows,
        SUPPORT_PARTITION_COLUMNS,
    )
    _assert_json(path / "manifests/support_partition_lock.json", partitions.lock_payload)

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

    capability = load_development_prediction_capability(path)
    validate_development_prediction_store(
        capability.store,
        source_cache_lock_hash=source_lock_hash,
        partition_lock_hash=partitions.lock_hash,
    )
    validate_global_development_seal(capability)

    features = produce_label_free_fixed_bank_features(
        source_cache,
        frame,
        partitions,
        load_metadata_similarity(resolved),
        capability,
    )
    feature_lock = build_fixed_bank_feature_lock(
        features,
        partition_lock_hash=partitions.lock_hash,
        development_prediction_seal_hash=capability.seal.prediction_seal_hash,
    )
    _assert_payload_csv(path / "tables/fixed_bank_feature_rows.csv", features.rows)
    _assert_json(path / "manifests/fixed_bank_feature_lock.json", feature_lock)
    persisted_feature_lock = read_json(path / "manifests/fixed_bank_feature_lock.json")
    _assert_json(
        path / "reports/phase_01_prelabel_surfaces_complete.json",
        prelabel_phase_payload(
            config_contract_hash=resolved.contract_hash,
            source_cache_lock_hash=source_lock_hash,
            development_prediction_seal_hash=capability.seal.prediction_seal_hash,
            feature_lock_hash=str(feature_lock["fixed_bank_feature_lock_hash"]),
        ),
    )

    labels = open_globally_sealed_development_labels(
        resolved.test_manifest_path, partitions, capability=capability
    )
    _assert_json(
        path / "reports/development_label_access_report.json",
        development_label_access_report_payload(labels),
    )
    responses = produce_fixed_bank_responses(
        features, persisted_feature_lock, capability, labels, partitions
    )
    response_lock = build_fixed_bank_response_lock(
        responses,
        feature_lock_hash=str(feature_lock["fixed_bank_feature_lock_hash"]),
        prediction_seal_hash=capability.seal.prediction_seal_hash,
    )
    _assert_payload_csv(path / "tables/fixed_bank_response_rows.csv", responses.rows)
    _assert_mapping_csv(
        path / "tables/source_inner_seed_diagnostics.csv",
        responses.descriptive_seed_rows,
    )
    _assert_json(path / "manifests/fixed_bank_response_lock.json", response_lock)

    dataset = build_fixed_bank_dataset(features.rows, responses.rows)
    audit = _run_core(dataset, include_smooth_descriptive=True)
    smooth = audit.smooth_descriptive_crossfit
    if smooth is None:
        raise ProtocolError("Fixed-bank canonical reconstruction lacks smooth output.")
    exact_lock = build_exact_crossfit_lock(audit.exact_crossfit)
    smooth_lock = build_smooth_descriptive_crossfit_lock(
        smooth, exact_crossfit_hash=audit.exact_crossfit.result_hash
    )
    audit_payload = audit.to_payload()
    _assert_json(path / "manifests/exact_crossfit_lock.json", exact_lock)
    _assert_json(path / "manifests/smooth_descriptive_crossfit_lock.json", smooth_lock)
    _assert_json(path / "manifests/audit_result.json", audit_payload)
    _assert_mapping_csv(
        path / "tables/exact_crossfit_predictions.csv",
        audit.exact_crossfit.table_rows,
    )
    _assert_mapping_csv(
        path / "tables/exact_crossfit_fold_audits.csv",
        audit.exact_crossfit.fold_table_rows,
    )
    _assert_mapping_csv(path / "tables/exact_query_metrics.csv", audit.query_metric_table_rows)
    _assert_mapping_csv(path / "tables/exact_outer_metrics.csv", audit.outer_metric_table_rows)
    _assert_mapping_csv(path / "tables/exact_family_summary.csv", audit.family_summary_table_rows)
    _assert_mapping_csv(
        path / "tables/abstention_decisions.csv",
        audit.abstention_decision_table_rows,
    )
    _assert_mapping_csv(
        path / "tables/abstention_summary.csv", audit.abstention_summary_table_rows
    )
    _assert_mapping_csv(
        path / "tables/smooth_descriptive_crossfit_predictions.csv", smooth.table_rows
    )
    _assert_mapping_csv(
        path / "tables/smooth_descriptive_crossfit_fold_audits.csv",
        smooth.fold_table_rows,
    )

    leakage = leakage_report_payload(
        support_partition_lock_hash=partitions.lock_hash,
        development_prediction_seal_hash=capability.seal.prediction_seal_hash,
        feature_lock_hash=str(feature_lock["fixed_bank_feature_lock_hash"]),
        response_lock_hash=str(response_lock["fixed_bank_response_lock_hash"]),
        exact_crossfit_lock_hash=str(exact_lock["exact_crossfit_lock_hash"]),
        smooth_crossfit_lock_hash=str(
            smooth_lock["smooth_descriptive_crossfit_lock_hash"]
        ),
    )
    _assert_json(path / "reports/leakage_report.json", leakage)
    _assert_json(
        path / "reports/publication_decision.json",
        publication_decision_payload(audit_payload),
    )
    _validate_runtime_summary(read_json(path / "reports/runtime_summary.json"))

    checks = {
        "schema_version": "midogpp_stage90_fixed_bank_validation_report_v1",
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_cache_lock_hash": source_lock_hash,
        "development_prediction_seal_hash": capability.seal.prediction_seal_hash,
        "fixed_bank_feature_lock_hash": str(
            feature_lock["fixed_bank_feature_lock_hash"]
        ),
        "fixed_bank_response_lock_hash": str(
            response_lock["fixed_bank_response_lock_hash"]
        ),
        "exact_crossfit_lock_hash": str(exact_lock["exact_crossfit_lock_hash"]),
        "smooth_descriptive_crossfit_lock_hash": str(
            smooth_lock["smooth_descriptive_crossfit_lock_hash"]
        ),
        "exact_decision_hash": audit.exact_decision_hash,
        "audit_result_hash": audit.result_hash,
        "primary_exact_gate_passed": audit.primary_exact_gate_passed,
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "feature_row_count": len(features.rows),
        "response_row_count": len(responses.rows),
        "descriptive_seed_row_count": len(responses.descriptive_seed_rows),
        "exact_prediction_row_count": len(audit.exact_crossfit.predictions),
        "exact_fold_audit_row_count": len(audit.exact_crossfit.fold_audits),
        "exact_query_metric_row_count": len(audit.query_metrics),
        "exact_outer_metric_row_count": len(audit.outer_metrics),
        "exact_family_summary_row_count": len(audit.family_summaries),
        "abstention_decision_row_count": len(audit.abstention_decisions),
        "abstention_summary_row_count": len(audit.abstention_summaries),
        "smooth_prediction_row_count": len(smooth.predictions),
        "smooth_fold_audit_row_count": len(smooth.fold_audits),
        "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
        "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
        "strict_training_row_count": EXPECTED_STRICT_TRAINING_ROW_COUNT,
        "candidate_e_history_retained_when_legal": True,
        "candidate_pool_excludes_H_and_q": True,
        "exact_response_is_primary": True,
        "smooth_response_is_isolated_descriptive_only": True,
        "smooth_influences_exact_decision": False,
        "features_sealed_before_test_label_access": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "known_fixed_bank_reuse": True,
        "unseen_expert_transfer_claim": False,
        "routing_quality_claimed": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "prior_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "terminal_diagnostic_only": True,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
    }
    report_path = path / "reports/validation_report.json"
    if not allow_pending and read_json(report_path) != checks:
        raise ProtocolError("Persisted fixed-bank validation report drifted.")
    if not allow_pending:
        state = read_json(path / "reports/run_state.json")
        if state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
            raise ProtocolError("Completed fixed-bank bundle lacks COMPLETE state.")
    return checks


def _run_core(*args: object, **kwargs: object) -> object:
    from .audit import run_fixed_bank_decision_core

    return run_fixed_bank_decision_core(*args, **kwargs)


def _assert_json(path: Path, expected: Mapping[str, object]) -> None:
    if read_json(path) != _json_ready(expected):
        raise ProtocolError(f"Fixed-bank JSON reconstruction drifted: {path}.")


def _assert_payload_csv(path: Path, rows: Sequence[object]) -> None:
    payloads = tuple(_payload_row(row) for row in rows)
    if not payloads:
        raise ProtocolError(f"Fixed-bank reconstructed table is empty: {path}.")
    _assert_csv(path, payloads, tuple(payloads[0]))


def _assert_mapping_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    payloads = tuple(_payload_row(row) for row in rows)
    if not payloads:
        raise ProtocolError(f"Fixed-bank reconstructed table is empty: {path}.")
    _assert_csv(path, payloads, tuple(payloads[0]))


def _assert_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    expected = render_csv(rows, columns).encode("utf-8")
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"Cannot read fixed-bank CSV: {path}.") from exc
    if observed != expected:
        raise ProtocolError(f"Fixed-bank CSV reconstruction drifted: {path}.")


def _payload_row(value: object) -> dict[str, object]:
    raw = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(raw, Mapping):
        raise ProtocolError("Fixed-bank table row is not a mapping.")
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
    exact_family_count = len(EXACT_FAMILY_IDS)
    if (
        payload.get("schema_version")
        != "midogpp_stage90_fixed_bank_runtime_summary_v1"
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or not isinstance(staging, Mapping)
        or int(payload.get("source_stream_count", -1))
        != len(CENTERS) * SEED_PAIR_COUNT
        or int(payload.get("development_prediction_cell_count", -1))
        != EXPECTED_DEVELOPMENT_CELL_COUNT
        or int(payload.get("feature_row_count", -1))
        != EXPECTED_FEATURE_ROW_COUNT
        or int(payload.get("response_row_count", -1))
        != EXPECTED_RESPONSE_ROW_COUNT
        or int(payload.get("descriptive_seed_row_count", -1))
        != EXPECTED_DESCRIPTIVE_SEED_ROW_COUNT
        or int(payload.get("exact_prediction_row_count", -1))
        != EXPECTED_EXACT_PREDICTION_COUNT
        or int(payload.get("exact_fold_audit_row_count", -1))
        != EXPECTED_EXACT_FOLD_COUNT
        or int(payload.get("exact_query_metric_row_count", -1))
        != exact_family_count * EXPECTED_QUERY_COUNT
        or int(payload.get("exact_outer_metric_row_count", -1))
        != exact_family_count * len(CENTERS)
        or int(payload.get("exact_family_summary_row_count", -1))
        != exact_family_count
        or int(payload.get("abstention_decision_row_count", -1))
        != exact_family_count * EXPECTED_QUERY_COUNT
        or int(payload.get("abstention_summary_row_count", -1))
        != exact_family_count
        or int(payload.get("smooth_prediction_row_count", -1))
        != EXPECTED_SMOOTH_PREDICTION_COUNT
        or int(payload.get("smooth_fold_audit_row_count", -1))
        != EXPECTED_SMOOTH_FOLD_COUNT
        or int(payload.get("target_task_count", -1)) != 0
        or int(payload.get("target_action_count", -1)) != 0
        or payload.get("test_labels_opened_after_prediction_and_feature_seals") is not True
        or payload.get("support_labels_opened") is not False
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("persistent_source_workers") is not True
        or int(payload.get("classifier_workers", -1)) != 4
        or int(payload.get("classifier_threads_per_worker", -1)) != 3
        or payload.get("scratch_preference")
        != ["/data/local/fixed_bank_decision_audit_v1", "artifact_parent"]
        or payload.get("hash_validated_resume") is not True
        or payload.get("gpu_and_cpu_phases_disjoint") is not True
        or payload.get("features_persisted_before_test_label_access") is not True
        or payload.get("smooth_isolated_from_exact_decision") is not True
        or payload.get("terminal_consumed_test_diagnostic_only") is not True
        or payload.get("action_selection_authorized") is not False
        or payload.get("policy_update_authorized") is not False
    ):
        raise ProtocolError("Fixed-bank runtime summary semantics drifted.")


__all__ = ("validate_fixed_bank_decision_audit_bundle",)
