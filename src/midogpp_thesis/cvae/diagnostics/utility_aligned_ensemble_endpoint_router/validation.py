"""Reconstructive closed-world validation for the Stage-90 sibling."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import build_ensemble_endpoint_action_library
from .artifact_io import read_json
from .bundle import assert_closed_world, validate_content_index
from .combined_prediction_io import read_combined_store
from .config import EnsembleEndpointRouterConfig, load_utility_aligned_ensemble_endpoint_router_config
from .development_label_access import open_globally_sealed_development_labels
from .development_prediction_execution import (
    DEVELOPMENT_ARRAY_MEMBER, DEVELOPMENT_INDEX_MEMBER,
    validate_development_prediction_store,
)
from .development_scoring import score_development_ensemble_endpoints
from .development_seal import (
    GLOBAL_DEVELOPMENT_SEAL_MEMBER, DevelopmentPredictionCapability,
    GlobalDevelopmentPredictionSeal, validate_global_development_seal,
)
from .diagnostic_plan import build_stage90_ensemble_diagnostic_plan_set
from .feature_production import produce_label_free_seed_features
from .features import build_stage90_ensemble_feature_surface_set
from .inference import build_center_contrasts, infer_center_contrasts
from .inputs import (
    load_label_free_validation_frame, load_metadata_similarity, load_validated_locks,
    validate_active_diagnostic_workspace_binding, validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .modeling import fit_stage90_ensemble_models
from .partitions import build_case_fold_surface, build_fixed_partition_surface
from .reports import (
    leakage_report_payload, publication_decision_payload, scoring_summary_payload,
)
from .runner_persistence import (
    persist_development_and_router_surfaces, persist_feature_surfaces,
    persist_initial_surfaces, persist_target_seal_phase, persist_terminal_surfaces,
)
from .source_cache import load_source_cache, validate_source_cache_lock
from .support_shifts import (
    build_and_persist_source_inner_support_shifts,
    build_and_persist_target_support_shifts,
)
from .target_label_access import open_target_labels_after_global_seal
from .target_prediction_execution import (
    TARGET_ARRAY_MEMBER, TARGET_INDEX_MEMBER, materialize_target_predictions,
    materialize_target_probe_predictions,
)
from .target_seal import validate_global_target_prediction_seal
from .terminal_scoring import score_terminal_target_predictions


def validate_utility_aligned_ensemble_endpoint_router_bundle(
    root: str | Path, *, config: EnsembleEndpointRouterConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root).resolve()
    assert_closed_world(path, allow_incomplete=False, allow_pending_validation=allow_pending)
    resolved = load_utility_aligned_ensemble_endpoint_router_config(path / "config.resolved.yaml")
    if config is not None and (
        resolved.contract_hash != config.contract_hash
        or resolved.artifact_root.resolve() != config.artifact_root.resolve()
        or resolved.input_artifact_ids != config.input_artifact_ids
    ):
        raise ProtocolError("Ensemble-endpoint supplied/resolved config drifted.")
    # Byte tamper must fail before any reconstruction helper is allowed to
    # compare or materialize a scientific surface.
    validate_content_index(path, config_contract_hash=resolved.contract_hash)
    workspace = validate_active_diagnostic_workspace_binding(resolved)
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_validation_frame(resolved)
    firewall = {**validate_pre_gpu_firewall(resolved, frame), "workspace_binding": workspace}
    partitions = build_fixed_partition_surface(frame, config_contract_hash=resolved.contract_hash)
    case_folds = build_case_fold_surface(partitions, config_contract_hash=resolved.contract_hash)
    persist_initial_surfaces(
        path, config=resolved, provenance=provenance, frame=frame, firewall=firewall,
        partitions=partitions, case_folds=case_folds,
    )

    source_cache = load_source_cache(path)
    source_lock = validate_source_cache_lock(
        path, config=resolved, generation_lock=locks.generation, frame=frame,
        partitions=partitions, source_cache=source_cache,
    )
    source_lock_hash = str(source_lock["source_cache_lock_hash"])
    seed_features = produce_label_free_seed_features(
        source_cache, frame, partitions, load_metadata_similarity(resolved)
    )
    development_store = read_combined_store(
        path / DEVELOPMENT_ARRAY_MEMBER, path / DEVELOPMENT_INDEX_MEMBER
    )
    validate_development_prediction_store(
        development_store, source_cache_lock_hash=source_lock_hash,
        partition_lock_hash=partitions.lock_hash,
    )
    development = DevelopmentPredictionCapability(
        store=development_store,
        seal=GlobalDevelopmentPredictionSeal(read_json(path / GLOBAL_DEVELOPMENT_SEAL_MEMBER)),
        seal_path=path / GLOBAL_DEVELOPMENT_SEAL_MEMBER,
        prediction_index_path=path / DEVELOPMENT_INDEX_MEMBER,
        prediction_arrays_path=path / DEVELOPMENT_ARRAY_MEMBER,
    )
    validate_global_development_seal(development)
    inner_shifts = build_and_persist_source_inner_support_shifts(path, development)
    probe = materialize_target_probe_predictions(
        resolved, source_cache, frame, partitions,
        source_cache_lock_hash=source_lock_hash, root=path,
    )
    target_shifts = build_and_persist_target_support_shifts(path, probe, partitions)
    probe_seal = read_json(path / "manifests/ensemble_endpoint_target_probe_seal.json")
    features = build_stage90_ensemble_feature_surface_set(
        seed_features.inner_rows, seed_features.target_rows,
        inner_support_shift_by_candidate=inner_shifts.by_candidate,
        target_support_shift_by_candidate=target_shifts.by_candidate,
        inner_support_shift_lock_hash=inner_shifts.lock_hash,
        target_support_shift_lock_hash=target_shifts.lock_hash,
        target_probe_seal_hash=str(probe_seal["probe_seal_hash"]),
    )
    persist_feature_surfaces(
        path, config_contract_hash=resolved.contract_hash,
        source_cache_lock_hash=source_lock_hash, seed_production=seed_features,
        features=features, inner_shift_lock_hash=inner_shifts.lock_hash,
        target_shift_lock_hash=target_shifts.lock_hash,
        target_probe_seal_hash=str(probe_seal["probe_seal_hash"]),
    )

    development_labels = open_globally_sealed_development_labels(
        resolved.validation_manifest_path, partitions, capability=development
    )
    utility, development_seed_rows = score_development_ensemble_endpoints(
        development, development_labels, partitions
    )
    models = fit_stage90_ensemble_models(features, utility)
    plans = build_stage90_ensemble_diagnostic_plan_set(models, features)
    actions = build_ensemble_endpoint_action_library(plans)
    persist_development_and_router_surfaces(
        path, config_contract_hash=resolved.contract_hash,
        development_labels=development_labels, utility_surface=utility,
        seed_rows=development_seed_rows, models=models, plans=plans, actions=actions,
        development_prediction_seal_hash=development.seal.prediction_seal_hash,
    )
    predictions = materialize_target_predictions(
        resolved, source_cache, probe, actions, frame, partitions, case_folds,
        source_cache_lock_hash=source_lock_hash, root=path,
    )
    capability = validate_global_target_prediction_seal(
        path, config_contract_hash=resolved.contract_hash,
        source_cache_lock_hash=source_lock_hash, partitions=partitions,
        case_folds=case_folds, library=actions, predictions=predictions,
        target_support_shift_lock_hash=target_shifts.lock_hash,
    )
    persist_target_seal_phase(
        path, config_contract_hash=resolved.contract_hash, capability=capability,
        prediction_cell_count=len(predictions.cells),
        unique_classifier_fit_count=predictions.unique_classifier_fit_count,
    )
    labels_by_sample, target_label_report = open_target_labels_after_global_seal(
        resolved, partitions, root=path, capability=capability
    )
    scores, target_seed_rows, oracle_rows = score_terminal_target_predictions(
        predictions, labels_by_sample, partitions, actions, plans, capability.payload
    )
    center_rows = build_center_contrasts(scores)
    inference_rows = infer_center_contrasts(center_rows)
    scoring_summary = scoring_summary_payload(scores.rows, inference_rows, oracle_rows)
    leakage = leakage_report_payload(
        support_partition_lock_hash=partitions.lock_hash,
        case_fold_lock_hash=case_folds.lock_hash,
        development_prediction_seal_hash=development.seal.prediction_seal_hash,
        source_inner_support_shift_lock_hash=inner_shifts.lock_hash,
        target_probe_seal_hash=str(probe_seal["probe_seal_hash"]),
        target_support_shift_lock_hash=target_shifts.lock_hash,
        model_set_hash=models.model_set_hash, plan_set_hash=plans.plan_set_hash,
        action_library_hash=actions.action_library_hash,
        target_prediction_seal_hash=capability.seal_hash,
    )
    runtime_summary = read_json(path / "reports/runtime_summary.json")
    _validate_runtime_summary(runtime_summary)
    persist_terminal_surfaces(
        path, config_contract_hash=resolved.contract_hash,
        target_label_report=target_label_report, seed_rows=target_seed_rows,
        scores=scores, center_contrasts=center_rows, inference_rows=inference_rows,
        oracle_rows=oracle_rows, leakage_report=leakage,
        scoring_summary=scoring_summary,
        publication_decision=publication_decision_payload(scoring_summary),
        runtime_summary=runtime_summary, target_seal_hash=capability.seal_hash,
    )
    checks = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_validation_report_v1",
        "status": "PASS", "config_contract_hash": resolved.contract_hash,
        "source_cache_lock_hash": source_lock_hash,
        "development_prediction_seal_hash": development.seal.prediction_seal_hash,
        "source_inner_support_shift_lock_hash": inner_shifts.lock_hash,
        "target_probe_seal_hash": str(probe_seal["probe_seal_hash"]),
        "target_support_shift_lock_hash": target_shifts.lock_hash,
        "ensemble_utility_surface_hash": utility.surface_hash,
        "feature_surface_set_hash": features.surface_hash,
        "model_set_hash": models.model_set_hash, "plan_set_hash": plans.plan_set_hash,
        "action_library_hash": actions.action_library_hash,
        "global_target_prediction_seal_hash": capability.seal_hash,
        "primary_development_response_count": len(utility.rows),
        "descriptive_development_seed_row_count": len(development_seed_rows),
        "target_prediction_cell_count": len(predictions.cells),
        "target_unique_classifier_fit_count": predictions.unique_classifier_fit_count,
        "target_endpoint_count": len(scores.rows),
        "previous_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "diagnostic_only": True,
    }
    report_path = path / "reports/validation_report.json"
    if not allow_pending and report_path.is_file() and read_json(report_path) != checks:
        raise ProtocolError("Persisted ensemble-endpoint validation report drifted.")
    if not allow_pending:
        state = read_json(path / "reports/run_state.json")
        if state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
            raise ProtocolError("Completed ensemble-endpoint bundle lacks COMPLETE run state.")
    return checks


def _validate_runtime_summary(payload: Mapping[str, object]) -> None:
    counts = payload.get("counts")
    preflight = payload.get("workstation_preflight")
    expected_counts = {
        "source_stream_count": 81,
        "development_prediction_cell_count": 5184,
        "primary_development_response_count": 504,
        "descriptive_development_seed_row_count": 4536,
        "target_probe_cell_count": 729,
        "target_prediction_cell_count": 1053,
        "target_unique_classifier_fit_count": 810,
    }
    if (
        payload.get("schema_version") != "midogpp_stage90_ensemble_endpoint_runtime_summary_v1"
        or not isinstance(counts, Mapping)
        or {str(key): int(value) for key, value in counts.items()} != expected_counts
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("classifier_workers") != 4
        or payload.get("classifier_threads_per_worker") != 3
        or payload.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or payload.get("hash_validated_resume") is not True
        or payload.get("target_unique_classifier_fit_count") != 810
        or payload.get("tf32_enabled") is not False
        or payload.get("amp_enabled") is not False
    ):
        raise ProtocolError("Ensemble-endpoint runtime summary semantics drifted.")


__all__ = ("validate_utility_aligned_ensemble_endpoint_router_bundle",)
