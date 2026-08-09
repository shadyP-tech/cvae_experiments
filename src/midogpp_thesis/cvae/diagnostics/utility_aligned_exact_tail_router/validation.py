"""Independent reconstructive validation of the exact-tail Stage-90 bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import build_exact_tail_action_library
from .artifact_io import json_ready, read_json, render_csv
from .bundle import assert_closed_world, validate_content_index
from .config import (
    UtilityAlignedExactTailRouterConfig,
    load_utility_aligned_exact_tail_router_config,
)
from .contracts import CENTERS
from .development_label_access import open_globally_sealed_development_labels
from .development_prediction_store import (
    DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
    DEVELOPMENT_PREDICTION_INDEX_MEMBER,
    load_development_prediction_store,
)
from .development_scoring import score_exact_tail_development_utility
from .development_seal import (
    GLOBAL_DEVELOPMENT_SEAL_MEMBER,
    DevelopmentPredictionCapability,
    load_global_development_seal,
    validate_global_development_seal,
)
from .feature_production import ACTION_BINDING_COLUMNS, produce_label_free_features
from .inference import (
    CENTER_CONTRAST_COLUMNS,
    CONTRAST_INFERENCE_COLUMNS,
    build_center_contrasts,
    infer_center_contrasts,
)
from .inputs import (
    load_label_free_validation_frame,
    load_metadata_similarity,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .modeling import fit_stage90_models
from .partitions import (
    SUPPORT_PARTITION_COLUMNS,
    build_case_fold_surface,
    build_fixed_partition_surface,
)
from .r2_policy import build_stage90_r2_plan_set
from .reports import (
    development_label_access_payload,
    leakage_report_payload,
    phase_completion_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    runtime_summary_payload,
    scoring_summary_payload,
)
from .runner_persistence import (
    CASE_FOLD_COLUMNS,
    FEATURE_ROW_COLUMNS,
    MODEL_SUMMARY_COLUMNS,
    R2_PLAN_COLUMNS,
    TARGET_ACTION_COLUMNS,
    UTILITY_ROW_COLUMNS,
    case_fold_rows,
    feature_table_rows,
    model_summary_rows,
    r2_plan_rows,
    target_action_rows,
    utility_table_rows,
)
from .scoring import (
    ENSEMBLE_METRIC_COLUMNS,
    ORACLE_DIAGNOSTIC_COLUMNS,
    SEED_METRIC_COLUMNS,
    build_hxe_oracle_diagnostics,
    score_target_probability_ensembles,
    score_target_seed_cells,
)
from .source_cache import load_source_cache
from .source_cache_validation import validate_source_cache_lock
from .target_label_access import open_target_labels_after_global_seal
from .target_prediction_store import read_target_prediction_store
from .target_seal import validate_global_target_prediction_seal


def validate_utility_aligned_exact_tail_router_bundle(
    root: str | Path,
    *,
    config: UtilityAlignedExactTailRouterConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Recompute every derived scientific surface and reject byte drift."""

    path = Path(root).resolve()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending,
    )
    resolved = load_utility_aligned_exact_tail_router_config(
        path / "config.resolved.yaml"
    )
    if config is not None:
        _validate_config_equivalence(resolved, config)

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
    case_folds = build_case_fold_surface(
        partitions, config_contract_hash=resolved.contract_hash
    )
    _assert_csv(
        path / "tables/support_partitions.csv",
        partitions.table_rows,
        SUPPORT_PARTITION_COLUMNS,
    )
    _assert_json(
        path / "manifests/support_partition_lock.json", partitions.lock_payload
    )
    _assert_json(path / "manifests/case_fold_lock.json", case_folds.lock_payload)
    _assert_csv(
        path / "tables/case_folds.csv", case_fold_rows(case_folds), CASE_FOLD_COLUMNS
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
            validation_cache_binding_hash=frame.cache_binding_hash,
            firewall=firewall,
        ),
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
    metadata = load_metadata_similarity(resolved)
    production = produce_label_free_features(
        source_cache, frame, partitions, metadata
    )
    _assert_json(
        path / "manifests/feature_production_lock.json", production.to_payload()
    )
    _assert_json(
        path / "manifests/feature_surface_set.json", production.surfaces.to_payload()
    )
    _assert_csv(
        path / "tables/inner_candidate_features.csv",
        feature_table_rows(production.inner_rows),
        FEATURE_ROW_COLUMNS,
    )
    _assert_csv(
        path / "tables/target_candidate_features.csv",
        feature_table_rows(production.target_rows),
        FEATURE_ROW_COLUMNS,
    )
    _assert_csv(
        path / "tables/development_action_bindings.csv",
        production.development_action_bindings,
        ACTION_BINDING_COLUMNS,
    )

    development = _load_development_capability(path)
    validate_global_development_seal(
        development,
        config=resolved,
        generation_lock=locks.generation,
        source_cache=source_cache,
        partitions=partitions,
        root=path,
    )
    development_labels = open_globally_sealed_development_labels(
        resolved.validation_manifest_path,
        partitions,
        capability=development,
    )
    _assert_json(
        path / "reports/development_label_access_report.json",
        development_label_access_payload(development_labels),
    )
    exact_utility = score_exact_tail_development_utility(
        development, development_labels, partitions
    )
    if (
        tuple(development_labels.labels_by_center) != CENTERS
        or any(row.outer_target_id == row.query_id for row in exact_utility)
    ):
        raise ProtocolError(
            "Utility-aligned all-center development disclosure/H exclusion drifted."
        )
    _assert_csv(
        path / "tables/exact_tail_development_utility.csv",
        utility_table_rows(exact_utility),
        UTILITY_ROW_COLUMNS,
    )
    models = fit_stage90_models(production.surfaces, exact_utility)
    plans = build_stage90_r2_plan_set(models, production.surfaces)
    actions = build_exact_tail_action_library(plans)
    _assert_json(path / "manifests/model_set.json", models.to_payload())
    _assert_csv(
        path / "tables/model_summary.csv",
        model_summary_rows(models),
        MODEL_SUMMARY_COLUMNS,
    )
    _assert_json(path / "manifests/r2_plan_set.json", plans.to_payload())
    _assert_csv(
        path / "tables/r2_plans.csv", r2_plan_rows(plans), R2_PLAN_COLUMNS
    )
    _assert_json(path / "manifests/action_library.json", actions.to_payload())
    _assert_csv(
        path / "tables/target_actions.csv",
        target_action_rows(actions),
        TARGET_ACTION_COLUMNS,
    )

    predictions = read_target_prediction_store(
        path,
        library=actions,
        source_cache_lock_hash=source_lock_hash,
        case_fold_lock_hash=case_folds.lock_hash,
    )
    target_seal = validate_global_target_prediction_seal(
        path,
        config_contract_hash=resolved.contract_hash,
        source_cache_lock_hash=source_lock_hash,
        partitions=partitions,
        case_folds=case_folds,
        library=actions,
        predictions=predictions,
    )
    labels_by_sample, target_label_report = open_target_labels_after_global_seal(
        resolved, partitions, root=path
    )
    _assert_json(
        path / "reports/target_label_access_report.json", target_label_report
    )
    seed_rows = score_target_seed_cells(predictions, labels_by_sample, partitions)
    ensemble_rows = score_target_probability_ensembles(
        predictions, labels_by_sample, partitions
    )
    center_rows = build_center_contrasts(ensemble_rows)
    inference_rows = infer_center_contrasts(center_rows)
    oracle_rows = build_hxe_oracle_diagnostics(ensemble_rows, plans)
    _assert_csv(
        path / "tables/target_seed_metrics.csv", seed_rows, SEED_METRIC_COLUMNS
    )
    _assert_csv(
        path / "tables/target_ensemble_metrics.csv",
        ensemble_rows,
        ENSEMBLE_METRIC_COLUMNS,
    )
    _assert_csv(
        path / "tables/center_contrasts.csv", center_rows, CENTER_CONTRAST_COLUMNS
    )
    _assert_csv(
        path / "tables/contrast_inference.csv",
        inference_rows,
        CONTRAST_INFERENCE_COLUMNS,
    )
    _assert_csv(
        path / "tables/oracle_hxe_diagnostics.csv",
        oracle_rows,
        ORACLE_DIAGNOSTIC_COLUMNS,
    )

    scoring_summary = scoring_summary_payload(
        ensemble_rows, inference_rows, oracle_rows
    )
    leakage = leakage_report_payload(
        support_partition_lock_hash=partitions.lock_hash,
        case_fold_lock_hash=case_folds.lock_hash,
        development_prediction_seal_hash=development.seal.prediction_seal_hash,
        model_set_hash=models.model_set_hash,
        plan_set_hash=plans.plan_set_hash,
        action_library_hash=actions.action_library_hash,
        target_prediction_seal_hash=str(target_seal["seal_hash"]),
        firewall=firewall,
    )
    _assert_json(path / "reports/leakage_report.json", leakage)
    _assert_json(path / "reports/scoring_summary.json", scoring_summary)
    _assert_json(
        path / "reports/publication_decision.json",
        publication_decision_payload(scoring_summary),
    )
    _validate_phase_reports(
        path,
        resolved=resolved,
        source_lock_hash=source_lock_hash,
        production=production,
        development=development,
        exact_utility=exact_utility,
        models=models,
        plans=plans,
        actions=actions,
        predictions=predictions,
        target_seal=target_seal,
        seed_rows=seed_rows,
        ensemble_rows=ensemble_rows,
        center_rows=center_rows,
        inference_rows=inference_rows,
        oracle_rows=oracle_rows,
    )
    _validate_runtime_report(
        path,
        source_cache=source_cache,
        production=production,
        development=development,
        exact_utility=exact_utility,
        actions=actions,
        predictions=predictions,
    )
    checks = {
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "workspace_and_provenance_verified": True,
        "pre_gpu_firewall_verified": True,
        "fixed_two_case_partitions_reconstructed": True,
        "source_cache_and_label_free_features_reconstructed": True,
        "global_development_prediction_seal_verified": True,
        "development_all_center_label_capability_verified": True,
        "outer_H_query_rows_absent_from_plan_H_utility": True,
        "strict_H_q_e_models_reconstructed": True,
        "R2_plan_and_action_library_reconstructed": True,
        "global_target_prediction_seal_verified": True,
        "terminal_scores_inference_and_oracle_recomputed": True,
        "content_index_verified": True,
        "development_utility_row_count": len(exact_utility),
        "target_prediction_cell_count": len(predictions.cells),
        "routing_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
        "policy_or_promotion_authorized": False,
    }
    validate_content_index(path, config_contract_hash=resolved.contract_hash)
    _validate_final_state(path, allow_pending=allow_pending, expected_checks=checks)
    return checks


def _load_development_capability(path: Path) -> DevelopmentPredictionCapability:
    seal_path = path / GLOBAL_DEVELOPMENT_SEAL_MEMBER
    return DevelopmentPredictionCapability(
        store=load_development_prediction_store(path),
        seal=load_global_development_seal(seal_path),
        seal_path=seal_path,
        prediction_index_path=path / DEVELOPMENT_PREDICTION_INDEX_MEMBER,
        prediction_arrays_path=path / DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
    )


def _validate_phase_reports(
    path: Path,
    *,
    resolved: object,
    source_lock_hash: str,
    production: object,
    development: object,
    exact_utility: Sequence[object],
    models: object,
    plans: object,
    actions: object,
    predictions: object,
    target_seal: Mapping[str, object],
    seed_rows: Sequence[object],
    ensemble_rows: Sequence[object],
    center_rows: Sequence[object],
    inference_rows: Sequence[object],
    oracle_rows: Sequence[object],
) -> None:
    contract_hash = str(resolved.contract_hash)
    _assert_json(
        path / "reports/phase_01_source_cache_and_features_complete.json",
        phase_completion_payload(
            "phase_01_source_cache_and_features_complete",
            config_contract_hash=contract_hash,
            bindings={
                "source_cache_lock_hash": source_lock_hash,
                "feature_production_hash": production.production_hash,
                "feature_surface_set_hash": production.surfaces.surface_hash,
                "canonical_inner_action_library_hash": production.canonical_inner_action_library_hash,
                "action_binding_hash": production.action_binding_hash,
            },
            counts={
                "inner_feature_row_count": len(production.inner_rows),
                "target_feature_row_count": len(production.target_rows),
                "development_action_binding_count": len(production.development_action_bindings),
            },
            development_labels_opened=False,
            terminal_target_scoring_opened=False,
        ),
    )
    _assert_json(
        path / "reports/phase_02_development_scoring_and_action_lock_complete.json",
        phase_completion_payload(
            "phase_02_development_scoring_and_action_lock_complete",
            config_contract_hash=contract_hash,
            bindings={
                "development_prediction_seal_hash": development.seal.prediction_seal_hash,
                "model_set_hash": models.model_set_hash,
                "r2_plan_set_hash": plans.plan_set_hash,
                "action_library_hash": actions.action_library_hash,
            },
            counts={
                "exact_tail_utility_row_count": len(exact_utility),
                "heldout_model_count": len(models.by_target),
                "r2_plan_count": len(plans.by_target),
                "target_action_count": actions.action_count,
            },
            development_labels_opened=True,
            terminal_target_scoring_opened=False,
        ),
    )
    _assert_json(
        path / "reports/phase_03_global_target_prediction_seal_complete.json",
        phase_completion_payload(
            "phase_03_global_target_prediction_seal_complete",
            config_contract_hash=contract_hash,
            bindings={
                "action_library_hash": actions.action_library_hash,
                "global_target_prediction_seal_hash": str(target_seal["seal_hash"]),
            },
            counts={
                "target_prediction_cell_count": len(predictions.cells),
                "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
            },
            development_labels_opened=True,
            terminal_target_scoring_opened=False,
        ),
    )
    _assert_json(
        path / "reports/phase_04_terminal_scoring_complete.json",
        phase_completion_payload(
            "phase_04_terminal_scoring_complete",
            config_contract_hash=contract_hash,
            bindings={"global_target_prediction_seal_hash": str(target_seal["seal_hash"])},
            counts={
                "target_seed_metric_count": len(seed_rows),
                "target_ensemble_metric_count": len(ensemble_rows),
                "center_contrast_count": len(center_rows),
                "contrast_inference_count": len(inference_rows),
                "oracle_target_count": len(oracle_rows),
            },
            development_labels_opened=True,
            terminal_target_scoring_opened=True,
        ),
    )


def _validate_runtime_report(
    path: Path,
    *,
    source_cache: object,
    production: object,
    development: object,
    exact_utility: Sequence[object],
    actions: object,
    predictions: object,
) -> None:
    observed = read_json(path / "reports/runtime_summary.json")
    preflight = observed.get("workstation_preflight")
    staging = observed.get("source_cache_staging")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or not isinstance(staging, Mapping)
        or staging.get("attempted") is not True
        or staging.get("used") not in {True, False}
        or staging.get("status")
        not in {
            "STAGED_LOCAL_CPU_CACHE",
            "CANONICAL_ALREADY_LOCAL",
            "CANONICAL_FALLBACK",
        }
    ):
        raise ProtocolError("Utility-aligned runtime/staging report drifted.")
    counts = {
        "source_stream_count": len(source_cache.source_records),
        "support_component_count": len(source_cache.component_records),
        "inner_feature_row_count": len(production.inner_rows),
        "target_feature_row_count": len(production.target_rows),
        "development_prediction_cell_count": development.seal.cell_count,
        "exact_tail_utility_row_count": len(exact_utility),
        "target_action_count": actions.action_count,
        "target_prediction_cell_count": len(predictions.cells),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
    }
    expected = runtime_summary_payload(
        preflight, counts=counts, source_cache_staging=staging
    )
    if observed != expected:
        raise ProtocolError("Utility-aligned runtime summary drifted.")


def _validate_config_equivalence(
    resolved: UtilityAlignedExactTailRouterConfig,
    supplied: UtilityAlignedExactTailRouterConfig,
) -> None:
    path_fields = (
        "artifact_root", "expert_bank_root", "generation_lock_root",
        "equal_union_policy_root", "validation_cache_root",
        "validation_manifest_path", "metadata_profile_root",
    )
    if resolved.contract_hash != supplied.contract_hash or any(
        Path(getattr(resolved, field)).resolve()
        != Path(getattr(supplied, field)).resolve()
        for field in path_fields
    ):
        raise ProtocolError("Utility-aligned supplied/resolved config drifted.")


def _validate_final_state(
    path: Path,
    *,
    allow_pending: bool,
    expected_checks: Mapping[str, object],
) -> None:
    state = read_json(path / "reports/run_state.json")
    if (
        state.get("schema_version")
        != "midogpp_utility_aligned_stage90_run_state_v1"
        or state.get("diagnostic_only") is not True
        or state.get("promotion_eligible") is not False
    ):
        raise ProtocolError("Utility-aligned run-state claim boundary drifted.")
    if allow_pending:
        if state.get("status") not in {"RUNNING", "COMPLETE"}:
            raise ProtocolError("Utility-aligned pending validation state drifted.")
    elif state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
        raise ProtocolError("Utility-aligned final run state is not COMPLETE.")
    if not allow_pending:
        report = read_json(path / "reports/validation_report.json")
        expected = {
            "schema_version": "midogpp_utility_aligned_stage90_validation_report_v1",
            "status": "PASS",
            "validator": "validate_utility_aligned_exact_tail_router_bundle",
            "checks": dict(expected_checks),
        }
        if report != json_ready(expected):
            raise ProtocolError("Utility-aligned validation report drifted.")


def _assert_json(path: Path, expected: Mapping[str, object]) -> None:
    if read_json(path) != json_ready(expected):
        raise ProtocolError(f"Utility-aligned derived JSON drifted: {path.name}.")


def _assert_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"Cannot read utility-aligned CSV: {path}.") from exc
    expected = render_csv(rows, columns).encode("utf-8")
    if observed != expected:
        raise ProtocolError(f"Utility-aligned derived CSV drifted: {path.name}.")


__all__ = ("validate_utility_aligned_exact_tail_router_bundle",)
