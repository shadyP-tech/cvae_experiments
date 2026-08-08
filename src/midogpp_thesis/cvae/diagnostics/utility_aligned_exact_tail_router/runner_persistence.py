"""Deterministic phase persistence for the consumed exact-tail diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .artifact_io import atomic_json, persist_or_validate_csv, persist_or_validate_json
from .contracts import CENTERS
from .feature_production import ACTION_BINDING_COLUMNS, Stage90FeatureProduction
from .inference import CENTER_CONTRAST_COLUMNS, CONTRAST_INFERENCE_COLUMNS
from .input_contracts import row_identity_hash
from .partitions import SUPPORT_PARTITION_COLUMNS
from .reports import (
    development_label_access_payload,
    phase_completion_payload,
    protocol_manifest_payload,
    run_state_payload,
)
from .scoring import (
    ENSEMBLE_METRIC_COLUMNS,
    ORACLE_DIAGNOSTIC_COLUMNS,
    SEED_METRIC_COLUMNS,
)


FEATURE_ROW_COLUMNS = (
    "schema_version", "role", "outer_target_id", "query_id",
    "candidate_source", "training_seed", "generation_seed", "replicate_id",
    "candidate_source_count", "support_partition_hash", "support_case_count",
    "reconstruction_mean", "reconstruction_std", "reconstruction_q25",
    "reconstruction_q50", "reconstruction_q75", "kl_mean", "kl_std",
    "kl_q25", "kl_q50", "kl_q75", "replica_disagreement",
    "distribution_mmd", "metadata_similarity", "feature_semantics", "row_hash",
)
UTILITY_ROW_COLUMNS = (
    "schema_version", "outer_target_id", "query_id", "candidate_source",
    "training_seed", "generation_seed", "replicate_id",
    "candidate_source_count", "support_partition_hash",
    "evaluation_partition_hash", "prediction_seal_hash",
    "base_prediction_hash", "tail_prediction_hash", "base_bacc", "tail_bacc",
    "utility_delta", "support_eval_disjoint", "predictions_sealed_before_labels",
    "source_expert_frozen", "target_labels_used_for_routing",
    "utility_semantics", "row_hash",
)
MODEL_SUMMARY_COLUMNS = (
    "schema_version", "target_center", "model_hash",
    "global_and_interaction_model_hash", "permuted_interaction_model_hash",
    "training_query_ids_json", "training_source_ids_json",
    "strict_H_q_e_exclusion", "heldout_target_labels_used_for_fit",
    "target_support_labels_used_for_fit", "diagnostic_only",
)
R2_PLAN_COLUMNS = (
    "schema_version", "target_center", "plan_hash", "G_delta_source",
    "R2_source", "P_source", "mean_predictions_json", "seed_predictions_json",
    "support_case_count", "all_nine_seed_pairs_retained", "routing_status",
    "development_crossfit_labels_previously_opened",
    "outer_H_development_rows_excluded_from_plan_H",
    "predictions_frozen_before_terminal_target_scoring",
    "outer_H_development_label_rows_used_for_plan_H",
    "terminal_target_labels_used_for_plan",
    "policy_authorized", "fallback_authorized", "promotion_authorized",
    "deployment_authorized", "diagnostic_only",
)
TARGET_ACTION_COLUMNS = (
    "schema_version", "target_center", "action_id", "action_hash",
    "selected_source", "base_per_source_per_class", "topup_total_per_class",
    "final_total_per_class", "topup_counts_json", "final_counts_json",
    "router_plan_hash", "action_geometry_label_free",
    "crossfit_development_utility_labels_used_for_route",
    "outer_H_development_rows_used_for_route",
    "target_support_labels_used_for_route", "terminal_target_labels_used_for_route",
    "policy_authorized", "fallback_authorized",
    "promotion_authorized", "deployment_authorized", "diagnostic_only",
)
CASE_FOLD_COLUMNS = (
    "schema_version", "fold_ordinal", "fold_id", "target_center",
    "heldout_case_id", "fixed_support_case_ids_json",
    "fixed_support_row_identity_hash", "heldout_row_identity_hash", "fold_hash",
    "support_labels_used", "evaluation_labels_used_for_route",
    "other_evaluation_embeddings_used_for_route",
)


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partitions: object,
    case_folds: object,
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
            validation_cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
            firewall=firewall,
        ),
    )
    persist_or_validate_csv(
        root / "tables/support_partitions.csv",
        getattr(partitions, "table_rows"),
        SUPPORT_PARTITION_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/support_partition_lock.json",
        getattr(partitions, "lock_payload"),
    )
    persist_or_validate_json(
        root / "manifests/case_fold_lock.json", getattr(case_folds, "lock_payload")
    )
    persist_or_validate_csv(
        root / "tables/case_folds.csv", case_fold_rows(case_folds), CASE_FOLD_COLUMNS
    )


def persist_source_and_feature_surfaces(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    production: Stage90FeatureProduction,
) -> None:
    persist_or_validate_json(
        root / "manifests/feature_production_lock.json", production.to_payload()
    )
    persist_or_validate_json(
        root / "manifests/feature_surface_set.json", production.surfaces.to_payload()
    )
    persist_or_validate_csv(
        root / "tables/inner_candidate_features.csv",
        feature_table_rows(production.inner_rows),
        FEATURE_ROW_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/target_candidate_features.csv",
        feature_table_rows(production.target_rows),
        FEATURE_ROW_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/development_action_bindings.csv",
        production.development_action_bindings,
        ACTION_BINDING_COLUMNS,
    )
    persist_or_validate_json(
        root / "reports/phase_01_source_cache_and_features_complete.json",
        phase_completion_payload(
            "phase_01_source_cache_and_features_complete",
            config_contract_hash=config_contract_hash,
            bindings={
                "source_cache_lock_hash": source_cache_lock_hash,
                "feature_production_hash": production.production_hash,
                "feature_surface_set_hash": production.surfaces.surface_hash,
                "canonical_inner_action_library_hash": (
                    production.canonical_inner_action_library_hash
                ),
                "action_binding_hash": production.action_binding_hash,
            },
            counts={
                "inner_feature_row_count": len(production.inner_rows),
                "target_feature_row_count": len(production.target_rows),
                "development_action_binding_count": len(
                    production.development_action_bindings
                ),
            },
            development_labels_opened=False,
            terminal_target_scoring_opened=False,
        ),
    )


def persist_development_and_router_surfaces(
    root: Path,
    *,
    config_contract_hash: str,
    development_labels: object,
    utility_rows: Sequence[object],
    models: object,
    plans: object,
    actions: object,
    development_prediction_seal_hash: str,
) -> None:
    persist_or_validate_json(
        root / "reports/development_label_access_report.json",
        development_label_access_payload(development_labels),
    )
    persist_or_validate_csv(
        root / "tables/exact_tail_development_utility.csv",
        utility_table_rows(utility_rows),
        UTILITY_ROW_COLUMNS,
    )
    persist_or_validate_json(root / "manifests/model_set.json", models.to_payload())
    persist_or_validate_csv(
        root / "tables/model_summary.csv", model_summary_rows(models), MODEL_SUMMARY_COLUMNS
    )
    persist_or_validate_json(root / "manifests/r2_plan_set.json", plans.to_payload())
    persist_or_validate_csv(
        root / "tables/r2_plans.csv", r2_plan_rows(plans), R2_PLAN_COLUMNS
    )
    persist_or_validate_json(
        root / "manifests/action_library.json", actions.to_payload()
    )
    persist_or_validate_csv(
        root / "tables/target_actions.csv", target_action_rows(actions), TARGET_ACTION_COLUMNS
    )
    persist_or_validate_json(
        root / "reports/phase_02_development_scoring_and_action_lock_complete.json",
        phase_completion_payload(
            "phase_02_development_scoring_and_action_lock_complete",
            config_contract_hash=config_contract_hash,
            bindings={
                "development_prediction_seal_hash": development_prediction_seal_hash,
                "model_set_hash": str(models.model_set_hash),
                "r2_plan_set_hash": str(plans.plan_set_hash),
                "action_library_hash": str(actions.action_library_hash),
            },
            counts={
                "exact_tail_utility_row_count": len(utility_rows),
                "heldout_model_count": len(models.by_target),
                "r2_plan_count": len(plans.by_target),
                "target_action_count": int(actions.action_count),
            },
            development_labels_opened=True,
            terminal_target_scoring_opened=False,
        ),
    )


def persist_target_seal_phase(
    root: Path,
    *,
    config_contract_hash: str,
    action_library_hash: str,
    target_seal: Mapping[str, object],
    prediction_cell_count: int,
    unique_classifier_fit_count: int,
) -> None:
    persist_or_validate_json(
        root / "reports/phase_03_global_target_prediction_seal_complete.json",
        phase_completion_payload(
            "phase_03_global_target_prediction_seal_complete",
            config_contract_hash=config_contract_hash,
            bindings={
                "action_library_hash": action_library_hash,
                "global_target_prediction_seal_hash": str(target_seal["seal_hash"]),
            },
            counts={
                "target_prediction_cell_count": prediction_cell_count,
                "unique_classifier_fit_count": unique_classifier_fit_count,
            },
            development_labels_opened=True,
            terminal_target_scoring_opened=False,
        ),
    )


def persist_terminal_surfaces(
    root: Path,
    *,
    config_contract_hash: str,
    target_label_report: Mapping[str, object],
    seed_rows: Sequence[Mapping[str, object]],
    ensemble_rows: Sequence[Mapping[str, object]],
    center_contrasts: Sequence[Mapping[str, object]],
    inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    leakage_report: Mapping[str, object],
    scoring_summary: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    target_seal_hash: str,
) -> None:
    persist_or_validate_json(
        root / "reports/target_label_access_report.json", target_label_report
    )
    persist_or_validate_csv(
        root / "tables/target_seed_metrics.csv", seed_rows, SEED_METRIC_COLUMNS
    )
    persist_or_validate_csv(
        root / "tables/target_ensemble_metrics.csv",
        ensemble_rows,
        ENSEMBLE_METRIC_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/center_contrasts.csv",
        center_contrasts,
        CENTER_CONTRAST_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/contrast_inference.csv",
        inference_rows,
        CONTRAST_INFERENCE_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/oracle_hxe_diagnostics.csv",
        oracle_rows,
        ORACLE_DIAGNOSTIC_COLUMNS,
    )
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(root / "reports/scoring_summary.json", scoring_summary)
    persist_or_validate_json(
        root / "reports/publication_decision.json", publication_decision
    )
    # Launch telemetry may legitimately change on a hash-validated resume.
    atomic_json(root / "reports/runtime_summary.json", runtime_summary)
    persist_or_validate_json(
        root / "reports/phase_04_terminal_scoring_complete.json",
        phase_completion_payload(
            "phase_04_terminal_scoring_complete",
            config_contract_hash=config_contract_hash,
            bindings={"global_target_prediction_seal_hash": target_seal_hash},
            counts={
                "target_seed_metric_count": len(seed_rows),
                "target_ensemble_metric_count": len(ensemble_rows),
                "center_contrast_count": len(center_contrasts),
                "contrast_inference_count": len(inference_rows),
                "oracle_target_count": len(oracle_rows),
            },
            development_labels_opened=True,
            terminal_target_scoring_opened=True,
        ),
    )


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    persist_or_validate_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_utility_aligned_stage90_validation_report_v1",
            "status": "PASS",
            "validator": "validate_utility_aligned_exact_tail_router_bundle",
            "checks": dict(checks),
        },
    )


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _row_with_hash(row: object) -> dict[str, object]:
    return {**getattr(row, "to_payload")(), "row_hash": str(getattr(row, "row_hash"))}


def feature_table_rows(rows: Sequence[object]) -> tuple[dict[str, object], ...]:
    return tuple(_row_with_hash(row) for row in rows)


def case_fold_rows(surface: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema_version": "midogpp_utility_aligned_stage90_case_fold_row_v1",
            "fold_ordinal": fold.fold_ordinal,
            "fold_id": fold.fold_id,
            "target_center": fold.target_center,
            "heldout_case_id": fold.heldout_case_id,
            "fixed_support_case_ids_json": _json(
                sorted({row.case_id for row in fold.fixed_support_rows})
            ),
            "fixed_support_row_identity_hash": row_identity_hash(
                fold.fixed_support_rows
            ),
            "heldout_row_identity_hash": fold.heldout_row_identity_hash,
            "fold_hash": fold.fold_hash,
            "support_labels_used": False,
            "evaluation_labels_used_for_route": False,
            "other_evaluation_embeddings_used_for_route": False,
        }
        for fold in surface.folds
    )


def utility_table_rows(rows: Sequence[object]) -> tuple[dict[str, object], ...]:
    return tuple(_row_with_hash(row) for row in rows)


def model_summary_rows(models: object) -> tuple[dict[str, object], ...]:
    output = []
    for target in CENTERS:
        model = models.by_target[target]
        standard = model.global_and_interaction
        output.append({
            "schema_version": "midogpp_utility_aligned_stage90_model_summary_v1",
            "target_center": target,
            "model_hash": model.model_hash,
            "global_and_interaction_model_hash": standard.model_hash,
            "permuted_interaction_model_hash": model.permuted_interaction.model_hash,
            "training_query_ids_json": _json(model.training_query_ids),
            "training_source_ids_json": _json(model.training_source_ids),
            "strict_H_q_e_exclusion": True,
            "heldout_target_labels_used_for_fit": False,
            "target_support_labels_used_for_fit": False,
            "diagnostic_only": True,
        })
    return tuple(output)


def r2_plan_rows(plans: object) -> tuple[dict[str, object], ...]:
    output = []
    for target in CENTERS:
        plan = plans.by_target[target]
        selected = plan.proposed_source_by_router
        output.append({
            "schema_version": "midogpp_utility_aligned_stage90_r2_plan_summary_v1",
            "target_center": target,
            "plan_hash": plan.plan_hash,
            "G_delta_source": selected["G_delta"],
            "R2_source": selected["R2"],
            "P_source": selected["P"],
            "mean_predictions_json": _json(plan.mean_prediction_by_router_source),
            "seed_predictions_json": _json(plan.seed_predictions_by_router_source),
            "support_case_count": plan.support_case_count,
            "all_nine_seed_pairs_retained": True,
            "routing_status": plan.routing_status,
            "development_crossfit_labels_previously_opened": (
                plan.development_crossfit_labels_previously_opened
            ),
            "outer_H_development_rows_excluded_from_plan_H": (
                plan.outer_H_development_rows_excluded_from_plan_H
            ),
            "predictions_frozen_before_terminal_target_scoring": (
                plan.predictions_frozen_before_terminal_target_scoring
            ),
            "outer_H_development_label_rows_used_for_plan_H": (
                plan.outer_H_development_label_rows_used_for_plan_H
            ),
            "terminal_target_labels_used_for_plan": (
                plan.terminal_target_labels_used_for_plan
            ),
            "policy_authorized": False,
            "fallback_authorized": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "diagnostic_only": True,
        })
    return tuple(output)


def target_action_rows(library: object) -> tuple[dict[str, object], ...]:
    output = []
    for target in CENTERS:
        for action in library.actions_by_target[target]:
            label_provenance = action.to_payload()
            output.append({
                "schema_version": "midogpp_utility_aligned_stage90_target_action_summary_v1",
                "target_center": target,
                "action_id": action.action_id,
                "action_hash": action.action_hash,
                "selected_source": action.selected_source,
                "base_per_source_per_class": action.base_per_source_per_class,
                "topup_total_per_class": action.topup_total_per_class,
                "final_total_per_class": action.final_total_per_class,
                "topup_counts_json": _json(action.topup_counts_by_source),
                "final_counts_json": _json(action.final_counts_by_class),
                "router_plan_hash": action.router_plan_hash,
                "action_geometry_label_free": label_provenance["action_geometry_label_free"],
                "crossfit_development_utility_labels_used_for_route": label_provenance[
                    "crossfit_development_utility_labels_used_for_route"
                ],
                "outer_H_development_rows_used_for_route": label_provenance[
                    "outer_H_development_rows_used_for_route"
                ],
                "target_support_labels_used_for_route": label_provenance[
                    "target_support_labels_used_for_route"
                ],
                "terminal_target_labels_used_for_route": label_provenance[
                    "terminal_target_labels_used_for_route"
                ],
                "policy_authorized": False,
                "fallback_authorized": False,
                "promotion_authorized": False,
                "deployment_authorized": False,
                "diagnostic_only": True,
            })
    return tuple(output)


def _json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = (
    "CASE_FOLD_COLUMNS", "FEATURE_ROW_COLUMNS", "MODEL_SUMMARY_COLUMNS", "R2_PLAN_COLUMNS",
    "TARGET_ACTION_COLUMNS", "UTILITY_ROW_COLUMNS", "feature_table_rows",
    "case_fold_rows", "utility_table_rows", "model_summary_rows", "r2_plan_rows",
    "target_action_rows", "persist_initial_surfaces",
    "persist_source_and_feature_surfaces", "persist_development_and_router_surfaces",
    "persist_target_seal_phase", "persist_terminal_surfaces",
    "persist_validation_report", "write_run_state",
)
