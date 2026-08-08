"""Closed-world bundle writer for the fresh fixed-policy Stage-70 study."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle_io import (
    atomic_json,
    load_workspace_provenance,
    write_content_index,
    write_dataclass_csv,
    write_resolved_config,
    write_validation_report,
)
from .bundle_schema import (
    CENTER_CONTRAST_COLUMNS,
    CONTENT_INDEX_EXCLUSIONS,
    ENSEMBLE_METRIC_COLUMNS,
    INFERENCE_COLUMNS,
    ORACLE_COLUMNS,
    REQUIRED_FILES,
    SEED_METRIC_COLUMNS,
    STATIC_CONTENT_INDEX_MEMBERS,
)
from .config import ResidualTopupFreshConfig
from .contracts import (
    EXPECTED_ENSEMBLE_METRIC_COUNT,
    EXPECTED_PLAN_CELL_COUNT,
    EvaluationPlan,
    FreshEvaluationReport,
)
from .execution import (
    FrozenPolicySurface,
    PredictionCache,
    write_prediction_index,
)
from .prediction_seal import PredictionSealCapability, validate_prediction_seal
from .publication_decision import build_publication_decision
from .source_cache import FreshSourceCache
from .target_cache import FreshTargetSurface


def write_residual_topup_fresh_bundle(
    root: str | Path,
    *,
    config: ResidualTopupFreshConfig,
    policy: FrozenPolicySurface,
    target_surface: FreshTargetSurface,
    source_cache: FreshSourceCache,
    prediction_cache: PredictionCache,
    plan: EvaluationPlan,
    prediction_seal: PredictionSealCapability,
    report: FreshEvaluationReport,
    workstation_report: Mapping[str, object],
) -> dict[str, object]:
    """Write the immutable reports/tables after labels have opened for scoring."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    summary = validate_prediction_seal(prediction_seal, expected_plan=plan)
    if (
        report.policy_update_emitted is not False
        or report.prediction_seal_hash != summary.seal_hash
        or len(report.scored.seed_cell_metrics) != EXPECTED_PLAN_CELL_COUNT
        or len(report.scored.ensemble_metrics) != EXPECTED_ENSEMBLE_METRIC_COUNT
        or prediction_cache.plan_hash != plan.plan_hash
    ):
        raise ProtocolError("Fresh Stage-70 bundle inputs escaped their frozen seal.")

    write_resolved_config(output / "config.resolved.yaml", config.source_path)
    provenance = load_workspace_provenance(
        output / "provenance/input_artifacts.json",
        config=config,
    )
    protocol = {
        "schema_version": "midogpp_residual_topup_fresh_protocol_manifest_v1",
        "experiment_id": config.experiment_id,
        "output_artifact_id": config.output_artifact_id,
        "claim_scope": "synthetic_downstream_utility",
        "config_contract_hash": config.contract_hash,
        "input_binding_hash": stable_hash(provenance),
        "policy_lock_hash": policy.policy_lock_hash,
        "action_library_hash": policy.action_library_hash,
        "reservation_id": target_surface.reservation.reservation_id,
        "reservation_hash": target_surface.reservation.reservation_hash,
        "target_cache_content_hash": target_surface.cache_content_hash,
        "target_cache_protocol_hash": target_surface.cache_protocol_hash,
        "scoring_manifest_sha256": target_surface.scoring_manifest_sha256,
        "source_cache_hash": source_cache.cache_hash,
        "prediction_cache_hash": prediction_cache.cache_hash,
        "evaluation_plan_hash": plan.plan_hash,
        "prediction_seal_hash": summary.seal_hash,
        "protocol": dict(config.protocol),
        "evaluation": dict(config.evaluation),
        "classifier": config.classifier.to_payload(),
        "runtime": dict(config.runtime),
        "workstation_preflight": dict(workstation_report),
        "all_actions_frozen_before_target_cache_extraction": True,
        "all_predictions_sealed_before_labels": True,
        "labels_used_for_scoring_only": True,
        "policy_update_emitted": False,
        "oracle_action_exported": False,
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    atomic_json(output / "manifests/protocol_manifest.json", protocol)
    policy_binding = {
        "schema_version": "midogpp_residual_topup_fresh_policy_binding_v1",
        "policy_lock_hash": policy.policy_lock_hash,
        "action_library_hash": policy.action_library_hash,
        "reservation_id": target_surface.reservation.reservation_id,
        "support_case_ids_by_center": {
            center: list(values)
            for center, values in target_surface.reservation.support_case_ids_by_center.items()
        },
        "evaluation_case_ids_by_center": {
            center: list(values)
            for center, values in target_surface.reservation.evaluation_case_ids_by_center.items()
        },
        "policy_frozen_before_target_cache_extraction": True,
        "policy_update_emitted": False,
    }
    policy_binding["policy_binding_hash"] = stable_hash(policy_binding)
    atomic_json(output / "manifests/policy_binding.json", policy_binding)
    plan_payload = {
        "schema_version": "midogpp_residual_topup_fresh_evaluation_plan_v1",
        "plan_hash": plan.plan_hash,
        "actions_by_target": {
            target: [action.to_payload() for action in plan.actions_by_target[target]]
            for target in plan.actions_by_target
        },
        "cells": [
            {
                "target_center": cell.target_center,
                "training_seed": cell.training_seed,
                "generation_seed": cell.generation_seed,
                "action_id": cell.action_id,
                "action_hash": cell.action_hash,
            }
            for cell in plan.cells
        ],
        "evaluation_row_ids_by_target": {
            target: list(rows)
            for target, rows in plan.evaluation_row_ids_by_target.items()
        },
        "primary_endpoint": plan.primary_endpoint,
        "seed_cell_endpoint_role": plan.seed_cell_endpoint_role,
        "labels_available_to_planning": False,
    }
    atomic_json(output / "manifests/evaluation_plan.json", plan_payload)
    seal_payload = {
        "schema_version": "midogpp_residual_topup_fresh_prediction_seal_v1",
        "prediction_seal_hash": summary.seal_hash,
        "plan_hash": summary.plan_hash,
        "prediction_cell_count": summary.prediction_cell_count,
        "target_count": summary.target_count,
        "action_seed_coverage_complete": summary.action_seed_coverage_complete,
        "row_coverage_complete": summary.row_coverage_complete,
        "all_predictions_sealed_before_labels": True,
        "labels_opened_at_seal_time": False,
    }
    atomic_json(output / "manifests/prediction_seal.json", seal_payload)

    write_prediction_index(
        output / "tables/prediction_index.csv", prediction_cache.index_rows
    )
    write_dataclass_csv(
        output / "tables/seed_cell_metrics.csv",
        report.scored.seed_cell_metrics,
        SEED_METRIC_COLUMNS,
    )
    write_dataclass_csv(
        output / "tables/ensemble_metrics.csv",
        report.scored.ensemble_metrics,
        ENSEMBLE_METRIC_COLUMNS,
    )
    write_dataclass_csv(
        output / "tables/center_contrasts.csv",
        report.center_contrasts,
        CENTER_CONTRAST_COLUMNS,
    )
    write_dataclass_csv(
        output / "tables/contrast_inference.csv",
        report.contrast_inference,
        INFERENCE_COLUMNS,
    )
    write_dataclass_csv(
        output / "tables/oracle_diagnostics.csv",
        report.oracle_diagnostics,
        ORACLE_COLUMNS,
    )
    atomic_json(
        output / "reports/label_access_report.json",
        {
            "schema_version": "midogpp_residual_topup_fresh_label_access_v1",
            "status": "PASS",
            "prediction_seal_hash": summary.seal_hash,
            "prediction_cell_count_before_label_access": summary.prediction_cell_count,
            "scoring_manifest_sha256": target_surface.scoring_manifest_sha256,
            "labels_opened_only_after_global_prediction_seal": True,
            "labels_available_to_generation": False,
            "labels_available_to_fit_or_predict": False,
            "labels_used_for_scoring_only": True,
        },
    )
    atomic_json(
        output / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_residual_topup_fresh_leakage_v1",
            "status": "PASS",
            "fresh_unconsumed_surface": True,
            "support_evaluation_cases_globally_disjoint": True,
            "policy_frozen_before_target_cache_extraction": True,
            "target_expert_excluded": True,
            "consumed_test_used": False,
            "consumed_validation_used": False,
            "consumed_stage70_used": False,
            "consumed_stage90_used": False,
            "seed_selection_performed": False,
            "policy_update_emitted": False,
            "oracle_action_exported": False,
        },
    )
    decision = build_publication_decision(report)
    atomic_json(output / "reports/publication_decision.json", decision)
    atomic_json(
        output / "reports/run_state.json",
        {
            "schema_version": "midogpp_residual_topup_fresh_run_state_v1",
            "status": "COMPLETE",
            "claim_scope": "synthetic_downstream_utility",
            "prediction_seal_hash": summary.seal_hash,
            "policy_update_emitted": False,
        },
    )
    return write_content_index(output)


__all__ = (
    "CENTER_CONTRAST_COLUMNS",
    "CONTENT_INDEX_EXCLUSIONS",
    "ENSEMBLE_METRIC_COLUMNS",
    "INFERENCE_COLUMNS",
    "ORACLE_COLUMNS",
    "REQUIRED_FILES",
    "SEED_METRIC_COLUMNS",
    "STATIC_CONTENT_INDEX_MEMBERS",
    "write_content_index",
    "write_residual_topup_fresh_bundle",
    "write_validation_report",
)
