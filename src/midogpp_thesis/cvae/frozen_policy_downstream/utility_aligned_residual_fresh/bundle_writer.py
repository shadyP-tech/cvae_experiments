"""Canonical writer for a sealed Stage-70 utility-aligned result."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .bundle_io import (
    atomic_copy,
    atomic_json,
    primary_result_payload,
    read_json,
    structural_checks,
    write_content_index,
    write_table,
)
from .config import INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID, UtilityAlignedResidualFreshConfig
from .contracts import CENTERS, EvaluationPlan, FreshEvaluationReport
from .policy_loading import FrozenUtilityAlignedPolicySurface
from .prediction_cache import PredictionCache, write_prediction_index
from .prediction_seal import (
    PredictionSealCapability,
    PredictionSealSummary,
    validate_prediction_seal,
)
from .source_cache import FreshSourceCache
from .target_surface import FreshTargetSurface


def write_utility_aligned_residual_fresh_bundle(
    root: str | Path,
    *,
    config: UtilityAlignedResidualFreshConfig,
    policy: FrozenUtilityAlignedPolicySurface,
    target_surface: FreshTargetSurface,
    source_cache: FreshSourceCache,
    prediction_cache: PredictionCache,
    plan: EvaluationPlan,
    prediction_seal: PredictionSealCapability,
    report: FreshEvaluationReport,
    workstation_report: Mapping[str, object],
) -> Mapping[str, object]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    summary = validate_prediction_seal(prediction_seal, expected_plan=plan)
    if report.prediction_seal_hash != summary.seal_hash:
        from ...protocol import ProtocolError

        raise ProtocolError("Utility-aligned report escaped its prediction seal.")
    atomic_copy(config.source_path, output / "config.resolved.yaml")
    _write_manifests(
        output,
        config=config,
        policy=policy,
        target=target_surface,
        source=source_cache,
        prediction=prediction_cache,
        plan=plan,
        summary=summary,
    )
    atomic_json(output / "reports/workstation_preflight.json", workstation_report)
    atomic_json(
        output / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_leakage_v1",
            "status": "PASS",
            "support_evaluation_case_disjoint": True,
            "target_expert_excluded": True,
            "target_labels_available_before_prediction_seal": False,
            "source_experts_updated": False,
            "consumed_stage70_used": False,
            "consumed_stage90_used": False,
            "oracle_policy_update_emitted": False,
        },
    )
    write_prediction_index(output / "tables/prediction_index.csv", prediction_cache)
    atomic_json(
        output / "reports/label_access.json",
        {
            "schema_version": "midogpp_utility_aligned_label_access_v1",
            "prediction_seal_hash": summary.seal_hash,
            "scoring_manifest_sha256": target_surface.scoring_manifest_sha256,
            "evaluation_row_count": sum(
                len(target_surface.frames_by_center[center].evaluation_row_ids)
                for center in CENTERS
            ),
            "labels_opened_after_complete_global_prediction_seal": True,
            "labels_available_to_generation_or_prediction": False,
        },
    )
    _write_result_tables(output, report)
    atomic_json(output / "reports/primary_result.json", primary_result_payload(report))
    atomic_json(
        output / "reports/run_state.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_run_state_v1",
            "status": "COMPLETE",
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "prediction_seal_hash": summary.seal_hash,
            "policy_update_emitted": False,
        },
    )
    checks = structural_checks()
    atomic_json(
        output / "reports/validation_report.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_validation_v1",
            "status": "PENDING_RECONSTRUCTION",
            "validator": "validate_utility_aligned_residual_fresh_bundle",
            "checks": checks,
        },
    )
    write_content_index(output)
    from .bundle_validation import validate_utility_aligned_residual_fresh_bundle

    reconstructed = validate_utility_aligned_residual_fresh_bundle(
        output, config=config, allow_pending=True
    )
    final_checks = {
        **checks,
        **dict(reconstructed),
        "reconstructive_validation_passed": True,
        "content_index_hash": read_json(
            output / "manifests/content_index.json"
        )["content_hash"],
    }
    atomic_json(
        output / "reports/validation_report.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_validation_v1",
            "status": "PASS",
            "validator": "validate_utility_aligned_residual_fresh_bundle",
            "checks": final_checks,
        },
    )
    return validate_utility_aligned_residual_fresh_bundle(output, config=config)


def _write_manifests(
    output: Path,
    *,
    config: UtilityAlignedResidualFreshConfig,
    policy: FrozenUtilityAlignedPolicySurface,
    target: FreshTargetSurface,
    source: FreshSourceCache,
    prediction: PredictionCache,
    plan: EvaluationPlan,
    summary: PredictionSealSummary,
) -> None:
    atomic_json(
        output / "provenance/input_artifacts.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_inputs_v1",
            "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
            "policy_lock_hash": policy.policy_lock_hash,
            "action_library_hash": policy.action_library_hash,
            "target_policy_lock_hash": policy.policy_payload.get("target_policy_lock_hash"),
            "exact_tail_utility_surface_lock_hash": policy.exact_tail_utility_surface_lock_hash,
            "reservation_id": target.reservation.reservation_id,
            "reservation_hash": target.reservation.reservation_hash,
            "target_evaluation_binding_hash": target.reservation.target_evaluation_binding_hash,
            "target_cache_content_hash": target.cache_content_hash,
            "source_cache_hash": source.cache_hash,
            "prediction_cache_hash": prediction.cache_hash,
            "consumed_stage70_used": False,
            "consumed_stage90_used": False,
        },
    )
    atomic_json(
        output / "manifests/protocol_manifest.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_protocol_v1",
            "dataset_family": "MIDOG++",
            "stage": "70_frozen_policy_downstream",
            "claim_scope": "synthetic_downstream_utility",
            "config_contract_hash": config.contract_hash,
            "target_support_evaluation_case_disjoint": True,
            "minimum_independent_support_cases_per_target": 8,
            "typed_case_bootstrap_plan_validated": True,
            "target_feature_geometry_validated": True,
            "bootstrap_surfaces_validated": True,
            "outer_target_expert_excluded": True,
            "all_logical_predictions_sealed_before_labels": True,
            "logical_prediction_count": summary.logical_prediction_count,
            "unique_composition_fit_count": prediction.unique_composition_fit_count,
            "inference_unit": "target_center",
            "inference_center_count": len(CENTERS),
            "seed_cells_are_technical_repetitions": True,
            "prior_cardinality_transfer_role": "eligibility_only",
            "prior_expected_improvement_claimed": False,
            "fresh_stage70_router_contrasts": ["R-G_delta", "R-U", "R-B", "R-P"],
            "oracle_diagnostics_terminal_only": True,
            "policy_update_emitted": False,
        },
    )
    atomic_json(
        output / "manifests/policy_binding.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_policy_binding_v1",
            "policy_lock_hash": policy.policy_lock_hash,
            "action_library_hash": policy.action_library_hash,
            "target_policy_lock_hash": policy.policy_payload.get("target_policy_lock_hash"),
            "target_feature_surface_hash": policy.policy_payload.get("feature_surface_hash"),
            "target_feature_schema_hash": policy.policy_payload.get("feature_schema_hash"),
            "target_reservation_artifact_id": policy.policy_payload.get("target_reservation_artifact_id"),
            "target_reservation_hash": policy.policy_payload.get("target_reservation_hash"),
            "target_evaluation_binding_hash": policy.policy_payload.get("target_evaluation_binding_hash"),
            "frozen_before_label_access": True,
            "target_evaluation_labels_used": False,
        },
    )
    atomic_json(
        output / "manifests/evaluation_plan.json",
        {
            "schema_version": "midogpp_utility_aligned_fresh_plan_summary_v1",
            "plan_hash": plan.plan_hash,
            "logical_prediction_count": len(plan.logical_cells),
            "unique_composition_cell_count": len(plan.composition_cells),
            "logical_actions_by_target": {
                target_id: [action.to_payload() for action in plan.actions_by_target[target_id]]
                for target_id in CENTERS
            },
            "composition_deduplication_does_not_collapse_logical_actions": True,
        },
    )
    atomic_json(
        output / "manifests/prediction_seal.json",
        {
            "schema_version": "midogpp_utility_aligned_prediction_seal_summary_v1",
            **asdict(summary),
        },
    )


def _write_result_tables(output: Path, report: FreshEvaluationReport) -> None:
    for name, rows in (
        ("seed_cell_metrics", report.scored.seed_cell_metrics),
        ("ensemble_metrics", report.scored.ensemble_metrics),
        ("center_contrasts", report.center_contrasts),
        ("contrast_inference", report.contrast_inference),
        ("oracle_diagnostics", report.oracle_diagnostics),
    ):
        write_table(output / f"tables/{name}.csv", [asdict(row) for row in rows])


__all__ = ("write_utility_aligned_residual_fresh_bundle",)
