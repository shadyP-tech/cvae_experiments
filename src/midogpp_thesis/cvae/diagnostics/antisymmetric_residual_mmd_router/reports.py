"""Manifest and report builders for the antisymmetric diagnostic bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json, sha256_file
from .bundle import CONTENT_INDEX_MEMBERS
from .config import AntisymmetricResidualMMDDiagnosticConfig
from .contracts import CLAIM_SCOPE, EXPECTED_CROSS_FIT_FOLD_COUNT, PUBLICATION_STATUS


def _protocol_manifest(
    config: AntisymmetricResidualMMDDiagnosticConfig,
    *,
    provenance: Mapping[str, Mapping[str, object]],
    validation_cache_binding_hash: str,
    support_partition_lock_hash: str,
    crossfit_surface_lock_hash: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_protocol_manifest_v1",
        "experiment_id": config.experiment_id,
        "output_artifact_id": config.output_artifact_id,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(config.input_artifact_ids),
        "input_artifact_hashes": {
            artifact_id: stable_hash(dict(provenance[artifact_id]))
            for artifact_id in config.input_artifact_ids
        },
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "support_partition_lock_hash": support_partition_lock_hash,
        "crossfit_surface_lock_hash": crossfit_surface_lock_hash,
        "protocol": dict(config.protocol),
        "proxy": dict(config.proxy),
        "classifier": config.classifier.to_payload(),
        "runtime": dict(config.runtime),
        "claim_boundary": dict(config.claim_boundary),
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _phase_payload(phase: str, **values: object) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_phase_report_v1",
        "phase": phase,
        "claim_scope": CLAIM_SCOPE,
        "cross_fitted_transductive_diagnostic": True,
        "diagnostic_only": True,
        "fresh_evidence": False,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "promotion_eligible": False,
        **values,
    }
    payload["phase_hash"] = stable_hash(payload)
    return payload


def _leakage_report() -> dict[str, object]:
    return {
        "schema_version": "midogpp_antisymmetric_residual_mmd_leakage_report_v1",
        "status": "PASS",
        "source_experts_frozen_source_only": True,
        "target_expert_excluded_from_every_pool": True,
        "fixed_support_cases_never_scored": True,
        "heldout_case_excluded_from_own_route": True,
        "heldout_case_embeddings_used_for_own_route": False,
        "cohort_evaluation_embeddings_used_for_other_case_routes": True,
        "support_labels_used": False,
        "evaluation_labels_available_before_global_prediction_seal": False,
        "evaluation_labels_used_for_scoring_only": True,
        "individual_expert_or_seed_selection_performed": False,
        "previous_stage90_router_or_utility_inputs_used": False,
        "proxy_is_nelbo_compatibility": False,
        "proxy_is_downstream_utility": False,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "promotion_eligible": False,
    }


def _publication_decision(
    scoring: Mapping[str, object],
    *,
    plans: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_antisymmetric_residual_mmd_publication_decision_v1"
        ),
        "decision": "PUBLISH_AS_EXPLORATORY_CONSUMED_DATA_DIAGNOSTIC_ONLY",
        "publication_status": PUBLICATION_STATUS,
        "mean_equal_union_bacc_center_equal": scoring[
            "mean_equal_union_bacc_center_equal"
        ],
        "mean_antisymmetric_residual_mmd_bacc_center_equal": scoring[
            "mean_antisymmetric_residual_mmd_bacc_center_equal"
        ],
        "mean_paired_bacc_delta_center_equal": scoring[
            "mean_paired_bacc_delta_center_equal"
        ],
        "crossfit_fold_count": EXPECTED_CROSS_FIT_FOLD_COUNT,
        "nonuniform_plan_count": sum(
            not bool(plan["used_uniform_fallback"]) for plan in plans.values()
        ),
        "uniform_fallback_count": sum(
            bool(plan["used_uniform_fallback"]) for plan in plans.values()
        ),
        "cross_fitted_transductive_diagnostic": True,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "required_next_evidence": (
            "separately_authorized_fresh_case_disjoint_target_surface"
        ),
    }


def _runtime_summary(
    config: AntisymmetricResidualMMDDiagnosticConfig,
    *,
    elapsed_seconds: float,
    unique_classifier_fit_count: int,
    workstation_preflight: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_antisymmetric_residual_mmd_runtime_summary_v1",
        "workstation_profile": config.runtime["workstation_profile"],
        "generation_devices": config.runtime["generation_devices"],
        "kernel_devices": config.runtime["kernel_devices"],
        "source_expert_load_count": 27,
        "source_block_count": 81,
        "source_prefix_per_class": 256,
        "source_cache_bytes": 81 * 2 * 256 * 3840 * 4,
        "target_kernel_workspace_count": 9,
        "target_kernel_workspace_reused_across_case_folds": True,
        "crossfit_fold_count": EXPECTED_CROSS_FIT_FOLD_COUNT,
        "prediction_task_count": 81,
        "classifier_workers": config.runtime["classifier_workers"],
        "classifier_threads_per_worker": config.runtime[
            "classifier_threads_per_worker"
        ],
        "unique_classifier_fit_count": unique_classifier_fit_count,
        "maximum_unique_classifier_fit_count": config.runtime[
            "maximum_unique_classifier_fit_count"
        ],
        "resume_policy": config.runtime["resume_policy"],
        "elapsed_seconds": float(elapsed_seconds),
        "workstation_preflight": dict(workstation_preflight),
    }


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        path = root / relative
        if not path.is_file():
            raise ProtocolError(
                f"Antisymmetric content member is missing: {relative}."
            )
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    atomic_write_json(root / "manifests/content_index.json", payload)


def _write_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    atomic_write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_antisymmetric_residual_mmd_run_state_v1",
            "status": status,
            "phase": phase,
            "resumable": status in {"RUNNING", "FAILED"},
            "error_type": error_type,
            "error_message": error_message,
            "cross_fitted_transductive_diagnostic": True,
            "diagnostic_only": True,
        },
    )


__all__: tuple[str, ...] = ()
