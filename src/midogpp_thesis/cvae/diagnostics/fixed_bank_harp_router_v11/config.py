"""Strict, path-independent configuration for the fenced HARP diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .identity import (
    CLAIM_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    claim_boundary_payload,
)


INPUT_ARTIFACT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_harp_source_train_full_test_cache_v11",
    "midogpp_stage90_harp_source_train_label_capability_v11",
    "midogpp_stage90_harp_full_test_evaluation_release_v11",
    "midogpp_uniform_b_test_consumption_ledger_harp_parent_v11",
    "midogpp_uniform_b_test_consumption_ledger_harp_execution_amendment_v11",
)

_TOP = frozenset({"experiment", "inputs", "protocol", "model", "runtime", "claim_boundary"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOCATION_ROLES = (
    "expert_bank_root",
    "generation_lock_root",
    "test_cache_root",
    "development_manifest_path",
    "evaluation_manifest_path",
    "parent_ledger_path",
    "execution_amendment_path",
)
_HASH_ROLES = (
    "test_cache_content_sha256",
    "development_manifest_sha256",
    "evaluation_manifest_sha256",
    "parent_ledger_sha256",
    "execution_amendment_sha256",
)


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP v11 config section is malformed: {key}.")
    return value


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"HARP v11 {name} must be null or SHA-256.")
    return value


def _location(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"HARP v11 {name} is not a canonical location.")
    lowered = value.lower()
    forbidden = (
        "fixed_bank_sceptre",
        "source_inner_candidate_utility",
        "fixed_bank_harp_router/v1",
        "fixed_bank_harp_router/v2",
        "fixed_bank_harp_router/v3",
        "fixed_bank_harp_router/v4",
        "fixed_bank_harp_router/v5",
        "fixed_bank_harp_router/v6",
        "fixed_bank_harp_router/v7",
        "fixed_bank_harp_router/v8",
        "fixed_bank_harp_router/v9",
        "fixed_bank_harp_router/v10",
        "harp_router_v1",
        "harp_router_v2",
        "harp_router_v3",
        "harp_router_v4",
        "harp_router_v5",
        "harp_router_v6",
        "harp_router_v7",
        "harp_router_v8",
        "harp_router_v9",
        "harp_router_v10",
        "harp_consumed_test_cache_v1",
        "harp_consumed_test_cache_v2",
        "harp_consumed_test_cache_v3",
        "harp_consumed_test_cache_v4",
        "harp_consumed_test_cache_v5",
        "harp_consumed_test_cache_v6",
        "harp_consumed_test_cache_v7",
        "harp_consumed_test_cache_v8",
        "harp_consumed_test_cache_v9",
        "harp_consumed_test_cache_v10",
        "harp_source_train_full_test_cache_v9",
        "harp_source_train_label_capability_v9",
        "harp_full_test_evaluation_release_v9",
        "harp_source_train_full_test_cache_v10",
        "harp_source_train_label_capability_v10",
        "harp_full_test_evaluation_release_v10",
        "harp_v1_execution",
        "harp_v2_execution",
        "harp_v3_execution",
        "harp_v4_execution",
        "harp_v5_execution",
        "harp_v6_execution",
        "harp_v7_execution",
        "harp_v8_execution",
        "harp_v9_execution",
        "harp_v10_execution",
        "source_active_selective_router_v7",
        "baseline_inclusive_action_safe_router_v8",
        "policy_calibrated_residual_router_v9",
        "policy_calibrated_residual_router_v10",
        "dense_residual_soft_router",
        "compatibility_conditioned_directional_router",
    )
    if any(
        re.search(re.escape(fragment) + r"(?![0-9])", lowered) is not None
        for fragment in forbidden
    ):
        raise ProtocolError("HARP v11 cannot consume a predecessor path or policy surface.")
    return value


@dataclass(frozen=True, slots=True)
class HarpStage90V11Config:
    source_path: Path
    artifact_root: str
    input_locations: Mapping[str, str]
    expected_hashes: Mapping[str, str | None]
    execution_authorized: bool
    protocol: Mapping[str, object]
    model: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    config_hash: str
    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    execution_revision: str = EXECUTION_REVISION

    def resolved_path(self, role: str) -> Path:
        try:
            raw = self.input_locations[role]
        except KeyError as exc:
            raise ProtocolError(f"HARP v11 input role is unknown: {role}.") from exc
        if "://" in raw:
            raise ProtocolError("HARP v11 production requires workspace-resolved inputs.")
        return Path(raw).resolve()

    @property
    def expected_execution_amendment_sha256(self) -> str | None:
        return self.expected_hashes["execution_amendment_sha256"]


def load_config(path: str | Path) -> HarpStage90V11Config:
    """Parse without resolving or opening any referenced artifact path."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read HARP v11 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP:
        raise ProtocolError("HARP v11 top-level config drifted.")

    experiment = _section(raw, "experiment")
    fixed_experiment = {
        "schema_version": "midogpp_harp_stage90_experiment_v11",
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "execution_revision": EXECUTION_REVISION,
        "implementation_authorizes_execution": False,
        "single_use_execution_identity": True,
        "consumed_test_reuse": True,
    }
    if set(experiment) != {*fixed_experiment, "status", "execution_authorized", "artifact_root"}:
        raise ProtocolError("HARP v11 experiment schema drifted.")
    if any(experiment.get(key) != value for key, value in fixed_experiment.items()):
        raise ProtocolError("HARP v11 experiment identity drifted.")
    authorized = experiment.get("execution_authorized")
    if type(authorized) is not bool:
        raise ProtocolError("HARP v11 authorization flag must be Boolean.")
    expected_status = "diagnostic" if authorized else "planned"
    if experiment.get("status") != expected_status:
        raise ProtocolError("HARP v11 status/authorization state drifted.")
    artifact_root = _location(experiment.get("artifact_root"), name="artifact root")
    if artifact_root.startswith("output://") and artifact_root != f"output://{OUTPUT_ARTIFACT_ID}":
        raise ProtocolError("HARP v11 output identity drifted.")

    inputs = _section(raw, "inputs")
    if set(inputs) != {
        "schema_version",
        "direct_input_artifact_ids",
        "expert_bank_lock_hash",
        "generation_lock_hash",
        *_LOCATION_ROLES,
        *_HASH_ROLES,
    }:
        raise ProtocolError("HARP v11 input schema drifted.")
    if inputs.get("schema_version") != "midogpp_harp_stage90_exact_seven_inputs_v11" or tuple(
        inputs.get("direct_input_artifact_ids", ())
    ) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("HARP v11 direct input inventory drifted.")
    hashes: dict[str, str | None] = {
        role: _optional_sha256(inputs.get(role), name=role) for role in _HASH_ROLES
    }
    for role in ("expert_bank_lock_hash", "generation_lock_hash"):
        value = inputs.get(role)
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{16}", value) is None:
            raise ProtocolError(
                f"HARP v11 {role} must be an exact 16-hex semantic hash."
            )
        hashes[role] = value
    activated_input_roles = tuple(
        role for role in _HASH_ROLES if role != "execution_amendment_sha256"
    )
    if authorized and any(hashes[role] is None for role in activated_input_roles):
        raise ProtocolError("Activated HARP v11 execution requires every prepared-input hash.")
    if not authorized and any(hashes[role] is not None for role in _HASH_ROLES):
        raise ProtocolError("Planned HARP v11 config may not pre-bind execution hashes.")
    locations = {role: _location(inputs.get(role), name=role) for role in _LOCATION_ROLES}
    owned_locations = {
        "test_cache_root": "harp_source_train_full_test_cache_v11",
        "development_manifest_path": "harp_source_train_label_capability_v11",
        "evaluation_manifest_path": "harp_full_test_evaluation_release_v11",
        "parent_ledger_path": "harp_parent_v11",
        "execution_amendment_path": "harp_execution_amendment_v11",
    }
    for role, fragment in owned_locations.items():
        if "://" in locations[role] and fragment not in locations[role]:
            raise ProtocolError(f"HARP v11 {role} is not revision-owned.")

    protocol = dict(_section(raw, "protocol"))
    expected_protocol = {
        "schema_version": "midogpp_harp_stage90_terminal_protocol_v11",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": True,
        "compatibility_is_label_free_proxy_not_nelbo_or_utility": True,
        "generative_expert_compatibility_claimed": False,
        "centers": list(CENTERS),
        "strict_outer_center_exclusion": True,
        "source_development_split": "train",
        "source_development_row_count": 9648,
        "source_development_case_count": 216,
        "target_evaluation_split": "test",
        "target_evaluation_row_count": 9928,
        "target_evaluation_case_count": 218,
        "consumed_test_development_cases_used": False,
        "all_consumed_test_cases_reserved_for_terminal_evaluation": True,
        "source_and_target_cache_lineages_separately_authenticated": True,
        "source_and_target_physical_shards_separate": True,
        "source_fold_conditioned_physical_surface": "H_q_r",
        "source_prediction_candidate_pool": (
            "C_minus_outer_H_and_heldout_q"
        ),
        "source_calibration_candidate_pool": (
            "C_minus_outer_H_heldout_q_and_current_query_r"
        ),
        "target_candidate_pool": "C_minus_outer_H",
        "candidate_pool_receipt_required": True,
        "heldout_q_physically_excluded_before_classifier_fit": True,
        "current_query_r_physically_excluded_from_calibration_classifier_fit": True,
        "compatibility_reranked_inside_each_H_q_r_pool": True,
        "source_fold_prediction_context_count": 72,
        "source_fold_calibration_context_count": 504,
        "source_fold_context_count": 576,
        "source_fold_action_count": 4680,
        "source_fold_classifier_task_count": 5184,
        "source_fold_seed_cell_count": 42120,
        "source_label_capability_center_sharded": True,
        "source_label_capability_shard_count": 9,
        "source_label_fit_scope": "C_minus_outer_H_and_heldout_q",
        "source_label_fold_workers_spawn_isolated": True,
        "heldout_q_label_shard_unauthorized_and_not_opened_by_typed_loader_in_own_H_q_worker": True,
        "cross_fold_model_or_prediction_state_shared": False,
        "global_source_label_open_order_claimed": False,
        "source_fold_label_capability_seal_required": True,
        "source_prelabel_q_prediction_store_required": True,
        "nested_source_center_lodo": True,
        "pseudo_target_q_predictions_sealed_before_q_outcomes_joined_to_same_fold": True,
        "router_transforms_refit_inside_each_H_q_fold": True,
        "six_source_calibration_base_per_source": 168,
        "six_source_calibration_base_total_per_class": 1008,
        "six_source_calibration_topup_total_per_class": 126,
        "six_source_calibration_final_total_per_class": 1134,
        "resident_source_rows_per_class": 294,
        "target_max_required_rows_per_source_per_class": 256,
        "seven_source_max_required_rows_per_source_per_class": 270,
        "six_source_max_required_rows_per_source_per_class": 294,
        "global_max_required_rows_per_source_per_class": 294,
        "prelease_action_capacity_certificate_required": True,
        "six_source_pure_topup_maximum_source_weight": 7.0 / 27.0,
        "six_source_pure_topup_effective_source_count": 243.0 / 43.0,
        "generic_quarter_max_weight_claimed_for_six_source_surface": False,
        "generic_min_six_effective_sources_claimed_for_six_source_surface": False,
        "expert_frame_fit_scope_verified_source_center_only": True,
        "delete_donor_ensemble": False,
        "seed_cells_are_technical_replications": True,
        "complete_B_U_and_Hxe_physical_menu": True,
        "directional_opportunity_surfaces_complete": True,
        "shared_label_free_effective_menu": True,
        "effective_menu_applied_before_development_labels": True,
        "structural_noop_and_all_margins_excluded_from_bacc_lane": True,
        "exact_byte_duplicate_aliases_persisted": True,
        "target_case_label_free_inference_features": "own_case_embeddings_only",
        "separate_target_support_partition_used": False,
        "target_case_features_may_not_fit_or_calibrate_router": True,
        "target_case_labels_sealed_until_global_route_seal": True,
        "physical_expert_lambda_grid": [1.0],
        "case_level_decisions": True,
        "development_and_evaluation_cases_disjoint": True,
        "global_routes_sealed_before_evaluation_labels": True,
        "source_numeric_oof_policy_replay_required": True,
        "whole_policy_source_oof_required": True,
        "per_outer_local_admission": True,
        "target_routes_fresh_reconstructed_twice": True,
        "exact_b_byte_identical_fallback": True,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_or_rank_used": False,
        "may_feed_stage60_or_stage70": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    if protocol != expected_protocol:
        raise ProtocolError("HARP v11 protocol contract drifted.")

    model = dict(_section(raw, "model"))
    expected_model = {
        "schema_version": "midogpp_harp_stage90_policy_calibrated_pairwise_residual_router_v11",
        "decision_unit": "case",
        "action_families": ["B", "U", "Hxe"],
        "directional_actions": ["D01", "D10"],
        "physical_expert_lambda_grid": [1.0],
        "baseline_anchor": "exact_B_probability_vector",
        "effective_menu_transform": (
            "label_free_threshold_crossing_then_exact_byte_deduplication"
        ),
        "effective_menu_required_members": ["B"],
        "effective_menu_excluded_families": ["ALL_MARGINS", "STRUCTURAL_NOOP"],
        "exact_byte_duplicate_aliases_persisted": True,
        "deployed_action_kind": "exact_top1_physical_action",
        "unevaluated_action_mixtures_allowed": False,
        "endpoint_models": [
            "case_equal_bacc_contribution_gain",
            "harm_probability",
            "brier_delta",
            "log_loss_delta",
        ],
        "primary_estimand": (
            "equal_centers_equal_classes_equal_supporting_cases_"
            "recall_at_threshold_0_5"
        ),
        "oracle_headroom_capture_estimand": (
            "policy_gain_over_proper_loss_safe_oracle_gain"
        ),
        "single_class_case_rule": (
            "sole_class_recall_weighted_by_total_cases_over_twice_"
            "class_supporting_cases"
        ),
        "budget_residual_contrast": "U_minus_B",
        "allocation_residual_contrast": "Hxe_minus_U",
        "residual_targets": (
            "signed_bacc_gain_harm_indicator_brier_delta_log_delta"
        ),
        "virtual_baseline_candidate": "B",
        "ranker": "source_only_center_case_balanced_tie_aware_pairwise_ridge",
        "ranker_target": "within_case_downstream_utility_preference",
        "ranker_tie_rule": "exact_utility_tie_orders_B_then_action_id",
        "ranker_baselines": [
            "always_B",
            "uniform_U",
            "label_free_compatibility_rank",
        ],
        "selected_action_acceptor": (
            "cross_fitted_ranker_selected_action_positive_gain_and_risk"
        ),
        "acceptor_training_surface": (
            "inner_source_lodo_oof_selected_actions_only"
        ),
        "acceptor_target": (
            "selected_action_positive_gain_within_harm_brier_log_budgets"
        ),
        "acceptor_features": [
            "selected_pairwise_score",
            "budget_residual_score",
            "allocation_residual_score",
            "top_runner_up_margin",
            "menu_score_mean",
            "menu_score_std",
            "menu_score_max_abs",
            "physical_action_count",
            "action_kind",
            "direction",
        ],
        "per_action_worst_center_certificate_used": False,
        "individual_action_safety_claimed": False,
        "policy_family": "one_dimensional_acceptance_threshold_exact_top1_or_B",
        "whole_policy_admission_scope": (
            "all_held_source_cases_nested_route_or_exact_B"
        ),
        "policy_calibration": (
            "nested_source_center_lodo_complete_rank_accept_route_or_B_replay"
        ),
        "policy_calibration_unit": "held_source_center_case",
        "policy_replay_endpoints": [
            "case_equal_bacc_contribution_gain",
            "brier_delta",
            "log_loss_delta",
            "harmful_route_rate",
            "coverage",
            "regret",
            "oracle_headroom_capture",
        ],
        "selection_rule": (
            "pairwise_top1_if_crossfit_acceptance_score_meets_nested_policy_"
            "threshold_else_exact_B"
        ),
        "numeric_oof_replay": (
            "required_case_ids_pairwise_scores_selected_actions_acceptance_scores_"
            "policy_routes_endpoints_and_fold_hashes"
        ),
        "pairwise_alpha_grid": [1.0],
        "residual_alpha_grid": [1.0],
        "acceptor_alpha_grid": [1.0],
        "acceptance_threshold_grid": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
        "rank_margin_fixed_guard": 0.0,
        "policy": {
            "min_case_equal_bacc_gain": 0.0,
            "min_delete_center_bacc_gain": -0.005,
            "max_routed_harm_rate": 0.25,
            "max_case_equal_brier_delta": 0.002,
            "max_case_equal_log_loss_delta": 0.005,
            "min_coverage": 0.02,
            "min_routed_cases": 3,
        },
        "admission": {
            "min_pooled_top1_excess_over_always_b": 0.01,
            "min_delete_center_top1_excess_over_always_b": -0.02,
            "min_opportunity_top1_accuracy": 0.35,
            "min_opportunity_cases": 8,
        },
        "all_preprocessing_fit_inside_source_lodo": True,
        "policy_hyperparameters_selected_inside_source_lodo": False,
        "regularization_hyperparameters_predeclared_fixed_before_source_lodo": True,
        "acceptance_threshold_selected_inside_source_lodo": True,
        "target_thresholds_frozen_before_target_evaluation": True,
        "per_outer_policy_admission_required": True,
        "policy_admission_null": "always_exact_B_tie_aware",
        "exact_b_byte_identical_fallback": True,
    }
    if model != expected_model:
        raise ProtocolError("HARP v11 model contract drifted.")

    runtime = dict(_section(raw, "runtime"))
    expected_runtime = {
        "schema_version": "midogpp_harp_stage90_workstation_runtime_v11",
        "profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "global_parent_blas_threads": 1,
        "classifier_workers": 4,
        "classifier_blas_threads_per_worker": 3,
        "science_workers": 4,
        "science_blas_threads_per_worker": 1,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_created": False,
        "late_torch_interop_setter_used": False,
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "memory_mapped_surfaces": True,
        "bounded_inflight_batches_per_gpu": 2,
        "bounded_inflight_classifier_tasks_per_worker": 2,
        "bounded_inflight_science_tasks_per_worker": 1,
        "phase_disjoint_cpu_pools": True,
        "cuda_hidden_from_cpu_workers": True,
        "no_nested_process_pools": True,
        "scratch_root": "/data/local/fixed_bank_harp_router_v11",
    }
    if runtime != expected_runtime:
        raise ProtocolError("HARP v11 workstation runtime contract drifted.")
    boundary = dict(_section(raw, "claim_boundary"))
    if boundary != claim_boundary_payload(execution_authorized=authorized):
        raise ProtocolError("HARP v11 claim boundary drifted.")

    canonical = {
        "experiment": dict(experiment),
        "inputs": dict(inputs),
        "protocol": protocol,
        "model": model,
        "runtime": runtime,
        "claim_boundary": boundary,
    }
    return HarpStage90V11Config(
        source_path=source,
        artifact_root=artifact_root,
        input_locations=locations,
        expected_hashes=hashes,
        execution_authorized=authorized,
        protocol=protocol,
        model=model,
        runtime=runtime,
        claim_boundary=boundary,
        config_hash=canonical_hash(canonical),
    )


__all__ = ("HarpStage90V11Config", "INPUT_ARTIFACT_IDS", "load_config")
