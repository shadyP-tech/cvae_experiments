"""Strict config for the consumed MIDOG++ ensemble-endpoint diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation.config import (
    CACHE_NAME as VALIDATION_CACHE_SEMANTIC_ID,
    MANIFEST_SHA256 as EXPECTED_MANIFEST_SHA256,
    REPRESENTATION_ID as VALIDATION_CACHE_REPRESENTATION_ID,
)
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...generation.contracts import EXPECTED_BANK_LOCK_HASH, EXPECTED_GENERATION_LOCK_HASH
from ...protocol import ProtocolError
from ...routing.metadata_compatibility.contracts import DOMAIN_MAPPING_SHA256
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
)
from .contracts import (
    CENTERS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    METADATA_PROFILE_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    STAGE_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)


_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "actions",
        "model",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    }
)


CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "validation_split": "val",
        "centers": list(CENTERS),
        "excluded_center": "4",
        "training_seeds": [17, 42, 101],
        "generation_seeds": [17, 42, 101],
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "total_case_count": 44,
        "fixed_support_case_count_per_center": 2,
        "fresh_policy_minimum_support_case_count": 8,
        "fresh_policy_minimum_case_bootstrap_replicates": 32,
        "low_support_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
        "expected_case_fold_count": 26,
        "support_split_seed": 20260806,
        "support_partition_namespace": (
            "midogpp_utility_aligned_ensemble_endpoint_router_support_v1"
        ),
        "cross_fit_mode": (
            "outer_H_excluded_candidate_ensemble_utility_then_fixed_two_case_R2E"
        ),
        "primary_development_response_unit": "candidate_H_q_e",
        "primary_development_response_count": 504,
        "primary_development_response": (
            "tail_exact_nine_probability_ensemble_BACC_minus_"
            "base_exact_nine_probability_ensemble_BACC"
        ),
        "descriptive_per_seed_utility_row_count": 4536,
        "descriptive_per_seed_rows_may_feed_model": False,
        "probabilities_averaged_before_single_threshold": True,
        "ensemble_probability_threshold": 0.5,
        "target_local_scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
        "target_local_scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
        "target_local_scalar_is_ensemble_first": True,
        "per_seed_support_shifts_are_descriptive_only": True,
        "whole_case_support_evaluation_disjoint": True,
        "each_evaluation_case_scored_exactly_once": True,
        "outer_target_H_excluded_from_development_query_and_source_roles": True,
        "pseudoquery_q_excluded_from_candidate_source_role": True,
        "other_evaluation_embeddings_available_to_router": False,
        "support_labels_used": False,
        "development_predictions_sealed_before_development_labels": True,
        "development_label_phase_contains_all_centers_but_outer_H_rows_excluded_per_model": True,
        "target_actions_locked_from_outer_H_excluded_utility_rows_before_terminal_scoring": True,
        "target_predictions_globally_sealed_before_terminal_target_scoring": True,
        "source_expert_updated": False,
        "target_expert_excluded": True,
        "previous_stage90_outputs_used": False,
        "stage60_policy_or_surface_inputs_used": False,
        "stage70_target_or_scoring_inputs_used": False,
    }


def canonical_actions_payload() -> dict[str, object]:
    return {
        "family": "utility_aligned_ensemble_endpoint_additive_tail_stage90_v1",
        "action_ids": ["B", "U", "G_delta", "R2E", "P"],
        "single_source_tail_prefix": "Hxe::",
        "action_count_per_target": 13,
        "frozen_action_count": 117,
        "inner_source_count": 7,
        "inner_base_per_source_per_class": 144,
        "inner_topup_total_per_class": 126,
        "inner_base_total_per_class": 1008,
        "inner_matched_total_per_class": 1134,
        "target_source_count": 8,
        "target_base_per_source_per_class": 128,
        "target_topup_total_per_class": 128,
        "target_base_total_per_class": 1024,
        "target_matched_total_per_class": 1152,
        "source_cache_prefix_per_class": 270,
        "B_role": "immutable_equal_union_base",
        "U_role": "matched_uniform_additive_tail_control",
        "G_delta_role": "candidate_ensemble_global_source_control_diagnostic",
        "R2E_role": "fixed_two_case_ensemble_first_target_shift_diagnostic",
        "P_role": "cyclic_target_shift_permutation_control",
        "Hxe_role": "terminal_single_source_tail_oracle_diagnostic",
        "R2E_fallback_or_abstention_authorized": False,
        "no_action_budget_temperature_strength_or_seed_search": True,
    }


def canonical_model_payload() -> dict[str, object]:
    return {
        "family": "candidate_specific_low_capacity_ridge_ensemble_endpoint_v1",
        "ridge_alpha_grid": [0.01, 0.1, 1.0, 10.0],
        "inner_selection": "strict_nested_leave_query_and_source_domain_out",
        "routing_tuning_endpoint": "mean_normalized_oracle_regret",
        "target_or_query_identity_predictors_used": False,
        "M0_predictors": ["global_source_control"],
        "M1_predictors": [
            "global_source_control",
            f"target_local::{SUPPORT_ACTION_PROBABILITY_SHIFT_NAME}",
        ],
        "response": (
            "candidate_exact_nine_probability_ensemble_BACC_delta"
        ),
        "response_row_count": 504,
        "per_seed_utility_row_count": 4536,
        "per_seed_utility_rows_may_feed_model": False,
        "permutation_seed": 90902026,
        "exact_nine_seed_cells_collapsed_before_model_fit": True,
        "technical_seed_cells_are_not_independent_units": True,
        "query_domains_are_model_selection_units": True,
        "R2E_selected_by_one_candidate_level_M1_prediction_per_source": True,
        "G_delta_selected_by_one_candidate_level_M0_prediction_per_source": True,
        "P_selected_from_cyclically_permuted_target_shift": True,
        "cardinality_transfer_role": "eligibility_diagnostic_only",
        "eligibility_does_not_authorize_policy": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "secondary_endpoint": "per_seed_bacc_descriptive_only",
        "primary_contrasts": ["R2E-B", "R2E-G_delta", "R2E-U", "R2E-P"],
        "secondary_contrasts": ["U-B", "G_delta-B"],
        "confidence_level": 0.95,
        "inference_unit": "target_center",
        "inference_center_count": 9,
        "technical_seed_repeats_are_not_independent_units": True,
        "oracle_diagnostics": [
            "R2E_Hxe_top1_agreement",
            "predicted_gain_utility_spearman",
            "normalized_oracle_gap",
        ],
        "Hxe_oracle_role": "terminal_only_no_policy_update",
        "target_scoring_capability_requires_global_target_seal": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "generated_cache_format": "float32_npy_memmap",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107374182400,
        "minimum_artifact_disk_free_bytes": 8589934592,
        "minimum_gpu_free_mib_per_device": 18000,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "development_coarse_task_count": 648,
        "development_classifier_fit_count": 5184,
        "target_task_count": 81,
        "target_action_identity_count": 1053,
        "target_unique_classifier_fit_count": 810,
        "router_action_ids_reuse_selected_Hxe_predictions": True,
        "maximum_total_classifier_fit_count": 5994,
        "scratch_preference": ["/data/local", "artifact_parent"],
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "consumed_validation_data": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "cross_fitted_fixed_support_diagnostic": True,
        "fixed_two_case_R2E_is_insufficient_for_policy": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "target_specific_router_success_claimed": False,
        "proxy_is_nelbo": False,
        "label_free_features_are_downstream_utility": False,
        "promotion_eligible": False,
        "may_update_policy": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


@dataclass(frozen=True)
class EnsembleEndpointRouterConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    metadata_profile_root: Path
    protocol: Mapping[str, object]
    actions: Mapping[str, object]
    model: Mapping[str, object]
    classifier: ClassifierSpec
    evaluation: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_ID

    @property
    def output_artifact_id(self) -> str:
        return OUTPUT_ARTIFACT_ID

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS

    @property
    def expected_manifest_sha256(self) -> str:
        return EXPECTED_MANIFEST_SHA256


def load_utility_aligned_ensemble_endpoint_router_config(
    path: str | Path,
) -> EnsembleEndpointRouterConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read ensemble-endpoint Stage-90 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(_TOP_LEVEL):
        raise ProtocolError("Ensemble-endpoint Stage-90 top-level config drifted.")
    _reject_pending(raw)
    experiment = _mapping(raw, "experiment")
    inputs = _mapping(raw, "inputs")
    protocol = _mapping(raw, "protocol")
    actions = _mapping(raw, "actions")
    model = _mapping(raw, "model")
    classifier_raw = _mapping(raw, "classifier")
    evaluation = _mapping(raw, "evaluation")
    runtime = _mapping(raw, "runtime")
    claim = _mapping(raw, "claim_boundary")
    _exact(
        experiment,
        {
            "id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "artifact_root": experiment.get("artifact_root"),
            "claim_scope": "diagnostic_only",
            "status": PUBLICATION_STATUS,
        },
        "experiment",
    )
    fixed_inputs = {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "validation_cache_artifact_id": VALIDATION_CACHE_ARTIFACT_ID,
        "validation_manifest_artifact_id": VALIDATION_MANIFEST_ARTIFACT_ID,
        "metadata_profile_artifact_id": METADATA_PROFILE_ARTIFACT_ID,
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_validation_cache_semantic_id": VALIDATION_CACHE_SEMANTIC_ID,
        "expected_validation_cache_representation_id": VALIDATION_CACHE_REPRESENTATION_ID,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_metadata_profile_sha256": DOMAIN_MAPPING_SHA256,
    }
    locations = {
        "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),
        "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),
        "validation_cache_root": (VALIDATION_CACHE_ARTIFACT_ID, ""),
        "validation_manifest_path": (VALIDATION_MANIFEST_ARTIFACT_ID, "manifest.csv"),
        "metadata_profile_root": (METADATA_PROFILE_ARTIFACT_ID, ""),
    }
    if set(inputs) != set(fixed_inputs).union(locations):
        raise ProtocolError("Ensemble-endpoint Stage-90 input schema drifted.")
    for key, value in fixed_inputs.items():
        _exact(inputs.get(key), value, f"input {key}")
    for key, (artifact_id, member) in locations.items():
        _artifact_uri(inputs[key], artifact_id=artifact_id, member=member)
    _exact(protocol, canonical_protocol_payload(), "protocol")
    _exact(actions, canonical_actions_payload(), "actions")
    _exact(model, canonical_model_payload(), "model")
    _exact(evaluation, canonical_evaluation_payload(), "evaluation")
    _exact(runtime, canonical_runtime_payload(), "runtime")
    _exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(classifier_raw)
    if classifier != CLASSIFIER:
        raise ProtocolError("Ensemble-endpoint Stage-90 classifier drifted.")
    artifact_root_text = _text(experiment["artifact_root"], "artifact root")
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("Ensemble-endpoint output identity drifted.")
    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "actions": dict(actions),
        "model": dict(model),
        "classifier": classifier.to_payload(),
        "evaluation": dict(evaluation),
        "claim_boundary": dict(claim),
    }
    return EnsembleEndpointRouterConfig(
        source_path=source,
        artifact_root=_path(source.parent, artifact_root_text),
        expert_bank_root=_path(
            source.parent, _text(inputs["expert_bank_root"], "bank root")
        ),
        generation_lock_root=_path(
            source.parent,
            _text(inputs["generation_lock_root"], "generation root"),
        ),
        validation_cache_root=_path(
            source.parent, _text(inputs["validation_cache_root"], "cache root")
        ),
        validation_manifest_path=_path(
            source.parent,
            _text(inputs["validation_manifest_path"], "manifest"),
        ),
        metadata_profile_root=_path(
            source.parent, _text(inputs["metadata_profile_root"], "metadata root")
        ),
        protocol=dict(protocol),
        actions=dict(actions),
        model=dict(model),
        classifier=classifier,
        evaluation=dict(evaluation),
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Ensemble-endpoint config section {key!r} is absent.")
    return value


def _exact(observed: object, expected: object, role: str) -> None:
    if observed != expected:
        raise ProtocolError(f"Ensemble-endpoint config {role} drifted.")


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Ensemble-endpoint config {role} must be text.")
    return value


def _artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    text = _text(value, artifact_id)
    if text.startswith("artifact://") and text != expected:
        raise ProtocolError(f"Ensemble-endpoint artifact URI drifted: {artifact_id}.")


def _path(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None if raw["class_weight"] is None else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Ensemble-endpoint classifier payload is malformed.") from exc


def _reject_pending(raw: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            _reject_pending(value, (*trail, str(key)))
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            _reject_pending(value, (*trail, str(index)))
    elif isinstance(raw, str) and ("pending://" in raw or "PENDING" in raw):
        raise ProtocolError(
            f"Ensemble-endpoint config contains pending value at {'.'.join(trail)}."
        )


__all__ = (
    "CLASSIFIER",
    "EnsembleEndpointRouterConfig",
    "canonical_actions_payload",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_model_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "load_utility_aligned_ensemble_endpoint_router_config",
)
