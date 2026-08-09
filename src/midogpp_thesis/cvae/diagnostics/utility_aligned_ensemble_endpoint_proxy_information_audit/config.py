"""Strict config for the independent consumed-data proxy-information audit."""

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
        "proxy_features",
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
        "support_split_seed": 20260806,
        "support_partition_namespace": (
            "midogpp_utility_aligned_ensemble_endpoint_proxy_information_audit_support_v1"
        ),
        "cross_fit_mode": "strict_all_role_H_q_e_domain_holdout",
        "strict_crossfit_training_row_count": 120,
        "primary_development_response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "primary_development_response_count": 504,
        "primary_development_response": (
            "tail_exact_nine_probability_ensemble_BACC_minus_"
            "base_exact_nine_probability_ensemble_BACC"
        ),
        "descriptive_per_seed_utility_row_count": 4536,
        "descriptive_per_seed_rows_may_feed_model": False,
        "probabilities_averaged_before_single_threshold": True,
        "ensemble_probability_threshold": 0.5,
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction": True,
        "outer_target_H_excluded_from_fit_scaling_and_prediction": True,
        "pseudoquery_q_excluded_from_fit_scaling_and_prediction": True,
        "candidate_source_e_excluded_from_fit_scaling_and_prediction": True,
        "whole_case_support_evaluation_disjoint": True,
        "fixed_two_case_support_is_diagnostic_only": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "development_predictions_sealed_before_development_labels": True,
        "development_labels_are_scoring_only": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "target_actions_built": False,
        "target_labels_opened": False,
        "target_predictions_materialized": False,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_outputs_used": False,
        "previous_stage90_outputs_used": False,
        "historical_or_quarantined_inputs_used": False,
    }


def canonical_proxy_features_payload() -> dict[str, object]:
    return {
        "family": "predeclared_compact_ensemble_endpoint_proxy_information_v1",
        "feature_row_unit": "candidate_H_q_e",
        "feature_row_count": 504,
        "fixed_support_case_count_per_center": 2,
        "support_probabilities_are_label_free": True,
        "support_probabilities_averaged_across_exact_nine_seed_cells": True,
        "evaluation_probabilities_used": False,
        "primitive_names": [
            "metadata_similarity",
            "absolute_ensemble_shift",
            "reconstruction_mean_within_query_z",
            "kl_mean_within_query_z",
            "log_distribution_mmd_within_query_z",
            "signed_margin_projection",
            "threshold_flip_rate",
            "mean_entropy_change",
        ],
        "primitive_formulas": {
            "metadata_similarity": (
                "exact_match_count_tumor_type_lab_or_origin_scanner_model_div_3"
            ),
            "absolute_ensemble_shift": (
                "mean_support_rows_abs(mean9_p_tail_minus_mean9_p_base)"
            ),
            "reconstruction_mean_within_query_z": (
                "center_candidate_mean_reconstruction_over_current_7_candidate_"
                "list_divide_by_population_rms_else_zero"
            ),
            "kl_mean_within_query_z": (
                "center_candidate_mean_analytic_kl_over_current_7_candidate_"
                "list_divide_by_population_rms_else_zero"
            ),
            "log_distribution_mmd_within_query_z": (
                "center_log1p_candidate_mean_distribution_mmd_over_current_7_"
                "candidate_list_divide_by_population_rms_else_zero"
            ),
            "signed_margin_projection": (
                "mean_support_rows((mean9_p_tail_minus_mean9_p_base)*"
                "where(mean9_p_base_gte_0.5,1,-1))"
            ),
            "threshold_flip_rate": (
                "mean_support_rows(indicator((mean9_p_base_minus_0.5)*"
                "(mean9_p_tail_minus_0.5)<0))"
            ),
            "mean_entropy_change": (
                "mean_support_rows(binary_entropy(mean9_p_tail)-"
                "binary_entropy(mean9_p_base))"
            ),
        },
        "within_query_standardization_uses_only_current_label_free_candidate_list": True,
        "within_query_standardization_uses_utility_or_evaluation_labels": False,
        "zero_variance_standardized_value": 0.0,
        "cyclic_directional_permutation": (
            "canonical_allowed_source_order_nonzero_rotation_by_one"
        ),
        "cyclic_directional_permutation_seed": 90902026,
        "cyclic_directional_permutation_shift": 1,
        "technical_seed_rows_are_features": False,
    }


def canonical_model_payload() -> dict[str, object]:
    return {
        "family": "fixed_alpha_cluster_weighted_ridge_proxy_information_v1",
        "ridge_alpha": 1.0,
        "hyperparameter_selection": "none_predeclared_before_labels",
        "maximum_predictors_per_family": 3,
        "scaling_fit_on_training_fold_only": True,
        "ridge_cluster_unit": "outer_target_query",
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction": True,
        "response": "candidate_exact_nine_probability_ensemble_BACC_delta",
        "response_row_count": 504,
        "descriptive_seed_row_count": 4536,
        "descriptive_seed_rows_may_feed_model": False,
        "family_ids": [
            "equal_union_null",
            "metadata_only_control",
            "absolute_shift_control",
            "rich_distributional_compact",
            "directional_action_compact",
            "hybrid_compact",
            "cyclic_directional_permutation_control",
        ],
        "family_predictors": {
            "equal_union_null": [],
            "metadata_only_control": ["metadata_similarity"],
            "absolute_shift_control": ["absolute_ensemble_shift"],
            "rich_distributional_compact": [
                "reconstruction_mean_within_query_z",
                "kl_mean_within_query_z",
                "log_distribution_mmd_within_query_z",
            ],
            "directional_action_compact": [
                "signed_margin_projection",
                "threshold_flip_rate",
                "mean_entropy_change",
            ],
            "hybrid_compact": [
                "metadata_similarity",
                "log_distribution_mmd_within_query_z",
                "signed_margin_projection",
            ],
            "cyclic_directional_permutation_control": [
                "cyclic_signed_margin_projection",
                "cyclic_threshold_flip_rate",
                "cyclic_mean_entropy_change",
            ],
        },
        "outer_target_centers_are_independent_units": True,
        "query_domains_are_nested_descriptive_units": True,
        "seed_or_patch_rows_are_independent_units": False,
        "target_or_query_identity_predictors_used": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "outer_target_center_proxy_information_screen",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": 9,
        "query_metric_row_count": 72,
        "query_metrics_are_descriptive_nested_within_centers": True,
        "metrics": [
            "spearman_proxy_utility",
            "pairwise_order_accuracy",
            "normalized_regret",
        ],
        "confidence_level": 0.95,
        "screening_candidate_family_ids": [
            "rich_distributional_compact",
            "directional_action_compact",
            "hybrid_compact",
        ],
        "control_family_ids": [
            "metadata_only_control",
            "absolute_shift_control",
            "cyclic_directional_permutation_control",
        ],
        "screening_gate": {
            "outer_center_mean_spearman_ci95_lower_strictly_above": 0.0,
            "outer_center_pairwise_accuracy_ci95_lower_strictly_above": 0.5,
            "outer_center_normalized_regret_ci95_upper_strictly_below": 0.5,
            "mean_regret_strictly_below_each_control_family": True,
            "all_conditions_required": True,
        },
        "screening_gate_may_authorize_policy": False,
        "no_target_action_or_target_performance_evaluation": True,
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
        "phase_order": "two_gpu_source_streams_then_four_by_three_cpu_development",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107374182400,
        "minimum_artifact_disk_free_bytes": 8589934592,
        "minimum_gpu_free_mib_per_device": 18000,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "development_coarse_task_count": 648,
        "development_classifier_fit_count": 5184,
        "target_task_count": 0,
        "target_action_count": 0,
        "target_classifier_fit_count": 0,
        "maximum_total_classifier_fit_count": 5184,
        "scratch_preference": ["/data/local", "artifact_parent"],
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "consumed_validation_data": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "proxy_information_audit_only": True,
        "cross_fitted_fixed_support_diagnostic": True,
        "fixed_two_case_support_is_insufficient_for_policy": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "target_specific_router_success_claimed": False,
        "proxy_is_nelbo": False,
        "proxy_is_downstream_utility": False,
        "target_actions_built": False,
        "target_labels_opened": False,
        "screening_gate_may_authorize_policy": False,
        "policy_update_authorized": False,
        "may_update_policy": False,
        "action_selection_authorized": False,
        "promotion_eligible": False,
        "oracle_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "may_feed_another_stage90_experiment": False,
    }


@dataclass(frozen=True)
class ProxyInformationAuditConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    metadata_profile_root: Path
    protocol: Mapping[str, object]
    proxy_features: Mapping[str, object]
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


def load_utility_aligned_ensemble_endpoint_proxy_information_audit_config(
    path: str | Path,
) -> ProxyInformationAuditConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read proxy-information Stage-90 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(_TOP_LEVEL):
        raise ProtocolError("Proxy-information Stage-90 top-level config drifted.")
    _reject_pending(raw)
    experiment = _mapping(raw, "experiment")
    inputs = _mapping(raw, "inputs")
    protocol = _mapping(raw, "protocol")
    proxy_features = _mapping(raw, "proxy_features")
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
        raise ProtocolError("Proxy-information Stage-90 input schema drifted.")
    for key, value in fixed_inputs.items():
        _exact(inputs.get(key), value, f"input {key}")
    for key, (artifact_id, member) in locations.items():
        _artifact_uri(inputs[key], artifact_id=artifact_id, member=member)
    _exact(protocol, canonical_protocol_payload(), "protocol")
    _exact(proxy_features, canonical_proxy_features_payload(), "proxy features")
    _exact(model, canonical_model_payload(), "model")
    _exact(evaluation, canonical_evaluation_payload(), "evaluation")
    _exact(runtime, canonical_runtime_payload(), "runtime")
    _exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(classifier_raw)
    if classifier != CLASSIFIER:
        raise ProtocolError("Proxy-information Stage-90 classifier drifted.")
    artifact_root_text = _text(experiment["artifact_root"], "artifact root")
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("Proxy-information output identity drifted.")
    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "proxy_features": dict(proxy_features),
        "model": dict(model),
        "classifier": classifier.to_payload(),
        "evaluation": dict(evaluation),
        "claim_boundary": dict(claim),
    }
    return ProxyInformationAuditConfig(
        source_path=source,
        artifact_root=_path(source.parent, artifact_root_text),
        expert_bank_root=_path(
            source.parent, _text(inputs["expert_bank_root"], "bank root")
        ),
        generation_lock_root=_path(
            source.parent, _text(inputs["generation_lock_root"], "generation root")
        ),
        validation_cache_root=_path(
            source.parent, _text(inputs["validation_cache_root"], "cache root")
        ),
        validation_manifest_path=_path(
            source.parent, _text(inputs["validation_manifest_path"], "manifest")
        ),
        metadata_profile_root=_path(
            source.parent, _text(inputs["metadata_profile_root"], "metadata root")
        ),
        protocol=dict(protocol),
        proxy_features=dict(proxy_features),
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
        raise ProtocolError(f"Proxy-information config section {key!r} is absent.")
    return value


def _exact(observed: object, expected: object, role: str) -> None:
    if observed != expected:
        raise ProtocolError(f"Proxy-information config {role} drifted.")


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Proxy-information config {role} must be text.")
    return value


def _artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    text = _text(value, artifact_id)
    if text.startswith("artifact://") and text != expected:
        raise ProtocolError(f"Proxy-information artifact URI drifted: {artifact_id}.")


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
            class_weight=(
                None if raw["class_weight"] is None else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(None if raw["l1_ratio"] is None else float(raw["l1_ratio"])),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Proxy-information classifier payload is malformed.") from exc


def _reject_pending(raw: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            _reject_pending(value, (*trail, str(key)))
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            _reject_pending(value, (*trail, str(index)))
    elif isinstance(raw, str) and ("pending://" in raw or "PENDING" in raw):
        raise ProtocolError(
            f"Proxy-information config contains pending value at {'.'.join(trail)}."
        )


__all__ = (
    "CLASSIFIER",
    "ProxyInformationAuditConfig",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_model_payload",
    "canonical_protocol_payload",
    "canonical_proxy_features_payload",
    "canonical_runtime_payload",
    "load_utility_aligned_ensemble_endpoint_proxy_information_audit_config",
)
