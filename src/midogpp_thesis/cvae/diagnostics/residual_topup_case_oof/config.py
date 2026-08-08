"""Strict configuration for the Stage-90 B/U/G/S case-OOF diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation.config import (
    CACHE_NAME as VALIDATION_CACHE_SEMANTIC_ID,
    MANIFEST_SHA256 as EXPECTED_MANIFEST_SHA256,
    REPRESENTATION_ID as VALIDATION_CACHE_REPRESENTATION_ID,
)
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
)
from ...protocol import ProtocolError
from ...routing.contracts import (
    EXPECTED_POLICY_LOCK_HASH as EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
)
from ...routing.residual_topup.contracts import (
    MAX_FINAL_SOURCE_WEIGHT,
    MIN_FINAL_EFFECTIVE_SOURCES,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PERMUTATION_ACTION_ID,
    PRIMARY_ACTION_IDS,
    PUBLICATION_STATUS,
    SINGLE_SOURCE_TAIL_PREFIX,
    STAGE_ID,
    SUPPORT_ACTION_ID,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)


_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "actions",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    }
)
CLAIM_SCOPE = "diagnostic_only"

DOWNSTREAM_CLASSIFIER = ClassifierSpec(
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
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_report_all_nine_no_seed_selection",
        "total_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "expected_case_oof_fold_count": EXPECTED_CASE_OOF_FOLD_COUNT,
        "support_split_seed": 20260806,
        "support_partition_namespace": "midogpp_residual_topup_support_v1",
        "case_oof_namespace": "midogpp_residual_topup_b_u_g_s_case_oof_v1",
        "cross_fit_mode": (
            "fixed_two_case_support_plus_whole_evaluation_case_scoring_oof"
        ),
        "cross_fitted_fixed_support_diagnostic": True,
        "cross_fitted_transductive_diagnostic": False,
        "whole_case_support_evaluation_disjoint": True,
        "each_evaluation_case_held_out_exactly_once": True,
        "heldout_case_excluded_from_own_route": True,
        "other_evaluation_embeddings_available_to_router": False,
        "global_proxy_uses_fixed_support_cases_only": True,
        "global_proxy_excludes_outer_target_H_and_query_q": True,
        "target_support_proxy_uses_fixed_S_H_only": True,
        "support_labels_used": False,
        "evaluation_labels_available_before_global_prediction_seal": False,
        "evaluation_labels_available_to_action_prediction": False,
        "evaluation_embeddings_available_to_router": False,
        "source_expert_updated": False,
        "target_expert_excluded": True,
        "all_actions_predictions_globally_sealed_before_any_label_access": True,
        "previous_stage90_router_or_utility_inputs_used": False,
        "stage50_target_utility_inputs_used": False,
        "stage60_policy_inputs_used": False,
        "stage70_target_or_scoring_inputs_used": False,
    }


def canonical_actions_payload() -> dict[str, object]:
    return {
        "family": (
            "immutable_equal_union_backbone_with_fixed_rank_"
            "residual_topup_case_oof_v1"
        ),
        "primary_action_ids": list(PRIMARY_ACTION_IDS),
        "single_source_tail_prefix": SINGLE_SOURCE_TAIL_PREFIX,
        "action_count_per_target": EXPECTED_ACTION_COUNT_PER_TARGET,
        "frozen_action_count": EXPECTED_FROZEN_ACTION_COUNT,
        "base_only_role": "original_equal_union_budget_reference",
        "uniform_topup_role": "matched_budget_control",
        "global_rank_topup_role": "leave_H_and_q_out_global_source_preference",
        "support_rank_topup_role": "fixed_unlabeled_target_support_routing",
        "permutation_role": "source_identity_permutation_diagnostic",
        "single_source_tail_role": "sealed_oracle_headroom_diagnostic_only",
        "target_source_count": 8,
        "base_per_source_per_class": 128,
        "base_total_per_class": 1024,
        "topup_total_per_class": 128,
        "matched_total_per_class": 1152,
        "source_cache_prefix_per_class": 256,
        "base_window_start": 0,
        "topup_window_starts_after_base": True,
        "class_agnostic_topup": True,
        "replica_aggregation": (
            "mean_three_training_replicas_before_each_case_ballot"
        ),
        "ballot_semantics": "true_normalized_midranks_lower_is_better",
        "borda_direction_semantics": (
            "explicit_one_minus_mean_normalized_midrank"
        ),
        "global_aggregation": "equal_case_ballots_leave_H_and_q_out",
        "support_aggregation": "equal_fixed_support_case_ballots_for_H_only",
        "source_identity_permutation_scheme": (
            "canonical_source_order_nonzero_cyclic_rotation"
        ),
        "source_identity_permutation_index": 1,
        "maximum_final_source_weight": MAX_FINAL_SOURCE_WEIGHT,
        "minimum_final_effective_sources": MIN_FINAL_EFFECTIVE_SOURCES,
        "no_action_budget_temperature_or_strength_search": True,
        "no_selector_or_fallback_gate": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "descriptive_seed_endpoint": "paired_seed_cell_mean_bacc",
        "primary_contrasts": ["S-U", "S-G"],
        "secondary_contrasts": ["G-U", "U-B", "S-B"],
        "permutation_contrast": "S-P",
        "confidence_level": 0.95,
        "inference_unit": "target_center",
        "inference_center_count": len(CENTERS),
        "technical_seed_repeats_are_not_independent_units": True,
        "oracle_diagnostics": [
            "support_utility_spearman",
            "top1_agreement",
            "normalized_oracle_gap",
        ],
        "oracle_matrix_role": "diagnostic_only_no_policy_update",
        "target_labels_used_for_scoring_only_after_global_seal": True,
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
        "tf32_disabled_in_gpu_workers": True,
        "dependency_version_policy": "presence_gate_versions_report_only",
        "generated_cache_format": "float32_npy_memmap",
        "one_expert_per_gpu_at_a_time": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107374182400,
        "minimum_artifact_disk_free_bytes": 8589934592,
        "minimum_gpu_free_mib_per_device": 18000,
        "maximum_unique_classifier_fit_count": 1053,
        "resume_policy": (
            "hash_validated_source_and_prediction_task_checkpoints_"
            "with_completed_product_reuse"
        ),
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "consumed_validation_data": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "cross_fitted_fixed_support_diagnostic": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "target_specific_router_success_claimed": False,
        "proxy_is_nelbo_compatibility": False,
        "proxy_is_downstream_utility": False,
        "promotion_eligible": False,
        "oracle_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


@dataclass(frozen=True)
class ResidualTopupCaseOOFConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    protocol: Mapping[str, object]
    actions: Mapping[str, object]
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


def load_residual_topup_case_oof_config(
    path: str | Path,
) -> ResidualTopupCaseOOFConfig:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read residual top-up case-OOF config.") from exc
    if not isinstance(payload, Mapping) or set(payload) != set(_TOP_LEVEL):
        raise ProtocolError("Residual top-up case-OOF top-level config drifted.")
    _reject_pending(payload)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    actions = _mapping(payload, "actions")
    classifier_payload = _mapping(payload, "classifier")
    evaluation = _mapping(payload, "evaluation")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")

    expected_experiment = {
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "artifact_root": experiment.get("artifact_root"),
        "claim_scope": CLAIM_SCOPE,
        "status": PUBLICATION_STATUS,
    }
    _require_exact(experiment, expected_experiment, "experiment identity")
    expected_inputs = {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "equal_union_policy_artifact_id": EQUAL_UNION_POLICY_ARTIFACT_ID,
        "validation_cache_artifact_id": VALIDATION_CACHE_ARTIFACT_ID,
        "validation_manifest_artifact_id": VALIDATION_MANIFEST_ARTIFACT_ID,
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_equal_union_policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
        "expected_validation_cache_semantic_id": VALIDATION_CACHE_SEMANTIC_ID,
        "expected_validation_cache_representation_id": VALIDATION_CACHE_REPRESENTATION_ID,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    location_keys = {
        "expert_bank_root",
        "generation_lock_root",
        "equal_union_policy_root",
        "validation_cache_root",
        "validation_manifest_path",
    }
    if set(inputs) != set(expected_inputs).union(location_keys):
        raise ProtocolError("Residual top-up case-OOF input schema drifted.")
    for key, value in expected_inputs.items():
        _require_exact(inputs.get(key), value, f"input {key}")

    _require_exact(protocol, canonical_protocol_payload(), "protocol")
    _require_exact(actions, canonical_actions_payload(), "actions")
    _require_exact(evaluation, canonical_evaluation_payload(), "evaluation")
    _require_exact(runtime, canonical_runtime_payload(), "runtime")
    _require_exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(classifier_payload)
    if classifier != DOWNSTREAM_CLASSIFIER:
        raise ProtocolError("Residual top-up case-OOF classifier drifted.")

    locations = {
        "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),
        "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),
        "equal_union_policy_root": (EQUAL_UNION_POLICY_ARTIFACT_ID, ""),
        "validation_cache_root": (VALIDATION_CACHE_ARTIFACT_ID, ""),
        "validation_manifest_path": (VALIDATION_MANIFEST_ARTIFACT_ID, "manifest.csv"),
    }
    for key, (artifact_id, member) in locations.items():
        _validate_artifact_uri(inputs[key], artifact_id=artifact_id, member=member)
    artifact_root_value = _string(experiment["artifact_root"], "artifact root")
    expected_output_uri = f"output://{OUTPUT_ARTIFACT_ID}"
    if artifact_root_value.startswith("output://") and artifact_root_value != expected_output_uri:
        raise ProtocolError("Residual top-up case-OOF output identity drifted.")

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "actions": dict(actions),
        "classifier": classifier.to_payload(),
        "evaluation": dict(evaluation),
        "claim_boundary": dict(claim),
    }
    return ResidualTopupCaseOOFConfig(
        source_path=source,
        artifact_root=_path(source.parent, artifact_root_value),
        expert_bank_root=_path(source.parent, inputs["expert_bank_root"]),
        generation_lock_root=_path(source.parent, inputs["generation_lock_root"]),
        equal_union_policy_root=_path(source.parent, inputs["equal_union_policy_root"]),
        validation_cache_root=_path(source.parent, inputs["validation_cache_root"]),
        validation_manifest_path=_path(source.parent, inputs["validation_manifest_path"]),
        protocol=dict(protocol),
        actions=dict(actions),
        classifier=classifier,
        evaluation=dict(evaluation),
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Residual top-up case-OOF config lacks {key!r}.")
    return value


def _classifier(payload: Mapping[str, object]) -> ClassifierSpec:
    expected_keys = {
        "family",
        "C",
        "penalty",
        "solver",
        "max_iter",
        "class_weight",
        "random_state",
        "l1_ratio",
        "threshold_policy",
        "scaler_fit",
    }
    if set(payload) != expected_keys:
        raise ProtocolError("Residual top-up case-OOF classifier schema drifted.")
    try:
        return ClassifierSpec(
            family=_string(payload["family"], "classifier family"),
            C=_number(payload["C"], "classifier C"),
            penalty=_string(payload["penalty"], "classifier penalty"),
            solver=_string(payload["solver"], "classifier solver"),
            max_iter=_integer(payload["max_iter"], "classifier max_iter"),
            class_weight=(
                None
                if payload["class_weight"] is None
                else _string(payload["class_weight"], "class weight")
            ),
            random_state=_integer(
                payload["random_state"], "classifier random state"
            ),
            l1_ratio=(
                None
                if payload["l1_ratio"] is None
                else _number(payload["l1_ratio"], "l1 ratio")
            ),
            threshold_policy=_string(
                payload["threshold_policy"], "threshold policy"
            ),
            scaler_fit=_string(payload["scaler_fit"], "scaler fit"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Residual top-up case-OOF classifier is invalid.") from exc


def _validate_artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    rendered = _string(value, "artifact location")
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    if rendered.startswith("artifact://") and rendered != expected:
        raise ProtocolError(f"Artifact URI must be {expected!r}.")


def _path(base: Path, value: object) -> Path:
    rendered = _string(value, "path")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    raw = Path(rendered).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def _require_exact(observed: object, expected: object, role: str) -> None:
    if not _strict_equal(observed, expected):
        raise ProtocolError(f"Residual top-up case-OOF {role} drifted.")


def _strict_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, Mapping):
        return isinstance(observed, Mapping) and set(observed) == set(expected) and all(
            _strict_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and len(observed) == len(expected)
            and all(
                _strict_equal(left, right)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    return observed == expected


def _string(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f"{role} must be a nonempty string.")
    return value


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{role} must be numeric.")
    return float(value)


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{role} must be an integer.")
    return value


def _reject_pending(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {"todo", "tbd", "pending"}:
                raise ProtocolError("Residual top-up case-OOF config is pending.")
            _reject_pending(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_pending(item)
    elif isinstance(value, str) and value.strip().lower() in {"todo", "tbd", "pending"}:
        raise ProtocolError("Residual top-up case-OOF config is pending.")


__all__ = (
    "ResidualTopupCaseOOFConfig",
    "canonical_actions_payload",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "load_residual_topup_case_oof_config",
)
