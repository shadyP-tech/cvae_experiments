"""Deterministic lock and action-library serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...protocol import ProtocolError
from ..utility_aligned.target_features import target_sources
from ..utility_aligned_identities import (
    CENTERS,
    DEVELOPMENT_RESERVATION_ARTIFACT_ID,
    METADATA_PROFILE_SHA256,
    TARGET_SUPPORT_SURFACE_ARTIFACT_ID,
)
from ..residual_topup.hashing import canonical_sha256
from .config import UtilityAlignedResidualPolicyConfig
from .contracts import (
    ACTION_LIBRARY_SCHEMA,
    BASE_LONG_ID,
    EXPECTED_ACTION_COUNT,
    EXPERIMENT_ID,
    GLOBAL_LONG_ID,
    ORACLE_PREFIX,
    OUTPUT_ARTIFACT_ID,
    PERMUTATION_LONG_ID,
    POLICY_LOCK_SCHEMA,
    ROUTED_LONG_ID,
    TARGET_POLICY_LOCK_SCHEMA,
    TARGET_RESERVATION_ARTIFACT_ID,
    UNIFORM_LONG_ID,
    LockedResidualAction,
    build_locked_action,
)
from .inputs import PolicyInputs
from .model_workers import TargetFitResult


@dataclass(frozen=True)
class BuiltPolicyBundle:
    model_lock: Mapping[str, object]
    global_ablation_lock: Mapping[str, object]
    cardinality_transfer_lock: Mapping[str, object]
    target_policy_lock: Mapping[str, object]
    action_library: Mapping[str, object]
    policy_lock: Mapping[str, object]


def build_policy_artifacts(
    config: UtilityAlignedResidualPolicyConfig,
    inputs: PolicyInputs,
    completed: Mapping[str, TargetFitResult],
) -> BuiltPolicyBundle:
    models_by_target: dict[str, Mapping[str, object]] = {}
    permutation_models_by_target: dict[str, Mapping[str, object]] = {}
    transfer_by_target: dict[str, Mapping[str, object]] = {}
    permutation_transfer_by_target: dict[str, Mapping[str, object]] = {}
    policies: dict[tuple[str, str], Mapping[str, object]] = {}
    if tuple(completed) != CENTERS:
        raise ProtocolError("Utility-aligned target model coverage drifted.")
    for target in CENTERS:
        result = completed[target]
        models = result.model_payload
        permutation_models = result.permutation_model_payload
        transfer = result.transfer_payload
        permutation_transfer = result.permutation_transfer_payload
        policies[(target, "G_delta")] = result.global_policy_payload
        policies[(target, "R")] = result.routed_policy_payload
        policies[(target, "P")] = result.permutation_policy_payload
        models_by_target[target] = models
        permutation_models_by_target[target] = permutation_models
        transfer_by_target[target] = transfer
        permutation_transfer_by_target[target] = permutation_transfer

    feature_schema_hash = canonical_sha256(
        {
            "schema_version": "midogpp_utility_aligned_feature_schema_lock_v1",
            "inner_global_feature_names": list(
                inputs.inner_feature_surfaces[CENTERS[0]].global_feature_names
            ),
            "inner_interaction_feature_names": list(
                inputs.inner_feature_surfaces[CENTERS[0]].interaction_feature_names
            ),
            "target_global_feature_names_by_target": {
                target: list(
                    inputs.target_features_by_target[target].point_surface.global_feature_names
                )
                for target in CENTERS
            },
            "target_interaction_feature_names_by_target": {
                target: list(
                    inputs.target_features_by_target[
                        target
                    ].point_surface.interaction_feature_names
                )
                for target in CENTERS
            },
            "labels_used": False,
        }
    )
    feature_surface_hash = canonical_sha256(
        {
            "schema_version": "midogpp_utility_aligned_feature_surface_set_v1",
            "inner": {
                target: inputs.inner_feature_surfaces[target].surface_hash
                for target in CENTERS
            },
            "target_point": {
                target: inputs.target_features_by_target[target].point_surface.surface_hash
                for target in CENTERS
            },
            "target_bootstrap": {
                target: [
                    surface.surface_hash
                    for surface in inputs.target_features_by_target[
                        target
                    ].bootstrap_surfaces
                ]
                for target in CENTERS
            },
        }
    )
    model_lock = _hashed(
        {
            "schema_version": "midogpp_utility_aligned_model_lock_v1",
            "exact_tail_surface_lock_hash": inputs.exact_lock.surface_lock_hash,
            "development_case_manifest_hash": inputs.development_case_manifest_hash,
            "development_support_case_ids_by_query": {
                target: list(inputs.development_support_case_ids_by_query[target])
                for target in CENTERS
            },
            "development_evaluation_case_ids_by_query": {
                target: list(inputs.development_evaluation_case_ids_by_query[target])
                for target in CENTERS
            },
            "development_target_evaluation_case_ids_by_target": {
                target: list(
                    inputs.development_target_evaluation_case_ids_by_target[target]
                )
                for target in CENTERS
            },
            "development_partition_hashes_by_query": {
                target: inputs.development_partition_hashes_by_query[target]
                for target in CENTERS
            },
            "feature_surface_hash": feature_surface_hash,
            "feature_schema_hash": feature_schema_hash,
            "models": [
                dict(models_by_target[target]) for target in CENTERS
            ],
            "permutation_models": [
                dict(permutation_models_by_target[target])
                for target in CENTERS
            ],
            "outer_target_excluded_from_fit": True,
            "query_domains_are_uncertainty_units": True,
            "seed_selection_performed": False,
        },
        "model_lock_hash",
    )
    global_lock = _hashed(
        {
            "schema_version": "midogpp_utility_aligned_global_ablation_lock_v1",
            "model_lock_hash": model_lock["model_lock_hash"],
            "policies": [
                dict(policies[(target, "G_delta")]) for target in CENTERS
            ],
            "global_gate_passed_by_target": {
                target: bool(transfer_by_target[target]["global_gate_passed"])
                for target in CENTERS
            },
            "target_bootstrap_features_used": False,
        },
        "global_ablation_lock_hash",
    )
    transfer_lock = _hashed(
        {
            "schema_version": "midogpp_utility_aligned_cardinality_transfer_lock_v1",
            "model_lock_hash": model_lock["model_lock_hash"],
            "results": [
                dict(transfer_by_target[target]) for target in CENTERS
            ],
            "permutation_results": [
                dict(permutation_transfer_by_target[target])
                for target in CENTERS
            ],
            "claim_role": "eligibility_for_fresh_7_to_8_evaluation_not_evidence",
            "query_domains_are_independent_units": True,
            "seed_cells_are_independent_units": False,
        },
        "cardinality_transfer_lock_hash",
    )
    target_policy_lock = _target_policy_lock(
        inputs=inputs,
        policies=policies,
    )
    actions = _actions(policies)
    shared = _shared_bindings(
        inputs=inputs,
        feature_surface_hash=feature_surface_hash,
        feature_schema_hash=feature_schema_hash,
        model_lock_hash=str(model_lock["model_lock_hash"]),
        global_ablation_lock_hash=str(global_lock["global_ablation_lock_hash"]),
        cardinality_transfer_lock_hash=str(
            transfer_lock["cardinality_transfer_lock_hash"]
        ),
        target_policy_lock_hash=str(target_policy_lock["target_policy_lock_hash"]),
    )
    action_library = _hashed(
        {
            "schema_version": ACTION_LIBRARY_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            **shared,
            "action_ids": [action.action_id for action in actions],
            "actions": [action.to_payload() for action in actions],
            "action_count": len(actions),
        },
        "action_library_hash",
    )
    if len(actions) != EXPECTED_ACTION_COUNT:
        raise ProtocolError("Utility-aligned action library count drifted.")
    policy_lock = _hashed(
        {
            "schema_version": POLICY_LOCK_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            **shared,
            "action_library_hash": action_library["action_library_hash"],
            "candidate_centers": list(CENTERS),
            "primary_contrasts": ["R-B", "R-G_delta", "R-U"],
            "permutation_contrast": "R-P",
            "success_requires_positive_one_sided_lcb": [
                "R-B", "R-G_delta", "R-U", "R-P"
            ],
            "policy_family": (
                "utility_aligned_exact_additive_tail_with_global_and_permutation_controls_v1"
            ),
            "fallback_policy": "exact_B",
            "outer_target_excluded_from_fit": True,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "seed_selection_performed": False,
            "minimum_independent_support_cases_per_target": 8,
            "support_bootstrap_count": 32,
        },
        "policy_lock_hash",
    )
    return BuiltPolicyBundle(
        model_lock=model_lock,
        global_ablation_lock=global_lock,
        cardinality_transfer_lock=transfer_lock,
        target_policy_lock=target_policy_lock,
        action_library=action_library,
        policy_lock=policy_lock,
    )


def _target_policy_lock(
    *,
    inputs: PolicyInputs,
    policies: Mapping[tuple[str, str], Mapping[str, object]],
) -> Mapping[str, object]:
    feature_locks = []
    for target in CENTERS:
        features = inputs.target_features_by_target[target]
        bootstrap_hashes = [surface.surface_hash for surface in features.bootstrap_surfaces]
        feature_locks.append(
            _hashed(
                {
                    "target_id": target,
                    "case_bootstrap_plan": features.plan.to_payload(),
                    "target_feature_surface_hash": features.point_surface.surface_hash,
                    "target_feature_row_count": len(features.point_surface.rows),
                    "bootstrap_surface_hashes": bootstrap_hashes,
                    "bootstrap_surface_hashes_hash": canonical_sha256(
                        bootstrap_hashes
                    ),
                    "candidate_sources": list(features.point_surface.candidate_sources),
                    "training_seeds": [17, 42, 101],
                    "generation_seeds": [17, 42, 101],
                    "case_level_resampling": True,
                    "labels_used": False,
                },
                "target_feature_lock_hash",
            )
        )
    return _hashed(
        {
            "schema_version": TARGET_POLICY_LOCK_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "exact_tail_surface_lock_hash": inputs.exact_lock.surface_lock_hash,
            "target_support_surface_artifact_id": TARGET_SUPPORT_SURFACE_ARTIFACT_ID,
            "target_support_surface_hash": inputs.target_support_surface_hash,
            "target_support_parent_reservation_artifact_id": inputs.target_support_parent_reservation_artifact_id,
            "target_support_parent_reservation_hash": inputs.target_support_parent_reservation_hash,
            "target_reservation_artifact_id": TARGET_RESERVATION_ARTIFACT_ID,
            "target_reservation_hash": inputs.target_reservation_hash,
            "target_support_case_ids_by_target": {
                target: list(inputs.support_case_ids_by_target[target]) for target in CENTERS
            },
            "target_evaluation_case_ids_by_target": {
                target: list(inputs.evaluation_case_ids_by_target[target]) for target in CENTERS
            },
            "target_evaluation_binding_hash": inputs.target_evaluation_binding_hash,
            "metadata_profile_sha256": METADATA_PROFILE_SHA256,
            "target_feature_locks": feature_locks,
            "policies": [
                dict(policies[(target, role)])
                for target in CENTERS
                for role in ("G_delta", "R", "P")
            ],
        },
        "target_policy_lock_hash",
    )


def _actions(
    policies: Mapping[tuple[str, str], Mapping[str, object]]
) -> tuple[LockedResidualAction, ...]:
    actions: list[LockedResidualAction] = []
    for target in CENTERS:
        actions.append(
            build_locked_action(
                target_id=target,
                action_id=BASE_LONG_ID,
                selected_source=None,
            )
        )
        actions.append(
            build_locked_action(
                target_id=target,
                action_id=UNIFORM_LONG_ID,
                selected_source=None,
            )
        )
        for role, long_id in (
            ("G_delta", GLOBAL_LONG_ID),
            ("R", ROUTED_LONG_ID),
            ("P", PERMUTATION_LONG_ID),
        ):
            policy = policies[(target, role)]
            actions.append(
                build_locked_action(
                    target_id=target,
                    action_id=long_id,
                    selected_source=(None if policy["selected_source"] is None else str(policy["selected_source"])),
                    fallback_reason=(None if policy["fallback_reason"] is None else str(policy["fallback_reason"])),
                )
            )
        for source in target_sources(target):
            actions.append(
                build_locked_action(
                    target_id=target,
                    action_id=f"{ORACLE_PREFIX}{source}",
                    selected_source=source,
                )
            )
    return tuple(actions)


def _shared_bindings(
    *,
    inputs: PolicyInputs,
    feature_surface_hash: str,
    feature_schema_hash: str,
    model_lock_hash: str,
    global_ablation_lock_hash: str,
    cardinality_transfer_lock_hash: str,
    target_policy_lock_hash: str,
) -> dict[str, object]:
    return {
        "exact_tail_surface_lock_hash": inputs.exact_lock.surface_lock_hash,
        "equal_union_policy_lock_hash": inputs.equal_union_lock_hash,
        "metadata_profile_sha256": METADATA_PROFILE_SHA256,
        "development_reservation_artifact_id": DEVELOPMENT_RESERVATION_ARTIFACT_ID,
        "development_reservation_hash": inputs.exact_lock.reservation_index_hash,
        "development_case_manifest_hash": inputs.development_case_manifest_hash,
        "development_support_case_ids_by_query": {
            target: list(inputs.development_support_case_ids_by_query[target])
            for target in CENTERS
        },
        "development_evaluation_case_ids_by_query": {
            target: list(inputs.development_evaluation_case_ids_by_query[target])
            for target in CENTERS
        },
        "development_target_evaluation_case_ids_by_target": {
            target: list(
                inputs.development_target_evaluation_case_ids_by_target[target]
            )
            for target in CENTERS
        },
        "development_partition_hashes_by_query": {
            target: inputs.development_partition_hashes_by_query[target]
            for target in CENTERS
        },
        "target_support_surface_artifact_id": TARGET_SUPPORT_SURFACE_ARTIFACT_ID,
        "target_support_surface_hash": inputs.target_support_surface_hash,
        "target_support_parent_reservation_artifact_id": inputs.target_support_parent_reservation_artifact_id,
        "target_support_parent_reservation_hash": inputs.target_support_parent_reservation_hash,
        "target_reservation_artifact_id": TARGET_RESERVATION_ARTIFACT_ID,
        "target_reservation_hash": inputs.target_reservation_hash,
        "target_support_case_ids_by_target": {
            target: list(inputs.support_case_ids_by_target[target]) for target in CENTERS
        },
        "target_evaluation_case_ids_by_target": {
            target: list(inputs.evaluation_case_ids_by_target[target]) for target in CENTERS
        },
        "target_evaluation_binding_hash": inputs.target_evaluation_binding_hash,
        "feature_surface_hash": feature_surface_hash,
        "feature_schema_hash": feature_schema_hash,
        "model_lock_hash": model_lock_hash,
        "global_ablation_lock_hash": global_ablation_lock_hash,
        "cardinality_transfer_lock_hash": cardinality_transfer_lock_hash,
        "target_policy_lock_hash": target_policy_lock_hash,
    }


def _hashed(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    values = dict(payload)
    values[key] = canonical_sha256(values)
    return values


__all__ = ("BuiltPolicyBundle", "build_policy_artifacts")
