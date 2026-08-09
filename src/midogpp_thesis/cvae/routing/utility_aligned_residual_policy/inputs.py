"""Thin orchestration facade for independently validated policy inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ..exact_tail_utility_surface.bundle import ExactTailUtilitySurfaceLock
from ..utility_aligned import FeatureSurface, TargetSupportActionShiftCase
from .config import UtilityAlignedResidualPolicyConfig
from .ensemble_inputs import EnsembleEndpointInputs, load_ensemble_endpoint_inputs
from .exact_inputs import load_equal_union, load_exact_ensemble_policy_inputs
from .target_inputs import (
    TARGET_RESERVATION_MEMBER,
    TARGET_SUPPORT_LOCK_MEMBER,
    TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID,
    TargetFeatureSet,
    load_target_inputs,
    parse_target_feature_set,
)


@dataclass(frozen=True)
class PolicyInputs:
    exact_lock: ExactTailUtilitySurfaceLock
    ensemble_endpoint: EnsembleEndpointInputs
    inner_feature_surfaces: Mapping[str, FeatureSurface]
    equal_union_lock_hash: str
    target_support_surface_hash: str
    development_case_manifest_hash: str
    development_support_case_ids_by_query: Mapping[str, tuple[str, ...]]
    development_evaluation_case_ids_by_query: Mapping[str, tuple[str, ...]]
    development_target_evaluation_case_ids_by_target: Mapping[str, tuple[str, ...]]
    development_partition_hashes_by_query: Mapping[str, str]
    target_support_parent_reservation_artifact_id: str
    target_support_parent_reservation_hash: str
    target_reservation_hash: str
    target_evaluation_binding_hash: str
    support_case_ids_by_target: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_target: Mapping[str, tuple[str, ...]]
    target_features_by_target: Mapping[str, TargetFeatureSet]
    target_action_shift_bindings: Mapping[str, object]
    target_action_shift_cases_by_target: Mapping[
        str, tuple[TargetSupportActionShiftCase, ...]
    ]


def load_policy_inputs(config: UtilityAlignedResidualPolicyConfig) -> PolicyInputs:
    exact = load_exact_ensemble_policy_inputs(config)
    lock = exact.lock
    inner = exact.inner_feature_surfaces
    development = exact.development_manifest
    ensemble_endpoint = load_ensemble_endpoint_inputs(config.exact_tail_surface_root)
    target = load_target_inputs(
        support_surface_root=config.target_support_surface_root,
        parent_reservation_root=config.target_support_parent_reservation_root,
        target_reservation_root=config.target_reservation_root,
    )
    if dict(development.target_evaluation_case_ids_by_center) != dict(
        target.evaluation_case_ids_by_target
    ):
        raise ProtocolError(
            "Exact-tail excluded target-evaluation cases differ from Stage-70 reservation."
        )
    development_opened = {
        case_id
        for mapping in (
            development.support_case_ids_by_center,
            development.evaluation_case_ids_by_center,
        )
        for values in mapping.values()
        for case_id in values
    }
    fresh_target = {
        case_id
        for mapping in (
            target.support_case_ids_by_target,
            target.evaluation_case_ids_by_target,
        )
        for values in mapping.values()
        for case_id in values
    }
    if development_opened & fresh_target:
        raise ProtocolError(
            "Exact-tail development cases overlap fresh target support/evaluation."
        )
    configured_scalar = str(config.model.get("target_local_scalar_name", ""))
    if target.action_shift_bindings.get("target_local_scalar_name") != configured_scalar:
        raise ProtocolError("Configured target-local scalar differs from target-support lock.")
    if (
        ensemble_endpoint.endpoint_lock.endpoint_lock_hash
        != lock.ensemble_endpoint_lock_hash
        or ensemble_endpoint.endpoint_lock.endpoint_table_sha256
        != lock.ensemble_endpoint_table_sha256
        or ensemble_endpoint.endpoint_lock.endpoint_row_hashes_hash
        != lock.ensemble_endpoint_row_hashes_hash
        or ensemble_endpoint.support_shift_lock.shift_lock_hash
        != lock.support_shift_lock_hash
        or ensemble_endpoint.support_shift_lock.shift_table_sha256
        != lock.support_shift_table_sha256
        or ensemble_endpoint.support_shift_lock.shift_row_hashes_hash
        != lock.support_shift_row_hashes_hash
        or any(
            row.support_partition_hash
            != development.partition_hashes_by_center[row.pseudo_query]
            for row in ensemble_endpoint.rows
        )
    ):
        raise ProtocolError(
            "Ensemble endpoint/support-shift inputs escaped the exact surface lock."
        )
    return PolicyInputs(
        exact_lock=lock,
        ensemble_endpoint=ensemble_endpoint,
        inner_feature_surfaces=MappingProxyType(inner),
        equal_union_lock_hash=load_equal_union(config.equal_union_policy_root),
        target_support_surface_hash=target.surface_hash,
        development_case_manifest_hash=development.case_manifest_hash,
        development_support_case_ids_by_query=development.support_case_ids_by_center,
        development_evaluation_case_ids_by_query=development.evaluation_case_ids_by_center,
        development_target_evaluation_case_ids_by_target=(
            development.target_evaluation_case_ids_by_center
        ),
        development_partition_hashes_by_query=development.partition_hashes_by_center,
        target_support_parent_reservation_artifact_id=target.parent_artifact_id,
        target_support_parent_reservation_hash=target.parent_hash,
        target_reservation_hash=target.reservation_hash,
        target_evaluation_binding_hash=target.evaluation_binding_hash,
        support_case_ids_by_target=target.support_case_ids_by_target,
        evaluation_case_ids_by_target=target.evaluation_case_ids_by_target,
        target_features_by_target=target.feature_sets,
        target_action_shift_bindings=target.action_shift_bindings,
        target_action_shift_cases_by_target=target.action_shift_cases_by_target,
    )


__all__ = (
    "PolicyInputs", "TARGET_RESERVATION_MEMBER", "TARGET_SUPPORT_LOCK_MEMBER",
    "TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID", "TargetFeatureSet", "load_policy_inputs",
)
