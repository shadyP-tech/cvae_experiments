"""Target-static G/R/P plans with the neutral R-or-exact-B fallback."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import EnsembleUtilityPolicy, build_ensemble_utility_policy
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    SUPPORT_BOOTSTRAP_REPLICATES,
    candidate_sources,
)
from .features import TargetFeatureSurfaces
from .models import EndpointRouterModelSet, EndpointRouterModels


@dataclass(frozen=True)
class FrozenTargetPolicy:
    """One pre-label, center-static policy and its diagnostic G/P proposals."""

    target_id: str
    core_policy: EnsembleUtilityPolicy
    model_hash: str
    target_feature_hash: str
    support_partition_lock_hash: str
    target_policy_seal_hash: str
    policy_hash: str

    def __post_init__(self) -> None:
        target = str(self.target_id)
        sources = candidate_sources(target)
        if (
            target not in CENTERS
            or not isinstance(self.core_policy, EnsembleUtilityPolicy)
            or self.core_policy.target_id != target
            or set(self.core_policy.role_prediction_by_source) != {
                GLOBAL_ACTION_ID,
                ROUTED_ACTION_ID,
                PERMUTATION_ACTION_ID,
            }
            or any(
                set(values) != set(sources)
                for values in self.core_policy.role_prediction_by_source.values()
            )
            or len(self.core_policy.bootstrap_feature_surface_hashes)
            != SUPPORT_BOOTSTRAP_REPLICATES
            or self.core_policy.selected_action_role not in {
                BASE_ACTION_ID,
                ROUTED_ACTION_ID,
            }
            or self.core_policy.exact_b_fallback
            != (self.core_policy.selected_action_role == BASE_ACTION_ID)
            or not all(
                _text(value)
                for value in (
                    self.model_hash,
                    self.target_feature_hash,
                    self.support_partition_lock_hash,
                    self.target_policy_seal_hash,
                )
            )
        ):
            raise ProtocolError("Frozen target policy boundary drifted.")
        proposal = self.routed_candidate_source
        if proposal not in sources:
            raise ProtocolError("Frozen target routed proposal is outside the candidate bank.")
        if self.policy_hash != canonical_sha256(self._unhashed_payload(target)):
            raise ProtocolError("Frozen target policy hash drifted.")
        object.__setattr__(self, "target_id", target)

    @property
    def routed_candidate_source(self) -> str:
        predictions = self.core_policy.role_prediction_by_source[ROUTED_ACTION_ID]
        return min(predictions, key=lambda source: (-predictions[source], source))

    @property
    def executed_routed_source(self) -> str | None:
        return self.core_policy.selected_source

    @property
    def exact_b_fallback(self) -> bool:
        return self.core_policy.exact_b_fallback

    @property
    def selected_action_id(self) -> str:
        return self.core_policy.selected_action_role

    def _unhashed_payload(self, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_frozen_target_policy_v1",
            "target_id": target or self.target_id,
            "core_policy_hash": self.core_policy.policy_hash,
            "model_hash": self.model_hash,
            "target_feature_hash": self.target_feature_hash,
            "support_partition_lock_hash": self.support_partition_lock_hash,
            "target_policy_seal_hash": self.target_policy_seal_hash,
            "routed_candidate_source": self.routed_candidate_source,
            "executed_routed_source": self.executed_routed_source,
            "selected_action_id": self.selected_action_id,
            "exact_B_fallback": self.exact_b_fallback,
            "fallback_reason": self.core_policy.fallback_reason,
            "source_inner_transfer_authorized": (
                self.core_policy.fallback_reason
                != "source_inner_cardinality_or_capacity_gate_failed"
            ),
            "target_static": True,
            "case_router_used": False,
            "support_labels_used": False,
            "same_outer_H_evaluation_labels_used": False,
            "target_utility_used": False,
            "may_update_from_terminal_scores": False,
            "diagnostic_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "policy_hash": self.policy_hash}


@dataclass(frozen=True)
class FrozenTargetPolicySet:
    by_target: Mapping[str, FrozenTargetPolicy]
    model_set_hash: str
    policy_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(value.target_id != target for target, value in values.items())
        ):
            raise ProtocolError("Frozen target policy set is incomplete.")
        payload = _policy_set_payload(values, self.model_set_hash)
        if self.policy_set_hash != canonical_sha256(payload):
            raise ProtocolError("Frozen target policy-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {
            **_policy_set_payload(self.by_target, self.model_set_hash),
            "policy_set_hash": self.policy_set_hash,
        }


def build_target_policy(
    models: EndpointRouterModels,
    target_features: TargetFeatureSurfaces,
    *,
    target_policy_seal_hash: str,
) -> FrozenTargetPolicy:
    """Build the only target plan; the signature deliberately has no labels."""

    if (
        not isinstance(models, EndpointRouterModels)
        or not isinstance(target_features, TargetFeatureSurfaces)
        or models.outer_target_id != target_features.target_id
        or models.source_feature_surface_hash
        != target_features.source_feature_surface_hash
        or not _text(target_policy_seal_hash)
    ):
        raise ProtocolError("Target policy model/feature binding drifted.")
    core = build_ensemble_utility_policy(
        models.global_model,
        models.routed_model,
        models.permutation_model,
        target_features.point_surface,
        target_features.bootstrap_surfaces,
        models.cardinality_transfer,
    )
    target = models.outer_target_id
    routed_predictions = core.role_prediction_by_source[ROUTED_ACTION_ID]
    proposal = min(
        routed_predictions,
        key=lambda source: (-routed_predictions[source], source),
    )
    payload = {
        "schema_version": "midogpp_consumed_test_frozen_target_policy_v1",
        "target_id": target,
        "core_policy_hash": core.policy_hash,
        "model_hash": models.model_hash,
        "target_feature_hash": target_features.feature_hash,
        "support_partition_lock_hash": target_features.support_partition_lock_hash,
        "target_policy_seal_hash": target_policy_seal_hash,
        "routed_candidate_source": proposal,
        "executed_routed_source": core.selected_source,
        "selected_action_id": core.selected_action_role,
        "exact_B_fallback": core.exact_b_fallback,
        "fallback_reason": core.fallback_reason,
        "source_inner_transfer_authorized": (
            core.fallback_reason
            != "source_inner_cardinality_or_capacity_gate_failed"
        ),
        "target_static": True,
        "case_router_used": False,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used": False,
        "target_utility_used": False,
        "may_update_from_terminal_scores": False,
        "diagnostic_only": True,
    }
    return FrozenTargetPolicy(
        target_id=target,
        core_policy=core,
        model_hash=models.model_hash,
        target_feature_hash=target_features.feature_hash,
        support_partition_lock_hash=target_features.support_partition_lock_hash,
        target_policy_seal_hash=target_policy_seal_hash,
        policy_hash=canonical_sha256(payload),
    )


def build_target_policy_set(
    models: EndpointRouterModelSet,
    target_features_by_target: Mapping[str, TargetFeatureSurfaces],
    *,
    target_policy_seal_hash: str,
) -> FrozenTargetPolicySet:
    if not isinstance(models, EndpointRouterModelSet):
        raise ProtocolError("Target policy set requires the typed model set.")
    features = {target: target_features_by_target[target] for target in CENTERS}
    values = {
        target: build_target_policy(
            models.by_target[target],
            features[target],
            target_policy_seal_hash=target_policy_seal_hash,
        )
        for target in CENTERS
    }
    payload = _policy_set_payload(values, models.model_set_hash)
    return FrozenTargetPolicySet(
        by_target=values,
        model_set_hash=models.model_set_hash,
        policy_set_hash=canonical_sha256(payload),
    )


def _policy_set_payload(
    values: Mapping[str, FrozenTargetPolicy], model_set_hash: str
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_frozen_target_policy_set_v1",
        "centers": list(CENTERS),
        "policy_hashes_by_target": {
            target: values[target].policy_hash for target in CENTERS
        },
        "model_set_hash": model_set_hash,
        "one_static_action_per_target": True,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used": False,
        "terminal_scores_may_update_policy": False,
    }


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


__all__ = (
    "FrozenTargetPolicy",
    "FrozenTargetPolicySet",
    "build_target_policy",
    "build_target_policy_set",
)
