"""Hash-bound scientific phase products for the target-static diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .actions import FrozenTargetActionLibrary
from .contracts import CENTERS
from .endpoint_adapter import DevelopmentEndpointResponseSet
from .features import SourceInnerFeatureSurfaceSet, TargetFeatureSurfaces
from .inference import TerminalEndpointScoreSet, TerminalInferenceProducts
from .models import EndpointRouterModelSet
from .partitions import ConsumedTestPartitionSurface
from .policy import FrozenTargetPolicySet


@dataclass(frozen=True)
class PrelabelScientificProducts:
    """All science objects frozen before any same-target evaluation label opens."""

    partitions: ConsumedTestPartitionSurface
    development_responses: DevelopmentEndpointResponseSet
    source_features: SourceInnerFeatureSurfaceSet
    models: EndpointRouterModelSet
    target_features_by_target: Mapping[str, TargetFeatureSurfaces]
    policies: FrozenTargetPolicySet
    actions: FrozenTargetActionLibrary
    global_prelabel_seal_hash: str
    product_hash: str

    def __post_init__(self) -> None:
        features = {
            str(target): value
            for target, value in self.target_features_by_target.items()
        }
        if (
            not isinstance(self.partitions, ConsumedTestPartitionSurface)
            or not isinstance(self.development_responses, DevelopmentEndpointResponseSet)
            or not isinstance(self.source_features, SourceInnerFeatureSurfaceSet)
            or not isinstance(self.models, EndpointRouterModelSet)
            or not isinstance(self.policies, FrozenTargetPolicySet)
            or not isinstance(self.actions, FrozenTargetActionLibrary)
            or tuple(features) != CENTERS
            or self.models.source_feature_surface_set_hash
            != self.source_features.surface_set_hash
            or self.models.development_response_set_hash
            != self.development_responses.response_set_hash
            or self.policies.model_set_hash != self.models.model_set_hash
            or self.actions.policy_set_hash != self.policies.policy_set_hash
            or not _text(self.global_prelabel_seal_hash)
        ):
            raise ProtocolError("Prelabel scientific product boundary drifted.")
        for target in CENTERS:
            feature = features[target]
            model = self.models.by_target[target]
            policy = self.policies.by_target[target]
            action_set = self.actions.by_target[target]
            if (
                not isinstance(feature, TargetFeatureSurfaces)
                or feature.target_id != target
                or feature.source_feature_surface_hash
                != self.source_features.by_target[target].surface_hash
                or feature.support_partition_lock_hash != self.partitions.lock_hash
                or model.source_feature_surface_hash
                != self.source_features.by_target[target].surface_hash
                or policy.model_hash != model.model_hash
                or policy.target_feature_hash != feature.feature_hash
                or policy.support_partition_lock_hash != self.partitions.lock_hash
                or action_set.policy_hash != policy.policy_hash
            ):
                raise ProtocolError("Prelabel target-wise lineage drifted.")
        if self.product_hash != canonical_sha256(
            _prelabel_payload(
                self.partitions,
                self.development_responses,
                self.source_features,
                self.models,
                features,
                self.policies,
                self.actions,
                self.global_prelabel_seal_hash,
            )
        ):
            raise ProtocolError("Prelabel scientific product hash drifted.")
        object.__setattr__(
            self, "target_features_by_target", MappingProxyType(features)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **_prelabel_payload(
                self.partitions,
                self.development_responses,
                self.source_features,
                self.models,
                self.target_features_by_target,
                self.policies,
                self.actions,
                self.global_prelabel_seal_hash,
            ),
            "product_hash": self.product_hash,
        }


@dataclass(frozen=True)
class CoreScientificProducts:
    """Complete terminal product; it exposes no route or model update method."""

    prelabel: PrelabelScientificProducts
    terminal_scores: TerminalEndpointScoreSet
    terminal_inference: TerminalInferenceProducts
    product_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prelabel, PrelabelScientificProducts)
            or not isinstance(self.terminal_scores, TerminalEndpointScoreSet)
            or not isinstance(self.terminal_inference, TerminalInferenceProducts)
            or self.terminal_scores.action_library_hash
            != self.prelabel.actions.action_library_hash
            or self.terminal_scores.policy_set_hash
            != self.prelabel.policies.policy_set_hash
            or self.terminal_scores.global_prelabel_seal_hash
            != self.prelabel.global_prelabel_seal_hash
            or self.terminal_inference.score_set_hash
            != self.terminal_scores.score_set_hash
            or self.product_hash != canonical_sha256(self._unhashed_payload())
        ):
            raise ProtocolError("Core scientific product lineage drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_endpoint_router_core_products_v1",
            "prelabel_product_hash": self.prelabel.product_hash,
            "terminal_score_set_hash": self.terminal_scores.score_set_hash,
            "terminal_inference_hash": self.terminal_inference.inference_hash,
            "target_static": True,
            "support_labels_used": False,
            "terminal_scores_may_update_plan": False,
            "consumed_test_diagnostic_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "product_hash": self.product_hash}


def build_prelabel_scientific_products(
    *,
    partitions: ConsumedTestPartitionSurface,
    development_responses: DevelopmentEndpointResponseSet,
    source_features: SourceInnerFeatureSurfaceSet,
    models: EndpointRouterModelSet,
    target_features_by_target: Mapping[str, TargetFeatureSurfaces],
    policies: FrozenTargetPolicySet,
    actions: FrozenTargetActionLibrary,
    global_prelabel_seal_hash: str,
) -> PrelabelScientificProducts:
    features = {target: target_features_by_target[target] for target in CENTERS}
    payload = _prelabel_payload(
        partitions,
        development_responses,
        source_features,
        models,
        features,
        policies,
        actions,
        global_prelabel_seal_hash,
    )
    return PrelabelScientificProducts(
        partitions=partitions,
        development_responses=development_responses,
        source_features=source_features,
        models=models,
        target_features_by_target=features,
        policies=policies,
        actions=actions,
        global_prelabel_seal_hash=global_prelabel_seal_hash,
        product_hash=canonical_sha256(payload),
    )


def build_core_scientific_products(
    *,
    prelabel: PrelabelScientificProducts,
    terminal_scores: TerminalEndpointScoreSet,
    terminal_inference: TerminalInferenceProducts,
) -> CoreScientificProducts:
    payload = {
        "schema_version": "midogpp_consumed_test_endpoint_router_core_products_v1",
        "prelabel_product_hash": prelabel.product_hash,
        "terminal_score_set_hash": terminal_scores.score_set_hash,
        "terminal_inference_hash": terminal_inference.inference_hash,
        "target_static": True,
        "support_labels_used": False,
        "terminal_scores_may_update_plan": False,
        "consumed_test_diagnostic_only": True,
    }
    return CoreScientificProducts(
        prelabel=prelabel,
        terminal_scores=terminal_scores,
        terminal_inference=terminal_inference,
        product_hash=canonical_sha256(payload),
    )


def _prelabel_payload(
    partitions: ConsumedTestPartitionSurface,
    development_responses: DevelopmentEndpointResponseSet,
    source_features: SourceInnerFeatureSurfaceSet,
    models: EndpointRouterModelSet,
    target_features_by_target: Mapping[str, TargetFeatureSurfaces],
    policies: FrozenTargetPolicySet,
    actions: FrozenTargetActionLibrary,
    global_prelabel_seal_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_endpoint_router_prelabel_products_v1",
        "support_partition_lock_hash": partitions.lock_hash,
        "development_response_set_hash": development_responses.response_set_hash,
        "source_feature_surface_set_hash": source_features.surface_set_hash,
        "model_set_hash": models.model_set_hash,
        "target_feature_hashes_by_target": {
            target: target_features_by_target[target].feature_hash for target in CENTERS
        },
        "policy_set_hash": policies.policy_set_hash,
        "action_library_hash": actions.action_library_hash,
        "global_prelabel_seal_hash": global_prelabel_seal_hash,
        "target_count": len(CENTERS),
        "target_static": True,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "terminal_scores_used": False,
        "consumed_test_diagnostic_only": True,
    }


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


__all__ = (
    "CoreScientificProducts",
    "PrelabelScientificProducts",
    "build_core_scientific_products",
    "build_prelabel_scientific_products",
)
