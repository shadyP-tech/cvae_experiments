"""Two-case target proposals with an explicit non-policy status.

``R2`` is useful as a terminal consumed-data diagnostic: it asks whether the
target-interaction model would choose a better exact tail than global source
quality or a feature permutation.  Two independent support cases cannot meet
the frozen fresh-policy minimum, so this module never emits a policy,
fallback authorization, or promotion flag.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import FeatureSurface, permute_interaction_features
from .contracts import (
    CENTERS,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GLOBAL_DELTA_ACTION_ID,
    MINIMUM_FRESH_POLICY_BOOTSTRAPS,
    MINIMUM_FRESH_POLICY_SUPPORT_CASES,
    PERMUTATION_ACTION_ID,
    PERMUTATION_SEED,
    R2_ACTION_ID,
    ROUTING_STATUS,
    SEED_PAIR_COUNT,
    candidate_sources,
    seed_pairs,
)
from .features import HeldoutFeatureSurfaces, Stage90FeatureSurfaceSet
from .modeling import HeldoutRouterModels, Stage90RouterModelSet


ROUTER_DIAGNOSTIC_IDS = (
    GLOBAL_DELTA_ACTION_ID,
    R2_ACTION_ID,
    PERMUTATION_ACTION_ID,
)


@dataclass(frozen=True)
class FrozenR2DiagnosticPlan:
    """One target's three pre-terminal-scoring diagnostic proposals."""

    target_id: str
    candidate_sources: tuple[str, ...]
    proposed_source_by_router: Mapping[str, str]
    mean_prediction_by_router_source: Mapping[str, Mapping[str, float]]
    seed_predictions_by_router_source: Mapping[
        str, Mapping[str, tuple[float, ...]]
    ]
    support_case_count: int
    model_hash: str
    target_feature_surface_hash: str
    plan_hash: str
    routing_status: str = ROUTING_STATUS
    development_crossfit_labels_previously_opened: bool = True
    outer_H_development_rows_excluded_from_plan_H: bool = True
    predictions_frozen_before_terminal_target_scoring: bool = True
    target_support_labels_used: bool = False
    target_evaluation_embeddings_used: bool = False
    outer_H_development_label_rows_used_for_plan_H: bool = False
    terminal_target_labels_used_for_plan: bool = False
    seed_selection_performed: bool = False
    policy_authorized: bool = False
    fallback_authorized: bool = False
    promotion_authorized: bool = False
    deployment_authorized: bool = False
    diagnostic_action_materialization_authorized: bool = True

    def __post_init__(self) -> None:
        target = str(self.target_id)
        sources = candidate_sources(target)
        selected = {
            str(key): str(value) for key, value in self.proposed_source_by_router.items()
        }
        means = _float_nested_mapping(self.mean_prediction_by_router_source)
        predictions = _prediction_nested_mapping(
            self.seed_predictions_by_router_source
        )
        if (
            self.candidate_sources != sources
            or tuple(selected) != ROUTER_DIAGNOSTIC_IDS
            or tuple(means) != ROUTER_DIAGNOSTIC_IDS
            or tuple(predictions) != ROUTER_DIAGNOSTIC_IDS
            or any(selected[router] not in sources for router in ROUTER_DIAGNOSTIC_IDS)
            or any(tuple(means[router]) != sources for router in ROUTER_DIAGNOSTIC_IDS)
            or any(
                tuple(predictions[router]) != sources
                for router in ROUTER_DIAGNOSTIC_IDS
            )
            or any(
                len(predictions[router][source]) != SEED_PAIR_COUNT
                for router in ROUTER_DIAGNOSTIC_IDS
                for source in sources
            )
            or any(
                abs(
                    means[router][source]
                    - float(np.mean(predictions[router][source], dtype=np.float64))
                )
                > 1.0e-15
                for router in ROUTER_DIAGNOSTIC_IDS
                for source in sources
            )
            or any(
                selected[router]
                != min(sources, key=lambda source: (-means[router][source], source))
                for router in ROUTER_DIAGNOSTIC_IDS
            )
            or self.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or self.routing_status != ROUTING_STATUS
            or self.development_crossfit_labels_previously_opened is not True
            or self.outer_H_development_rows_excluded_from_plan_H is not True
            or self.predictions_frozen_before_terminal_target_scoring is not True
            or self.target_support_labels_used is not False
            or self.target_evaluation_embeddings_used is not False
            or self.outer_H_development_label_rows_used_for_plan_H is not False
            or self.terminal_target_labels_used_for_plan is not False
            or self.seed_selection_performed is not False
            or self.policy_authorized is not False
            or self.fallback_authorized is not False
            or self.promotion_authorized is not False
            or self.deployment_authorized is not False
            or self.diagnostic_action_materialization_authorized is not True
        ):
            raise ProtocolError("Stage-90 R2 diagnostic plan boundary drifted.")
        expected = canonical_sha256(
            self._unhashed_payload(
                target=target,
                selected=selected,
                means=means,
                predictions=predictions,
            )
        )
        if self.plan_hash != expected:
            raise ProtocolError("Stage-90 R2 diagnostic plan hash drifted.")
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "proposed_source_by_router", MappingProxyType(selected))
        object.__setattr__(
            self,
            "mean_prediction_by_router_source",
            MappingProxyType(
                {key: MappingProxyType(value) for key, value in means.items()}
            ),
        )
        object.__setattr__(
            self,
            "seed_predictions_by_router_source",
            MappingProxyType(
                {key: MappingProxyType(value) for key, value in predictions.items()}
            ),
        )

    @property
    def proposed_global_source(self) -> str:
        return self.proposed_source_by_router[GLOBAL_DELTA_ACTION_ID]

    @property
    def proposed_r2_source(self) -> str:
        return self.proposed_source_by_router[R2_ACTION_ID]

    @property
    def proposed_permutation_source(self) -> str:
        return self.proposed_source_by_router[PERMUTATION_ACTION_ID]

    def _unhashed_payload(
        self,
        *,
        target: str | None = None,
        selected: Mapping[str, str] | None = None,
        means: Mapping[str, Mapping[str, float]] | None = None,
        predictions: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None,
    ) -> dict[str, object]:
        selected_values = selected or self.proposed_source_by_router
        mean_values = means or self.mean_prediction_by_router_source
        prediction_values = predictions or self.seed_predictions_by_router_source
        return {
            "schema_version": "midogpp_utility_aligned_stage90_r2_plan_v1",
            "target_id": target or self.target_id,
            "candidate_sources": list(self.candidate_sources),
            "proposed_source_by_router": dict(selected_values),
            "mean_prediction_by_router_source": {
                router: dict(mean_values[router]) for router in ROUTER_DIAGNOSTIC_IDS
            },
            "seed_predictions_by_router_source": {
                router: {
                    source: list(prediction_values[router][source])
                    for source in self.candidate_sources
                }
                for router in ROUTER_DIAGNOSTIC_IDS
            },
            "seed_pair_order": [list(pair) for pair in seed_pairs()],
            "seed_pair_count": SEED_PAIR_COUNT,
            "support_case_count": self.support_case_count,
            "minimum_fresh_policy_support_cases": MINIMUM_FRESH_POLICY_SUPPORT_CASES,
            "minimum_fresh_policy_bootstraps": MINIMUM_FRESH_POLICY_BOOTSTRAPS,
            "routing_status": ROUTING_STATUS,
            "model_hash": self.model_hash,
            "target_feature_surface_hash": self.target_feature_surface_hash,
            "development_crossfit_labels_previously_opened": True,
            "outer_H_development_rows_excluded_from_plan_H": True,
            "predictions_frozen_before_terminal_target_scoring": True,
            "target_support_labels_used": False,
            "target_evaluation_embeddings_used": False,
            "outer_H_development_label_rows_used_for_plan_H": False,
            "terminal_target_labels_used_for_plan": False,
            "seed_selection_performed": False,
            "policy_authorized": False,
            "fallback_authorized": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "diagnostic_action_materialization_authorized": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "plan_hash": self.plan_hash}


@dataclass(frozen=True)
class Stage90R2PlanSet:
    by_target: Mapping[str, FrozenR2DiagnosticPlan]
    model_set_hash: str
    feature_surface_set_hash: str
    plan_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(
                not isinstance(value, FrozenR2DiagnosticPlan)
                or value.target_id != target
                for target, value in values.items()
            )
        ):
            raise ProtocolError("Stage-90 R2 plan set is incomplete.")
        expected = canonical_sha256(_plan_set_payload(
            values,
            model_set_hash=self.model_set_hash,
            feature_surface_set_hash=self.feature_surface_set_hash,
        ))
        if self.plan_set_hash != expected:
            raise ProtocolError("Stage-90 R2 plan-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {
            **_plan_set_payload(
                self.by_target,
                model_set_hash=self.model_set_hash,
                feature_surface_set_hash=self.feature_surface_set_hash,
            ),
            "plan_set_hash": self.plan_set_hash,
        }


def build_r2_diagnostic_plan(
    models: HeldoutRouterModels,
    features: HeldoutFeatureSurfaces,
) -> FrozenR2DiagnosticPlan:
    """Freeze G_delta/R2/P proposals by averaging every predeclared seed cell."""

    if (
        not isinstance(models, HeldoutRouterModels)
        or not isinstance(features, HeldoutFeatureSurfaces)
        or models.outer_target_id != features.outer_target_id
    ):
        raise ProtocolError("Stage-90 R2 planning inputs do not align.")
    target_surface = features.target
    permuted_target = permute_interaction_features(
        target_surface,
        permutation_seed=PERMUTATION_SEED,
    )
    standard = models.global_and_interaction
    permuted = models.permuted_interaction
    raw_predictions = {
        GLOBAL_DELTA_ACTION_ID: standard.global_model.predict(
            target_surface.global_values
        ),
        R2_ACTION_ID: standard.interaction_model.predict(
            target_surface.interaction_values
        ),
        PERMUTATION_ACTION_ID: permuted.interaction_model.predict(
            permuted_target.interaction_values
        ),
    }
    seed_predictions: dict[str, dict[str, tuple[float, ...]]] = {}
    means: dict[str, dict[str, float]] = {}
    selected: dict[str, str] = {}
    for router in ROUTER_DIAGNOSTIC_IDS:
        by_source = _predictions_by_source(
            target_surface,
            raw_predictions[router],
        )
        seed_predictions[router] = by_source
        means[router] = {
            source: float(np.mean(by_source[source], dtype=np.float64))
            for source in target_surface.candidate_sources
        }
        selected[router] = min(
            target_surface.candidate_sources,
            key=lambda source: (-means[router][source], source),
        )
    support_counts = {row.support_case_count for row in target_surface.rows}
    if support_counts != {FIXED_SUPPORT_CASE_COUNT_PER_CENTER}:
        raise ProtocolError("Stage-90 R2 plan does not use exactly two support cases.")
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_r2_plan_v1",
        "target_id": models.outer_target_id,
        "candidate_sources": list(target_surface.candidate_sources),
        "proposed_source_by_router": selected,
        "mean_prediction_by_router_source": means,
        "seed_predictions_by_router_source": {
            router: {
                source: list(seed_predictions[router][source])
                for source in target_surface.candidate_sources
            }
            for router in ROUTER_DIAGNOSTIC_IDS
        },
        "seed_pair_order": [list(pair) for pair in seed_pairs()],
        "seed_pair_count": SEED_PAIR_COUNT,
        "support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "minimum_fresh_policy_support_cases": MINIMUM_FRESH_POLICY_SUPPORT_CASES,
        "minimum_fresh_policy_bootstraps": MINIMUM_FRESH_POLICY_BOOTSTRAPS,
        "routing_status": ROUTING_STATUS,
        "model_hash": models.model_hash,
        "target_feature_surface_hash": target_surface.surface_hash,
        "development_crossfit_labels_previously_opened": True,
        "outer_H_development_rows_excluded_from_plan_H": True,
        "predictions_frozen_before_terminal_target_scoring": True,
        "target_support_labels_used": False,
        "target_evaluation_embeddings_used": False,
        "outer_H_development_label_rows_used_for_plan_H": False,
        "terminal_target_labels_used_for_plan": False,
        "seed_selection_performed": False,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "diagnostic_action_materialization_authorized": True,
    }
    return FrozenR2DiagnosticPlan(
        target_id=models.outer_target_id,
        candidate_sources=target_surface.candidate_sources,
        proposed_source_by_router=selected,
        mean_prediction_by_router_source=means,
        seed_predictions_by_router_source=seed_predictions,
        support_case_count=FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        model_hash=models.model_hash,
        target_feature_surface_hash=target_surface.surface_hash,
        plan_hash=canonical_sha256(unhashed),
    )


def build_stage90_r2_plan_set(
    models: Stage90RouterModelSet,
    features: Stage90FeatureSurfaceSet,
) -> Stage90R2PlanSet:
    if (
        not isinstance(models, Stage90RouterModelSet)
        or not isinstance(features, Stage90FeatureSurfaceSet)
        or models.feature_surface_set_hash != features.surface_hash
    ):
        raise ProtocolError("Stage-90 R2 model/feature sets do not align.")
    by_target = {
        target: build_r2_diagnostic_plan(
            models.by_target[target],
            features.by_target[target],
        )
        for target in CENTERS
    }
    payload = _plan_set_payload(
        by_target,
        model_set_hash=models.model_set_hash,
        feature_surface_set_hash=features.surface_hash,
    )
    return Stage90R2PlanSet(
        by_target=by_target,
        model_set_hash=models.model_set_hash,
        feature_surface_set_hash=features.surface_hash,
        plan_set_hash=canonical_sha256(payload),
    )


def _predictions_by_source(
    surface: FeatureSurface,
    values: np.ndarray,
) -> dict[str, tuple[float, ...]]:
    predictions = np.asarray(values, dtype=np.float64)
    if predictions.shape != (len(surface.rows),) or not np.isfinite(predictions).all():
        raise ProtocolError("Stage-90 target predictions are malformed.")
    grouped: dict[str, list[tuple[tuple[int, int], float]]] = defaultdict(list)
    for row, value in zip(surface.rows, predictions, strict=True):
        grouped[row.candidate_source].append(
            ((row.training_seed, row.generation_seed), float(value))
        )
    result: dict[str, tuple[float, ...]] = {}
    expected_pairs = seed_pairs()
    for source in surface.candidate_sources:
        ordered = tuple(sorted(grouped[source], key=lambda item: item[0]))
        if tuple(pair for pair, _value in ordered) != expected_pairs:
            raise ProtocolError("Stage-90 route requires every frozen seed pair exactly once.")
        result[source] = tuple(value for _pair, value in ordered)
    return result


def _float_nested_mapping(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    result = {
        str(router): {str(source): float(value) for source, value in rows.items()}
        for router, rows in values.items()
    }
    if any(not math.isfinite(value) for rows in result.values() for value in rows.values()):
        raise ProtocolError("Stage-90 mean route predictions must be finite.")
    return result


def _prediction_nested_mapping(
    values: Mapping[str, Mapping[str, tuple[float, ...]]],
) -> dict[str, dict[str, tuple[float, ...]]]:
    result = {
        str(router): {
            str(source): tuple(float(value) for value in predictions)
            for source, predictions in rows.items()
        }
        for router, rows in values.items()
    }
    if any(
        not math.isfinite(value)
        for rows in result.values()
        for predictions in rows.values()
        for value in predictions
    ):
        raise ProtocolError("Stage-90 seed route predictions must be finite.")
    return result


def _plan_set_payload(
    values: Mapping[str, FrozenR2DiagnosticPlan],
    *,
    model_set_hash: str,
    feature_surface_set_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_r2_plan_set_v1",
        "centers": list(CENTERS),
        "plan_hashes_by_target": {
            target: values[target].plan_hash for target in CENTERS
        },
        "model_set_hash": model_set_hash,
        "feature_surface_set_hash": feature_surface_set_hash,
        "routing_status": ROUTING_STATUS,
        "diagnostic_only": True,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "development_crossfit_labels_previously_opened": True,
        "outer_H_development_label_rows_used_for_plan_H": False,
        "terminal_target_labels_used_for_plans": False,
    }


__all__ = (
    "ROUTER_DIAGNOSTIC_IDS",
    "FrozenR2DiagnosticPlan",
    "Stage90R2PlanSet",
    "build_r2_diagnostic_plan",
    "build_stage90_r2_plan_set",
)
