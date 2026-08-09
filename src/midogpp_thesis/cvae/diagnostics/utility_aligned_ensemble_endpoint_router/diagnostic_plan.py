"""Frozen, non-actionable two-case ensemble-endpoint proposals."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.ensemble_feature_contracts import EnsembleFeatureSurface
from ...routing.utility_aligned.ensemble_model_contracts import EnsembleUtilityModel
from .contracts import (
    CENTERS,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GLOBAL_DELTA_ACTION_ID,
    MINIMUM_FRESH_POLICY_BOOTSTRAPS,
    MINIMUM_FRESH_POLICY_SUPPORT_CASES,
    PERMUTATION_ACTION_ID,
    ROUTED_ENSEMBLE_ACTION_ID,
    ROUTER_DIAGNOSTIC_IDS,
    ROUTING_STATUS,
    candidate_sources,
)
from .features import HeldoutEnsembleFeatureSurfaces, Stage90EnsembleFeatureSurfaceSet
from .modeling import HeldoutEnsembleRouterModels, Stage90EnsembleRouterModelSet


@dataclass(frozen=True)
class FrozenEnsembleEndpointDiagnosticPlan:
    target_id: str
    candidate_sources: tuple[str, ...]
    proposed_source_by_router: Mapping[str, str]
    prediction_by_router_source: Mapping[str, Mapping[str, float]]
    technical_seed_spread_by_source: Mapping[str, float]
    model_hash_by_router: Mapping[str, str]
    target_feature_surface_hash_by_router: Mapping[str, str]
    support_case_count: int
    model_bundle_hash: str
    target_feature_bundle_hash: str
    inner_support_shift_lock_hash: str
    target_support_shift_lock_hash: str
    target_probe_seal_hash: str
    plan_hash: str
    routing_status: str = ROUTING_STATUS
    may_update_policy: bool = False
    policy_authorized: bool = False
    fallback_authorized: bool = False
    promotion_authorized: bool = False
    deployment_authorized: bool = False
    diagnostic_action_materialization_authorized: bool = True

    def __post_init__(self) -> None:
        target = str(self.target_id)
        sources = candidate_sources(target)
        selected = {str(key): str(value) for key, value in self.proposed_source_by_router.items()}
        predictions = {
            str(router): MappingProxyType(
                {str(source): float(value) for source, value in by_source.items()}
            )
            for router, by_source in self.prediction_by_router_source.items()
        }
        spread = {
            str(source): float(value)
            for source, value in self.technical_seed_spread_by_source.items()
        }
        model_hashes = {str(key): str(value) for key, value in self.model_hash_by_router.items()}
        feature_hashes = {
            str(key): str(value)
            for key, value in self.target_feature_surface_hash_by_router.items()
        }
        if (
            self.candidate_sources != sources
            or tuple(selected) != ROUTER_DIAGNOSTIC_IDS
            or tuple(predictions) != ROUTER_DIAGNOSTIC_IDS
            or tuple(model_hashes) != ROUTER_DIAGNOSTIC_IDS
            or tuple(feature_hashes) != ROUTER_DIAGNOSTIC_IDS
            or any(set(values) != set(sources) for values in predictions.values())
            or set(spread) != set(sources)
            or any(not np.isfinite(tuple(values.values())).all() for values in predictions.values())
            or any(not np.isfinite(value) or value < 0.0 for value in spread.values())
            or any(selected[router] not in sources for router in ROUTER_DIAGNOSTIC_IDS)
            or any(
                selected[router]
                != min(sources, key=lambda source: (-predictions[router][source], source))
                for router in ROUTER_DIAGNOSTIC_IDS
            )
            or self.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or self.routing_status != ROUTING_STATUS
            or self.may_update_policy is not False
            or self.policy_authorized is not False
            or self.fallback_authorized is not False
            or self.promotion_authorized is not False
            or self.deployment_authorized is not False
            or self.diagnostic_action_materialization_authorized is not True
            or any(
                not _is_hash(value)
                for value in (
                    self.inner_support_shift_lock_hash,
                    self.target_support_shift_lock_hash,
                    self.target_probe_seal_hash,
                )
            )
        ):
            raise ProtocolError("Ensemble endpoint diagnostic plan boundary drifted.")
        expected = canonical_sha256(
            self._unhashed_payload(
                target=target,
                selected=selected,
                predictions=predictions,
                spread=spread,
                model_hashes=model_hashes,
                feature_hashes=feature_hashes,
            )
        )
        if self.plan_hash != expected:
            raise ProtocolError("Ensemble endpoint diagnostic plan hash drifted.")
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "proposed_source_by_router", MappingProxyType(selected))
        object.__setattr__(self, "prediction_by_router_source", MappingProxyType(predictions))
        object.__setattr__(self, "technical_seed_spread_by_source", MappingProxyType(spread))
        object.__setattr__(self, "model_hash_by_router", MappingProxyType(model_hashes))
        object.__setattr__(
            self,
            "target_feature_surface_hash_by_router",
            MappingProxyType(feature_hashes),
        )

    def _unhashed_payload(
        self,
        *,
        target: str | None = None,
        selected: Mapping[str, str] | None = None,
        predictions: Mapping[str, Mapping[str, float]] | None = None,
        spread: Mapping[str, float] | None = None,
        model_hashes: Mapping[str, str] | None = None,
        feature_hashes: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        resolved_predictions = predictions or self.prediction_by_router_source
        return {
            "schema_version": "midogpp_utility_aligned_stage90_ensemble_plan_v1",
            "target_id": target or self.target_id,
            "candidate_sources": list(self.candidate_sources),
            "proposed_source_by_router": dict(selected or self.proposed_source_by_router),
            "prediction_by_router_source": {
                router: dict(resolved_predictions[router])
                for router in ROUTER_DIAGNOSTIC_IDS
            },
            "technical_seed_spread_by_source": dict(
                spread or self.technical_seed_spread_by_source
            ),
            "technical_seed_spread_role": "descriptive_only_non_decision",
            "technical_seed_spread_values_may_feed_plan": False,
            "model_hash_by_router": dict(model_hashes or self.model_hash_by_router),
            "target_feature_surface_hash_by_router": dict(
                feature_hashes or self.target_feature_surface_hash_by_router
            ),
            "support_case_count": self.support_case_count,
            "minimum_fresh_policy_support_cases": MINIMUM_FRESH_POLICY_SUPPORT_CASES,
            "minimum_fresh_policy_bootstraps": MINIMUM_FRESH_POLICY_BOOTSTRAPS,
            "routing_status": ROUTING_STATUS,
            "model_bundle_hash": self.model_bundle_hash,
            "target_feature_bundle_hash": self.target_feature_bundle_hash,
            "inner_support_shift_lock_hash": self.inner_support_shift_lock_hash,
            "target_support_shift_lock_hash": self.target_support_shift_lock_hash,
            "target_probe_seal_hash": self.target_probe_seal_hash,
            "development_response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
            "development_crossfit_labels_previously_opened": True,
            "outer_H_development_rows_excluded_from_plan_H": True,
            "predictions_frozen_before_terminal_target_scoring": True,
            "target_support_labels_used": False,
            "target_evaluation_embeddings_used": False,
            "outer_H_development_label_rows_used_for_plan_H": False,
            "terminal_target_labels_used_for_plan": False,
            "seed_selection_performed": False,
            "may_update_policy": False,
            "policy_authorized": False,
            "fallback_authorized": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "diagnostic_action_materialization_authorized": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "plan_hash": self.plan_hash}


@dataclass(frozen=True)
class Stage90EnsembleDiagnosticPlanSet:
    by_target: Mapping[str, FrozenEnsembleEndpointDiagnosticPlan]
    model_set_hash: str
    feature_surface_set_hash: str
    plan_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(
                not isinstance(value, FrozenEnsembleEndpointDiagnosticPlan)
                or value.target_id != target
                for target, value in values.items()
            )
        ):
            raise ProtocolError("Stage-90 ensemble plan set is incomplete.")
        expected = canonical_sha256(
            _plan_set_payload(
                values,
                model_set_hash=self.model_set_hash,
                feature_surface_set_hash=self.feature_surface_set_hash,
            )
        )
        if self.plan_set_hash != expected:
            raise ProtocolError("Stage-90 ensemble plan-set hash drifted.")
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


def build_ensemble_endpoint_diagnostic_plan(
    models: HeldoutEnsembleRouterModels,
    features: HeldoutEnsembleFeatureSurfaces,
) -> FrozenEnsembleEndpointDiagnosticPlan:
    """Freeze G/R2E/P point proposals without uncertainty authorization."""

    if (
        not isinstance(models, HeldoutEnsembleRouterModels)
        or not isinstance(features, HeldoutEnsembleFeatureSurfaces)
        or models.outer_target_id != features.outer_target_id
        or models.feature_bundle_hash != features.surface_hash
    ):
        raise ProtocolError("Ensemble endpoint planning inputs do not align.")
    surfaces = {
        GLOBAL_DELTA_ACTION_ID: features.target_m0,
        ROUTED_ENSEMBLE_ACTION_ID: features.target_m1,
        PERMUTATION_ACTION_ID: features.target_permuted,
    }
    predictions: dict[str, dict[str, float]] = {}
    selected: dict[str, str] = {}
    for router in ROUTER_DIAGNOSTIC_IDS:
        by_source = _predict_by_source(models.by_router[router], surfaces[router])
        predictions[router] = by_source
        selected[router] = min(
            features.target_m1.candidate_sources,
            key=lambda source: (-by_source[source], source),
        )
    spread = {
        row.candidate_source: float(row.target_local_scalar_seed_standard_deviation)
        for row in features.target_m1.rows
        if row.target_local_scalar_seed_standard_deviation is not None
    }
    model_hashes = {
        router: models.by_router[router].model_hash for router in ROUTER_DIAGNOSTIC_IDS
    }
    feature_hashes = {
        router: surfaces[router].surface_hash for router in ROUTER_DIAGNOSTIC_IDS
    }
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_plan_v1",
        "target_id": models.outer_target_id,
        "candidate_sources": list(features.target_m1.candidate_sources),
        "proposed_source_by_router": selected,
        "prediction_by_router_source": predictions,
        "technical_seed_spread_by_source": spread,
        "technical_seed_spread_role": "descriptive_only_non_decision",
        "technical_seed_spread_values_may_feed_plan": False,
        "model_hash_by_router": model_hashes,
        "target_feature_surface_hash_by_router": feature_hashes,
        "support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "minimum_fresh_policy_support_cases": MINIMUM_FRESH_POLICY_SUPPORT_CASES,
        "minimum_fresh_policy_bootstraps": MINIMUM_FRESH_POLICY_BOOTSTRAPS,
        "routing_status": ROUTING_STATUS,
        "model_bundle_hash": models.model_hash,
        "target_feature_bundle_hash": features.surface_hash,
        "inner_support_shift_lock_hash": features.inner_support_shift_lock_hash,
        "target_support_shift_lock_hash": features.target_support_shift_lock_hash,
        "target_probe_seal_hash": features.target_probe_seal_hash,
        "development_response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "development_crossfit_labels_previously_opened": True,
        "outer_H_development_rows_excluded_from_plan_H": True,
        "predictions_frozen_before_terminal_target_scoring": True,
        "target_support_labels_used": False,
        "target_evaluation_embeddings_used": False,
        "outer_H_development_label_rows_used_for_plan_H": False,
        "terminal_target_labels_used_for_plan": False,
        "seed_selection_performed": False,
        "may_update_policy": False,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "diagnostic_action_materialization_authorized": True,
    }
    return FrozenEnsembleEndpointDiagnosticPlan(
        target_id=models.outer_target_id,
        candidate_sources=features.target_m1.candidate_sources,
        proposed_source_by_router=selected,
        prediction_by_router_source=predictions,
        technical_seed_spread_by_source=spread,
        model_hash_by_router=model_hashes,
        target_feature_surface_hash_by_router=feature_hashes,
        support_case_count=FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        model_bundle_hash=models.model_hash,
        target_feature_bundle_hash=features.surface_hash,
        inner_support_shift_lock_hash=features.inner_support_shift_lock_hash,
        target_support_shift_lock_hash=features.target_support_shift_lock_hash,
        target_probe_seal_hash=features.target_probe_seal_hash,
        plan_hash=canonical_sha256(unhashed),
    )


def build_stage90_ensemble_diagnostic_plan_set(
    models: Stage90EnsembleRouterModelSet,
    features: Stage90EnsembleFeatureSurfaceSet,
) -> Stage90EnsembleDiagnosticPlanSet:
    if (
        not isinstance(models, Stage90EnsembleRouterModelSet)
        or not isinstance(features, Stage90EnsembleFeatureSurfaceSet)
        or models.feature_surface_set_hash != features.surface_hash
    ):
        raise ProtocolError("Stage-90 ensemble plan-set inputs do not align.")
    values = {
        target: build_ensemble_endpoint_diagnostic_plan(
            models.by_target[target], features.by_target[target]
        )
        for target in CENTERS
    }
    payload = _plan_set_payload(
        values,
        model_set_hash=models.model_set_hash,
        feature_surface_set_hash=features.surface_hash,
    )
    return Stage90EnsembleDiagnosticPlanSet(
        by_target=values,
        model_set_hash=models.model_set_hash,
        feature_surface_set_hash=features.surface_hash,
        plan_set_hash=canonical_sha256(payload),
    )


def _predict_by_source(
    model: EnsembleUtilityModel,
    surface: EnsembleFeatureSurface,
) -> dict[str, float]:
    if (
        model.outer_target_id != surface.outer_target_id
        or model.feature_names != surface.feature_names
        or set(model.candidate_models) != set(surface.candidate_sources)
    ):
        raise ProtocolError("Ensemble point-prediction surface/model drifted.")
    output: dict[str, float] = {}
    for index, row in enumerate(surface.rows):
        value = float(
            model.candidate_models[row.candidate_source].predict(
                surface.values[index : index + 1]
            )[0]
        )
        if not np.isfinite(value):
            raise ProtocolError("Ensemble point prediction is non-finite.")
        output[row.candidate_source] = value
    if set(output) != set(surface.candidate_sources):
        raise ProtocolError("Ensemble point prediction candidate coverage drifted.")
    return output


def _plan_set_payload(
    values: Mapping[str, FrozenEnsembleEndpointDiagnosticPlan],
    *,
    model_set_hash: str,
    feature_surface_set_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_plan_set_v1",
        "centers": list(CENTERS),
        "plan_hashes_by_target": {
            target: values[target].plan_hash for target in CENTERS
        },
        "inner_support_shift_lock_hashes": sorted(
            {values[target].inner_support_shift_lock_hash for target in CENTERS}
        ),
        "target_support_shift_lock_hashes": sorted(
            {values[target].target_support_shift_lock_hash for target in CENTERS}
        ),
        "target_probe_seal_hashes": sorted(
            {values[target].target_probe_seal_hash for target in CENTERS}
        ),
        "model_set_hash": model_set_hash,
        "feature_surface_set_hash": feature_surface_set_hash,
        "support_case_count_per_target": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "routing_status": ROUTING_STATUS,
        "may_update_policy": False,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "diagnostic_action_materialization_authorized": True,
    }


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 16 and value.strip() == value


__all__ = (
    "FrozenEnsembleEndpointDiagnosticPlan",
    "Stage90EnsembleDiagnosticPlanSet",
    "build_ensemble_endpoint_diagnostic_plan",
    "build_stage90_ensemble_diagnostic_plan_set",
)
