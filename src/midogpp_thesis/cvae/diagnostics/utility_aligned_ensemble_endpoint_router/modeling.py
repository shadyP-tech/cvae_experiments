"""Strict candidate-level ensemble utility models for the Stage-90 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.local_marginal_utility.ridge import DEFAULT_RIDGE_ALPHAS
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.ensemble_endpoint import (
    validate_ensemble_utility_responses,
)
from ...routing.utility_aligned.ensemble_model_contracts import EnsembleUtilityModel
from ...routing.utility_aligned.ensemble_modeling import fit_ensemble_utility_model
from ...routing.utility_aligned.ensemble_utility_contracts import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
)
from .contracts import (
    CENTERS,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    GLOBAL_DELTA_ACTION_ID,
    PERMUTATION_ACTION_ID,
    PERMUTATION_SEED,
    ROUTED_ENSEMBLE_ACTION_ID,
    candidate_sources,
)
from .features import (
    HeldoutEnsembleFeatureSurfaces,
    Stage90EnsembleFeatureSurfaceSet,
)


@dataclass(frozen=True)
class HeldoutEnsembleRouterModels:
    """M0, M1 and permuted-M1 models whose response labels exclude ``H``."""

    outer_target_id: str
    global_model: EnsembleUtilityModel
    routed_model: EnsembleUtilityModel
    permutation_model: EnsembleUtilityModel
    feature_bundle_hash: str
    inner_support_shift_lock_hash: str
    target_support_shift_lock_hash: str
    target_probe_seal_hash: str
    training_query_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]
    model_hash: str
    strict_outer_target_query_source_exclusion: bool = True
    heldout_target_labels_used_for_fit: bool = False
    target_support_labels_used_for_fit: bool = False
    target_evaluation_used_for_fit: bool = False

    def __post_init__(self) -> None:
        target = str(self.outer_target_id)
        sources = candidate_sources(target)
        models = (self.global_model, self.routed_model, self.permutation_model)
        if (
            any(not isinstance(model, EnsembleUtilityModel) for model in models)
            or any(model.outer_target_id != target for model in models)
            or self.global_model.feature_names != ("global_source_control",)
            or len(self.routed_model.feature_names) != 2
            or self.permutation_model.feature_names != self.routed_model.feature_names
            or self.global_model.permutation_seed is not None
            or self.routed_model.permutation_seed is not None
            or self.permutation_model.permutation_seed != PERMUTATION_SEED
            or self.training_query_ids != sources
            or self.training_source_ids != sources
            or target in self.training_query_ids
            or target in self.training_source_ids
            or self.strict_outer_target_query_source_exclusion is not True
            or self.heldout_target_labels_used_for_fit is not False
            or self.target_support_labels_used_for_fit is not False
            or self.target_evaluation_used_for_fit is not False
            or any(
                not _is_hash(value)
                for value in (
                    self.inner_support_shift_lock_hash,
                    self.target_support_shift_lock_hash,
                    self.target_probe_seal_hash,
                )
            )
        ):
            raise ProtocolError("Held-out ensemble model boundary drifted.")
        for model in models:
            _validate_model_exclusions(model, outer_target=target)
        expected = canonical_sha256(self._unhashed_payload(target=target))
        if self.model_hash != expected:
            raise ProtocolError("Held-out ensemble model hash drifted.")
        object.__setattr__(self, "outer_target_id", target)

    @property
    def by_router(self) -> Mapping[str, EnsembleUtilityModel]:
        return MappingProxyType(
            {
                GLOBAL_DELTA_ACTION_ID: self.global_model,
                ROUTED_ENSEMBLE_ACTION_ID: self.routed_model,
                PERMUTATION_ACTION_ID: self.permutation_model,
            }
        )

    def _unhashed_payload(self, *, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_stage90_ensemble_models_v1",
            "outer_target_id": target or self.outer_target_id,
            "model_hashes_by_router": {
                GLOBAL_DELTA_ACTION_ID: self.global_model.model_hash,
                ROUTED_ENSEMBLE_ACTION_ID: self.routed_model.model_hash,
                PERMUTATION_ACTION_ID: self.permutation_model.model_hash,
            },
            "feature_bundle_hash": self.feature_bundle_hash,
            "inner_support_shift_lock_hash": self.inner_support_shift_lock_hash,
            "target_support_shift_lock_hash": self.target_support_shift_lock_hash,
            "target_probe_seal_hash": self.target_probe_seal_hash,
            "training_query_ids": list(self.training_query_ids),
            "training_source_ids": list(self.training_source_ids),
            "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
            "seed_rows_are_independent_observations": False,
            "strict_outer_target_query_source_exclusion": True,
            "nested_query_source_exclusion": True,
            "heldout_target_labels_used_for_fit": False,
            "target_support_labels_used_for_fit": False,
            "target_evaluation_used_for_fit": False,
            "seed_selection_performed": False,
            "permutation_seed": PERMUTATION_SEED,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "model_hash": self.model_hash}


@dataclass(frozen=True)
class Stage90EnsembleRouterModelSet:
    by_target: Mapping[str, HeldoutEnsembleRouterModels]
    feature_surface_set_hash: str
    ensemble_utility_surface_hash: str
    model_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(
                not isinstance(value, HeldoutEnsembleRouterModels)
                or value.outer_target_id != target
                for target, value in values.items()
            )
        ):
            raise ProtocolError("Stage-90 ensemble model set is incomplete.")
        expected = canonical_sha256(
            _model_set_payload(
                values,
                feature_surface_set_hash=self.feature_surface_set_hash,
                ensemble_utility_surface_hash=self.ensemble_utility_surface_hash,
            )
        )
        if self.model_set_hash != expected:
            raise ProtocolError("Stage-90 ensemble model-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {
            **_model_set_payload(
                self.by_target,
                feature_surface_set_hash=self.feature_surface_set_hash,
                ensemble_utility_surface_hash=self.ensemble_utility_surface_hash,
            ),
            "model_set_hash": self.model_set_hash,
        }


def fit_stage90_heldout_ensemble_models(
    features: HeldoutEnsembleFeatureSurfaces,
    utility: EnsembleUtilitySurface
    | Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    permutation_seed: int = PERMUTATION_SEED,
) -> HeldoutEnsembleRouterModels:
    """Fit one H-excluded M0/M1/P bundle from candidate-level responses."""

    if not isinstance(features, HeldoutEnsembleFeatureSurfaces):
        raise ProtocolError("Stage-90 ensemble fitting requires typed features.")
    if permutation_seed != PERMUTATION_SEED:
        raise ProtocolError("Stage-90 ensemble permutation seed cannot be tuned.")
    surface = (
        utility
        if isinstance(utility, EnsembleUtilitySurface)
        else validate_ensemble_utility_responses(utility)
    )
    target = features.outer_target_id
    heldout_rows = surface.rows_for_outer_target(target)
    if len(heldout_rows) != 56:
        raise ProtocolError("Held-out ensemble model requires 56 candidate responses.")
    global_model = fit_ensemble_utility_model(
        features.inner_m0, surface, alphas=alphas
    )
    routed_model = fit_ensemble_utility_model(
        features.inner_m1, surface, alphas=alphas
    )
    permutation_model = fit_ensemble_utility_model(
        features.inner_permuted, surface, alphas=alphas
    )
    queries = tuple(sorted({row.query_id for row in features.inner_m1.rows}))
    sources = tuple(sorted({row.candidate_source for row in features.inner_m1.rows}))
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_models_v1",
        "outer_target_id": target,
        "model_hashes_by_router": {
            GLOBAL_DELTA_ACTION_ID: global_model.model_hash,
            ROUTED_ENSEMBLE_ACTION_ID: routed_model.model_hash,
            PERMUTATION_ACTION_ID: permutation_model.model_hash,
        },
        "feature_bundle_hash": features.surface_hash,
        "inner_support_shift_lock_hash": features.inner_support_shift_lock_hash,
        "target_support_shift_lock_hash": features.target_support_shift_lock_hash,
        "target_probe_seal_hash": features.target_probe_seal_hash,
        "training_query_ids": list(queries),
        "training_source_ids": list(sources),
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "seed_rows_are_independent_observations": False,
        "strict_outer_target_query_source_exclusion": True,
        "nested_query_source_exclusion": True,
        "heldout_target_labels_used_for_fit": False,
        "target_support_labels_used_for_fit": False,
        "target_evaluation_used_for_fit": False,
        "seed_selection_performed": False,
        "permutation_seed": PERMUTATION_SEED,
    }
    return HeldoutEnsembleRouterModels(
        outer_target_id=target,
        global_model=global_model,
        routed_model=routed_model,
        permutation_model=permutation_model,
        feature_bundle_hash=features.surface_hash,
        inner_support_shift_lock_hash=features.inner_support_shift_lock_hash,
        target_support_shift_lock_hash=features.target_support_shift_lock_hash,
        target_probe_seal_hash=features.target_probe_seal_hash,
        training_query_ids=queries,
        training_source_ids=sources,
        model_hash=canonical_sha256(unhashed),
    )


def fit_stage90_ensemble_models(
    features: Stage90EnsembleFeatureSurfaceSet,
    utility: EnsembleUtilitySurface
    | Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
) -> Stage90EnsembleRouterModelSet:
    """Fit all nine independent outer-H model bundles."""

    if not isinstance(features, Stage90EnsembleFeatureSurfaceSet):
        raise ProtocolError("Stage-90 ensemble fitting requires a complete feature set.")
    surface = (
        utility
        if isinstance(utility, EnsembleUtilitySurface)
        else validate_ensemble_utility_responses(utility)
    )
    if (
        surface.outer_target_ids != CENTERS
        or len(surface.rows) != EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT
    ):
        raise ProtocolError("Stage-90 ensemble utility surface is incomplete.")
    by_target = {
        target: fit_stage90_heldout_ensemble_models(
            features.by_target[target], surface, alphas=alphas
        )
        for target in CENTERS
    }
    payload = _model_set_payload(
        by_target,
        feature_surface_set_hash=features.surface_hash,
        ensemble_utility_surface_hash=surface.surface_hash,
    )
    return Stage90EnsembleRouterModelSet(
        by_target=by_target,
        feature_surface_set_hash=features.surface_hash,
        ensemble_utility_surface_hash=surface.surface_hash,
        model_set_hash=canonical_sha256(payload),
    )


def _validate_model_exclusions(
    model: EnsembleUtilityModel,
    *,
    outer_target: str,
) -> None:
    expected_queries = set(candidate_sources(outer_target))
    if {audit.predicted_row_key[1] for audit in model.fold_audits} != expected_queries:
        raise ProtocolError("Ensemble model nested query coverage drifted.")
    for audit in model.fold_audits:
        query = audit.predicted_row_key[1]
        source = audit.predicted_row_key[2]
        required = {outer_target, query, source}
        if (
            not required.issubset(set(audit.excluded_domain_ids))
            or required.intersection(audit.training_query_ids)
            or required.intersection(audit.training_source_ids)
            or audit.strict_h_q_e_exclusion is not True
        ):
            raise ProtocolError("Ensemble model strict H/q/e exclusion failed.")


def _model_set_payload(
    values: Mapping[str, HeldoutEnsembleRouterModels],
    *,
    feature_surface_set_hash: str,
    ensemble_utility_surface_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_model_set_v1",
        "centers": list(CENTERS),
        "model_hashes_by_target": {
            target: values[target].model_hash for target in CENTERS
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
        "feature_surface_set_hash": feature_surface_set_hash,
        "ensemble_utility_surface_hash": ensemble_utility_surface_hash,
        "primary_response_count": EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "seed_rows_are_independent_observations": False,
        "strict_outer_target_query_source_exclusion": True,
        "heldout_target_labels_used_for_fit": False,
        "target_support_labels_used_for_fit": False,
        "target_evaluation_used_for_fit": False,
        "diagnostic_only": True,
    }


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 16 and value.strip() == value


__all__ = (
    "HeldoutEnsembleRouterModels",
    "Stage90EnsembleRouterModelSet",
    "fit_stage90_ensemble_models",
    "fit_stage90_heldout_ensemble_models",
)
