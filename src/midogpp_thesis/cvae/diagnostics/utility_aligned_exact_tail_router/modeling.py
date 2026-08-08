"""Strict per-H exact-tail utility models for the consumed diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.local_marginal_utility.ridge import DEFAULT_RIDGE_ALPHAS
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    ExactTailUtilityRow,
    ExactTailUtilitySurface,
    UtilityAlignedModels,
    fit_utility_aligned_models,
    validate_exact_tail_utility_rows,
)
from .contracts import (
    CENTERS,
    EXPECTED_INNER_UTILITY_ROW_COUNT,
    PERMUTATION_SEED,
    candidate_sources,
)
from .features import HeldoutFeatureSurfaces, Stage90FeatureSurfaceSet


@dataclass(frozen=True)
class HeldoutRouterModels:
    """Global, interaction, and permuted models whose labels exclude H."""

    outer_target_id: str
    global_and_interaction: UtilityAlignedModels
    permuted_interaction: UtilityAlignedModels
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
        standard = self.global_and_interaction
        permuted = self.permuted_interaction
        if (
            not isinstance(standard, UtilityAlignedModels)
            or not isinstance(permuted, UtilityAlignedModels)
            or standard.outer_target_id != target
            or permuted.outer_target_id != target
            or standard.candidate_sources != sources
            or permuted.candidate_sources != sources
            or standard.permutation_seed is not None
            or permuted.permutation_seed != PERMUTATION_SEED
            or self.training_query_ids != sources
            or self.training_source_ids != sources
            or target in self.training_query_ids
            or target in self.training_source_ids
            or self.strict_outer_target_query_source_exclusion is not True
            or self.heldout_target_labels_used_for_fit is not False
            or self.target_support_labels_used_for_fit is not False
            or self.target_evaluation_used_for_fit is not False
        ):
            raise ProtocolError("Stage-90 held-out model boundary drifted.")
        _validate_crossfit_exclusions(standard, outer_target=target)
        _validate_crossfit_exclusions(permuted, outer_target=target)
        expected = canonical_sha256(self._unhashed_payload(target=target))
        if self.model_hash != expected:
            raise ProtocolError("Stage-90 held-out model hash drifted.")
        object.__setattr__(self, "outer_target_id", target)

    def _unhashed_payload(self, *, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_stage90_heldout_models_v1",
            "outer_target_id": target or self.outer_target_id,
            "candidate_sources": list(self.global_and_interaction.candidate_sources),
            "global_and_interaction_model_hash": self.global_and_interaction.model_hash,
            "permuted_interaction_model_hash": self.permuted_interaction.model_hash,
            "training_query_ids": list(self.training_query_ids),
            "training_source_ids": list(self.training_source_ids),
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
class Stage90RouterModelSet:
    by_target: Mapping[str, HeldoutRouterModels]
    feature_surface_set_hash: str
    exact_utility_surface_hash: str
    model_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(
                not isinstance(value, HeldoutRouterModels)
                or value.outer_target_id != target
                for target, value in values.items()
            )
        ):
            raise ProtocolError("Stage-90 model set is incomplete.")
        expected = canonical_sha256(_model_set_payload(
            values,
            feature_surface_set_hash=self.feature_surface_set_hash,
            exact_utility_surface_hash=self.exact_utility_surface_hash,
        ))
        if self.model_set_hash != expected:
            raise ProtocolError("Stage-90 model-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {
            **_model_set_payload(
                self.by_target,
                feature_surface_set_hash=self.feature_surface_set_hash,
                exact_utility_surface_hash=self.exact_utility_surface_hash,
            ),
            "model_set_hash": self.model_set_hash,
        }


def fit_stage90_heldout_models(
    features: HeldoutFeatureSurfaces,
    utility: ExactTailUtilitySurface | Sequence[ExactTailUtilityRow],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    permutation_seed: int = PERMUTATION_SEED,
) -> HeldoutRouterModels:
    """Fit one H model pair while keeping H out of query/source label roles."""

    if not isinstance(features, HeldoutFeatureSurfaces):
        raise ProtocolError("Stage-90 model fitting requires typed feature surfaces.")
    if permutation_seed != PERMUTATION_SEED:
        raise ProtocolError("Stage-90 permutation seed is frozen and cannot be tuned.")
    utility_surface = (
        utility
        if isinstance(utility, ExactTailUtilitySurface)
        else validate_exact_tail_utility_rows(utility)
    )
    standard = fit_utility_aligned_models(
        features.inner,
        utility_surface,
        alphas=alphas,
    )
    permuted = fit_utility_aligned_models(
        features.inner,
        utility_surface,
        alphas=alphas,
        permutation_seed=PERMUTATION_SEED,
    )
    target = features.outer_target_id
    queries = tuple(sorted({row.query_id for row in features.inner.rows}))
    sources = tuple(sorted({row.candidate_source for row in features.inner.rows}))
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_heldout_models_v1",
        "outer_target_id": target,
        "candidate_sources": list(standard.candidate_sources),
        "global_and_interaction_model_hash": standard.model_hash,
        "permuted_interaction_model_hash": permuted.model_hash,
        "training_query_ids": list(queries),
        "training_source_ids": list(sources),
        "strict_outer_target_query_source_exclusion": True,
        "nested_query_source_exclusion": True,
        "heldout_target_labels_used_for_fit": False,
        "target_support_labels_used_for_fit": False,
        "target_evaluation_used_for_fit": False,
        "seed_selection_performed": False,
        "permutation_seed": PERMUTATION_SEED,
    }
    return HeldoutRouterModels(
        outer_target_id=target,
        global_and_interaction=standard,
        permuted_interaction=permuted,
        training_query_ids=queries,
        training_source_ids=sources,
        model_hash=canonical_sha256(unhashed),
    )


def fit_stage90_models(
    features: Stage90FeatureSurfaceSet,
    utility: ExactTailUtilitySurface | Sequence[ExactTailUtilityRow],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    permutation_seed: int = PERMUTATION_SEED,
) -> Stage90RouterModelSet:
    """Fit all nine independent outer-target model bundles."""

    if not isinstance(features, Stage90FeatureSurfaceSet):
        raise ProtocolError("Stage-90 model fitting requires a complete feature set.")
    utility_surface = (
        utility
        if isinstance(utility, ExactTailUtilitySurface)
        else validate_exact_tail_utility_rows(utility)
    )
    if (
        utility_surface.outer_target_ids != CENTERS
        or len(utility_surface.rows) != EXPECTED_INNER_UTILITY_ROW_COUNT
    ):
        raise ProtocolError("Stage-90 exact-tail utility surface is incomplete.")
    by_target = {
        target: fit_stage90_heldout_models(
            features.by_target[target],
            utility_surface,
            alphas=alphas,
            permutation_seed=permutation_seed,
        )
        for target in CENTERS
    }
    payload = _model_set_payload(
        by_target,
        feature_surface_set_hash=features.surface_hash,
        exact_utility_surface_hash=utility_surface.surface_hash,
    )
    return Stage90RouterModelSet(
        by_target=by_target,
        feature_surface_set_hash=features.surface_hash,
        exact_utility_surface_hash=utility_surface.surface_hash,
        model_set_hash=canonical_sha256(payload),
    )


def _validate_crossfit_exclusions(
    models: UtilityAlignedModels,
    *,
    outer_target: str,
) -> None:
    for result in (models.global_crossfit, models.interaction_crossfit):
        expected_queries = set(candidate_sources(outer_target))
        if {fold.heldout_query_id for fold in result.folds} != expected_queries:
            raise ProtocolError("Stage-90 nested query coverage drifted.")
        for fold in result.folds:
            if (
                outer_target in fold.training_query_ids
                or outer_target in fold.training_source_ids
                or fold.heldout_query_id in fold.training_query_ids
                or fold.heldout_query_id in fold.training_source_ids
                or fold.strict_query_source_exclusion is not True
            ):
                raise ProtocolError("Stage-90 nested H/q/e exclusion failed.")


def _model_set_payload(
    values: Mapping[str, HeldoutRouterModels],
    *,
    feature_surface_set_hash: str,
    exact_utility_surface_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_model_set_v1",
        "centers": list(CENTERS),
        "model_hashes_by_target": {
            target: values[target].model_hash for target in CENTERS
        },
        "feature_surface_set_hash": feature_surface_set_hash,
        "exact_utility_surface_hash": exact_utility_surface_hash,
        "strict_outer_target_query_source_exclusion": True,
        "heldout_target_labels_used_for_fit": False,
        "target_support_labels_used_for_fit": False,
        "target_evaluation_used_for_fit": False,
        "diagnostic_only": True,
    }


__all__ = (
    "HeldoutRouterModels",
    "Stage90RouterModelSet",
    "fit_stage90_heldout_models",
    "fit_stage90_models",
)
