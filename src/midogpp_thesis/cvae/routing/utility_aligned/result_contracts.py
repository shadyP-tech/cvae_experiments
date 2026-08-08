"""Immutable fitted-model and ranking-result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..local_marginal_utility.ridge import ClusterWeightedRidgeModel
from .row_contracts import INNER_CANDIDATE_COUNT


SOURCE_INNER_TOP1_CHANCE = 1.0 / INNER_CANDIDATE_COUNT
MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP = 0.46
CARDINALITY_CLAIM_ROLE = "source_inner_6_to_7_eligibility_only_not_7_to_8_evidence"
MODEL_SEMANTICS = (
    "global_candidate_source_effect_plus_low_capacity_target_source_interaction"
)


@dataclass(frozen=True)
class FoldAudit:
    heldout_query_id: str
    heldout_row_indices: tuple[int, ...]
    training_query_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]
    training_candidate_count_per_query: int
    selected_alpha: float
    inner_loss_by_alpha: Mapping[float, float]
    observation_count: int
    strict_query_source_exclusion: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inner_loss_by_alpha",
            MappingProxyType(
                {float(key): float(value) for key, value in self.inner_loss_by_alpha.items()}
            ),
        )


@dataclass(frozen=True)
class CrossfitResult:
    model_role: str
    predictions: np.ndarray
    standard_errors: np.ndarray
    folds: tuple[FoldAudit, ...]
    crossfit_hash: str


@dataclass(frozen=True)
class RankingMetrics:
    query_count: int
    seed_pair_count: int
    top1_oracle_agreement: float
    top1_lower_bound: float
    mean_spearman: float
    spearman_lower_bound: float
    defined_spearman_queries: int
    mean_normalized_oracle_gap: float
    normalized_oracle_gap_upper_bound: float
    pairwise_accuracy: float
    mean_selected_utility_delta: float
    selected_utility_lower_bound: float
    mean_oracle_utility_delta: float
    positive_selected_gain_rate: float
    metrics_hash: str


@dataclass(frozen=True)
class CardinalityTransferResult:
    outer_target_id: str
    candidate_sources: tuple[str, ...]
    training_candidate_count: int
    evaluation_candidate_count: int
    deployment_candidate_count: int
    global_metrics: RankingMetrics
    interaction_metrics: RankingMetrics
    top1_delta: float
    spearman_delta: float
    normalized_gap_reduction: float
    normalized_gap_reduction_lower_bound: float
    pairwise_accuracy_delta: float
    selected_utility_delta: float
    selected_utility_delta_lower_bound: float
    global_gate_passed: bool
    global_gate_reason: str
    eligibility_passed: bool
    eligibility_reason: str
    claim_role: str
    model_hash: str
    result_hash: str


@dataclass(frozen=True)
class UtilityAlignedModels:
    outer_target_id: str
    candidate_sources: tuple[str, ...]
    global_model: ClusterWeightedRidgeModel
    interaction_model: ClusterWeightedRidgeModel
    global_crossfit: CrossfitResult
    interaction_crossfit: CrossfitResult
    global_selected_alpha: float
    interaction_selected_alpha: float
    feature_surface_hash: str
    utility_surface_hash: str
    permutation_seed: int | None
    model_semantics: str
    model_hash: str


__all__ = (
    "CARDINALITY_CLAIM_ROLE",
    "MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP",
    "MODEL_SEMANTICS",
    "SOURCE_INNER_TOP1_CHANCE",
    "CardinalityTransferResult",
    "CrossfitResult",
    "FoldAudit",
    "RankingMetrics",
    "UtilityAlignedModels",
)
