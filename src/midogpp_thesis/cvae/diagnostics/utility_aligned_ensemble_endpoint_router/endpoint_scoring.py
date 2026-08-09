"""Stage-90 adapters for the neutral exact-nine ensemble endpoint.

Only this module accepts development evaluation labels.  Its output is one
candidate-level ``(H, q, e)`` response; raw per-seed BACC rows are not a model
input and are intentionally absent from this API.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.utility_aligned.ensemble_endpoint import (
    build_ensemble_utility_response,
    score_nine_seed_probability_ensemble,
    support_action_probability_shift,
    validate_ensemble_utility_responses,
)
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    ProbabilityEnsembleEndpoint,
    SeedProbabilityVector,
    SupportActionProbabilityShift,
)
from ...routing.utility_aligned.ensemble_utility_contracts import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
)
from .contracts import (
    CENTERS,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    INNER_CANDIDATE_COUNT,
    candidate_sources,
)


def build_source_inner_ensemble_response(
    *,
    outer_target_id: str,
    query_id: str,
    candidate_source: str,
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
    labels: Sequence[int] | np.ndarray,
    support_partition_hash: str,
    evaluation_partition_hash: str,
    prediction_seal_hash: str,
    support_eval_disjoint: bool = True,
    predictions_sealed_before_labels: bool = True,
    source_expert_frozen: bool = True,
) -> ScoredEnsembleUtilityResponse:
    """Score one exact-nine candidate response and discard its raw arrays."""

    response = build_ensemble_utility_response(
        outer_target_id=outer_target_id,
        query_id=query_id,
        candidate_source=candidate_source,
        base_vectors=base_vectors,
        tail_vectors=tail_vectors,
        labels=labels,
        support_partition_hash=support_partition_hash,
        evaluation_partition_hash=evaluation_partition_hash,
        prediction_seal_hash=prediction_seal_hash,
        support_eval_disjoint=support_eval_disjoint,
        predictions_sealed_before_labels=predictions_sealed_before_labels,
        source_expert_frozen=source_expert_frozen,
        target_labels_used_for_routing=False,
    )
    return response.to_scored_response()


def build_support_action_shift(
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
) -> SupportActionProbabilityShift:
    """Build the only model-eligible local scalar (ensemble first, then abs)."""

    return support_action_probability_shift(base_vectors, tail_vectors)


def validate_heldout_source_inner_ensemble_responses(
    rows: Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
    *,
    outer_target_id: object,
) -> EnsembleUtilitySurface:
    """Validate the complete 8x7 candidate response surface for one ``H``."""

    target = str(outer_target_id)
    if target not in CENTERS:
        raise ProtocolError("Held-out ensemble utility target is unknown.")
    surface = validate_ensemble_utility_responses(rows)
    expected = len(candidate_sources(target)) * INNER_CANDIDATE_COUNT
    if (
        surface.outer_target_ids != (target,)
        or len(surface.rows) != expected
        or {row.query_id for row in surface.rows} != set(candidate_sources(target))
    ):
        raise ProtocolError("Held-out ensemble utility response geometry drifted.")
    return surface


def validate_source_inner_ensemble_responses(
    rows: Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
) -> EnsembleUtilitySurface:
    """Validate all 504 primary candidate responses for the terminal study."""

    surface = validate_ensemble_utility_responses(rows)
    if (
        surface.outer_target_ids != CENTERS
        or len(surface.rows) != EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT
    ):
        raise ProtocolError(
            "Stage-90 ensemble utility requires exactly 504 candidate responses."
        )
    return surface


__all__ = (
    "EnsembleUtilityResponse",
    "EnsembleUtilitySurface",
    "ProbabilityEnsembleEndpoint",
    "ScoredEnsembleUtilityResponse",
    "SeedProbabilityVector",
    "SupportActionProbabilityShift",
    "build_source_inner_ensemble_response",
    "build_support_action_shift",
    "score_nine_seed_probability_ensemble",
    "validate_heldout_source_inner_ensemble_responses",
    "validate_source_inner_ensemble_responses",
)
