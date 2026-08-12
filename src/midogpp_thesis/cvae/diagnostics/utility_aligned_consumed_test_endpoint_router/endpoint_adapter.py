"""Adapters around the neutral exact-nine probability-ensemble endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
    SeedProbabilityVector,
    TargetSupportActionShiftCase,
    build_ensemble_utility_response,
    build_target_support_action_shift_case,
    score_nine_seed_probability_ensemble,
    validate_ensemble_utility_responses,
)
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    ProbabilityEnsembleEndpoint,
)
from .contracts import (
    CENTERS,
    DEVELOPMENT_RESPONSE_COUNT,
    INNER_CANDIDATE_COUNT,
    SEED_PAIR_COUNT,
    candidate_sources,
    inner_candidate_sources,
)


@dataclass(frozen=True)
class DevelopmentEndpointResponseSet:
    """Complete 504-row response surface after development prediction sealing."""

    surface: EnsembleUtilitySurface
    development_prediction_seal_hash: str
    response_set_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.surface, EnsembleUtilitySurface)
            or len(self.surface.rows) != DEVELOPMENT_RESPONSE_COUNT
            or self.surface.outer_target_ids != CENTERS
            or not _text(self.development_prediction_seal_hash)
        ):
            raise ProtocolError("Development endpoint response set is incomplete.")
        if any(
            row.prediction_seal_hash != self.development_prediction_seal_hash
            for row in self.surface.rows
        ):
            raise ProtocolError("Development responses do not share the sealed surface.")
        expected = canonical_sha256(self._unhashed_payload())
        if self.response_set_hash != expected:
            raise ProtocolError("Development endpoint response-set hash drifted.")

    @property
    def rows(self) -> tuple[ScoredEnsembleUtilityResponse, ...]:
        return self.surface.rows

    def rows_for_outer_target(
        self, outer_target_id: object
    ) -> tuple[ScoredEnsembleUtilityResponse, ...]:
        rows = self.surface.rows_for_outer_target(outer_target_id)
        if len(rows) != 56:
            raise ProtocolError("One outer H requires exactly 56 H/q/e responses.")
        return rows

    def binding_hash_for_outer_target(self, outer_target_id: object) -> str:
        """Bind only rows eligible for model ``H``.

        This per-``H`` hash intentionally excludes responses where that center
        serves as ``q`` for another outer fold, preserving the same-``H`` label
        poison-invariance required by the role-scoped development protocol.
        """

        rows = self.rows_for_outer_target(outer_target_id)
        return canonical_sha256(
            {
                "schema_version": (
                    "midogpp_consumed_test_outer_development_response_binding_v1"
                ),
                "outer_target_id": str(outer_target_id),
                "row_hashes": [row.row_hash for row in rows],
                "response_count": len(rows),
                "same_outer_H_evaluation_labels_used": False,
            }
        )

    def _unhashed_payload(self) -> dict[str, object]:
        return _development_response_set_payload(
            self.surface, self.development_prediction_seal_hash
        )

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "response_set_hash": self.response_set_hash}


def build_development_endpoint_response(
    *,
    outer_target_id: object,
    query_id: object,
    candidate_source: object,
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
    development_query_evaluation_labels: Sequence[int] | np.ndarray,
    support_partition_hash: str,
    evaluation_partition_hash: str,
    development_prediction_seal_hash: str,
) -> ScoredEnsembleUtilityResponse:
    """Score one legal ``(H,q,e)`` response after its development seal.

    The only label argument is explicitly scoped to evaluation rows of ``q``.
    There is no API through which support labels or same-``H`` labels can be
    supplied to the model-facing response for ``H``.
    """

    outer = str(outer_target_id)
    query = str(query_id)
    source = str(candidate_source)
    legal_sources = inner_candidate_sources(outer, query)
    if source not in legal_sources:
        raise ProtocolError("Development response violates strict H/q/e exclusion.")
    if len(tuple(base_vectors)) != SEED_PAIR_COUNT or len(tuple(tail_vectors)) != SEED_PAIR_COUNT:
        raise ProtocolError("Development response requires exact-nine probability cells.")
    response = build_ensemble_utility_response(
        outer_target_id=outer,
        query_id=query,
        candidate_source=source,
        base_vectors=base_vectors,
        tail_vectors=tail_vectors,
        labels=development_query_evaluation_labels,
        support_partition_hash=support_partition_hash,
        evaluation_partition_hash=evaluation_partition_hash,
        prediction_seal_hash=development_prediction_seal_hash,
        support_eval_disjoint=True,
        predictions_sealed_before_labels=True,
        source_expert_frozen=True,
        target_labels_used_for_routing=False,
    )
    return response.to_scored_response()


def validate_development_endpoint_responses(
    rows: Sequence[
        EnsembleUtilityResponse
        | ScoredEnsembleUtilityResponse
        | Mapping[str, object]
    ],
    *,
    development_prediction_seal_hash: str,
) -> DevelopmentEndpointResponseSet:
    """Validate exact 9x8x7 response coverage and one pre-label seal."""

    surface = validate_ensemble_utility_responses(rows)
    if len(surface.rows) != DEVELOPMENT_RESPONSE_COUNT or surface.outer_target_ids != CENTERS:
        raise ProtocolError("Development endpoint surface requires exactly 504 H/q/e rows.")
    expected_keys = {
        (outer, query, source)
        for outer in CENTERS
        for query in candidate_sources(outer)
        for source in inner_candidate_sources(outer, query)
    }
    if set(surface.row_keys) != expected_keys:
        raise ProtocolError("Development endpoint H/q/e coverage drifted.")
    if any(
        row.prediction_seal_hash != development_prediction_seal_hash
        for row in surface.rows
    ):
        raise ProtocolError("Development endpoint seal binding drifted.")
    payload = _development_response_set_payload(
        surface, development_prediction_seal_hash
    )
    return DevelopmentEndpointResponseSet(
        surface=surface,
        development_prediction_seal_hash=development_prediction_seal_hash,
        response_set_hash=canonical_sha256(payload),
    )


def build_label_free_support_case_shift(
    *,
    target_id: str,
    candidate_source: str,
    case_id: str,
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
) -> TargetSupportActionShiftCase:
    """Build one label-free case unit for target whole-case bootstrapping."""

    if str(candidate_source) not in candidate_sources(target_id):
        raise ProtocolError("Target expert cannot enter a support action shift.")
    return build_target_support_action_shift_case(
        target_id=str(target_id),
        candidate_source=str(candidate_source),
        case_id=str(case_id),
        base_vectors=base_vectors,
        tail_vectors=tail_vectors,
    )


def score_sealed_probability_ensemble(
    vectors: Sequence[SeedProbabilityVector],
    terminal_evaluation_labels: Sequence[int] | np.ndarray,
) -> ProbabilityEnsembleEndpoint:
    """Thin terminal-only adapter; no model or policy object is accepted."""

    return score_nine_seed_probability_ensemble(vectors, terminal_evaluation_labels)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _development_response_set_payload(
    surface: EnsembleUtilitySurface, seal_hash: str
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_development_response_set_v1",
        "surface_hash": surface.surface_hash,
        "development_prediction_seal_hash": seal_hash,
        "response_count": len(surface.rows),
        "response_unit": "candidate_H_q_e_after_exact_nine_probability_ensemble",
        "strict_H_q_e_exclusion": True,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "cross_center_evaluation_labels_used_as_development_q_labels": True,
        "seed_cells_are_independent_observations": False,
    }


__all__ = (
    "DevelopmentEndpointResponseSet",
    "build_development_endpoint_response",
    "build_label_free_support_case_shift",
    "score_sealed_probability_ensemble",
    "validate_development_endpoint_responses",
)
