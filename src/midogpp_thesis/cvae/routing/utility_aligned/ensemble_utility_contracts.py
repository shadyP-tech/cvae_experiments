"""Candidate-level utility-response contracts built from exact-nine endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .ensemble_endpoint_contracts import (
    ENSEMBLE_SEED_PAIR_COUNT,
    ENSEMBLE_UTILITY_SEMANTICS,
    ProbabilityEnsembleEndpoint,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    _bounded_utility,
    _canonical_text,
    _identifiers,
)


@dataclass(frozen=True)
class EnsembleUtilityResponse:
    """One candidate-level response keyed only by ``(H, q, e)``."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    candidate_source_count: int
    support_partition_hash: str
    evaluation_partition_hash: str
    prediction_seal_hash: str
    base_endpoint: ProbabilityEnsembleEndpoint
    tail_endpoint: ProbabilityEnsembleEndpoint
    support_eval_disjoint: bool
    predictions_sealed_before_labels: bool
    source_expert_frozen: bool
    target_labels_used_for_routing: bool = False
    utility_semantics: str = ENSEMBLE_UTILITY_SEMANTICS
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, query, source = _identifiers(
            self.outer_target_id, self.query_id, self.candidate_source
        )
        if outer == query or source in {outer, query}:
            raise ProtocolError("Ensemble response requires three distinct H/q/e domains.")
        if (
            isinstance(self.candidate_source_count, bool)
            or not isinstance(self.candidate_source_count, Integral)
            or int(self.candidate_source_count) != INNER_CANDIDATE_COUNT
        ):
            raise ProtocolError("Ensemble source-inner response requires seven candidates.")
        support_hash = _canonical_text(
            self.support_partition_hash, "support_partition_hash"
        )
        evaluation_hash = _canonical_text(
            self.evaluation_partition_hash, "evaluation_partition_hash"
        )
        seal_hash = _canonical_text(self.prediction_seal_hash, "prediction_seal_hash")
        if support_hash == evaluation_hash:
            raise ProtocolError("Support and evaluation partition hashes must differ.")
        if (
            not isinstance(self.base_endpoint, ProbabilityEnsembleEndpoint)
            or not isinstance(self.tail_endpoint, ProbabilityEnsembleEndpoint)
            or self.base_endpoint.row_identity_hash
            != self.tail_endpoint.row_identity_hash
            or self.base_endpoint.label_hash != self.tail_endpoint.label_hash
        ):
            raise ProtocolError("Base and tail ensemble endpoints are not row/label paired.")
        if (
            self.support_eval_disjoint is not True
            or self.predictions_sealed_before_labels is not True
            or self.source_expert_frozen is not True
            or self.target_labels_used_for_routing is not False
            or self.utility_semantics != ENSEMBLE_UTILITY_SEMANTICS
        ):
            raise ProtocolError("Ensemble utility response violates the label boundary.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "candidate_source_count", INNER_CANDIDATE_COUNT)
        object.__setattr__(self, "support_partition_hash", support_hash)
        object.__setattr__(self, "evaluation_partition_hash", evaluation_hash)
        object.__setattr__(self, "prediction_seal_hash", seal_hash)
        object.__setattr__(self, "row_hash", canonical_sha256(self.to_payload()))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    @property
    def base_bacc(self) -> float:
        return self.base_endpoint.balanced_accuracy

    @property
    def tail_bacc(self) -> float:
        return self.tail_endpoint.balanced_accuracy

    @property
    def base_endpoint_hash(self) -> str:
        return self.base_endpoint.endpoint_hash

    @property
    def tail_endpoint_hash(self) -> str:
        return self.tail_endpoint.endpoint_hash

    @property
    def utility_delta(self) -> float:
        return self.tail_bacc - self.base_bacc

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_ensemble_utility_response_v1"
            ),
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "base_endpoint_hash": self.base_endpoint.endpoint_hash,
            "tail_endpoint_hash": self.tail_endpoint.endpoint_hash,
            "evaluation_row_identity_hash": self.base_endpoint.row_identity_hash,
            "evaluation_label_hash": self.base_endpoint.label_hash,
            "base_component_vector_hashes": list(
                self.base_endpoint.component_vector_hashes
            ),
            "tail_component_vector_hashes": list(
                self.tail_endpoint.component_vector_hashes
            ),
            "base_bacc": self.base_bacc,
            "tail_bacc": self.tail_bacc,
            "utility_delta": self.utility_delta,
            "support_eval_disjoint": self.support_eval_disjoint,
            "predictions_sealed_before_labels": self.predictions_sealed_before_labels,
            "source_expert_frozen": self.source_expert_frozen,
            "target_labels_used_for_routing": self.target_labels_used_for_routing,
            "utility_semantics": self.utility_semantics,
        }

    def to_scored_response(self) -> ScoredEnsembleUtilityResponse:
        """Drop raw arrays while preserving the complete scored provenance."""

        return ScoredEnsembleUtilityResponse(
            outer_target_id=self.outer_target_id,
            query_id=self.query_id,
            candidate_source=self.candidate_source,
            candidate_source_count=self.candidate_source_count,
            support_partition_hash=self.support_partition_hash,
            evaluation_partition_hash=self.evaluation_partition_hash,
            prediction_seal_hash=self.prediction_seal_hash,
            evaluation_row_identity_hash=self.base_endpoint.row_identity_hash,
            evaluation_label_hash=self.base_endpoint.label_hash,
            base_endpoint_hash=self.base_endpoint_hash,
            tail_endpoint_hash=self.tail_endpoint_hash,
            base_probability_cell_hashes_hash=canonical_sha256(
                list(self.base_endpoint.component_vector_hashes)
            ),
            tail_probability_cell_hashes_hash=canonical_sha256(
                list(self.tail_endpoint.component_vector_hashes)
            ),
            base_ensemble_probability_hash=array_sha256(
                self.base_endpoint.mean_positive_probabilities
            ),
            tail_ensemble_probability_hash=array_sha256(
                self.tail_endpoint.mean_positive_probabilities
            ),
            base_ensemble_prediction_hash=array_sha256(
                self.base_endpoint.predictions
            ),
            tail_ensemble_prediction_hash=array_sha256(
                self.tail_endpoint.predictions
            ),
            source_response_hash=self.row_hash,
            source_endpoint_row_hash=None,
            base_component_vector_hashes=self.base_endpoint.component_vector_hashes,
            tail_component_vector_hashes=self.tail_endpoint.component_vector_hashes,
            base_bacc=self.base_bacc,
            tail_bacc=self.tail_bacc,
            support_eval_disjoint=self.support_eval_disjoint,
            predictions_sealed_before_labels=self.predictions_sealed_before_labels,
            source_expert_frozen=self.source_expert_frozen,
            target_labels_used_for_routing=self.target_labels_used_for_routing,
            utility_semantics=self.utility_semantics,
        )


@dataclass(frozen=True)
class ScoredEnsembleUtilityResponse:
    """Persistable BACC/hash-only candidate response.

    This is the model-facing DTO.  It deliberately cannot reconstruct or
    pretend to contain raw probability arrays after the scoring boundary.
    """

    outer_target_id: str
    query_id: str
    candidate_source: str
    candidate_source_count: int
    support_partition_hash: str
    evaluation_partition_hash: str
    prediction_seal_hash: str
    evaluation_row_identity_hash: str
    evaluation_label_hash: str | None
    base_endpoint_hash: str
    tail_endpoint_hash: str
    base_probability_cell_hashes_hash: str
    tail_probability_cell_hashes_hash: str
    base_ensemble_probability_hash: str
    tail_ensemble_probability_hash: str
    base_ensemble_prediction_hash: str
    tail_ensemble_prediction_hash: str
    source_response_hash: str | None
    source_endpoint_row_hash: str | None
    base_component_vector_hashes: tuple[str, ...]
    tail_component_vector_hashes: tuple[str, ...]
    base_bacc: float
    tail_bacc: float
    support_eval_disjoint: bool
    predictions_sealed_before_labels: bool
    source_expert_frozen: bool
    target_labels_used_for_routing: bool = False
    utility_semantics: str = ENSEMBLE_UTILITY_SEMANTICS
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, query, source = _identifiers(
            self.outer_target_id, self.query_id, self.candidate_source
        )
        if outer == query or source in {outer, query}:
            raise ProtocolError("Scored ensemble response requires distinct H/q/e.")
        if (
            isinstance(self.candidate_source_count, bool)
            or not isinstance(self.candidate_source_count, Integral)
            or int(self.candidate_source_count) != INNER_CANDIDATE_COUNT
        ):
            raise ProtocolError("Scored ensemble response requires seven candidates.")
        hashes = {
            name: _canonical_text(value, name)
            for name, value in {
                "support_partition_hash": self.support_partition_hash,
                "evaluation_partition_hash": self.evaluation_partition_hash,
                "prediction_seal_hash": self.prediction_seal_hash,
                "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
                "base_endpoint_hash": self.base_endpoint_hash,
                "tail_endpoint_hash": self.tail_endpoint_hash,
                "base_probability_cell_hashes_hash": (
                    self.base_probability_cell_hashes_hash
                ),
                "tail_probability_cell_hashes_hash": (
                    self.tail_probability_cell_hashes_hash
                ),
                "base_ensemble_probability_hash": self.base_ensemble_probability_hash,
                "tail_ensemble_probability_hash": self.tail_ensemble_probability_hash,
                "base_ensemble_prediction_hash": self.base_ensemble_prediction_hash,
                "tail_ensemble_prediction_hash": self.tail_ensemble_prediction_hash,
            }.items()
        }
        label_hash = (
            None
            if self.evaluation_label_hash is None
            else _canonical_text(self.evaluation_label_hash, "evaluation_label_hash")
        )
        source_response_hash = (
            None
            if self.source_response_hash is None
            else _canonical_text(self.source_response_hash, "source_response_hash")
        )
        source_endpoint_row_hash = (
            None
            if self.source_endpoint_row_hash is None
            else _canonical_text(
                self.source_endpoint_row_hash, "source_endpoint_row_hash"
            )
        )
        if hashes["support_partition_hash"] == hashes["evaluation_partition_hash"]:
            raise ProtocolError("Support and evaluation partition hashes must differ.")
        base_components = tuple(
            _canonical_text(value, "base_component_vector_hash")
            for value in self.base_component_vector_hashes
        )
        tail_components = tuple(
            _canonical_text(value, "tail_component_vector_hash")
            for value in self.tail_component_vector_hashes
        )
        if (base_components or tail_components) and (
            len(base_components) != ENSEMBLE_SEED_PAIR_COUNT
            or len(tail_components) != ENSEMBLE_SEED_PAIR_COUNT
            or len(set(base_components)) != ENSEMBLE_SEED_PAIR_COUNT
            or len(set(tail_components)) != ENSEMBLE_SEED_PAIR_COUNT
        ):
            raise ProtocolError(
                "Optional scored response components require exact-nine unique hashes."
            )
        base_bacc = _bounded_utility(self.base_bacc, "base_bacc")
        tail_bacc = _bounded_utility(self.tail_bacc, "tail_bacc")
        if (
            self.support_eval_disjoint is not True
            or self.predictions_sealed_before_labels is not True
            or self.source_expert_frozen is not True
            or self.target_labels_used_for_routing is not False
            or self.utility_semantics != ENSEMBLE_UTILITY_SEMANTICS
        ):
            raise ProtocolError("Scored ensemble response violates the label boundary.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "candidate_source_count", INNER_CANDIDATE_COUNT)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "evaluation_label_hash", label_hash)
        object.__setattr__(self, "source_response_hash", source_response_hash)
        object.__setattr__(
            self, "source_endpoint_row_hash", source_endpoint_row_hash
        )
        object.__setattr__(self, "base_component_vector_hashes", base_components)
        object.__setattr__(self, "tail_component_vector_hashes", tail_components)
        object.__setattr__(self, "base_bacc", base_bacc)
        object.__setattr__(self, "tail_bacc", tail_bacc)
        object.__setattr__(self, "row_hash", canonical_sha256(self.to_payload()))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    @property
    def utility_delta(self) -> float:
        return self.tail_bacc - self.base_bacc

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_scored_ensemble_utility_response_v1",
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "evaluation_label_hash": self.evaluation_label_hash,
            "base_endpoint_hash": self.base_endpoint_hash,
            "tail_endpoint_hash": self.tail_endpoint_hash,
            "base_probability_cell_hashes_hash": (
                self.base_probability_cell_hashes_hash
            ),
            "tail_probability_cell_hashes_hash": (
                self.tail_probability_cell_hashes_hash
            ),
            "base_ensemble_probability_hash": self.base_ensemble_probability_hash,
            "tail_ensemble_probability_hash": self.tail_ensemble_probability_hash,
            "base_ensemble_prediction_hash": self.base_ensemble_prediction_hash,
            "tail_ensemble_prediction_hash": self.tail_ensemble_prediction_hash,
            "source_response_hash": self.source_response_hash,
            "source_endpoint_row_hash": self.source_endpoint_row_hash,
            "base_component_vector_hashes": list(self.base_component_vector_hashes),
            "tail_component_vector_hashes": list(self.tail_component_vector_hashes),
            "base_bacc": self.base_bacc,
            "tail_bacc": self.tail_bacc,
            "utility_delta": self.utility_delta,
            "support_eval_disjoint": self.support_eval_disjoint,
            "predictions_sealed_before_labels": self.predictions_sealed_before_labels,
            "source_expert_frozen": self.source_expert_frozen,
            "target_labels_used_for_routing": self.target_labels_used_for_routing,
            "utility_semantics": self.utility_semantics,
        }


@dataclass(frozen=True)
class EnsembleUtilitySurface:
    rows: tuple[ScoredEnsembleUtilityResponse, ...]
    outer_target_ids: tuple[str, ...]
    row_keys: tuple[tuple[str, str, str], ...]
    surface_hash: str

    def rows_for_outer_target(
        self, outer_target_id: object
    ) -> tuple[ScoredEnsembleUtilityResponse, ...]:
        target = _canonical_text(outer_target_id, "outer_target_id")
        selected = tuple(row for row in self.rows if row.outer_target_id == target)
        if not selected:
            raise ProtocolError("Ensemble utility surface has no requested outer target.")
        return selected






__all__ = (
    "EnsembleUtilityResponse",
    "EnsembleUtilitySurface",
    "ScoredEnsembleUtilityResponse",
)

