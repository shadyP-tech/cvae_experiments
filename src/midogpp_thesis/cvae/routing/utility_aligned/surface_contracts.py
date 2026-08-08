"""Immutable label-free feature and sealed utility surface contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    CaseBootstrapReplicate,
    ExactTailUtilityRow,
    _canonical_text,
    _finite,
    _identifiers,
    _nonnegative,
    _seed,
)


FEATURE_SEMANTICS = (
    "label_free_support_distribution_summaries_within_query_normalized"
)


@dataclass(frozen=True)
class CandidateFeatureRow:
    """Label-free support summary for one candidate and paired seed cell.

    There is intentionally no label, utility, prediction, oracle, or target
    evaluation field in this constructor.
    """

    role: str
    outer_target_id: str
    query_id: str
    candidate_source: str
    training_seed: int
    generation_seed: int
    candidate_source_count: int
    support_partition_hash: str
    support_case_count: int
    reconstruction_mean: float
    reconstruction_std: float
    reconstruction_q25: float
    reconstruction_q50: float
    reconstruction_q75: float
    kl_mean: float
    kl_std: float
    kl_q25: float
    kl_q50: float
    kl_q75: float
    replica_disagreement: float
    distribution_mmd: float
    metadata_similarity: float
    feature_semantics: str = FEATURE_SEMANTICS
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, query, source = _identifiers(
            self.outer_target_id, self.query_id, self.candidate_source
        )
        if self.role not in {INNER_ROLE, TARGET_ROLE}:
            raise ProtocolError("Candidate feature row role is invalid.")
        expected_count = (
            INNER_CANDIDATE_COUNT if self.role == INNER_ROLE else TARGET_CANDIDATE_COUNT
        )
        if (
            isinstance(self.candidate_source_count, bool)
            or not isinstance(self.candidate_source_count, Integral)
            or int(self.candidate_source_count) != expected_count
        ):
            raise ProtocolError("Candidate feature row cardinality/role drifted.")
        if self.role == INNER_ROLE:
            if outer == query:
                raise ProtocolError("Source-inner feature rows require q != H.")
            if source in {outer, query}:
                raise ProtocolError("Source-inner feature candidate must exclude H and q.")
        else:
            if outer != query:
                raise ProtocolError("Fresh target feature rows require query == target H.")
            if source == outer:
                raise ProtocolError("Fresh target expert cannot enter the candidate pool.")
        training_seed = _seed(self.training_seed, TRAINING_SEEDS, "training_seed")
        generation_seed = _seed(
            self.generation_seed, GENERATION_SEEDS, "generation_seed"
        )
        support_hash = _canonical_text(
            self.support_partition_hash, "support_partition_hash"
        )
        if (
            isinstance(self.support_case_count, bool)
            or not isinstance(self.support_case_count, Integral)
            or int(self.support_case_count) <= 0
        ):
            raise ProtocolError("Support case count must be a positive integer.")
        values = {
            "reconstruction_mean": _nonnegative(
                self.reconstruction_mean, "reconstruction_mean"
            ),
            "reconstruction_std": _nonnegative(
                self.reconstruction_std, "reconstruction_std"
            ),
            "reconstruction_q25": _nonnegative(
                self.reconstruction_q25, "reconstruction_q25"
            ),
            "reconstruction_q50": _nonnegative(
                self.reconstruction_q50, "reconstruction_q50"
            ),
            "reconstruction_q75": _nonnegative(
                self.reconstruction_q75, "reconstruction_q75"
            ),
            "kl_mean": _nonnegative(self.kl_mean, "kl_mean"),
            "kl_std": _nonnegative(self.kl_std, "kl_std"),
            "kl_q25": _nonnegative(self.kl_q25, "kl_q25"),
            "kl_q50": _nonnegative(self.kl_q50, "kl_q50"),
            "kl_q75": _nonnegative(self.kl_q75, "kl_q75"),
            "replica_disagreement": _nonnegative(
                self.replica_disagreement, "replica_disagreement"
            ),
            "distribution_mmd": _nonnegative(
                self.distribution_mmd, "distribution_mmd"
            ),
            "metadata_similarity": _finite(
                self.metadata_similarity, "metadata_similarity"
            ),
        }
        if not (
            values["reconstruction_q25"]
            <= values["reconstruction_q50"]
            <= values["reconstruction_q75"]
            and values["kl_q25"] <= values["kl_q50"] <= values["kl_q75"]
        ):
            raise ProtocolError("Distributional feature quantiles are not ordered.")
        if not 0.0 <= values["metadata_similarity"] <= 1.0:
            raise ProtocolError("Metadata similarity must lie in [0, 1].")
        if self.feature_semantics != FEATURE_SEMANTICS:
            raise ProtocolError("Candidate feature semantics drifted.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "training_seed", training_seed)
        object.__setattr__(self, "generation_seed", generation_seed)
        object.__setattr__(self, "candidate_source_count", expected_count)
        object.__setattr__(self, "support_partition_hash", support_hash)
        object.__setattr__(self, "support_case_count", int(self.support_case_count))
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "row_hash", canonical_sha256(self.to_payload()))

    @property
    def replicate_id(self) -> str:
        return f"training_{self.training_seed}__generation_{self.generation_seed}"

    @property
    def row_key(self) -> tuple[str, str, str, int, int]:
        return (
            self.outer_target_id,
            self.query_id,
            self.candidate_source,
            self.training_seed,
            self.generation_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_candidate_feature_row_v1",
            "role": self.role,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "replicate_id": self.replicate_id,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "support_case_count": self.support_case_count,
            "reconstruction_mean": self.reconstruction_mean,
            "reconstruction_std": self.reconstruction_std,
            "reconstruction_q25": self.reconstruction_q25,
            "reconstruction_q50": self.reconstruction_q50,
            "reconstruction_q75": self.reconstruction_q75,
            "kl_mean": self.kl_mean,
            "kl_std": self.kl_std,
            "kl_q25": self.kl_q25,
            "kl_q50": self.kl_q50,
            "kl_q75": self.kl_q75,
            "replica_disagreement": self.replica_disagreement,
            "distribution_mmd": self.distribution_mmd,
            "metadata_similarity": self.metadata_similarity,
            "feature_semantics": self.feature_semantics,
        }


@dataclass(frozen=True)
class ExactTailUtilitySurface:
    rows: tuple[ExactTailUtilityRow, ...]
    outer_target_ids: tuple[str, ...]
    row_keys: tuple[tuple[str, str, str, int, int], ...]
    surface_hash: str

    def rows_for_outer_target(self, outer_target_id: object) -> tuple[ExactTailUtilityRow, ...]:
        target = _canonical_text(outer_target_id, "outer_target_id")
        selected = tuple(row for row in self.rows if row.outer_target_id == target)
        if not selected:
            raise ProtocolError("Exact-tail utility surface has no requested outer target.")
        return selected


@dataclass(frozen=True)
class FeatureSurface:
    role: str
    outer_target_id: str
    candidate_sources: tuple[str, ...]
    rows: tuple[CandidateFeatureRow, ...]
    row_keys: tuple[tuple[str, str, str, int, int], ...]
    global_feature_names: tuple[str, ...]
    interaction_feature_names: tuple[str, ...]
    global_values: np.ndarray
    interaction_values: np.ndarray
    permutation_seed: int | None
    case_bootstrap_replicate: CaseBootstrapReplicate | None
    surface_hash: str

    @property
    def query_clusters(self) -> tuple[str, ...]:
        return tuple(row.query_id for row in self.rows)

    @property
    def source_clusters(self) -> tuple[str, ...]:
        return tuple(row.candidate_source for row in self.rows)

    @property
    def candidate_source_count(self) -> int:
        return (
            INNER_CANDIDATE_COUNT if self.role == INNER_ROLE else TARGET_CANDIDATE_COUNT
        )


@dataclass(frozen=True)
class PairwisePreference:
    outer_target_id: str
    query_id: str
    training_seed: int
    generation_seed: int
    left_source: str
    right_source: str
    left_utility_delta: float
    right_utility_delta: float
    preferred_source: str | None
    utility_margin: float
    preference_hash: str


def _immutable_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy()
    if not np.isfinite(array).all():
        raise ProtocolError("Utility-aligned array must be finite.")
    array.setflags(write=False)
    return array


__all__ = (
    "FEATURE_SEMANTICS",
    "CandidateFeatureRow",
    "ExactTailUtilitySurface",
    "FeatureSurface",
    "PairwisePreference",
)
