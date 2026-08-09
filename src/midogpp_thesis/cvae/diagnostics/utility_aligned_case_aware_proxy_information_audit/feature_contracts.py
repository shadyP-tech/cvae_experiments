"""Label-free support-case and feature-surface DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .constants import (
    CENTERS,
    FEATURE_ROW_SCHEMA,
    MIN_SUPPORT_CASE_COUNT_PER_CENTER,
    candidate_sources,
)
from .contract_validation import (
    bounded,
    finite,
    hash_matrix,
    hash_sequence,
    hash_token,
    probability_matrix,
    vector_hashes,
)


@dataclass(frozen=True, eq=False)
class SupportCaseVectors:
    """Exact-nine support predictions and case-level proxy summaries."""

    case_id: str
    case_hash: str
    row_hash: str
    provenance_hash: str
    base_probabilities: np.ndarray
    tail_probabilities: np.ndarray
    reconstruction_summary: float
    kl_summary: float
    # Case-level mean of the nine per-stream log1p linear-kernel MMD values;
    # equal-case aggregation must not apply a second transform.
    log_mmd_summary: float
    base_vector_hashes: tuple[str, ...] = ()
    tail_vector_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ProtocolError("Support case_id must be a nonempty string.")
        case_hash = hash_token(self.case_hash, "case_hash")
        row_hash = hash_token(self.row_hash, "row_hash")
        provenance_hash = hash_token(self.provenance_hash, "provenance_hash")
        base = probability_matrix(self.base_probabilities, "base_probabilities")
        tail = probability_matrix(self.tail_probabilities, "tail_probabilities")
        if base.shape != tail.shape:
            raise ProtocolError("Support base/tail exact-nine geometry drifted.")
        base_hashes = vector_hashes(
            self.base_vector_hashes, base, "base_vector_hashes"
        )
        tail_hashes = vector_hashes(
            self.tail_vector_hashes, tail, "tail_vector_hashes"
        )
        object.__setattr__(self, "case_hash", case_hash)
        object.__setattr__(self, "row_hash", row_hash)
        object.__setattr__(self, "provenance_hash", provenance_hash)
        object.__setattr__(self, "base_probabilities", base)
        object.__setattr__(self, "tail_probabilities", tail)
        object.__setattr__(self, "base_vector_hashes", base_hashes)
        object.__setattr__(self, "tail_vector_hashes", tail_hashes)
        object.__setattr__(
            self,
            "reconstruction_summary",
            finite(self.reconstruction_summary, "reconstruction_summary"),
        )
        object.__setattr__(self, "kl_summary", finite(self.kl_summary, "kl_summary"))
        log_mmd = finite(self.log_mmd_summary, "log_mmd_summary")
        if log_mmd < 0.0:
            raise ProtocolError("Distribution-MMD support summaries must be nonnegative.")
        object.__setattr__(self, "log_mmd_summary", log_mmd)

    @property
    def row_count(self) -> int:
        return int(self.base_probabilities.shape[1])


@dataclass(frozen=True)
class CaseAwareProxyFeatureRow:
    """One sealed label-free candidate row after equal-case aggregation."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    candidate_source_count: int
    support_partition_hash: str
    prediction_seal_hash: str
    support_case_count: int
    support_row_count: int
    support_case_hashes: tuple[str, ...]
    support_row_hashes: tuple[str, ...]
    support_provenance_hashes: tuple[str, ...]
    base_vector_hashes_by_case: tuple[tuple[str, ...], ...]
    tail_vector_hashes_by_case: tuple[tuple[str, ...], ...]
    metadata_similarity: float
    pooled_row_weighted_abs_shift: float
    equal_case_abs_shift: float
    case_abs_shift_sd: float
    equal_case_signed_margin: float
    case_balanced_flip_rate: float
    case_balanced_entropy_change: float
    case_balanced_reconstruction: float
    case_balanced_kl: float
    case_balanced_log_mmd: float
    probability_role_used: str = "support_only"
    labels_used: bool = False
    evaluation_probabilities_used_as_features: bool = False
    technical_seed_rows_are_independent_observations: bool = False
    feature_row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.outer_target_id, "outer_target_id"),
            (self.query_id, "query_id"),
            (self.candidate_source, "candidate_source"),
        ):
            if value not in CENTERS:
                raise ProtocolError(f"{name} is outside the frozen center geometry.")
        outer = self.outer_target_id
        query = self.query_id
        source = self.candidate_source
        if outer == query or source in {outer, query}:
            raise ProtocolError("Feature row requires distinct H/q/e domains.")
        expected_candidates = len(candidate_sources(outer, query))
        if (
            type(self.candidate_source_count) is not int
            or self.candidate_source_count != expected_candidates
        ):
            raise ProtocolError("Feature candidate-source geometry drifted.")
        if (
            type(self.support_case_count) is not int
            or self.support_case_count < MIN_SUPPORT_CASE_COUNT_PER_CENTER
            or type(self.support_row_count) is not int
            or self.support_row_count < self.support_case_count
        ):
            raise ProtocolError("Feature row requires at least eight whole support cases.")
        case_hashes = hash_sequence(
            self.support_case_hashes, "support_case_hashes", self.support_case_count
        )
        row_hashes = hash_sequence(
            self.support_row_hashes, "support_row_hashes", self.support_case_count
        )
        provenance = hash_sequence(
            self.support_provenance_hashes,
            "support_provenance_hashes",
            self.support_case_count,
        )
        if len(set(case_hashes)) != len(case_hashes):
            raise ProtocolError("Support case hashes must identify whole distinct cases.")
        base_hashes = hash_matrix(
            self.base_vector_hashes_by_case,
            "base_vector_hashes_by_case",
            self.support_case_count,
        )
        tail_hashes = hash_matrix(
            self.tail_vector_hashes_by_case,
            "tail_vector_hashes_by_case",
            self.support_case_count,
        )
        support_hash = hash_token(
            self.support_partition_hash, "support_partition_hash"
        )
        prediction_hash = hash_token(self.prediction_seal_hash, "prediction_seal_hash")
        values = {
            "metadata_similarity": bounded(
                self.metadata_similarity, "metadata_similarity", 0.0, 1.0
            ),
            "pooled_row_weighted_abs_shift": bounded(
                self.pooled_row_weighted_abs_shift,
                "pooled_row_weighted_abs_shift",
                0.0,
                1.0,
            ),
            "equal_case_abs_shift": bounded(
                self.equal_case_abs_shift, "equal_case_abs_shift", 0.0, 1.0
            ),
            "case_abs_shift_sd": bounded(
                self.case_abs_shift_sd, "case_abs_shift_sd", 0.0, 1.0
            ),
            "equal_case_signed_margin": bounded(
                self.equal_case_signed_margin,
                "equal_case_signed_margin",
                -1.0,
                1.0,
            ),
            "case_balanced_flip_rate": bounded(
                self.case_balanced_flip_rate,
                "case_balanced_flip_rate",
                0.0,
                1.0,
            ),
            "case_balanced_entropy_change": bounded(
                self.case_balanced_entropy_change,
                "case_balanced_entropy_change",
                -float(np.log(2.0)),
                float(np.log(2.0)),
            ),
            "case_balanced_reconstruction": finite(
                self.case_balanced_reconstruction, "case_balanced_reconstruction"
            ),
            "case_balanced_kl": finite(self.case_balanced_kl, "case_balanced_kl"),
            "case_balanced_log_mmd": finite(
                self.case_balanced_log_mmd, "case_balanced_log_mmd"
            ),
        }
        if (
            self.probability_role_used != "support_only"
            or self.labels_used is not False
            or self.evaluation_probabilities_used_as_features is not False
            or self.technical_seed_rows_are_independent_observations is not False
        ):
            raise ProtocolError("Feature row crossed the label/probability boundary.")
        object.__setattr__(self, "support_partition_hash", support_hash)
        object.__setattr__(self, "prediction_seal_hash", prediction_hash)
        object.__setattr__(self, "support_case_hashes", case_hashes)
        object.__setattr__(self, "support_row_hashes", row_hashes)
        object.__setattr__(self, "support_provenance_hashes", provenance)
        object.__setattr__(self, "base_vector_hashes_by_case", base_hashes)
        object.__setattr__(self, "tail_vector_hashes_by_case", tail_hashes)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "feature_row_hash", canonical_sha256(self._unhashed_payload())
        )

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_ROW_SCHEMA,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "support_case_count": self.support_case_count,
            "support_row_count": self.support_row_count,
            "support_case_hashes": list(self.support_case_hashes),
            "support_row_hashes": list(self.support_row_hashes),
            "support_provenance_hashes": list(self.support_provenance_hashes),
            "base_vector_hashes_by_case": [
                list(values) for values in self.base_vector_hashes_by_case
            ],
            "tail_vector_hashes_by_case": [
                list(values) for values in self.tail_vector_hashes_by_case
            ],
            "metadata_similarity": self.metadata_similarity,
            "pooled_row_weighted_abs_shift": self.pooled_row_weighted_abs_shift,
            "equal_case_abs_shift": self.equal_case_abs_shift,
            "case_abs_shift_sd": self.case_abs_shift_sd,
            "equal_case_signed_margin": self.equal_case_signed_margin,
            "case_balanced_flip_rate": self.case_balanced_flip_rate,
            "case_balanced_entropy_change": self.case_balanced_entropy_change,
            "case_balanced_reconstruction": self.case_balanced_reconstruction,
            "case_balanced_kl": self.case_balanced_kl,
            "case_balanced_log_mmd": self.case_balanced_log_mmd,
            "probability_role_used": self.probability_role_used,
            "labels_used": self.labels_used,
            "evaluation_probabilities_used_as_features": (
                self.evaluation_probabilities_used_as_features
            ),
            "technical_seed_rows_are_independent_observations": (
                self.technical_seed_rows_are_independent_observations
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "feature_row_hash": self.feature_row_hash}


@dataclass(frozen=True)
class CaseAwareFeatureSurface:
    rows: tuple[CaseAwareProxyFeatureRow, ...]
    row_keys: tuple[tuple[str, str, str], ...]
    surface_hash: str


__all__ = (
    "CaseAwareFeatureSurface",
    "CaseAwareProxyFeatureRow",
    "SupportCaseVectors",
)
