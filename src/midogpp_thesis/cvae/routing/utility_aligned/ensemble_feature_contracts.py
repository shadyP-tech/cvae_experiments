"""Feature and fresh-target contracts for candidate-level ensemble routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .ensemble_endpoint_contracts import (
    ENSEMBLE_SEED_KEYS,
    ENSEMBLE_SEED_PAIR_COUNT,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    _bounded_utility,
    _canonical_text,
    _finite,
    _identifiers,
    _nonnegative,
)


GLOBAL_SOURCE_CONTROL_SEMANTICS = (
    "label_free_mean_metadata_similarity_over_outer_H_source_inner_"
    "pseudoqueries_and_exact_nine_seed_cells_v1"
)
GLOBAL_SOURCE_CONTROL_NAME = "global_source_control"

AGGREGATED_FEATURE_NAMES = (
    "reconstruction_mean",
    "reconstruction_std",
    "reconstruction_q25",
    "reconstruction_q50",
    "reconstruction_q75",
    "kl_mean",
    "kl_std",
    "kl_q25",
    "kl_q50",
    "kl_q75",
    "replica_disagreement",
    "distribution_mmd",
    "metadata_similarity",
)


@dataclass(frozen=True)
class GlobalSourceControl:
    """Label-free M0 source control derived only from source-inner features."""

    outer_target_id: str
    value_by_source: Mapping[str, float]
    source_inner_seed_row_count: int
    input_row_hashes_hash: str
    provenance_hash: str
    semantics: str = GLOBAL_SOURCE_CONTROL_SEMANTICS

    def __post_init__(self) -> None:
        outer = _canonical_text(self.outer_target_id, "outer_target_id")
        values = {
            _canonical_text(source, "candidate_source"): _finite(
                value, "global_source_control"
            )
            for source, value in self.value_by_source.items()
        }
        if len(values) != TARGET_CANDIDATE_COUNT or outer in values:
            raise ProtocolError("Global source control requires eight H-excluded sources.")
        if (
            isinstance(self.source_inner_seed_row_count, bool)
            or not isinstance(self.source_inner_seed_row_count, Integral)
            or int(self.source_inner_seed_row_count)
            != TARGET_CANDIDATE_COUNT
            * INNER_CANDIDATE_COUNT
            * ENSEMBLE_SEED_PAIR_COUNT
        ):
            raise ProtocolError("Global source control source-inner coverage drifted.")
        input_hash = _canonical_text(
            self.input_row_hashes_hash, "input_row_hashes_hash"
        )
        provenance_hash = _canonical_text(self.provenance_hash, "provenance_hash")
        if self.semantics != GLOBAL_SOURCE_CONTROL_SEMANTICS:
            raise ProtocolError("Global source control semantics drifted.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "value_by_source", MappingProxyType(values))
        object.__setattr__(
            self, "source_inner_seed_row_count", int(self.source_inner_seed_row_count)
        )
        object.__setattr__(self, "input_row_hashes_hash", input_hash)
        object.__setattr__(self, "provenance_hash", provenance_hash)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_global_source_control_v1",
            "outer_target_id": self.outer_target_id,
            "value_by_source": dict(self.value_by_source),
            "source_inner_seed_row_count": self.source_inner_seed_row_count,
            "input_row_hashes_hash": self.input_row_hashes_hash,
            "semantics": self.semantics,
            "labels_used": False,
            "utility_responses_used": False,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class TargetSupportActionShiftCase:
    """One independent target-support case for the local action-shift scalar."""

    target_id: str
    candidate_source: str
    case_id: str
    support_row_identity_hash: str
    support_row_count: int
    seed_keys: tuple[tuple[int, int], ...]
    per_seed_mean_absolute_shifts: tuple[float, ...]
    base_component_vector_hashes: tuple[str, ...]
    tail_component_vector_hashes: tuple[str, ...]
    ensemble_mean_absolute_shift: float
    base_ensemble_probability_hash: str
    tail_ensemble_probability_hash: str
    ensemble_absolute_difference_hash: str
    case_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target, source, case_id = _identifiers(
            self.target_id, self.candidate_source, self.case_id
        )
        if target == source:
            raise ProtocolError("Target expert cannot enter support action-shift rows.")
        row_hash = _canonical_text(
            self.support_row_identity_hash, "support_row_identity_hash"
        )
        try:
            support_row_count = int(self.support_row_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Per-case action shift row count is invalid.") from exc
        if isinstance(self.support_row_count, bool) or support_row_count <= 0:
            raise ProtocolError("Per-case action shift row count must be positive.")
        seed_keys = tuple((int(left), int(right)) for left, right in self.seed_keys)
        if seed_keys != ENSEMBLE_SEED_KEYS:
            raise ProtocolError("Per-case action shift requires canonical exact-nine seeds.")
        shifts = tuple(
            _bounded_utility(value, "per_seed_mean_absolute_shift")
            for value in self.per_seed_mean_absolute_shifts
        )
        base_hashes = tuple(
            _canonical_text(value, "base_component_vector_hash")
            for value in self.base_component_vector_hashes
        )
        tail_hashes = tuple(
            _canonical_text(value, "tail_component_vector_hash")
            for value in self.tail_component_vector_hashes
        )
        if (
            len(shifts) != ENSEMBLE_SEED_PAIR_COUNT
            or len(base_hashes) != ENSEMBLE_SEED_PAIR_COUNT
            or len(tail_hashes) != ENSEMBLE_SEED_PAIR_COUNT
            or len(set(base_hashes)) != ENSEMBLE_SEED_PAIR_COUNT
            or len(set(tail_hashes)) != ENSEMBLE_SEED_PAIR_COUNT
        ):
            raise ProtocolError("Per-case action shift exact-nine provenance drifted.")
        ensemble_shift = _bounded_utility(
            self.ensemble_mean_absolute_shift,
            "ensemble_mean_absolute_shift",
        )
        if ensemble_shift > float(np.mean(np.asarray(shifts, dtype=np.float64))) + 1.0e-12:
            raise ProtocolError(
                "Per-case ensemble-first shift exceeds its technical-seed bound."
            )
        base_ensemble_hash = _canonical_text(
            self.base_ensemble_probability_hash,
            "base_ensemble_probability_hash",
        )
        tail_ensemble_hash = _canonical_text(
            self.tail_ensemble_probability_hash,
            "tail_ensemble_probability_hash",
        )
        difference_hash = _canonical_text(
            self.ensemble_absolute_difference_hash,
            "ensemble_absolute_difference_hash",
        )
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "support_row_identity_hash", row_hash)
        object.__setattr__(self, "support_row_count", support_row_count)
        object.__setattr__(self, "seed_keys", seed_keys)
        object.__setattr__(self, "per_seed_mean_absolute_shifts", shifts)
        object.__setattr__(self, "base_component_vector_hashes", base_hashes)
        object.__setattr__(self, "tail_component_vector_hashes", tail_hashes)
        object.__setattr__(self, "ensemble_mean_absolute_shift", ensemble_shift)
        object.__setattr__(
            self, "base_ensemble_probability_hash", base_ensemble_hash
        )
        object.__setattr__(
            self, "tail_ensemble_probability_hash", tail_ensemble_hash
        )
        object.__setattr__(
            self, "ensemble_absolute_difference_hash", difference_hash
        )
        object.__setattr__(self, "case_hash", canonical_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_target_support_action_shift_case_v2",
            "target_id": self.target_id,
            "candidate_source": self.candidate_source,
            "case_id": self.case_id,
            "support_row_identity_hash": self.support_row_identity_hash,
            "support_row_count": self.support_row_count,
            "seed_keys": [list(key) for key in self.seed_keys],
            "per_seed_mean_absolute_shifts": list(
                self.per_seed_mean_absolute_shifts
            ),
            "technical_seed_spread_semantics": (
                SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
            ),
            "technical_seed_values_may_feed_model": False,
            "base_component_vector_hashes": list(self.base_component_vector_hashes),
            "tail_component_vector_hashes": list(self.tail_component_vector_hashes),
            "ensemble_mean_absolute_shift": self.ensemble_mean_absolute_shift,
            "base_ensemble_probability_sha256": (
                self.base_ensemble_probability_hash
            ),
            "tail_ensemble_probability_sha256": (
                self.tail_ensemble_probability_hash
            ),
            "ensemble_absolute_difference_sha256": (
                self.ensemble_absolute_difference_hash
            ),
            "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            "scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
            "labels_used": False,
        }



@dataclass(frozen=True)
class EnsembleCandidateFeatureRow:
    """One candidate feature observation after deterministic nine-cell collapse.

    Seed standard deviations are descriptive diagnostics only.  The only
    optional local predictor is the single explicitly named scalar.
    """

    role: str
    outer_target_id: str
    query_id: str
    candidate_source: str
    candidate_source_count: int
    support_partition_hash: str
    support_case_count: int
    seed_row_hashes: tuple[str, ...]
    feature_mean_by_name: Mapping[str, float]
    feature_seed_standard_deviation_by_name: Mapping[str, float]
    target_local_scalar: float | None
    target_local_scalar_name: str | None
    target_local_scalar_semantics: str | None
    target_local_scalar_seed_standard_deviation: float | None
    target_local_scalar_provenance_hash: str | None
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, query, source = _identifiers(
            self.outer_target_id, self.query_id, self.candidate_source
        )
        if self.role not in {INNER_ROLE, TARGET_ROLE}:
            raise ProtocolError("Ensemble candidate feature role is invalid.")
        expected_count = (
            INNER_CANDIDATE_COUNT if self.role == INNER_ROLE else TARGET_CANDIDATE_COUNT
        )
        if (
            isinstance(self.candidate_source_count, bool)
            or not isinstance(self.candidate_source_count, Integral)
            or int(self.candidate_source_count) != expected_count
        ):
            raise ProtocolError("Ensemble candidate cardinality/role drifted.")
        if self.role == INNER_ROLE:
            if outer == query or source in {outer, query}:
                raise ProtocolError("Source-inner ensemble features require distinct H/q/e.")
        elif outer != query or source == outer:
            raise ProtocolError("Fresh-target ensemble features require q == H != e.")
        support_hash = _canonical_text(
            self.support_partition_hash, "support_partition_hash"
        )
        if (
            isinstance(self.support_case_count, bool)
            or not isinstance(self.support_case_count, Integral)
            or int(self.support_case_count) <= 0
        ):
            raise ProtocolError("Support case count must be a positive integer.")
        seed_hashes = tuple(
            _canonical_text(value, "seed_row_hash") for value in self.seed_row_hashes
        )
        if len(seed_hashes) != ENSEMBLE_SEED_PAIR_COUNT or len(set(seed_hashes)) != len(
            seed_hashes
        ):
            raise ProtocolError("Ensemble feature row requires nine unique seed row hashes.")
        means = _validated_feature_mapping(self.feature_mean_by_name, nonnegative=False)
        spread = _validated_feature_mapping(
            self.feature_seed_standard_deviation_by_name, nonnegative=True
        )
        scalar_fields = (
            self.target_local_scalar,
            self.target_local_scalar_name,
            self.target_local_scalar_semantics,
            self.target_local_scalar_seed_standard_deviation,
            self.target_local_scalar_provenance_hash,
        )
        if all(value is None for value in scalar_fields):
            scalar = scalar_name = scalar_semantics = scalar_sd = scalar_hash = None
        elif any(value is None for value in scalar_fields):
            raise ProtocolError("Target-local scalar contract must be wholly present or absent.")
        else:
            scalar = _finite(self.target_local_scalar, "target_local_scalar")
            scalar_name = _canonical_text(
                self.target_local_scalar_name, "target_local_scalar_name"
            )
            scalar_semantics = _canonical_text(
                self.target_local_scalar_semantics,
                "target_local_scalar_semantics",
            )
            scalar_sd = _nonnegative(
                self.target_local_scalar_seed_standard_deviation,
                "target_local_scalar_seed_standard_deviation",
            )
            scalar_hash = _canonical_text(
                self.target_local_scalar_provenance_hash,
                "target_local_scalar_provenance_hash",
            )
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "candidate_source_count", expected_count)
        object.__setattr__(self, "support_partition_hash", support_hash)
        object.__setattr__(self, "support_case_count", int(self.support_case_count))
        object.__setattr__(self, "seed_row_hashes", seed_hashes)
        object.__setattr__(self, "feature_mean_by_name", MappingProxyType(means))
        object.__setattr__(
            self,
            "feature_seed_standard_deviation_by_name",
            MappingProxyType(spread),
        )
        object.__setattr__(self, "target_local_scalar", scalar)
        object.__setattr__(self, "target_local_scalar_name", scalar_name)
        object.__setattr__(self, "target_local_scalar_semantics", scalar_semantics)
        object.__setattr__(
            self, "target_local_scalar_seed_standard_deviation", scalar_sd
        )
        object.__setattr__(self, "target_local_scalar_provenance_hash", scalar_hash)
        object.__setattr__(self, "row_hash", canonical_sha256(self.to_payload()))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    @property
    def seed_pair_count(self) -> int:
        return len(self.seed_row_hashes)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_ensemble_candidate_feature_row_v1"
            ),
            "role": self.role,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "support_case_count": self.support_case_count,
            "seed_pair_count": self.seed_pair_count,
            "seed_row_hashes": list(self.seed_row_hashes),
            "feature_mean_by_name": dict(self.feature_mean_by_name),
            "feature_seed_standard_deviation_by_name": dict(
                self.feature_seed_standard_deviation_by_name
            ),
            "target_local_scalar": self.target_local_scalar,
            "target_local_scalar_name": self.target_local_scalar_name,
            "target_local_scalar_semantics": self.target_local_scalar_semantics,
            "target_local_scalar_seed_standard_deviation": (
                self.target_local_scalar_seed_standard_deviation
            ),
            "target_local_scalar_provenance_hash": (
                self.target_local_scalar_provenance_hash
            ),
            "seed_rows_are_independent_observations": False,
        }


@dataclass(frozen=True)
class EnsembleFeatureSurface:
    role: str
    outer_target_id: str
    candidate_sources: tuple[str, ...]
    rows: tuple[EnsembleCandidateFeatureRow, ...]
    row_keys: tuple[tuple[str, str, str], ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    global_source_control_semantics: str
    global_source_control_provenance_hash: str
    target_local_scalar_name: str | None
    target_local_scalar_semantics: str | None
    permutation_seed: int | None
    surface_hash: str

    @property
    def query_clusters(self) -> tuple[str, ...]:
        return tuple(row.query_id for row in self.rows)

    @property
    def source_clusters(self) -> tuple[str, ...]:
        return tuple(row.candidate_source for row in self.rows)

    @property
    def independent_query_count(self) -> int:
        return len(set(self.query_clusters))



@dataclass(frozen=True)
class TargetEnsembleFeatureProduction:
    target_id: str
    global_source_control: GlobalSourceControl
    case_bootstrap_plan_hash: str
    per_case_shift_surface_hash: str
    point_surface: EnsembleFeatureSurface
    bootstrap_surfaces: tuple[EnsembleFeatureSurface, ...]
    production_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_target_ensemble_feature_production_v1"
            ),
            "target_id": self.target_id,
            "global_source_control_provenance_hash": (
                self.global_source_control.provenance_hash
            ),
            "case_bootstrap_plan_hash": self.case_bootstrap_plan_hash,
            "per_case_shift_surface_hash": self.per_case_shift_surface_hash,
            "point_surface_hash": self.point_surface.surface_hash,
            "bootstrap_surface_hashes": [
                surface.surface_hash for surface in self.bootstrap_surfaces
            ],
            "bootstrap_replicate_count": len(self.bootstrap_surfaces),
            "resampling_unit": "independent_target_support_case",
            "labels_used": False,
            "utility_responses_used": False,
            "production_hash": self.production_hash,
        }


def _validated_feature_mapping(
    values: Mapping[str, float], *, nonnegative: bool
) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(AGGREGATED_FEATURE_NAMES):
        raise ProtocolError("Aggregated feature diagnostics have schema drift.")
    output: dict[str, float] = {}
    for name in AGGREGATED_FEATURE_NAMES:
        output[name] = (
            _nonnegative(values[name], name) if nonnegative else _finite(values[name], name)
        )
    return output




__all__ = (
    "AGGREGATED_FEATURE_NAMES",
    "GLOBAL_SOURCE_CONTROL_NAME",
    "GLOBAL_SOURCE_CONTROL_SEMANTICS",
    "EnsembleCandidateFeatureRow",
    "EnsembleFeatureSurface",
    "GlobalSourceControl",
    "TargetEnsembleFeatureProduction",
    "TargetSupportActionShiftCase",
)
