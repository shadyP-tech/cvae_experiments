"""Deterministic candidate-level aggregation and low-capacity designs."""

from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .ensemble_endpoint_contracts import (
    ENSEMBLE_SEED_KEYS,
    SupportActionProbabilityShift,
)
from .ensemble_feature_contracts import (
    AGGREGATED_FEATURE_NAMES,
    GLOBAL_SOURCE_CONTROL_NAME,
    EnsembleCandidateFeatureRow,
    EnsembleFeatureSurface,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    _canonical_text,
    _finite,
)
from .surface_contracts import CandidateFeatureRow, _immutable_array


def aggregate_candidate_seed_features(
    rows: Sequence[CandidateFeatureRow],
    *,
    legacy_target_local_scalar_name: str | None = None,
    support_action_shift_by_candidate: (
        Mapping[tuple[str, str, str], SupportActionProbabilityShift] | None
    ) = None,
) -> tuple[EnsembleCandidateFeatureRow, ...]:
    """Collapse each exact-nine candidate group into one observation.

    All legacy feature means and seed standard deviations remain diagnostics.
    At most one scalar is admitted as a predictor: either one explicitly named
    legacy feature for compatibility/testing, or the versioned label-free
    support action shift.  The two modes are mutually exclusive.
    """

    if not rows or any(not isinstance(row, CandidateFeatureRow) for row in rows):
        raise ProtocolError("Ensemble aggregation requires typed, nonempty seed rows.")
    if legacy_target_local_scalar_name is not None and support_action_shift_by_candidate is not None:
        raise ProtocolError("Only one target-local scalar source may be configured.")
    legacy_name: str | None = None
    if legacy_target_local_scalar_name is not None:
        legacy_name = _canonical_text(
            legacy_target_local_scalar_name, "legacy_target_local_scalar_name"
        )
        if legacy_name not in AGGREGATED_FEATURE_NAMES:
            raise ProtocolError("Configured legacy target-local scalar is not supported.")
    ordered = tuple(sorted(rows, key=lambda row: row.row_key))
    seed_keys = tuple(
        (row.training_seed, row.generation_seed) for row in ordered
    )
    if len(set(row.row_key for row in ordered)) != len(ordered):
        raise ProtocolError("Candidate seed features contain duplicate cells.")
    by_candidate: dict[tuple[str, str, str], list[CandidateFeatureRow]] = defaultdict(list)
    for row in ordered:
        by_candidate[(row.outer_target_id, row.query_id, row.candidate_source)].append(row)
    configured_shift = dict(support_action_shift_by_candidate or {})
    if configured_shift and set(configured_shift) != set(by_candidate):
        raise ProtocolError("Support action shifts do not align to candidate H/q/e keys.")
    output: list[EnsembleCandidateFeatureRow] = []
    for candidate_key, candidate_rows_raw in sorted(by_candidate.items()):
        candidate_rows = tuple(
            sorted(
                candidate_rows_raw,
                key=lambda row: (row.training_seed, row.generation_seed),
            )
        )
        observed_keys = tuple(
            (row.training_seed, row.generation_seed) for row in candidate_rows
        )
        if observed_keys != ENSEMBLE_SEED_KEYS:
            raise ProtocolError(
                "Every ensemble candidate requires canonical exact-nine seed rows."
            )
        if len({row.role for row in candidate_rows}) != 1:
            raise ProtocolError("Candidate seed feature roles drifted.")
        if len({row.candidate_source_count for row in candidate_rows}) != 1:
            raise ProtocolError("Candidate seed feature cardinality drifted.")
        if len({row.support_partition_hash for row in candidate_rows}) != 1:
            raise ProtocolError("Candidate support partition drifted across seed rows.")
        if len({row.support_case_count for row in candidate_rows}) != 1:
            raise ProtocolError("Candidate support case count drifted across seed rows.")
        means: dict[str, float] = {}
        standard_deviations: dict[str, float] = {}
        for name in AGGREGATED_FEATURE_NAMES:
            values = np.asarray(
                [float(getattr(row, name)) for row in candidate_rows],
                dtype=np.float64,
            )
            means[name] = float(np.mean(values, dtype=np.float64))
            standard_deviations[name] = float(
                np.std(values, ddof=0, dtype=np.float64)
            )
        scalar: float | None = None
        scalar_name: str | None = None
        scalar_semantics: str | None = None
        scalar_standard_deviation: float | None = None
        scalar_provenance_hash: str | None = None
        if legacy_name is not None:
            scalar_values = np.asarray(
                [float(getattr(row, legacy_name)) for row in candidate_rows],
                dtype=np.float64,
            )
            scalar = float(np.mean(scalar_values, dtype=np.float64))
            scalar_name = legacy_name
            scalar_semantics = f"legacy_candidate_seed_mean::{legacy_name}::v1"
            scalar_standard_deviation = float(
                np.std(scalar_values, ddof=0, dtype=np.float64)
            )
            scalar_provenance_hash = canonical_sha256(
                {
                    "schema_version": "midogpp_utility_aligned_legacy_scalar_adapter_v1",
                    "candidate_key": list(candidate_key),
                    "scalar_name": scalar_name,
                    "scalar_semantics": scalar_semantics,
                    "seed_row_hashes": [row.row_hash for row in candidate_rows],
                    "value": scalar,
                    "seed_standard_deviation": scalar_standard_deviation,
                }
            )
        elif configured_shift:
            shift = configured_shift[candidate_key]
            if not isinstance(shift, SupportActionProbabilityShift):
                raise ProtocolError("Target-local action shift must use its typed contract.")
            scalar = shift.value
            scalar_name = shift.scalar_name
            scalar_semantics = shift.scalar_semantics
            scalar_standard_deviation = shift.seed_standard_deviation
            scalar_provenance_hash = shift.shift_hash
        first = candidate_rows[0]
        output.append(
            EnsembleCandidateFeatureRow(
                role=first.role,
                outer_target_id=first.outer_target_id,
                query_id=first.query_id,
                candidate_source=first.candidate_source,
                candidate_source_count=first.candidate_source_count,
                support_partition_hash=first.support_partition_hash,
                support_case_count=first.support_case_count,
                seed_row_hashes=tuple(row.row_hash for row in candidate_rows),
                feature_mean_by_name=means,
                feature_seed_standard_deviation_by_name=standard_deviations,
                target_local_scalar=scalar,
                target_local_scalar_name=scalar_name,
                target_local_scalar_semantics=scalar_semantics,
                target_local_scalar_seed_standard_deviation=(
                    scalar_standard_deviation
                ),
                target_local_scalar_provenance_hash=scalar_provenance_hash,
            )
        )
    return tuple(output)


def build_ensemble_feature_surface(
    rows: Sequence[EnsembleCandidateFeatureRow],
    *,
    global_source_control_by_source: Mapping[str, float],
    global_source_control_semantics: str,
    global_source_control_provenance_hash: str,
) -> EnsembleFeatureSurface:
    """Build M0/M1: one global source control plus zero/one local scalar."""

    if not rows or any(not isinstance(row, EnsembleCandidateFeatureRow) for row in rows):
        raise ProtocolError("Ensemble feature surface requires candidate-level rows.")
    ordered = tuple(sorted(rows, key=lambda row: row.row_key))
    row_keys = tuple(row.row_key for row in ordered)
    if len(set(row_keys)) != len(row_keys):
        raise ProtocolError("Ensemble candidate feature rows contain duplicates.")
    roles = {row.role for row in ordered}
    targets = {row.outer_target_id for row in ordered}
    if len(roles) != 1 or len(targets) != 1:
        raise ProtocolError("One ensemble feature surface requires one role and outer target.")
    role = next(iter(roles))
    outer = next(iter(targets))
    queries = tuple(sorted({row.query_id for row in ordered}))
    if role == INNER_ROLE:
        if len(queries) != TARGET_CANDIDATE_COUNT or outer in queries:
            raise ProtocolError("Source-inner ensemble features require eight pseudoqueries.")
        candidate_sources = queries
        for query in queries:
            query_rows = tuple(row for row in ordered if row.query_id == query)
            expected = {source for source in candidate_sources if source != query}
            if (
                len(query_rows) != INNER_CANDIDATE_COUNT
                or {row.candidate_source for row in query_rows} != expected
            ):
                raise ProtocolError("Source-inner ensemble candidate list is incomplete.")
    elif role == TARGET_ROLE:
        if queries != (outer,):
            raise ProtocolError("Target ensemble features require q == H.")
        candidate_sources = tuple(sorted({row.candidate_source for row in ordered}))
        if len(candidate_sources) != TARGET_CANDIDATE_COUNT or len(ordered) != TARGET_CANDIDATE_COUNT:
            raise ProtocolError("Target ensemble features require eight candidates.")
    else:
        raise ProtocolError("Ensemble feature role is invalid.")
    controls = {str(key): _finite(value, "global_source_control") for key, value in global_source_control_by_source.items()}
    if set(controls) != set(candidate_sources):
        raise ProtocolError("Global source controls do not align to the candidate universe.")
    control_semantics = _canonical_text(
        global_source_control_semantics, "global_source_control_semantics"
    )
    control_hash = _canonical_text(
        global_source_control_provenance_hash,
        "global_source_control_provenance_hash",
    )
    scalar_names = {row.target_local_scalar_name for row in ordered}
    scalar_semantics_values = {row.target_local_scalar_semantics for row in ordered}
    if len(scalar_names) != 1 or len(scalar_semantics_values) != 1:
        raise ProtocolError("Target-local scalar contract drifted across candidates.")
    scalar_name = next(iter(scalar_names))
    scalar_semantics = next(iter(scalar_semantics_values))
    if (scalar_name is None) != (scalar_semantics is None):
        raise ProtocolError("Target-local scalar name/semantics are misaligned.")
    feature_names = (GLOBAL_SOURCE_CONTROL_NAME,)
    matrix_rows: list[list[float]] = []
    if scalar_name is not None:
        feature_names += (f"target_local::{scalar_name}",)
    for row in ordered:
        values = [controls[row.candidate_source]]
        if scalar_name is not None:
            if row.target_local_scalar is None:
                raise ProtocolError("Target-local scalar value is absent.")
            values.append(row.target_local_scalar)
        matrix_rows.append(values)
    matrix = _immutable_array(np.asarray(matrix_rows, dtype=np.float64))
    if matrix.shape != (len(ordered), len(feature_names)) or len(feature_names) > 2:
        raise ProtocolError("Ensemble M0/M1 feature capacity drifted.")
    payload = {
        "schema_version": "midogpp_utility_aligned_ensemble_feature_surface_v1",
        "role": role,
        "outer_target_id": outer,
        "candidate_sources": list(candidate_sources),
        "row_hashes": [row.row_hash for row in ordered],
        "feature_names": list(feature_names),
        "values_sha256": array_sha256(matrix),
        "global_source_control_semantics": control_semantics,
        "global_source_control_provenance_hash": control_hash,
        "target_local_scalar_name": scalar_name,
        "target_local_scalar_semantics": scalar_semantics,
        "predictor_capacity": "one_global_source_control_plus_at_most_one_target_local_scalar",
        "seed_rows_are_independent_observations": False,
        "target_or_query_identity_features_used": False,
        "labels_used": False,
        "permutation_seed": None,
    }
    return EnsembleFeatureSurface(
        role=role,
        outer_target_id=outer,
        candidate_sources=candidate_sources,
        rows=ordered,
        row_keys=row_keys,
        feature_names=feature_names,
        values=matrix,
        global_source_control_semantics=control_semantics,
        global_source_control_provenance_hash=control_hash,
        target_local_scalar_name=scalar_name,
        target_local_scalar_semantics=scalar_semantics,
        permutation_seed=None,
        surface_hash=canonical_sha256(payload),
    )


def cyclically_permute_target_scalar(
    surface: EnsembleFeatureSurface, *, permutation_seed: int
) -> EnsembleFeatureSurface:
    """Cyclically reassign only the one local scalar within each query list."""

    if (
        not isinstance(surface, EnsembleFeatureSurface)
        or surface.permutation_seed is not None
        or len(surface.feature_names) != 2
        or surface.target_local_scalar_name is None
    ):
        raise ProtocolError("Ensemble permutation requires an unpermuted M1 surface.")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, Integral):
        raise ProtocolError("Permutation seed must be an integer.")
    seed = int(permutation_seed)
    values = np.asarray(surface.values, dtype=np.float64).copy()
    by_query: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(surface.rows):
        by_query[row.query_id].append(index)
    for indices in by_query.values():
        ordered_indices = sorted(
            indices, key=lambda index: surface.rows[index].candidate_source
        )
        count = len(ordered_indices)
        if count not in {INNER_CANDIDATE_COUNT, TARGET_CANDIDATE_COUNT}:
            raise ProtocolError("Permutation candidate list cardinality drifted.")
        shift = 1 + (abs(seed) % (count - 1))
        original = values[ordered_indices, 1].copy()
        values[ordered_indices, 1] = np.roll(original, -shift)
    permuted = _immutable_array(values)
    payload = {
        "schema_version": "midogpp_utility_aligned_ensemble_feature_surface_v1",
        "parent_surface_hash": surface.surface_hash,
        "role": surface.role,
        "outer_target_id": surface.outer_target_id,
        "candidate_sources": list(surface.candidate_sources),
        "row_hashes": [row.row_hash for row in surface.rows],
        "feature_names": list(surface.feature_names),
        "values_sha256": array_sha256(permuted),
        "global_source_control_semantics": surface.global_source_control_semantics,
        "global_source_control_provenance_hash": surface.global_source_control_provenance_hash,
        "target_local_scalar_name": surface.target_local_scalar_name,
        "target_local_scalar_semantics": surface.target_local_scalar_semantics,
        "permutation_kind": "cyclic_candidate_source_within_query",
        "permutation_seed": seed,
        "global_source_control_permuted": False,
        "labels_used": False,
    }
    return EnsembleFeatureSurface(
        role=surface.role,
        outer_target_id=surface.outer_target_id,
        candidate_sources=surface.candidate_sources,
        rows=surface.rows,
        row_keys=surface.row_keys,
        feature_names=surface.feature_names,
        values=permuted,
        global_source_control_semantics=surface.global_source_control_semantics,
        global_source_control_provenance_hash=surface.global_source_control_provenance_hash,
        target_local_scalar_name=surface.target_local_scalar_name,
        target_local_scalar_semantics=surface.target_local_scalar_semantics,
        permutation_seed=seed,
        surface_hash=canonical_sha256(payload),
    )


__all__ = (
    "aggregate_candidate_seed_features",
    "build_ensemble_feature_surface",
    "cyclically_permute_target_scalar",
)
