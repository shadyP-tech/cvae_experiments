"""Low-dimensional label-free feature surfaces for exact-tail routing."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from numbers import Integral
from typing import Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    CaseBootstrapReplicate,
)
from .surface_contracts import (
    CandidateFeatureRow,
    FeatureSurface,
    _immutable_array,
)


SOURCE_INDICATOR_PREFIX = "global_source_effect::"
NUMERIC_INTERACTION_FEATURE_NAMES = (
    "reconstruction_mean_within_query_z",
    "reconstruction_coefficient_of_variation",
    "reconstruction_iqr_ratio",
    "kl_mean_within_query_z",
    "kl_coefficient_of_variation",
    "kl_iqr_ratio",
    "replica_disagreement_energy_ratio",
    "distribution_mmd_within_query_z",
    "metadata_similarity",
    "total_energy_rank_fraction",
    "log1p_support_case_count",
)


def build_distributional_feature_surface(
    rows: Sequence[CandidateFeatureRow],
    *,
    case_bootstrap_replicate: CaseBootstrapReplicate | None = None,
) -> FeatureSurface:
    """Construct source-effect and interaction designs without label inputs.

    Normalization is performed independently within every query/seed candidate
    list.  This prevents the seven-source development cardinality from being
    silently standardized together with the eight-source deployment geometry.
    Target and query identifiers are retained only in row keys and are never
    emitted as predictive columns.
    """

    if not rows or any(not isinstance(row, CandidateFeatureRow) for row in rows):
        raise ProtocolError("Feature surface requires typed, nonempty rows.")
    if case_bootstrap_replicate is not None and not isinstance(
        case_bootstrap_replicate, CaseBootstrapReplicate
    ):
        raise ProtocolError("Case bootstrap provenance must use the typed replicate contract.")
    ordered = tuple(sorted(rows, key=lambda row: row.row_key))
    row_keys = tuple(row.row_key for row in ordered)
    if len(set(row_keys)) != len(row_keys):
        raise ProtocolError("Candidate feature rows contain duplicate cells.")
    roles = {row.role for row in ordered}
    targets = {row.outer_target_id for row in ordered}
    if len(roles) != 1 or len(targets) != 1:
        raise ProtocolError(
            "One feature surface must contain exactly one role and outer target."
        )
    role = next(iter(roles))
    outer_target = next(iter(targets))
    expected_count = (
        INNER_CANDIDATE_COUNT if role == INNER_ROLE else TARGET_CANDIDATE_COUNT
    )
    query_ids = tuple(sorted({row.query_id for row in ordered}))
    if role == INNER_ROLE:
        if len(query_ids) != TARGET_CANDIDATE_COUNT or outer_target in query_ids:
            raise ProtocolError(
                "Source-inner features require eight non-target pseudoqueries."
            )
        candidate_sources = query_ids
    elif role == TARGET_ROLE:
        if query_ids != (outer_target,):
            raise ProtocolError("Fresh-target features require one target query.")
        candidate_sources = tuple(sorted({row.candidate_source for row in ordered}))
        if len(candidate_sources) != TARGET_CANDIDATE_COUNT:
            raise ProtocolError("Fresh-target features require eight candidates.")
    else:  # guarded by CandidateFeatureRow, retained as a fail-closed boundary.
        raise ProtocolError("Feature surface role is invalid.")
    if case_bootstrap_replicate is not None:
        if role != TARGET_ROLE or case_bootstrap_replicate.target_id != outer_target:
            raise ProtocolError("Case-bootstrap replicate target/role drifted.")
        if any(
            row.support_partition_hash
            != case_bootstrap_replicate.support_partition_hash
            for row in ordered
        ):
            raise ProtocolError(
                "Bootstrap feature rows do not match the typed sampled-case partition."
            )
        if any(
            row.support_case_count != len(case_bootstrap_replicate.sampled_indices)
            for row in ordered
        ):
            raise ProtocolError("Bootstrap support-case count drifted from its plan.")

    expected_seed_pairs = set(product(TRAINING_SEEDS, GENERATION_SEEDS))
    by_group: dict[tuple[str, int, int], list[CandidateFeatureRow]] = defaultdict(list)
    for row in ordered:
        by_group[(row.query_id, row.training_seed, row.generation_seed)].append(row)
    for query in query_ids:
        query_rows = [row for row in ordered if row.query_id == query]
        if len({row.support_partition_hash for row in query_rows}) != 1:
            raise ProtocolError("Support partition drifted across candidate features.")
        if len({row.support_case_count for row in query_rows}) != 1:
            raise ProtocolError("Support case count drifted across candidate features.")
        expected_sources = (
            tuple(source for source in candidate_sources if source != query)
            if role == INNER_ROLE
            else candidate_sources
        )
        if len(expected_sources) != expected_count:
            raise ProtocolError("Candidate feature cardinality drifted.")
        for source in expected_sources:
            source_seed_pairs = {
                (row.training_seed, row.generation_seed)
                for row in query_rows
                if row.candidate_source == source
            }
            if source_seed_pairs != expected_seed_pairs:
                raise ProtocolError(
                    "Every feature candidate requires all nine paired seed cells."
                )
        if {row.candidate_source for row in query_rows} != set(expected_sources):
            raise ProtocolError("Feature candidate set is incomplete or illegal.")
    for group_rows in by_group.values():
        if len(group_rows) != expected_count:
            raise ProtocolError("Feature replicate candidate list is incomplete.")

    indicator_sources = candidate_sources[1:]
    global_names = tuple(f"{SOURCE_INDICATOR_PREFIX}{source}" for source in indicator_sources)
    interaction_names = global_names + NUMERIC_INTERACTION_FEATURE_NAMES
    global_rows: list[list[float]] = []
    numeric_by_key: dict[tuple[str, str, str, int, int], list[float]] = {}
    for group_key, group_rows in sorted(by_group.items()):
        group = tuple(sorted(group_rows, key=lambda row: row.candidate_source))
        rec_means = np.asarray([row.reconstruction_mean for row in group], dtype=np.float64)
        kl_means = np.asarray([row.kl_mean for row in group], dtype=np.float64)
        mmd = np.asarray([row.distribution_mmd for row in group], dtype=np.float64)
        energy = rec_means + kl_means
        rec_z = _within_group_z(rec_means)
        kl_z = _within_group_z(kl_means)
        mmd_z = _within_group_z(np.log1p(mmd))
        energy_rank = _average_rank_fraction(energy)
        for index, row in enumerate(group):
            rec_denominator = max(abs(row.reconstruction_mean), np.finfo(np.float64).eps)
            kl_denominator = max(abs(row.kl_mean), np.finfo(np.float64).eps)
            energy_denominator = max(abs(float(energy[index])), np.finfo(np.float64).eps)
            numeric_by_key[row.row_key] = [
                float(rec_z[index]),
                row.reconstruction_std / rec_denominator,
                (row.reconstruction_q75 - row.reconstruction_q25) / rec_denominator,
                float(kl_z[index]),
                row.kl_std / kl_denominator,
                (row.kl_q75 - row.kl_q25) / kl_denominator,
                row.replica_disagreement / energy_denominator,
                float(mmd_z[index]),
                row.metadata_similarity,
                float(energy_rank[index]),
                float(np.log1p(row.support_case_count)),
            ]
    interaction_rows: list[list[float]] = []
    for row in ordered:
        source_effect = [
            1.0 if row.candidate_source == source else 0.0
            for source in indicator_sources
        ]
        global_rows.append(source_effect)
        interaction_rows.append([*source_effect, *numeric_by_key[row.row_key]])
    global_values = _immutable_array(np.asarray(global_rows, dtype=np.float64))
    interaction_values = _immutable_array(np.asarray(interaction_rows, dtype=np.float64))
    if global_values.shape != (len(ordered), len(global_names)) or interaction_values.shape != (
        len(ordered),
        len(interaction_names),
    ):
        raise ProtocolError("Utility-aligned feature matrix geometry drifted.")
    payload = {
        "schema_version": "midogpp_utility_aligned_feature_surface_v1",
        "role": role,
        "outer_target_id": outer_target,
        "candidate_sources": list(candidate_sources),
        "row_hashes": [row.row_hash for row in ordered],
        "global_feature_names": list(global_names),
        "interaction_feature_names": list(interaction_names),
        "global_values_sha256": array_sha256(global_values),
        "interaction_values_sha256": array_sha256(interaction_values),
        "target_or_query_identity_features_used": False,
        "labels_used": False,
        "permutation_seed": None,
        "case_bootstrap_replicate_hash": (
            case_bootstrap_replicate.replicate_hash
            if case_bootstrap_replicate is not None
            else None
        ),
    }
    return FeatureSurface(
        role=role,
        outer_target_id=outer_target,
        candidate_sources=candidate_sources,
        rows=ordered,
        row_keys=row_keys,
        global_feature_names=global_names,
        interaction_feature_names=interaction_names,
        global_values=global_values,
        interaction_values=interaction_values,
        permutation_seed=None,
        case_bootstrap_replicate=case_bootstrap_replicate,
        surface_hash=canonical_sha256(payload),
    )


def permute_interaction_features(
    surface: FeatureSurface,
    *,
    permutation_seed: int,
) -> FeatureSurface:
    """Apply a deterministic cyclic source-identity negative control.

    Candidate-source global effects remain attached to source identity.  Only
    the label-free target-interaction columns are cyclically reassigned within
    each query/seed candidate list.  No utility value or evaluation label is
    accepted or read by this operation.
    """

    if not isinstance(surface, FeatureSurface) or surface.permutation_seed is not None:
        raise ProtocolError("Interaction permutation requires an unpermuted feature surface.")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, Integral):
        raise ProtocolError("Permutation seed must be an integer.")
    seed = int(permutation_seed)
    global_width = len(surface.global_feature_names)
    values = np.asarray(surface.interaction_values, dtype=np.float64).copy()
    by_group: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(surface.rows):
        by_group[(row.query_id, row.training_seed, row.generation_seed)].append(index)
    for indices in by_group.values():
        ordered_indices = sorted(indices, key=lambda index: surface.rows[index].candidate_source)
        count = len(ordered_indices)
        if count not in {INNER_CANDIDATE_COUNT, TARGET_CANDIDATE_COUNT}:
            raise ProtocolError("Permutation candidate list cardinality drifted.")
        shift = 1 + (abs(seed) % (count - 1))
        original = values[ordered_indices, global_width:].copy()
        for destination, source_index in enumerate(
            np.roll(np.arange(count, dtype=np.int64), -shift)
        ):
            values[ordered_indices[destination], global_width:] = original[source_index]
    permuted = _immutable_array(values)
    payload = {
        "schema_version": "midogpp_utility_aligned_feature_surface_v1",
        "parent_surface_hash": surface.surface_hash,
        "role": surface.role,
        "outer_target_id": surface.outer_target_id,
        "candidate_sources": list(surface.candidate_sources),
        "row_hashes": [row.row_hash for row in surface.rows],
        "global_feature_names": list(surface.global_feature_names),
        "interaction_feature_names": list(surface.interaction_feature_names),
        "global_values_sha256": array_sha256(surface.global_values),
        "interaction_values_sha256": array_sha256(permuted),
        "permutation_kind": "cyclic_source_identity_within_query_seed_list",
        "permutation_seed": seed,
        "case_bootstrap_replicate_hash": (
            surface.case_bootstrap_replicate.replicate_hash
            if surface.case_bootstrap_replicate is not None
            else None
        ),
        "target_or_query_identity_features_used": False,
        "labels_used": False,
    }
    return FeatureSurface(
        role=surface.role,
        outer_target_id=surface.outer_target_id,
        candidate_sources=surface.candidate_sources,
        rows=surface.rows,
        row_keys=surface.row_keys,
        global_feature_names=surface.global_feature_names,
        interaction_feature_names=surface.interaction_feature_names,
        global_values=surface.global_values,
        interaction_values=permuted,
        permutation_seed=seed,
        case_bootstrap_replicate=surface.case_bootstrap_replicate,
        surface_hash=canonical_sha256(payload),
    )


def _within_group_z(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values, dtype=np.float64))
    rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
    if rms <= np.sqrt(np.finfo(np.float64).eps):
        return np.zeros_like(centered)
    return centered / rms


def _average_rank_fraction(values: np.ndarray) -> np.ndarray:
    count = len(values)
    if count < 2:
        raise ProtocolError("Feature ranking requires at least two candidates.")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(count, dtype=np.float64)
    cursor = 0
    while cursor < count:
        end = cursor + 1
        while end < count and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * float(cursor + end - 1)
        cursor = end
    return ranks / float(count - 1)


__all__ = (
    "NUMERIC_INTERACTION_FEATURE_NAMES",
    "SOURCE_INDICATOR_PREFIX",
    "build_distributional_feature_surface",
    "permute_interaction_features",
)
